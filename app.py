"""
Ollama chat app: free chat + Q&A over uploaded documents (PDF / TXT / MD).

How document Q&A works (simple RAG, no extra dependencies):
  1. /upload extracts text page by page and splits it into overlapping chunks.
  2. On each question, the chunks of the selected documents are ranked with
     BM25 (keyword relevance) against the question.
  3. The best chunks are put in a system prompt, with [file p.N] labels so the
     model can cite pages.
Small documents are sent whole, so "summarize this" style questions also work.
"""

import json
import logging
import math
import os
import re
import threading
import uuid
from collections import Counter

import requests
from flask import Flask, Response, jsonify, render_template, request
from pypdf import PdfReader

# ── Config (override with environment variables) ─────────────────────────────
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"
MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
NUM_CTX = int(os.environ.get("NUM_CTX", 8192))          # model context window (tokens)

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", 20))
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}

CHUNK_SIZE = 1000            # characters per chunk
CHUNK_OVERLAP = 150          # characters shared between neighbouring chunks
TOP_K = 5                    # max chunks sent per question
MAX_CONTEXT_CHARS = 6000     # ~1.5k tokens of document text per question
MAX_HISTORY = 20             # last N messages sent to the model

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ollama-chat")

# In-memory document store: {doc_id: {...}}. Cleared when the server restarts.
DOCS = {}
DOCS_LOCK = threading.Lock()


# ── Text processing ──────────────────────────────────────────────────────────
TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text):
    return [t for t in TOKEN_RE.findall(text.lower()) if len(t) > 1]


def normalize(text):
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into ~size-char chunks, preferring paragraph/sentence breaks."""
    text = normalize(text)
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            for sep in ("\n\n", ". ", "\n", " "):
                cut = text.rfind(sep, start + size // 2, end)
                if cut != -1:
                    end = cut + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        # Start the next chunk `overlap` chars back, snapped to a word boundary
        nxt = max(end - overlap, start + 1)
        space = text.find(" ", nxt, end)
        start = space + 1 if space != -1 else nxt
    return chunks


def extract_pages(file_storage, ext):
    """Return [(page_number, text), ...] for an uploaded file."""
    if ext == ".pdf":
        reader = PdfReader(file_storage.stream)
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("This PDF is password-protected.")
        # extract_text() can return None (e.g. image-only pages)
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]

    raw = file_storage.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    return [(1, text)]


def retrieve(query, docs):
    """Pick the chunks most relevant to `query` (BM25), within the context budget."""
    chunks = [c for d in docs for c in d["chunks"]]
    if not chunks:
        return []

    # Small enough? Send everything, in document order.
    if sum(len(c["text"]) for c in chunks) <= MAX_CONTEXT_CHARS:
        return chunks

    q_terms = set(tokenize(query))
    n = len(chunks)
    avgdl = sum(c["length"] for c in chunks) / n
    df = Counter()
    for c in chunks:
        df.update(c["tf"].keys())

    k1, b = 1.5, 0.75
    scored = []
    for c in chunks:
        score = 0.0
        for term in q_terms:
            tf = c["tf"].get(term, 0)
            if not tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * c["length"] / avgdl))
        scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    candidates = [c for s, c in scored if s > 0] or chunks  # no match → start of docs

    selected, used = [], 0
    for c in candidates:
        if len(selected) >= TOP_K or used + len(c["text"]) > MAX_CONTEXT_CHARS:
            break
        selected.append(c)
        used += len(c["text"])

    # Present excerpts in reading order, not score order
    selected.sort(key=lambda c: (c["doc_order"], c["index"]))
    return selected


def build_system_prompt(chunks):
    excerpts = "\n\n".join(f"[{c['doc_name']} p.{c['page']}]\n{c['text']}" for c in chunks)
    return (
        "You are a helpful assistant answering questions about the user's documents.\n"
        "Use the document excerpts below as your main source. If the answer is not in "
        "them, say so clearly instead of guessing. When you use an excerpt, cite it "
        "like [file.pdf p.3].\n\n"
        f"=== DOCUMENT EXCERPTS ===\n{excerpts}\n=== END EXCERPTS ==="
    )


def clean_messages(raw):
    """Keep only well-formed user/assistant messages, trimmed to MAX_HISTORY."""
    if not isinstance(raw, list):
        return []
    msgs = [
        {"role": m["role"], "content": m["content"]}
        for m in raw
        if isinstance(m, dict)
        and m.get("role") in ("user", "assistant")
        and isinstance(m.get("content"), str)
        and m["content"].strip()
    ]
    msgs = msgs[-MAX_HISTORY:]
    while msgs and msgs[0]["role"] != "user":   # history should start with the user
        msgs.pop(0)
    return msgs


def public_doc(doc):
    return {k: doc[k] for k in ("id", "name", "pages", "chunks_count", "chars")}


def ndjson(obj):
    return json.dumps(obj, ensure_ascii=False) + "\n"


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", model=MODEL, max_upload_mb=MAX_UPLOAD_MB)


@app.post("/upload")
def upload():
    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify({"error": "No file uploaded"}), 400

    name = os.path.basename(f.filename.replace("\\", "/"))
    ext = os.path.splitext(name)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"}), 400

    try:
        pages = extract_pages(f, ext)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        log.exception("Failed to read %s", name)
        return jsonify({"error": "Could not read this file. Is it a valid, non-corrupted document?"}), 400

    with DOCS_LOCK:
        doc_order = len(DOCS)
    doc_id = uuid.uuid4().hex[:12]
    chunks = []
    for page_no, text in pages:
        for piece in chunk_text(text):
            tokens = tokenize(piece)
            chunks.append({
                "doc_id": doc_id, "doc_name": name, "doc_order": doc_order,
                "page": page_no, "index": len(chunks), "text": piece,
                "tf": Counter(tokens), "length": max(len(tokens), 1),
            })

    if not chunks:
        return jsonify({"error": "No text found. If this is a scanned PDF, it needs OCR first."}), 422

    doc = {
        "id": doc_id, "name": name, "pages": len(pages), "chunks": chunks,
        "chunks_count": len(chunks), "chars": sum(len(c["text"]) for c in chunks),
    }
    with DOCS_LOCK:
        DOCS[doc_id] = doc
    log.info("Indexed %s: %d pages, %d chunks", name, doc["pages"], len(chunks))
    return jsonify(public_doc(doc)), 201


@app.get("/documents")
def list_documents():
    with DOCS_LOCK:
        return jsonify([public_doc(d) for d in DOCS.values()])


@app.delete("/documents/<doc_id>")
def delete_document(doc_id):
    with DOCS_LOCK:
        if DOCS.pop(doc_id, None) is None:
            return jsonify({"error": "Document not found"}), 404
    return jsonify({"ok": True})


@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    messages = clean_messages(data.get("messages"))
    if not messages or messages[-1]["role"] != "user":
        return jsonify({"error": "The last message must be a non-empty user message."}), 400

    doc_ids = data.get("doc_ids") or []
    with DOCS_LOCK:
        docs = [DOCS[d] for d in doc_ids if isinstance(d, str) and d in DOCS]

    sources = []
    if docs:
        # Include the previous question so follow-ups ("explain more") still retrieve well
        user_turns = [m["content"] for m in messages if m["role"] == "user"]
        query = " ".join(user_turns[-2:])
        chunks = retrieve(query, docs)
        messages = [{"role": "system", "content": build_system_prompt(chunks)}] + messages
        seen = set()
        for c in chunks:
            key = (c["doc_name"], c["page"])
            if key not in seen:
                seen.add(key)
                sources.append({"name": c["doc_name"], "page": c["page"]})

    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": True,
        "options": {"num_ctx": NUM_CTX},
    }

    def generate():
        yield ndjson({"type": "sources", "sources": sources})
        try:
            # (connect timeout, read timeout between chunks)
            with requests.post(OLLAMA_CHAT_URL, json=payload, stream=True, timeout=(5, 300)) as r:
                if r.status_code != 200:
                    try:
                        err = r.json().get("error", r.text)
                    except ValueError:
                        err = r.text
                    if r.status_code == 404 and "not found" in err:
                        err += f" — run: ollama pull {MODEL}"
                    yield ndjson({"type": "error", "message": f"Ollama error ({r.status_code}): {err}"})
                    return

                for line in r.iter_lines():
                    if not line:
                        continue
                    part = json.loads(line)
                    if "error" in part:
                        yield ndjson({"type": "error", "message": part["error"]})
                        return
                    token = part.get("message", {}).get("content", "")
                    if token:
                        yield ndjson({"type": "token", "content": token})
                    if part.get("done"):
                        break
            yield ndjson({"type": "done"})

        except requests.exceptions.ConnectionError:
            yield ndjson({"type": "error", "message": "Ollama is not running. Start it with: ollama serve"})
        except requests.exceptions.Timeout:
            yield ndjson({"type": "error", "message": "The model timed out. Try a shorter message or restart Ollama."})
        except Exception as e:
            log.exception("Chat failed")
            yield ndjson({"type": "error", "message": f"Unexpected error: {e}"})

    return Response(
        generate(),   # uses only local variables, so no request context needed
        mimetype="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": f"File too large (max {MAX_UPLOAD_MB} MB)."}), 413


if __name__ == "__main__":
    print(f"Starting chat app with model: {MODEL}")
    print("Make sure Ollama is running: ollama serve")
    # Debug mode lets anyone who reaches the page run code — keep it opt-in.
    app.run(host="127.0.0.1", port=5000, debug=os.environ.get("FLASK_DEBUG") == "1", threaded=True)