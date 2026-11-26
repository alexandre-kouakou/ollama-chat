# Ollama Chat — local chatbot with document Q&A

A lightweight Flask web app to chat with a local LLM through [Ollama](https://ollama.com), and to ask questions about your own documents (PDF, TXT, MD). Everything runs on your machine: no API keys, no data sent to the cloud.

## Features

- 💬 **Free chat** with any Ollama model (default: `gemma3:4b`)
- 📄 **Document Q&A**: upload PDF/TXT/MD files with the 📎 button or drag & drop
- 🔎 **Retrieval (RAG)**: documents are split into chunks and the most relevant ones are picked with BM25, then given to the model
- 📌 **Page citations**: answers cite sources like `[manual.pdf p.3]`
- ⚡ **Streaming** replies, with a Stop button
- 🎛️ Turn individual documents on/off per question

## How it works

```
Upload ──► extract text per page (pypdf) ──► split into overlapping chunks
                                                     │
Question ──► BM25 ranks chunks ──► top chunks + question ──► Ollama ──► streamed answer
```

Small documents are sent whole, so questions like "summarize this" also work.

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com/download) installed and running

## Installation

```bash
git clone https://github.com/alexandre-kouakou/ollama-chat.git
cd ollama-chat

python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
ollama pull gemma3:4b
```

## Usage

```bash
ollama serve        # if Ollama isn't already running
python app.py
```

Open http://127.0.0.1:5000

## Configuration

Optional environment variables:

| Variable          | Default                  | Description                       |
|-------------------|--------------------------|-----------------------------------|
| `OLLAMA_MODEL`    | `gemma3:4b`              | Any model you have pulled         |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Where Ollama is running           |
| `NUM_CTX`         | `8192`                   | Model context window (tokens)     |
| `MAX_UPLOAD_MB`   | `20`                     | Maximum upload size               |
| `FLASK_DEBUG`     | off                      | Set to `1` for Flask debug mode   |

Example: `OLLAMA_MODEL=llama3.2 python app.py`

## Project structure

```
ollama-chat/
├── app.py              # Flask backend: upload, retrieval, streaming chat
├── templates/
│   └── index.html      # Chat UI (HTML/CSS/JS, no framework)
├── requirements.txt
└── README.md
```

## Limitations

- Uploaded documents are kept in memory and are cleared when the server restarts.
- Scanned (image-only) PDFs need OCR first.
- Retrieval is keyword-based (BM25); semantic search with embeddings is a planned improvement.

## License

MIT, see [LICENSE](LICENSE).
