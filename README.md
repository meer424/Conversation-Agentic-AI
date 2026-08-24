# Lab Manual Q&A Agent

A local, single-user Streamlit app that answers questions about an uploaded lab manual PDF, grounded strictly in its content, with page-level citations.

Full spec, architecture, and build rules: see [`AGENTS.md`](./AGENTS.md). If you're using Cursor, it also reads `.cursor/rules/project-rules.mdc` automatically on every request — you don't need to paste anything in manually.

## Project status

This repo is a **scaffold**, not a finished app. Every file under `src/` currently has real function signatures and docstrings but `raise NotImplementedError` bodies — that's intentional (see AGENTS.md §4, "interface-first"). Implement modules one at a time, in this order:

```
src/utils.py -> src/pdf_loader.py -> src/chunking.py -> src/vectorstore.py -> src/retrieval.py -> src/agent.py -> app.py
```

## Setup

### 1. Create and activate a virtual environment

macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

Windows (PowerShell):
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure your API key
```bash
cp .env.example .env
# then edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

### 4. Run the app
```bash
streamlit run app.py
```

## Working on this in Cursor

1. Open this folder in Cursor (`File > Open Folder`).
2. Open Agent/Composer mode.
3. Ask it to implement one module at a time — e.g. *"Implement `src/utils.py` per its docstrings, then show me the diff before moving on."* Don't ask it to implement everything in one shot; the build order above exists so each layer can be tested before the next depends on it.
4. After each module, actually run something against it (a quick script, or fill in the matching test in `tests/test_pipeline.py`) before moving to the next file.
5. Once `src/agent.py` works standalone, wire the `TODO`/`NotImplementedError` handlers in `app.py` one at a time and test in the browser after each.

## Troubleshooting

- **ChromaDB / sqlite version errors on import** — Chroma needs a reasonably modern `sqlite3`. On some Linux distros you may need `pip install pysqlite3-binary` and to alias it before importing chromadb; check ChromaDB's docs if you hit this.
- **`sentence-transformers` install is slow / large** — it pulls in `torch`. This is expected for local embeddings; there's no way around the download on first install.
- **First query after a fresh upload is slow** — embedding the manual and downloading the embedding model (first run only) both happen before the first answer. Subsequent runs on the same file reuse the cached Chroma collection (by content hash) and are fast.
- **"I couldn't find this in the uploaded lab manual" for something that's clearly in the PDF** — check `app.log` for what was actually extracted; scanned/image-only PDFs are detected and warned about, not OCR'd.
