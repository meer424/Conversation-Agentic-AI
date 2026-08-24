# AGENTS.md — Lab Manual Q&A Agent

This file is the single source of truth for any AI coding agent (Cursor Agent/Composer, Claude Code, etc.) working in this repository. Read this in full before writing or editing any file. If a request from the user conflicts with this file, ask before proceeding.

---

## 1. Product Definition

**What we're building:** A local, single-user Streamlit app that ingests one or more lab-manual PDFs and answers questions grounded strictly in that content, with page-level citations. Not a general chatbot. Not a multi-tenant SaaS product.

**Primary user:** A student/instructor running this on their own machine, uploading their own manual, asking things like "what's the procedure for titration in experiment 3?" or "what safety equipment does experiment 5 require?"

### 1.1 What the market already does well (informed the design below)
Researched against ChatPDF, Humata, AskYourPDF, NotebookLM, Atlas, and Paperguide (2026):
- Every credible tool shows **page-level, clickable citations** next to the answer — never a bare answer.
- The **#1 trust-killer across the category is a confident wrong answer with a plausible-looking citation**. Faithfulness has to be enforced in code, not just requested in the prompt.
- Side-by-side "source panel + chat panel" is the dominant UX pattern for single-document work.
- Multi-document workspaces ("chat with a collection") is a common upgrade path — build for it from day one, even in single-manual MVP.
- Retrieval quality — not the LLM — is the dominant driver of answer quality. Fixed-size-only chunking and pure dense search are known weak points for technical/procedural documents (numbered steps, apparatus names, exact chemical/part names).
- Practically all serious tools stream the response token-by-token — no full-blob wait state.
- A clean "I don't have this in the manual" fallback is a feature, not a failure — copy this pattern exactly.

### 1.2 What we are explicitly NOT building (yet)
User accounts, multi-tenant permissions, billing, cloud deployment, OCR for scanned/image PDFs (detect and warn instead), mobile app. Keep scope to a single local Streamlit session.

---

## 2. Tech Stack (fixed — do not substitute without asking)

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.11+ | |
| Frontend | Streamlit | `st.chat_message` / `st.chat_input`, no custom JS framework |
| Orchestration | LangChain (`langchain`, `langchain-community`, `langchain-anthropic`) | |
| LLM | Anthropic Claude via `anthropic` SDK | Model set via `.env` (`CLAUDE_MODEL`), default `claude-sonnet-5`. Do not hardcode the model string inside logic files — always read from `config.py`. Check Anthropic's docs for the current model list before assuming a string is valid. |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`), local, free | Swappable via config for API-based embeddings later |
| Vector store | ChromaDB, local persistent client | One collection per file-hash, so re-uploading the same PDF reuses the cached index |
| Keyword search | `rank_bm25` | Used for hybrid retrieval — see §3 |
| Reranking | `sentence-transformers` cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) | Local, no extra API cost — see §3 |
| PDF parsing | `pdfplumber` primary, `pypdf` fallback | |
| Env | `python-dotenv` | |

No FastAPI, no Docker, no database beyond Chroma's local persistence, no auth libraries. If you think a new dependency is needed, stop and ask — do not silently add it to `requirements.txt`.

---

## 3. Retrieval Architecture (this is the part most tutorials get wrong — follow it precisely)

Ship this in two phases. **Phase 1 (MVP) must work end-to-end before Phase 2 starts.**

**Phase 1 — MVP (dense-only):**
```
PDF → per-page extraction (page number kept as metadata)
    → RecursiveCharacterTextSplitter (chunk_size=1000, overlap=150, configurable)
    → embed with all-MiniLM-L6-v2 → Chroma persistent collection
Query → embed → similarity_search(k=6) → build context block → Claude → answer + citations
```

**Phase 2 — Hardening (hybrid + rerank), behind a `config.py` flag `ENABLE_HYBRID_RETRIEVAL` and `ENABLE_RERANK` so either can be toggled off for debugging:**
```
Query → [dense top-25 via Chroma]  +  [sparse top-25 via BM25 over chunk texts]
      → Reciprocal Rank Fusion → top-15 candidates
      → cross-encoder rerank (query, chunk) pairs → top 5–8
      → context builder (chunk text + source filename + page number)
      → Claude (streaming) → answer
```
Rationale: lab manuals are procedural/technical text — exact apparatus names, chemical formulas, and numbered steps are exactly what dense-only embeddings miss and BM25 catches. This is not gold-plating; it is the documented biggest lever for this document type.

**Chunking rule specific to lab manuals:** never let the splitter's overlap logic be the only thing standing between a chunk and a broken numbered step. Prefer splitting on paragraph/section boundaries before falling back to raw character count. If a chunk starts mid-step (e.g., begins with "3." with no preceding context), that's a bug, not a quirk.

**Faithfulness guardrail (mandatory, implemented in code, not just prompted):**
After the LLM returns an answer with cited page numbers, verify each cited page number actually appears in the metadata of the chunks that were retrieved for that query. If the model cites a page that was never retrieved, strip that citation and flag it in logs — do not silently trust it. This is the single most important correctness check in the whole app; do not skip it to save time.

---

## 4. Project Structure

```
lab_manual_agent/
├── AGENTS.md                    # this file
├── .cursor/rules/project-rules.mdc
├── app.py                       # Streamlit entry point
├── config.py                    # all constants, model names, chunk sizes, flags
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── src/
│   ├── __init__.py
│   ├── pdf_loader.py            # extract text + page metadata
│   ├── chunking.py              # structure-aware splitting
│   ├── vectorstore.py           # Chroma build/load/persist, embedding, caching by file hash
│   ├── retrieval.py             # dense + BM25 hybrid + rerank + fusion
│   ├── agent.py                 # prompt templates, Claude calls, streaming, citation verification
│   └── utils.py                 # file hashing, logging setup, helpers
├── data/
│   ├── uploaded_pdfs/            # gitignored
│   └── vector_store/              # gitignored, Chroma persistence
└── tests/
    └── test_pipeline.py
```

Every `src/*.py` file in this scaffold currently contains **function signatures, type hints, and docstrings only** — no implementation. This is intentional (interface-first). Implement one module at a time, in this order: `utils.py` → `pdf_loader.py` → `chunking.py` → `vectorstore.py` → `retrieval.py` → `agent.py` → `app.py`. Don't jump ahead to `app.py` before the pipeline underneath it actually works — test each module from a plain Python shell or `tests/test_pipeline.py` before wiring it into Streamlit.

---

## 5. Functional Requirements

### 5.1 Ingestion (`pdf_loader.py`, `chunking.py`, `vectorstore.py`)
- Accept multiple PDFs via `st.file_uploader(accept_multiple_files=True)`.
- Extract text per page; keep `page_number` + `source_filename` on every chunk.
- Detect empty/near-empty extraction (scanned/image PDF) → surface a clear warning in the UI, do not crash, do not silently index nothing.
- Hash uploaded file bytes; if a matching Chroma collection already exists on disk, reuse it instead of recomputing embeddings.
- Wrap ingestion in `st.cache_resource` appropriately so Streamlit reruns don't recompute embeddings.

### 5.2 Retrieval + Agent (`retrieval.py`, `agent.py`)
- Implement Phase 1 first, confirm it works, then add Phase 2 behind the config flags.
- System prompt must instruct Claude to: answer only from provided context; explicitly say "I couldn't find this in the uploaded lab manual" when context is insufficient; cite source filename + page number(s) at the end of every answer; give step-by-step structure when the manual describes a procedure.
- Maintain short conversational memory (last N turns, configurable) so follow-ups like "what about step 3?" resolve correctly.
- Stream the response.
- Run the faithfulness guardrail from §3 before displaying the answer.

### 5.3 Streamlit UI (`app.py`)
- Sidebar: multi-file uploader, "Process Manual(s)" button with progress spinner, list of currently indexed documents, "Clear Knowledge Base" button, masked API key input if not found in `.env`.
- Main: chat interface (`st.chat_message`, `st.chat_input`), session-persisted history, streamed responses.
- Under each assistant answer: an expandable "📄 Sources" section listing the exact chunks/pages used — this is not optional, it's the app's main trust signal per §1.1.
- Empty state: friendly instructions when no PDF is uploaded yet.
- All failure states (bad PDF, missing API key, empty extraction, API error) surface as `st.error`/`st.warning` with a specific, actionable message — never a raw traceback in the UI (full traceback goes to the log file instead).

### 5.4 Stretch (only after 5.1–5.3 are solid and tested)
- Multi-manual scoping (dropdown: one manual vs. all).
- Export chat transcript.
- "Quiz me" mode generating practice questions from the manual.
- Highlight the retrieved passage inline, not just cite the page number.

---

## 6. Negative Prompting — Things You Must NOT Do

This section exists because generic AI-generated scaffolds tend to accumulate the same three failure categories. Treat every item below as a hard constraint, not a style suggestion.

### 6.1 Unnecessary code / over-engineering
- No unused imports, dead code paths, or commented-out blocks left in committed files.
- No premature abstraction — e.g., don't build a generic "LLM provider plugin system" for a single-provider app; a simple config-driven switch is enough.
- No auth, user accounts, multi-tenancy, Docker, CI/CD, or cloud deployment scaffolding — out of scope per §1.2.
- No hand-rolled PDF parser, hand-rolled text splitter, or hand-rolled vector index — the chosen libraries already do this; use them.
- No adding a dependency that isn't already in `requirements.txt` without flagging it first.
- No speculative feature flags for features not listed in §5.

### 6.2 Broken endpoints / non-functional code
- No function that returns a hardcoded or mock value "for now" — if a function is included, it must actually work against a real PDF, the real Chroma store, and the real Claude API.
- No bare `except: pass` or `except Exception: pass` — every caught exception must be logged and, where user-facing, surfaced in the UI with a specific message.
- No `# TODO: implement later` left in code you claim is done. If something can't be finished, say so explicitly and explain why, rather than leaving a silent stub.
- No UI element (button, uploader, input) that isn't wired to a real, working handler.
- Before saying a feature works, actually run it (`streamlit run app.py`, or the relevant test) — do not assert success without execution.

### 6.3 AI slop
- No comments that just restate the function name (`# process the PDF` above `def process_pdf(...)`).
- No generic variable names — use domain terms (`retrieved_chunks`, `chunk_metadata`, `claude_response`, not `data`/`temp`/`result`).
- No copy-pasted tutorial boilerplate that doesn't match this repo's actual module layout.
- No inventing library functions or APIs that don't exist — verify with `pip show <package>` before importing something you're not 100% sure is installed.
- No padding the README or docstrings with marketing language ("revolutionary," "seamless," "cutting-edge"). Describe what the code does, plainly.
- No emoji in code, comments, or commit messages (the 📄/🧪 in the UI copy is the one intentional exception, per §5.3 — everywhere else, skip it).

---

## 7. Agent Working Rules (process, not code)

- **Interface-first for anything non-trivial:** when adding a new module or nontrivial function, propose the function signature + docstring first, get it confirmed, then implement.
- **Checkpoint before refactors:** before any multi-file change, `git add -A && git commit -m "checkpoint: before <task>"` so changes are easy to roll back.
- **Never delete or overwrite `.env`, `requirements.txt`, or anything in `data/`** without explicit confirmation.
- **Test as you go:** after implementing a `src/` module, exercise it directly (small script or `tests/test_pipeline.py`) before wiring it into `app.py`. Don't discover a broken vector store inside a Streamlit rerun loop.
- **Be concise in explanations.** Don't narrate standard library usage; explain the non-obvious decisions only (e.g., why hybrid retrieval, why the faithfulness check).

---

## 8. Code Quality Bar

- Type hints on every function signature.
- Google-style docstrings on every module and public function.
- `logging` module configured in `utils.py`, used across ingestion and query paths (not `print`).
- No hardcoded secrets — everything from `.env` via `python-dotenv`.
- Modular imports — `app.py` orchestrates, `src/` does the work. No business logic embedded directly in Streamlit callback bodies beyond thin glue code.

---

## 9. Definition of Done (MVP)

You're done with the MVP when: a user can upload a real lab manual PDF, click "Process," ask "what safety precautions are listed for experiment 2," get a streamed answer citing the correct filename + page number, get an honest "not found" for a question the manual doesn't answer, and re-upload the same PDF without waiting through re-embedding. If any of these five don't work, the MVP isn't done — regardless of how much other code exists.
