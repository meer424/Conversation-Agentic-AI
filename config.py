"""Central configuration for the Lab Manual Q&A Agent.

All tunable values live here so no module hardcodes a constant. Values are
read from environment variables (via .env) with sensible defaults, so the
app runs out of the box in MVP mode and can be hardened later by flipping
the two retrieval flags.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths -------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent
UPLOADED_PDFS_DIR: Path = BASE_DIR / "data" / "uploaded_pdfs"
VECTOR_STORE_DIR: Path = BASE_DIR / "data" / "vector_store"

UPLOADED_PDFS_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# --- LLM -----------------------------------------------------------------
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
# Maximum output tokens for Gemini responses.
MAX_RESPONSE_TOKENS: int = int(os.getenv("MAX_RESPONSE_TOKENS", "8096"))
# How many prior user/assistant turns to keep in the conversation buffer.
CONVERSATION_MEMORY_TURNS: int = int(os.getenv("CONVERSATION_MEMORY_TURNS", "6"))

# --- Embeddings ------------------------------------------------------------
EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
RERANKER_MODEL_NAME: str = os.getenv(
    "RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# --- Chunking --------------------------------------------------------------
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))

# --- Retrieval ---------------------------------------------------------
# Phase 1 (MVP): plain dense top-k similarity search.
RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "6"))

# Phase 2 (hardening): hybrid dense+BM25 retrieval fused via reciprocal
# rank fusion, then optionally reranked with a cross-encoder. Both are
# off by default so the MVP path is what you validate first. Flip these
# in .env only after Phase 1 works end-to-end. See AGENTS.md section 3.
ENABLE_HYBRID_RETRIEVAL: bool = os.getenv("ENABLE_HYBRID_RETRIEVAL", "false").lower() == "true"
ENABLE_RERANK: bool = os.getenv("ENABLE_RERANK", "false").lower() == "true"
HYBRID_CANDIDATE_POOL: int = int(os.getenv("HYBRID_CANDIDATE_POOL", "25"))
RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "8"))

# --- Logging -------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE: Path = BASE_DIR / "app.log"
