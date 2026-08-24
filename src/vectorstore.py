"""Chroma persistent vector store: build, load, and cache by content hash.

A collection name is derived from the hash of the uploaded file set (see
src/utils.py) so re-uploading the same manual(s) reuses the existing
embeddings instead of recomputing them.
"""

import config
from src.chunking import Chunk
from src.utils import get_logger

logger = get_logger(__name__)


def get_embedding_function():
    """Return a LangChain-compatible embedding function backed by
    sentence-transformers (config.EMBEDDING_MODEL_NAME).

    Should be constructed once and reused — do not reload the model on
    every call. Cache with st.cache_resource at the call site in app.py,
    not inside this function.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)


def _get_chroma_client():
    """Return a persistent Chroma client pointed at config.VECTOR_STORE_DIR."""
    import chromadb

    return chromadb.PersistentClient(path=str(config.VECTOR_STORE_DIR))


def collection_exists(collection_name: str) -> bool:
    """Return True if a persisted Chroma collection with this name already
    exists on disk under config.VECTOR_STORE_DIR.
    """
    client = _get_chroma_client()
    existing = [col.name for col in client.list_collections()]
    return collection_name in existing


def build_vectorstore(chunks: list[Chunk], collection_name: str):
    """Embed all chunks and persist them into a new Chroma collection.

    Requirements:
    - Store chunk_id, source_filename, and page_number as metadata on
      every embedded document — the agent's citation logic depends on
      this metadata being present and correct.
    - Persist to config.VECTOR_STORE_DIR so the collection survives
      across Streamlit reruns and app restarts.
    - Return the Chroma collection handle for immediate use.
    """
    from langchain_chroma import Chroma

    embedding_fn = get_embedding_function()

    texts = [chunk.text for chunk in chunks]
    metadatas = [
        {
            "chunk_id": chunk.chunk_id,
            "source_filename": chunk.source_filename,
            "page_number": chunk.page_number,
        }
        for chunk in chunks
    ]
    ids = [chunk.chunk_id for chunk in chunks]

    logger.info(
        "Building vectorstore collection '%s' with %d chunks.", collection_name, len(chunks)
    )

    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embedding_fn,
        metadatas=metadatas,
        ids=ids,
        collection_name=collection_name,
        persist_directory=str(config.VECTOR_STORE_DIR),
    )

    logger.info("Vectorstore collection '%s' built and persisted.", collection_name)
    return vectorstore


def load_vectorstore(collection_name: str):
    """Load an existing persisted Chroma collection by name.

    Raise a clear, catchable ValueError if the collection doesn't exist.
    Callers should check collection_exists() first, but don't rely on that alone.
    """
    from langchain_chroma import Chroma

    if not collection_exists(collection_name):
        raise ValueError(
            f"Chroma collection '{collection_name}' does not exist in "
            f"{config.VECTOR_STORE_DIR}. Call build_vectorstore() first."
        )

    embedding_fn = get_embedding_function()
    logger.info("Loading existing vectorstore collection '%s'.", collection_name)

    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_fn,
        persist_directory=str(config.VECTOR_STORE_DIR),
    )


def similarity_search(collection, query: str, k: int) -> list[dict]:
    """Run dense similarity search and return the top-k results.

    Each returned dict includes: text, source_filename, page_number,
    chunk_id, and score (lower = more similar for L2, higher for cosine).
    """
    results = collection.similarity_search_with_relevance_scores(query, k=k)

    retrieved: list[dict] = []
    for doc, score in results:
        retrieved.append(
            {
                "text": doc.page_content,
                "source_filename": doc.metadata.get("source_filename", "unknown"),
                "page_number": doc.metadata.get("page_number", 0),
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "score": score,
            }
        )

    logger.debug("Dense search returned %d results for query: '%s'.", len(retrieved), query[:60])
    return retrieved
