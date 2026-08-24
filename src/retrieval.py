"""Retrieval pipeline: MVP dense search, plus optional hybrid + rerank.

Phase 1 (MVP): callers can just use vectorstore.similarity_search directly.
Phase 2 (config.ENABLE_HYBRID_RETRIEVAL / config.ENABLE_RERANK): this module
adds BM25 sparse retrieval, reciprocal rank fusion, and cross-encoder
reranking on top. Implement Phase 1 first and confirm it works before
starting this module — see AGENTS.md section 3 and 4 for build order.
"""

import config
from src.chunking import Chunk
from src.utils import get_logger
from src.vectorstore import similarity_search

logger = get_logger(__name__)


def build_bm25_index(chunks: list[Chunk]):
    """Build an in-memory BM25 index (rank_bm25) over all chunk texts.

    Rebuild this whenever the underlying chunk set changes (i.e. whenever
    build_vectorstore is called for a new upload) — it is not persisted
    to disk like the Chroma collection is.
    """
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [chunk.text.lower().split() for chunk in chunks]
    return BM25Okapi(tokenized_corpus)


def bm25_search(bm25_index, chunks: list[Chunk], query: str, k: int) -> list[dict]:
    """Return the top-k chunks by BM25 keyword score for the query.

    Return shape matches vectorstore.similarity_search output (text,
    source_filename, page_number, chunk_id, score) so fuse_results()
    can combine them.
    """
    tokenized_query = query.lower().split()
    scores = bm25_index.get_scores(tokenized_query)

    # Pair each chunk with its BM25 score, sort descending, take top-k.
    scored_chunks = sorted(
        zip(scores, chunks), key=lambda pair: pair[0], reverse=True
    )[:k]

    results: list[dict] = []
    for score, chunk in scored_chunks:
        results.append(
            {
                "text": chunk.text,
                "source_filename": chunk.source_filename,
                "page_number": chunk.page_number,
                "chunk_id": chunk.chunk_id,
                "score": float(score),
            }
        )

    logger.debug("BM25 search returned %d results for query: '%s'.", len(results), query[:60])
    return results


def fuse_results(dense_results: list[dict], sparse_results: list[dict], k: int) -> list[dict]:
    """Combine dense and sparse result lists via Reciprocal Rank Fusion (RRF).

    Standard RRF: score(d) = sum(1 / (rank_constant + rank)) across lists.
    rank_constant=60 is the well-established default from the original RRF paper.
    Deduplicates by chunk_id before returning the top-k fused results.
    """
    rank_constant = 60
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for result_list in (dense_results, sparse_results):
        for rank, result in enumerate(result_list, start=1):
            chunk_id = result["chunk_id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (rank_constant + rank)
            chunk_map[chunk_id] = result  # last write wins for metadata, which is identical

    fused = sorted(chunk_map.values(), key=lambda r: rrf_scores[r["chunk_id"]], reverse=True)
    return fused[:k]


def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Rerank candidates with a local cross-encoder (config.RERANKER_MODEL_NAME).

    Score each (query, candidate_text) pair, sort descending, return top_k.
    This is a local sentence-transformers CrossEncoder — no external API call.
    """
    from sentence_transformers import CrossEncoder

    cross_encoder = CrossEncoder(config.RERANKER_MODEL_NAME)

    pairs = [(query, candidate["text"]) for candidate in candidates]
    scores = cross_encoder.predict(pairs)

    scored = sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)
    reranked = [candidate for _, candidate in scored[:top_k]]

    logger.debug("Cross-encoder reranked %d candidates → top %d.", len(candidates), len(reranked))
    return reranked


def retrieve(collection, bm25_index, chunks: list[Chunk], query: str) -> list[dict]:
    """Top-level retrieval entry point used by agent.py.

    If config.ENABLE_HYBRID_RETRIEVAL is False: just call
    vectorstore.similarity_search with config.RETRIEVAL_TOP_K and return.

    If True: run dense + BM25 each at config.HYBRID_CANDIDATE_POOL, fuse
    via RRF, then if config.ENABLE_RERANK is also True, rerank down to
    config.RERANK_TOP_K before returning. This is the single place agent.py
    calls into — it never needs to know which mode is active.
    """
    if not config.ENABLE_HYBRID_RETRIEVAL:
        logger.debug("Phase 1 retrieval: dense-only, top_k=%d.", config.RETRIEVAL_TOP_K)
        return similarity_search(collection, query, k=config.RETRIEVAL_TOP_K)

    # Phase 2: hybrid retrieval
    logger.debug(
        "Phase 2 retrieval: hybrid dense+BM25, candidate_pool=%d.", config.HYBRID_CANDIDATE_POOL
    )
    dense_results = similarity_search(collection, query, k=config.HYBRID_CANDIDATE_POOL)
    sparse_results = bm25_search(bm25_index, chunks, query, k=config.HYBRID_CANDIDATE_POOL)

    fused = fuse_results(dense_results, sparse_results, k=config.RERANK_TOP_K * 2)

    if config.ENABLE_RERANK:
        logger.debug("Reranking %d fused candidates → top %d.", len(fused), config.RERANK_TOP_K)
        return rerank(query, fused, top_k=config.RERANK_TOP_K)

    return fused[: config.RERANK_TOP_K]
