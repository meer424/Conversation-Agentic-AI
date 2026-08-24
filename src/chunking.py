"""Structure-aware chunking for lab manual pages.

Naive fixed-size splitting is known to break numbered procedure steps
mid-sentence (see AGENTS.md section 3). Prefer paragraph/section-boundary
splits before falling back to raw character-count splitting.
"""

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.pdf_loader import PageContent
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    """A single retrievable unit of text with citation metadata."""

    chunk_id: str
    source_filename: str
    page_number: int
    text: str


def chunk_pages(pages: list[PageContent], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Split extracted pages into overlapping, citation-tagged chunks.

    Requirements:
    - Use LangChain's RecursiveCharacterTextSplitter with separators ordered
      to prefer splitting on paragraph boundaries before raw character count.
    - Every returned Chunk carries the source_filename and page_number of the
      page it was extracted from.
    - chunk_id is deterministic: f"{source_filename}::p{page_number}::{index}"
      so the same input always produces the same IDs — required for Chroma
      upserts and the faithfulness check in agent.py.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Prefer splitting on structural boundaries before raw character count.
        # \n\n = paragraph break (most common section separator in lab manuals)
        # \n   = line break (numbered step boundary)
        # .    = sentence boundary
        # " "  = word boundary (last resort before character split)
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks: list[Chunk] = []
    for page in pages:
        if not page.text.strip():
            # Skip empty pages — they'd produce empty chunks that pollute retrieval.
            continue

        raw_chunks = splitter.split_text(page.text)
        for index, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            # Deterministic ID: stable across runs for the same input.
            chunk_id = f"{page.source_filename}::p{page.page_number}::{index}"

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    source_filename=page.source_filename,
                    page_number=page.page_number,
                    text=chunk_text,
                )
            )

    logger.info(
        "Chunked %d pages into %d chunks (chunk_size=%d, overlap=%d).",
        len(pages),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks
