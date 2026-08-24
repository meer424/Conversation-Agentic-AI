"""PDF text extraction with page-level metadata.

Primary parser: pdfplumber (better table/text fidelity). Fall back to
pypdf if pdfplumber raises or returns nothing for a given file.
"""

import io
from dataclasses import dataclass

from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class PageContent:
    """One page of extracted text plus the metadata needed for citations."""

    source_filename: str
    page_number: int  # 1-indexed, matches what a human would see in a PDF viewer
    text: str


def _extract_with_pdfplumber(file_bytes: bytes, source_filename: str) -> list[PageContent]:
    """Attempt extraction using pdfplumber. Returns empty list on any failure."""
    try:
        import pdfplumber

        pages: list[PageContent] = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                pages.append(PageContent(source_filename=source_filename, page_number=i, text=text))
        return pages
    except Exception as exc:
        logger.warning("pdfplumber failed on '%s': %s", source_filename, exc)
        return []


def _extract_with_pypdf(file_bytes: bytes, source_filename: str) -> list[PageContent]:
    """Attempt extraction using pypdf. Returns empty list on any failure."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(file_bytes))
        pages: list[PageContent] = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append(PageContent(source_filename=source_filename, page_number=i, text=text))
        return pages
    except Exception as exc:
        logger.warning("pypdf fallback failed on '%s': %s", source_filename, exc)
        return []


def _all_pages_empty(pages: list[PageContent]) -> bool:
    """Return True if every page yielded an empty or whitespace-only string."""
    return all(not p.text.strip() for p in pages)


def extract_pdf_pages(file_bytes: bytes, source_filename: str) -> list[PageContent]:
    """Extract text from every page of a single PDF.

    Requirements:
    - Try pdfplumber first; on failure (exception or all-empty pages),
      retry with pypdf before giving up.
    - Never raise on a malformed PDF — log the error and return an empty
      list so the caller (app.py) can show a specific st.error message.
    - Preserve page order and 1-indexed page numbers exactly as a human
      would see them in a PDF viewer, since these numbers are what gets
      shown in citations later.
    """
    pages = _extract_with_pdfplumber(file_bytes, source_filename)

    if not pages or _all_pages_empty(pages):
        logger.info(
            "pdfplumber returned no text for '%s'; trying pypdf fallback.", source_filename
        )
        pages = _extract_with_pypdf(file_bytes, source_filename)

    if not pages:
        logger.error("Both parsers failed to extract text from '%s'.", source_filename)

    return pages


def extract_multiple_pdfs(files: list[tuple[bytes, str]]) -> list[PageContent]:
    """Run extract_pdf_pages over multiple uploaded files and flatten the result.

    `files` is a list of (file_bytes, filename) tuples, e.g. from
    st.file_uploader(accept_multiple_files=True). Files that fail extraction
    should be skipped (with a logged reason), not abort the whole batch.
    """
    all_pages: list[PageContent] = []
    for file_bytes, filename in files:
        pages = extract_pdf_pages(file_bytes, filename)
        if not pages:
            logger.error("Skipping '%s' — extraction returned no pages.", filename)
        else:
            all_pages.extend(pages)
            logger.info("Extracted %d pages from '%s'.", len(pages), filename)
    return all_pages
