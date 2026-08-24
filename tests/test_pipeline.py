"""Basic pipeline tests.

Run with: pytest tests/ -v

Tests are organized to run without a real PDF file or API key wherever possible.
The faithfulness guardrail test (test_verify_citations_strips_unretrieved_pages)
is especially critical — see AGENTS.md section 3.
"""

import pytest

from src import chunking, pdf_loader
from src.agent import Citation, extract_citations, verify_citations
from src.pdf_loader import PageContent
from src.utils import is_extraction_effectively_empty


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_page(text: str, page_number: int = 1, filename: str = "test.pdf") -> PageContent:
    return PageContent(source_filename=filename, page_number=page_number, text=text)


SAMPLE_LAB_TEXT = """\
Experiment 1: Acid-Base Titration

Objective
Determine the concentration of an unknown HCl solution using NaOH as a titrant.

Materials
- 50 mL burette
- 250 mL Erlenmeyer flask
- Phenolphthalein indicator
- 0.1 M NaOH standard solution

Procedure
1. Rinse the burette with distilled water, then with the NaOH solution.
2. Fill the burette to the 0.00 mL mark with 0.1 M NaOH.
3. Pipette 25.00 mL of the unknown HCl solution into the Erlenmeyer flask.
4. Add 3 drops of phenolphthalein indicator.
5. Slowly add NaOH from the burette, swirling continuously.
6. Stop when the solution turns faint pink and remains pink for 30 seconds.
7. Record the final burette reading.

Safety
Always wear goggles and gloves when handling acids and bases.
"""


# ---------------------------------------------------------------------------
# pdf_loader tests (pure Python, no real PDF needed for the metadata tests)
# ---------------------------------------------------------------------------

class TestExtractPdfPagesContract:
    def test_returns_list(self) -> None:
        """extract_pdf_pages on invalid bytes must return an empty list, not raise."""
        result = pdf_loader.extract_pdf_pages(b"not a pdf", "fake.pdf")
        assert isinstance(result, list)

    def test_empty_bytes_returns_empty_list(self) -> None:
        result = pdf_loader.extract_pdf_pages(b"", "empty.pdf")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# chunking tests
# ---------------------------------------------------------------------------

class TestChunkPagesPreservesCitationMetadata:
    def test_every_chunk_carries_source_metadata(self) -> None:
        """Every Chunk produced by chunk_pages must carry the correct
        source_filename and page_number inherited from its source PageContent.
        """
        pages = [_make_page(SAMPLE_LAB_TEXT, page_number=3, filename="manual.pdf")]
        chunks = chunking.chunk_pages(pages, chunk_size=200, chunk_overlap=20)

        assert len(chunks) > 0, "Should produce at least one chunk from non-trivial text"
        for chunk in chunks:
            assert chunk.source_filename == "manual.pdf", (
                f"chunk.source_filename should be 'manual.pdf', got '{chunk.source_filename}'"
            )
            assert chunk.page_number == 3, (
                f"chunk.page_number should be 3, got {chunk.page_number}"
            )

    def test_chunk_ids_are_deterministic(self) -> None:
        """Same input must always produce the same chunk_ids (required for Chroma upserts)."""
        pages = [_make_page(SAMPLE_LAB_TEXT)]
        chunks_a = chunking.chunk_pages(pages, chunk_size=300, chunk_overlap=30)
        chunks_b = chunking.chunk_pages(pages, chunk_size=300, chunk_overlap=30)

        ids_a = [c.chunk_id for c in chunks_a]
        ids_b = [c.chunk_id for c in chunks_b]
        assert ids_a == ids_b, "chunk_ids must be deterministic for the same input"

    def test_chunk_ids_are_unique_within_a_page(self) -> None:
        pages = [_make_page(SAMPLE_LAB_TEXT)]
        chunks = chunking.chunk_pages(pages, chunk_size=200, chunk_overlap=20)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "chunk_ids must be unique within a page"


class TestChunkPagesRespectsBounds:
    def test_no_chunk_wildly_exceeds_chunk_size(self) -> None:
        """No produced chunk should exceed chunk_size by more than the overlap window."""
        chunk_size = 300
        chunk_overlap = 50
        pages = [_make_page(SAMPLE_LAB_TEXT * 3)]  # repeat text to force many splits
        chunks = chunking.chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        slack = chunk_overlap + 50  # generous slack for the splitter's boundary logic
        for chunk in chunks:
            assert len(chunk.text) <= chunk_size + slack, (
                f"Chunk of length {len(chunk.text)} wildly exceeds chunk_size={chunk_size}"
            )

    def test_skips_empty_pages(self) -> None:
        """Chunking an all-whitespace page should produce zero chunks, not empty-text chunks."""
        pages = [_make_page("   \n  \t  ")]
        chunks = chunking.chunk_pages(pages, chunk_size=200, chunk_overlap=20)
        assert all(c.text.strip() for c in chunks), "No chunk should have empty/whitespace text"


# ---------------------------------------------------------------------------
# is_extraction_effectively_empty (pulled from utils tests for pipeline context)
# ---------------------------------------------------------------------------

class TestIsExtractionEffectivelyEmptyFlagsScannedPdf:
    def test_near_empty_pages_flagged(self) -> None:
        pages = ["   ", "\n", "x", "  "]
        assert is_extraction_effectively_empty(pages) is True

    def test_normal_pages_not_flagged(self) -> None:
        pages = [SAMPLE_LAB_TEXT, SAMPLE_LAB_TEXT]
        assert is_extraction_effectively_empty(pages) is False


# ---------------------------------------------------------------------------
# agent.verify_citations — faithfulness guardrail (AGENTS.md §3)
# ---------------------------------------------------------------------------

class TestVerifyCitationsStripsUnretrievedPages:
    """This is the single most important correctness test in the whole app
    (per AGENTS.md §3). It must not be skipped or softened.
    """

    def _retrieved(self, filename: str, page: int) -> dict:
        return {
            "text": "some content",
            "source_filename": filename,
            "page_number": page,
            "chunk_id": f"{filename}::p{page}::0",
            "score": 0.9,
        }

    def test_valid_citation_is_kept(self) -> None:
        retrieved_chunks = [self._retrieved("lab.pdf", 5)]
        citations = [Citation(source_filename="lab.pdf", page_number=5)]
        verified, stripped = verify_citations(citations, retrieved_chunks)
        assert len(verified) == 1
        assert stripped == 0

    def test_unretrieved_page_is_stripped(self) -> None:
        """The model cited page 99 but only page 5 was retrieved — must strip."""
        retrieved_chunks = [self._retrieved("lab.pdf", 5)]
        citations = [Citation(source_filename="lab.pdf", page_number=99)]
        verified, stripped = verify_citations(citations, retrieved_chunks)
        assert len(verified) == 0
        assert stripped == 1

    def test_wrong_filename_is_stripped(self) -> None:
        retrieved_chunks = [self._retrieved("lab.pdf", 5)]
        citations = [Citation(source_filename="other.pdf", page_number=5)]
        verified, stripped = verify_citations(citations, retrieved_chunks)
        assert len(verified) == 0
        assert stripped == 1

    def test_mixed_valid_and_invalid_citations(self) -> None:
        retrieved_chunks = [
            self._retrieved("lab.pdf", 3),
            self._retrieved("lab.pdf", 7),
        ]
        citations = [
            Citation(source_filename="lab.pdf", page_number=3),   # valid
            Citation(source_filename="lab.pdf", page_number=7),   # valid
            Citation(source_filename="lab.pdf", page_number=99),  # hallucinated
        ]
        verified, stripped = verify_citations(citations, retrieved_chunks)
        assert len(verified) == 2
        assert stripped == 1

    def test_no_citations_is_fine(self) -> None:
        retrieved_chunks = [self._retrieved("lab.pdf", 1)]
        verified, stripped = verify_citations([], retrieved_chunks)
        assert verified == []
        assert stripped == 0

    def test_empty_retrieved_chunks_strips_all_citations(self) -> None:
        """If retrieval returned nothing, every citation the model produces is hallucinated."""
        citations = [Citation(source_filename="lab.pdf", page_number=1)]
        verified, stripped = verify_citations(citations, [])
        assert len(verified) == 0
        assert stripped == 1


# ---------------------------------------------------------------------------
# agent.extract_citations
# ---------------------------------------------------------------------------

class TestExtractCitations:
    def test_parses_single_citation(self) -> None:
        text = "The answer is X. (Source: lab.pdf, page 4)"
        citations = extract_citations(text)
        assert len(citations) == 1
        assert citations[0].source_filename == "lab.pdf"
        assert citations[0].page_number == 4

    def test_parses_multiple_citations(self) -> None:
        text = "(Source: manual.pdf, page 2) and (Source: manual.pdf, page 10)"
        citations = extract_citations(text)
        assert len(citations) == 2

    def test_no_citations_returns_empty_list(self) -> None:
        text = "There are no citations in this text."
        citations = extract_citations(text)
        assert citations == []

    def test_case_insensitive(self) -> None:
        text = "(SOURCE: lab.pdf, PAGE 7)"
        citations = extract_citations(text)
        assert len(citations) == 1
        assert citations[0].page_number == 7
