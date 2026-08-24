"""Tests for src/utils.py.

Run with: pytest tests/test_utils.py -v
"""

import logging

# pyrefly: ignore [missing-import]
import pytest

from src.utils import (
    get_logger,
    hash_file_bytes,
    hash_multiple_files,
    is_extraction_effectively_empty,
)


# ---------------------------------------------------------------------------
# get_logger
# ---------------------------------------------------------------------------


class TestGetLogger:
    def test_returns_logger_instance(self) -> None:
        logger = get_logger("test.basic")
        assert isinstance(logger, logging.Logger)

    def test_logger_has_handlers(self) -> None:
        logger = get_logger("test.handlers")
        assert len(logger.handlers) > 0, "Logger must have at least one handler attached"

    def test_no_duplicate_handlers_on_repeated_calls(self) -> None:
        """Calling get_logger twice with the same name must not double-attach handlers.

        This simulates the Streamlit rerun scenario where module-level code
        re-executes and get_logger is called again for an already-configured name.
        """
        name = "test.no_dupes"
        logger_first = get_logger(name)
        handler_count_first = len(logger_first.handlers)

        logger_second = get_logger(name)
        handler_count_second = len(logger_second.handlers)

        assert handler_count_first == handler_count_second, (
            f"Handler count changed from {handler_count_first} to "
            f"{handler_count_second} on repeated get_logger call — "
            "duplicate handlers would cause doubled log output on Streamlit reruns"
        )

    def test_does_not_propagate(self) -> None:
        logger = get_logger("test.propagate")
        assert logger.propagate is False


# ---------------------------------------------------------------------------
# hash_file_bytes
# ---------------------------------------------------------------------------


class TestHashFileBytes:
    def test_returns_hex_string(self) -> None:
        result = hash_file_bytes(b"hello")
        assert isinstance(result, str)
        # SHA-256 hex digest is always 64 characters
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_deterministic_same_input(self) -> None:
        data = b"some pdf bytes"
        assert hash_file_bytes(data) == hash_file_bytes(data)

    def test_different_inputs_produce_different_hashes(self) -> None:
        assert hash_file_bytes(b"file_a") != hash_file_bytes(b"file_b")

    def test_empty_bytes(self) -> None:
        """Empty bytes must not raise — used to detect accidentally empty reads."""
        result = hash_file_bytes(b"")
        assert isinstance(result, str)
        assert len(result) == 64


# ---------------------------------------------------------------------------
# hash_multiple_files
# ---------------------------------------------------------------------------


class TestHashMultipleFiles:
    def test_returns_hex_string(self) -> None:
        result = hash_multiple_files([b"file1", b"file2"])
        assert isinstance(result, str)
        assert len(result) == 64

    def test_order_independent(self) -> None:
        """The same set of files in different order must produce the same hash."""
        files = [b"alpha", b"beta", b"gamma"]
        hash_abc = hash_multiple_files(files)
        hash_cba = hash_multiple_files(list(reversed(files)))
        assert hash_abc == hash_cba, (
            "hash_multiple_files must be order-independent — same PDF set "
            "uploaded in different order must resolve to the same Chroma collection"
        )

    def test_different_file_sets_differ(self) -> None:
        assert hash_multiple_files([b"a", b"b"]) != hash_multiple_files([b"a", b"c"])

    def test_single_file_matches_hash_file_bytes(self) -> None:
        """A single-element list should produce a hash that is still stable across runs.

        Note: hash_multiple_files([x]) is NOT required to equal hash_file_bytes(x)
        because it hashes the combined sorted list of hex digests — this test just
        checks it's deterministic for a single file.
        """
        result_1 = hash_multiple_files([b"only_file"])
        result_2 = hash_multiple_files([b"only_file"])
        assert result_1 == result_2


# ---------------------------------------------------------------------------
# is_extraction_effectively_empty
# ---------------------------------------------------------------------------


class TestIsExtractionEffectivelyEmpty:
    def test_normal_text_returns_false(self) -> None:
        pages = [
            "Experiment 1: Titration\nAdd 10 mL of NaOH to the burette.",
            "Safety: Always wear gloves and goggles when handling acids.",
        ]
        assert is_extraction_effectively_empty(pages) is False

    def test_near_empty_pages_returns_true(self) -> None:
        """Pages with only a handful of characters per page → scanned PDF heuristic."""
        pages = ["   ", "\n", "x", "  "]
        assert is_extraction_effectively_empty(pages) is True

    def test_empty_list_returns_true(self) -> None:
        """No pages extracted at all must be treated as unextractable."""
        assert is_extraction_effectively_empty([]) is True

    def test_exactly_at_threshold_returns_false(self) -> None:
        """Average non-whitespace == min_chars_per_page should NOT be flagged empty."""
        # min_chars_per_page default is 20; one page with exactly 20 non-ws chars
        page = "a" * 20
        assert is_extraction_effectively_empty([page], min_chars_per_page=20) is False

    def test_one_below_threshold_returns_true(self) -> None:
        page = "a" * 19
        assert is_extraction_effectively_empty([page], min_chars_per_page=20) is True

    def test_custom_threshold(self) -> None:
        pages = ["short"]  # 5 non-ws chars
        assert is_extraction_effectively_empty(pages, min_chars_per_page=10) is True
        assert is_extraction_effectively_empty(pages, min_chars_per_page=3) is False

    def test_mixed_pages_uses_average(self) -> None:
        """One empty page and one rich page — average should determine the result."""
        pages = [
            "",  # 0 non-ws chars
            "a" * 60,  # 60 non-ws chars
        ]
        # average = (0 + 60) / 2 = 30 → above default threshold of 20 → not empty
        assert is_extraction_effectively_empty(pages) is False
