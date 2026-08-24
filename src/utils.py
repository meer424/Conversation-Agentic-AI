"""Shared helpers: logging setup and file hashing.

Implement this module first — pdf_loader, vectorstore, and agent all
depend on the logger and the hashing function defined here.
"""

import hashlib
import logging
from pathlib import Path

import config


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger that writes to both console and config.LOG_FILE.

    Requirements:
    - Read the log level from config.LOG_LEVEL.
    - Attach a handler once per logger name (avoid duplicate handlers on
      repeated Streamlit reruns — check `logger.handlers` before adding).
    - Use a formatter that includes timestamp, level, and module name.
    """
    logger = logging.getLogger(name)

    # Guard: if handlers are already attached this logger is already configured.
    # Streamlit can re-execute module-level code on reruns; without this guard
    # every rerun would add another handler and duplicate every log line.
    if logger.handlers:
        return logger

    log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler — creates the log file if it doesn't exist yet
    log_file: Path = config.LOG_FILE
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Prevent log records from propagating to the root logger, which would
    # cause a second round of output if the root logger also has handlers.
    logger.propagate = False

    return logger


def hash_file_bytes(file_bytes: bytes) -> str:
    """Return a stable content hash (e.g. sha256 hex digest) for a file's bytes.

    Used to name/reuse Chroma collections so re-uploading an identical PDF
    doesn't trigger re-embedding. Must be deterministic across runs.
    """
    return hashlib.sha256(file_bytes).hexdigest()


def hash_multiple_files(files_bytes: list[bytes]) -> str:
    """Return a single stable hash representing a set of uploaded files.

    Order-independent (sort the individual hashes before combining) so the
    same set of PDFs uploaded in a different order still resolves to the
    same collection.
    """
    individual_hashes = sorted(hash_file_bytes(fb) for fb in files_bytes)
    combined = "".join(individual_hashes)
    return hashlib.sha256(combined.encode()).hexdigest()


def is_extraction_effectively_empty(
    pages_text: list[str], min_chars_per_page: int = 20
) -> bool:
    """Return True if extracted text looks like a scanned/image-only PDF.

    Heuristic: if the average non-whitespace character count per page is
    below `min_chars_per_page`, treat the PDF as unextractable text (likely
    needs OCR, which this app does not support). The UI must warn the user
    instead of silently indexing near-nothing.
    """
    if not pages_text:
        return True

    total_non_ws_chars = sum(
        sum(1 for ch in page if not ch.isspace()) for page in pages_text
    )
    average_non_ws_per_page = total_non_ws_chars / len(pages_text)
    return average_non_ws_per_page < min_chars_per_page
