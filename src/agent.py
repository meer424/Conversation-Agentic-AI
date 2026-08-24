"""RAG agent logic: prompt construction, Gemini calls, streaming, and the
citation faithfulness guardrail described in AGENTS.md section 3.

This module is intentionally the last one implemented — it depends on
retrieval.py (or vectorstore.py directly in MVP mode) already working.
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass, field

import config
from src.utils import get_logger

logger = get_logger(__name__)


SYSTEM_PROMPT_TEMPLATE = """\
You are a Lab Manual Assistant. Answer the user's question using ONLY the
context passages below, drawn from the uploaded lab manual(s).

Rules:
- If the answer is not present in the context, say exactly:
  "I couldn't find this in the uploaded lab manual." Do not guess.
- When the manual describes a procedure, answer with clear numbered steps
  matching the manual's own structure.
- At the end of your answer, cite every source you used in the form:
  (Source: <filename>, page <page_number>).
- Never cite a page that isn't in the context passages provided to you.

Context passages:
{context_block}
"""


@dataclass
class Citation:
    source_filename: str
    page_number: int


@dataclass
class AgentAnswer:
    text: str
    citations: list[Citation]
    unverified_citations_stripped: int  # count of citations removed by the faithfulness check


def build_context_block(retrieved_chunks: list[dict]) -> str:
    """Format retrieved chunks into the context block injected into the
    system prompt. Include filename + page number inline with each passage
    so the model has an easy, unambiguous citation format to copy from.
    """
    parts: list[str] = []
    for chunk in retrieved_chunks:
        filename = chunk.get("source_filename", "unknown")
        page = chunk.get("page_number", "?")
        text = chunk.get("text", "").strip()
        parts.append(f"[Source: {filename}, page {page}]\n{text}")
    return "\n\n---\n\n".join(parts)


def build_messages(
    question: str,
    retrieved_chunks: list[dict],
    conversation_history: list[dict],
) -> tuple[list[dict], str]:
    """Assemble the full Gemini messages list: system prompt (with context)
    + trimmed conversation history (config.CONVERSATION_MEMORY_TURNS)
    + the new user question.

    conversation_history is a list of {"role": "user"|"assistant", "content": str}
    dicts from st.session_state.messages (excluding the current question).

    Gemini uses role "model" instead of "assistant" — this function converts.
    """
    context_block = build_context_block(retrieved_chunks)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context_block=context_block)

    # Trim history to the last N turns (each turn = one user + one assistant message).
    max_history_messages = config.CONVERSATION_MEMORY_TURNS * 2
    trimmed_history = conversation_history[-max_history_messages:]

    # Convert "assistant" role → "model" for Gemini's API format.
    gemini_history: list[dict] = []
    for msg in trimmed_history:
        role = "model" if msg["role"] == "assistant" else "user"
        gemini_history.append({"role": role, "parts": [msg["content"]]})

    # Add the current user question.
    gemini_history.append({"role": "user", "parts": [question]})

    return gemini_history, system_prompt


def call_gemini_streaming(messages: list[dict], system_prompt: str) -> Iterator[str]:
    """Call the Gemini API (config.GEMINI_MODEL) with streaming enabled and
    yield text deltas as they arrive, for use with st.write_stream.

    Requirements:
    - Read the API key from config.GEMINI_API_KEY; raise a clear,
      catchable ValueError if it is missing so app.py can show a specific
      st.error rather than an SDK traceback.
    - Use system_instruction to pass the RAG system prompt.
    - Respect config.MAX_RESPONSE_TOKENS.
    """
    import google.generativeai as genai

    api_key = config.GEMINI_API_KEY
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. Add it to your .env file or enter it in the sidebar."
        )

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
        model_name=config.GEMINI_MODEL,
        system_instruction=system_prompt,
    )

    logger.debug(
        "Streaming request to %s, max_output_tokens=%d.", config.GEMINI_MODEL, config.MAX_RESPONSE_TOKENS
    )

    # Convert messages list to Gemini's expected format.
    # The last message is the current user question; prior messages are history.
    history = messages[:-1]
    last_user_message = messages[-1]["parts"][0] if messages else ""

    chat = model.start_chat(history=history)

    response = chat.send_message(
        last_user_message,
        stream=True,
        generation_config=genai.GenerationConfig(
            max_output_tokens=config.MAX_RESPONSE_TOKENS,
        ),
    )

    for chunk in response:
        if chunk.text:
            yield chunk.text


def extract_citations(answer_text: str) -> list[Citation]:
    """Parse "(Source: <filename>, page <page_number>)" occurrences out of
    the model's answer text.
    """
    # Pattern: (Source: some_file.pdf, page 12)
    pattern = re.compile(
        r"\(Source:\s*(.+?),\s*page\s*(\d+)\)",
        re.IGNORECASE,
    )
    citations: list[Citation] = []
    for match in pattern.finditer(answer_text):
        filename = match.group(1).strip()
        page_number = int(match.group(2))
        citations.append(Citation(source_filename=filename, page_number=page_number))
    return citations


def verify_citations(
    citations: list[Citation], retrieved_chunks: list[dict]
) -> tuple[list[Citation], int]:
    """The faithfulness guardrail from AGENTS.md section 3.

    Keep only citations whose (source_filename, page_number) pair actually
    appears in the metadata of retrieved_chunks. Return the filtered list
    plus a count of how many were stripped, so the UI/logs can flag when
    the model hallucinated a citation.
    """
    # Build a set of (filename, page) pairs from the chunks that were actually retrieved.
    valid_pairs: set[tuple[str, int]] = {
        (chunk.get("source_filename", ""), chunk.get("page_number", -1))
        for chunk in retrieved_chunks
    }

    verified: list[Citation] = []
    stripped_count = 0

    for citation in citations:
        if (citation.source_filename, citation.page_number) in valid_pairs:
            verified.append(citation)
        else:
            stripped_count += 1
            logger.warning(
                "Faithfulness check: stripped hallucinated citation "
                "(Source: %s, page %d) — not in retrieved chunks.",
                citation.source_filename,
                citation.page_number,
            )

    return verified, stripped_count


def answer_question(
    question: str,
    retrieved_chunks: list[dict],
    conversation_history: list[dict],
) -> AgentAnswer:
    """Non-streaming convenience wrapper used by tests — calls the pieces
    above in order and returns a fully assembled, faithfulness-checked
    AgentAnswer. app.py should generally use call_gemini_streaming directly
    for the live UI and this function for tests/debugging.
    """
    messages, system_prompt = build_messages(question, retrieved_chunks, conversation_history)

    full_text = "".join(call_gemini_streaming(messages, system_prompt))

    raw_citations = extract_citations(full_text)
    verified_citations, stripped = verify_citations(raw_citations, retrieved_chunks)

    return AgentAnswer(
        text=full_text,
        citations=verified_citations,
        unverified_citations_stripped=stripped,
    )
