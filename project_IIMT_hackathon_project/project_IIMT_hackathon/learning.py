"""Local evidence retrieval and Ollama-backed study assistance."""

from __future__ import annotations

import json
import os
import re
import socket
from io import BytesIO
from typing import Any, Dict, List, Union
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pypdf import PdfReader
from pypdf.errors import PdfReadError


EvidencePassage = Dict[str, Union[str, int]]

_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "also",
    "am", "an", "and", "any", "are", "as", "at", "be", "because",
    "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "could", "did", "do", "does", "doing", "down", "during",
    "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself",
    "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself",
    "just", "me", "more", "most", "my", "myself", "no", "nor", "not",
    "of", "off", "on", "once", "only", "or", "other", "our", "ours",
    "ourselves", "out", "over", "own", "same", "she", "should", "so",
    "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was",
    "we", "were", "what", "when", "where", "which", "while", "who",
    "whom", "why", "with", "would", "you", "your", "yours",
}


class ModelServiceError(RuntimeError):
    """Raised when the local Ollama service cannot complete a request."""


class NotesFileError(ValueError):
    """Raised when an uploaded notes file cannot be read as study material."""


def extract_pdf_text(file_bytes: bytes) -> str:
    """Extract searchable text from a non-encrypted PDF, retaining page labels."""
    try:
        reader = PdfReader(BytesIO(file_bytes))
        if reader.is_encrypted:
            raise NotesFileError(
                "This PDF is encrypted. Upload an unencrypted copy of the notes."
            )
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if page_text:
                pages.append(f"[PDF page {page_number}]\n{page_text}")
    except PdfReadError as error:
        raise NotesFileError(
            "This PDF could not be read. Check that it is a valid, unencrypted PDF."
        ) from error
    if not pages:
        raise NotesFileError(
            "No selectable text was found in this PDF. Paste text or use a text-based PDF."
        )
    return "\n\n".join(pages)


def _tokens(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", text.casefold())
    return [token for token in tokens if token not in _STOP_WORDS]


def retrieve_evidence(
    notes: str, question: str, limit: int = 5
) -> List[EvidencePassage]:
    """Return source lines that share meaningful terms with the question."""
    query_tokens = set(_tokens(question))
    if not query_tokens:
        return []

    ranked: List[tuple[int, int, EvidencePassage]] = []
    for line_number, raw_line in enumerate(notes.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        candidate_sentences = re.split(
            r"(?<=[.!?])\s+(?=[A-Z0-9])", line
        )
        score = max(
            (
                len(query_tokens.intersection(_tokens(sentence)))
                for sentence in candidate_sentences
            ),
            default=0,
        )
        if score:
            ranked.append(
                (
                    score,
                    line_number,
                    {"line": line_number, "text": line},
                )
            )

    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [passage for _, _, passage in ranked[:limit]]


def _ollama_chat(
    prompt: str,
    system_prompt: str,
    *,
    json_response: bool = False,
    timeout: int = 120,
) -> str:
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    model = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
    payload: Dict[str, Any] = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    }
    if json_response:
        payload["format"] = "json"

    request = Request(
        f"{host}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            **(
                {"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"}
                if os.environ.get("OLLAMA_API_KEY")
                else {}
            ),
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        details = error.read(500).decode("utf-8", errors="replace")
        raise ModelServiceError(
            f"Ollama returned HTTP {error.code}: {details}"
        ) from error
    except (URLError, TimeoutError, socket.timeout) as error:
        raise ModelServiceError(
            f"Could not reach Ollama at {host}. Check that Ollama is running "
            f"and that model '{model}' is available."
        ) from error
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ModelServiceError(
            "Ollama returned a response that could not be read."
        ) from error

    message = result.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise ModelServiceError("Ollama returned an empty response.")
    return content.strip()


def answer_from_evidence(
    topic: str, question: str, evidence: List[EvidencePassage]
) -> str:
    """Ask the local model for an answer limited to retrieved source lines."""
    evidence_text = "\n".join(
        f"[Line {passage['line']}] {passage['text']}" for passage in evidence
    )
    return _ollama_chat(
        (
            f"Selected topic: {topic}\n"
            f"Student question: {question}\n\n"
            "Candidate source evidence:\n"
            f"{evidence_text}\n\n"
            "Answer the question directly in one or two clear sentences using "
            "only facts explicitly supported by the candidate source evidence. "
            "First check whether the evidence contains an explicit phrase that "
            "answers the question. If it does, state that answer immediately "
            "using the source wording; never say the notes lack a fact that is "
            "present in the evidence. Only say the notes do not provide enough "
            "information when the answer truly is absent. Do not add outside "
            "knowledge, unsupported comparisons, or alternative locations. "
            "Do not invent citations; the app displays the source lines "
            "separately."
        ),
        (
            "You are a careful study tutor. Treat the supplied notes as the "
            "only source of truth. Distinguish supported facts from missing "
            "information and never fill gaps with guesses."
        ),
    )


def validate_quiz_payload(
    payload: Any, notes: str
) -> Dict[str, Union[str, List[str]]]:
    """Validate model-generated quiz content against the source notes."""
    if not isinstance(payload, dict):
        raise ModelServiceError("The model did not return a quiz object.")
    if payload.get("error"):
        raise ModelServiceError(str(payload["error"]))

    question = payload.get("question")
    options = payload.get("options")
    answer = payload.get("answer")
    explanation = payload.get("explanation")
    source_quote = payload.get("source_quote")
    if (
        not isinstance(question, str)
        or not question.strip()
        or not isinstance(options, list)
        or not all(isinstance(option, str) and option.strip() for option in options)
        or not isinstance(answer, str)
        or not isinstance(explanation, str)
        or not explanation.strip()
        or not isinstance(source_quote, str)
        or not source_quote.strip()
    ):
        raise ModelServiceError(
            "The model returned an incomplete quiz. Please try generating it again."
        )

    clean_options = [option.strip() for option in options]
    if len(clean_options) != 4 or len({item.casefold() for item in clean_options}) != 4:
        raise ModelServiceError(
            "The generated quiz must contain four distinct answer options."
        )
    matching_answers = [
        option for option in clean_options if option.casefold() == answer.strip().casefold()
    ]
    if len(matching_answers) != 1:
        raise ModelServiceError(
            "The generated quiz answer does not match one of its options."
        )

    def normalize_whitespace(value: str) -> str:
        return " ".join(value.split()).casefold()

    normalized_notes = normalize_whitespace(notes)
    normalized_quote = normalize_whitespace(source_quote)
    if normalized_quote not in normalized_notes:
        raise ModelServiceError(
            "The generated quiz cited text that is not present in the notes."
        )
    if normalize_whitespace(matching_answers[0]) not in normalized_quote:
        raise ModelServiceError(
            "The generated answer is not stated in its supporting source quote."
        )

    return {
        "question": question.strip(),
        "options": clean_options,
        "answer": matching_answers[0],
        "explanation": f"The notes say: {source_quote.strip()}",
        "source_quote": source_quote.strip(),
    }


def generate_quiz(topic: str, notes: str) -> Dict[str, Union[str, List[str]]]:
    """Generate and source-validate one four-option practice question."""
    prompt = (
        f"Selected topic: {topic}\n\n"
        "Create one clear multiple-choice practice question answerable from "
        "the source notes below. Return exactly one JSON object with these "
        "keys: question, options, answer, explanation, source_quote. "
        "options must contain exactly four distinct strings. answer must be "
        "identical to one option and be an exact phrase found in source_quote. "
        "source_quote must be copied verbatim from the notes and directly "
        "support the answer. explanation must briefly explain the answer "
        "using the quote. If the notes do not support a suitable question, "
        'return {"error":"The notes do not contain enough information for a '
        'reliable practice question."}.\n\n'
        f"Source notes:\n{notes[:16000]}"
    )
    response = _ollama_chat(
        prompt,
        (
            "You create concise educational practice questions. Use only the "
            "source notes. Return valid JSON only and never invent facts, "
            "quotes, or answers."
        ),
        json_response=True,
    )
    try:
        payload = json.loads(response)
    except json.JSONDecodeError as error:
        raise ModelServiceError(
            "The model response was not valid JSON. Please try generating the quiz again."
        ) from error
    return validate_quiz_payload(payload, notes)
