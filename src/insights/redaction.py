"""PII redaction for FAQ question samples surfaced to PMs.

Two layers: a deterministic regex pass for structured PII (emails, phone
numbers) that always runs, and an LLM pass that replaces personal names —
regex alone can't recognise arbitrary names. The LLM pass degrades
gracefully: if the model is unavailable or its output can't be matched back
to the input, the regex-redacted text is returned rather than failing the
request over redaction.
"""

import json
import logging
import re

from pydantic import BaseModel, ValidationError

from llm.base import LLMClient, Message
from llm.errors import LLMUnavailableError
from llm.parsing import extract_json_object

logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"(?<!\w)(\+?\d[\d\-\s()]{6,}\d)(?!\w)")

_NAME_REDACTION_SYSTEM = (
    "You redact personal names from user-submitted questions for a PM-facing "
    "FAQ view. Replace every person's name (first name, last name, or full "
    "name) with [NAME]. Do not change anything else: keep wording, "
    "punctuation, and meaning identical otherwise. Do not redact product "
    "names, company names, or technical terms.\n\n"
    "Return STRICT JSON only (no prose, no markdown fences): "
    '{"texts": [str, ...]} with exactly one output per input, in the same '
    "order."
)


class _RedactPayload(BaseModel):
    texts: list[str] = []


def redact_structured(text: str) -> str:
    """Deterministic email/phone redaction, with no LLM involved.

    Public because it is the floor other modules rely on: anything that hands a
    question to a model, or takes text back from one, can apply this without a
    round-trip and without trusting the model to have done it.
    """
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _PHONE_RE.sub("[PHONE]", text)
    return text


def redact_pii(texts: list[str], llm: LLMClient) -> list[str]:
    """Redact PII from ``texts``, preserving order and length.

    Applies deterministic regex redaction first, then a single batched LLM
    call to redact personal names across all of ``texts`` at once.
    """
    structured = [redact_structured(t) for t in texts]
    if not structured:
        return structured

    messages: list[Message] = [
        {"role": "system", "content": _NAME_REDACTION_SYSTEM},
        {"role": "user", "content": json.dumps({"texts": structured})},
    ]

    try:
        raw = llm.generate(messages)
        payload = _RedactPayload.model_validate_json(extract_json_object(raw))
    except (
        LLMUnavailableError,
        ValidationError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        logger.warning("Name redaction failed, falling back to regex-only: %s", exc)
        return structured

    if len(payload.texts) != len(structured):
        logger.warning(
            "Name redaction returned %d items for %d inputs, "
            "falling back to regex-only",
            len(payload.texts),
            len(structured),
        )
        return structured

    return payload.texts
