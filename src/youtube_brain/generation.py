"""Grounded answer generation through a swappable local/API chat endpoint."""

import json
import os
import re
from collections.abc import Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from youtube_brain.config import Settings
from youtube_brain.models import Chapter, KeyClaim
from youtube_brain.retrieval import RetrievedChunk
from youtube_brain.timestamps import format_citation_timestamp

ABSTENTION = "I could not find this in the video"


def _summary_json_schema(timestamps: Sequence[str]) -> dict[str, object]:
    timestamp_schema = {"type": "string", "enum": list(timestamps)}
    return {
        "type": "object",
        "properties": {
            "chapters": {
                "type": "array",
                "minItems": 1,
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "timestamp": timestamp_schema,
                    },
                    "required": ["title", "timestamp"],
                    "additionalProperties": False,
                },
            },
            "claims": {
                "type": "array",
                "minItems": 0,
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string"},
                        "timestamp": timestamp_schema,
                        "quote": {"type": "string"},
                    },
                    "required": ["claim", "timestamp", "quote"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["chapters", "claims"],
        "additionalProperties": False,
    }


def _contains_verbatim_quote(text: str, quote: str) -> bool:
    text_words = re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
    quote_words = re.findall(r"\w+", quote.casefold(), flags=re.UNICODE)
    if not quote_words:
        return False
    quote_length = len(quote_words)
    return any(
        text_words[index:index + quote_length] == quote_words
        for index in range(len(text_words) - quote_length + 1)
    )


def _post_json(
    url: str,
    payload: dict[str, object],
    api_key: str | None = None,
    timeout_seconds: int | None = 90,
) -> dict[str, object]:
    """POST JSON to an LLM endpoint and report provider errors explicitly."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM provider returned HTTP {error.code}: {detail}") from error
    except (URLError, TimeoutError) as error:
        raise RuntimeError(f"Could not reach the configured LLM provider: {error}") from error
    if not isinstance(result, dict):
        raise RuntimeError("The LLM provider returned a malformed response.")
    return result


class LanguageModel:
    """Minimal LLM client for Ollama and OpenAI-compatible chat APIs."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def complete(
        self,
        system: str,
        prompt: str,
        json_schema: dict[str, object] | None = None,
    ) -> str:
        """Generate a completion using the configured provider."""
        if self.settings.llm_provider == "ollama":
            options: dict[str, object] = {"temperature": 0, "seed": 42}
            if json_schema is not None:
                options.update(
                    temperature=0.2,
                    repeat_penalty=1.15,
                    num_predict=1200,
                )
            payload: dict[str, object] = {
                "model": self.settings.llm_model,
                "stream": False,
                "options": options,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            }
            if json_schema is not None:
                payload["format"] = json_schema
            response = _post_json(
                f"{self.settings.llm_base_url}/api/chat",
                payload,
                timeout_seconds=self.settings.llm_timeout_seconds,
            )
            message = response.get("message")
            if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                raise RuntimeError("Ollama returned no message content.")
            return message["content"].strip()
        if self.settings.llm_provider == "openai-compatible":
            api_key = os.environ.get(self.settings.llm_api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"Set {self.settings.llm_api_key_env} to use the configured API provider."
                )
            payload = {
                "model": self.settings.llm_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
            }
            if json_schema is not None:
                payload["response_format"] = {"type": "json_object"}
            response = _post_json(
                f"{self.settings.llm_base_url}/chat/completions",
                payload,
                api_key,
                timeout_seconds=self.settings.llm_timeout_seconds,
            )
            choices = response.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise RuntimeError("The LLM provider returned no choices.")
            message = choices[0].get("message")
            if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                raise RuntimeError("The LLM provider returned no message content.")
            return message["content"].strip()
        raise ValueError(f"Unsupported LLM provider: {self.settings.llm_provider}")

    def answer(self, question: str, retrieved: Sequence[RetrievedChunk]) -> str:
        """Generate an answer and allow citations only at retrieved chunk times."""
        if not retrieved:
            return ABSTENTION
        context = "\n\n".join(
            f"[{format_citation_timestamp(item.chunk.start_time)}] "
            f"{item.chunk.text}"
            for item in retrieved
        )
        system = (
            "Answer the question using only the supplied video transcript passages. "
            "First check whether any passage directly answers it. If so, state that "
            "answer; do not abstain. Write in the question language and copy the "
            "supporting passage timestamp exactly, including its brackets. Never use "
            "a placeholder timestamp. Abstain only when none of the passages supports "
            "an answer. Do not invent facts or timestamps."
        )
        answer = self.complete(system, f"Question:\n{question}\n\nVideo context:\n{context}")
        allowed = {format_citation_timestamp(item.chunk.start_time) for item in retrieved}
        citations = re.findall(r"\[(\d{2,}:\d{2}(?::\d{2})?)\]", answer)
        valid_citations = list(dict.fromkeys(value for value in citations if value in allowed))
        if not valid_citations:
            return ABSTENTION
        answer = re.sub(r"\[(\d{2,}:\d{2}(?::\d{2})?)\]", "", answer).strip()
        if not answer or answer == ABSTENTION:
            return ABSTENTION
        return f"{answer} " + " ".join(f"[{value}]" for value in valid_citations[:3])

    def summarize(
        self, chunks: Sequence[RetrievedChunk]
    ) -> tuple[list[Chapter], list[KeyClaim]]:
        """Generate structured chapters and evidence-linked key claims."""
        allowed = {
            format_citation_timestamp(item.chunk.start_time): item.chunk for item in chunks
        }
        if not allowed:
            raise ValueError("Cannot summarize without transcript chunks.")
        context = "\n".join(
            f"[{format_citation_timestamp(item.chunk.start_time)}] {item.chunk.text}"
            for item in chunks
        )
        response = self.complete(
            (
                "You are a summarizer, not a transcription service. The transcript is "
                "input data; do not reproduce or convert it into another transcript format. "
                "Extract only information supported by it and return exactly the requested "
                "JSON object, with no extra keys or text."
            ),
            (
                "Create up to 5 concise chapters and up to 5 key claims, returning fewer "
                "when the transcript does not support more. Return an "
                "object with only these keys: 'chapters' and 'claims'. 'chapters' must be "
                "an array of objects with 'title' and 'timestamp'. 'claims' must be an "
                "array of objects with 'claim', 'timestamp', and 'quote'. Keep all text "
                "short; each quote must be a short verbatim excerpt, no more than 12 "
                "words. For every timestamp, select one of the exact values in square "
                "brackets below. Do not copy surrounding transcript text into a timestamp "
                "or include a 'transcription' key. "
                f"\nTranscript:\n{context}"
            ),
            json_schema=_summary_json_schema(list(allowed)),
        )
        json_response = response.strip()
        if json_response.startswith("```"):
            json_response = "\n".join(json_response.splitlines()[1:-1])
        try:
            data = json.loads(json_response)
        except json.JSONDecodeError as error:
            preview = response[:300] if response else "<empty response>"
            raise RuntimeError(
                "The LLM did not return valid chapter/claim JSON. "
                f"Response begins with: {preview!r}. Check the model response and retry."
            ) from error
        if not isinstance(data, dict) or not {"chapters", "claims"}.issubset(data):
            keys = sorted(data) if isinstance(data, dict) else type(data).__name__
            raise RuntimeError(
                "The LLM returned JSON in the wrong format; expected 'chapters' and "
                f"'claims', but received {keys!r}. Retry the video processing."
            )
        try:
            chapters = [
                Chapter(title=str(item["title"]), start_time=_parse_allowed_time(item["timestamp"], allowed))
                for item in data["chapters"]
                if item["timestamp"] in allowed and str(item["title"]).strip()
            ][:10]
            claims = [
                KeyClaim(
                    claim=str(item["claim"]),
                    timestamp=_parse_allowed_time(item["timestamp"], allowed),
                    quote=str(item["quote"]),
                )
                for item in data["claims"]
                if item["timestamp"] in allowed
                and str(item["claim"]).strip()
                and str(item["quote"]).strip()
            ][:10]
            claims = [
                claim
                for claim in claims
                if _contains_verbatim_quote(
                    allowed[
                        format_citation_timestamp(claim.timestamp)
                    ].text,
                    claim.quote,
                )
            ]
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"The LLM returned invalid chapter/claim JSON: {error}") from error
        if not chapters:
            raise RuntimeError(
                "The LLM did not return any valid chapters with transcript timestamps. "
                "Retry the video processing."
            )
        return chapters, claims


def _parse_allowed_time(value: str, allowed: dict[str, object]) -> float:
    """Convert an allowed timestamp into seconds."""
    if value not in allowed:
        raise ValueError(f"Unrecognized timestamp: {value}")
    pieces = [int(piece) for piece in value.split(":")]
    if len(pieces) == 2:
        return float(pieces[0] * 60 + pieces[1])
    if len(pieces) == 3:
        return float(pieces[0] * 3600 + pieces[1] * 60 + pieces[2])
    raise ValueError(f"Malformed timestamp: {value}")
