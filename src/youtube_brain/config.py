"""Configuration loading and strongly typed runtime settings."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Settings:
    cache_dir: Path
    max_upload_bytes: int
    chunk_seconds: int
    overlap_seconds: int
    embedding_model: str
    whisper_model: str
    languages: tuple[str, ...]
    retrieval_candidates: int
    rrf_k: int
    minimum_relevance: float
    llm_provider: str
    llm_base_url: str
    llm_model: str
    llm_api_key_env: str
    llm_timeout_seconds: int | None
    examples: tuple[dict[str, str], ...]


def load_settings(path: str | Path = "config.yaml") -> Settings:
    """Load application settings and reject invalid pipeline values."""
    with Path(path).open("r", encoding="utf-8") as config_file:
        raw: dict[str, Any] = yaml.safe_load(config_file)

    app = raw["app"]
    pipeline = raw["pipeline"]
    llm = raw["llm"]
    chunk_seconds = int(pipeline["chunk_seconds"])
    overlap_seconds = int(pipeline["overlap_seconds"])
    if chunk_seconds <= 0 or not 0 <= overlap_seconds < chunk_seconds:
        raise ValueError("Chunk overlap must be non-negative and shorter than the chunk.")
    timeout_value = llm.get("timeout_seconds", 300)
    llm_timeout_seconds = None if timeout_value is None else int(timeout_value)
    if llm_timeout_seconds is not None and llm_timeout_seconds <= 0:
        raise ValueError("LLM timeout must be a positive number of seconds.")

    return Settings(
        cache_dir=Path(app["cache_dir"]),
        max_upload_bytes=int(app["max_upload_mb"]) * 1024 * 1024,
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds,
        embedding_model=str(pipeline["embedding_model"]),
        whisper_model=str(pipeline["whisper_model"]),
        languages=tuple(str(item) for item in pipeline["languages"]),
        retrieval_candidates=int(pipeline["retrieval_candidates"]),
        rrf_k=int(pipeline["rrf_k"]),
        minimum_relevance=float(pipeline["minimum_relevance"]),
        llm_provider=str(llm["provider"]),
        llm_base_url=str(llm["base_url"]).rstrip("/"),
        llm_model=str(llm["model"]),
        llm_api_key_env=str(llm["api_key_env"]),
        llm_timeout_seconds=llm_timeout_seconds,
        examples=tuple(app["examples"]),
    )
