"""Small filesystem cache for transcripts, chunks, and computed insights."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from youtube_brain.models import Chapter, Chunk, KeyClaim, Segment


class CacheStore:
    """Persist per-video pipeline results as JSON under one cache directory."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, video_id: str) -> Path:
        if not video_id.replace("-", "").replace("_", "").isalnum():
            raise ValueError("Invalid cache identifier.")
        return self.root / f"{video_id}.json"

    def _embedding_path(self, video_id: str) -> Path:
        self._path(video_id)
        return self.root / f"{video_id}.npy"

    def load_embeddings(self, video_id: str) -> np.ndarray | None:
        """Load cached dense vectors when an index has already been built."""
        path = self._embedding_path(video_id)
        if not path.exists():
            return None
        return np.load(path, allow_pickle=False)

    def save_embeddings(self, video_id: str, embeddings: np.ndarray) -> None:
        """Atomically persist dense vectors for reuse between app sessions."""
        path = self._embedding_path(video_id)
        temporary_path = path.with_suffix(".tmp.npy")
        np.save(temporary_path, embeddings, allow_pickle=False)
        temporary_path.replace(path)

    def load(self, video_id: str) -> dict[str, Any] | None:
        """Return cached data, or None when this video has not been processed."""
        path = self._path(video_id)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as cache_file:
            payload = json.load(cache_file)
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError(f"Unsupported or malformed cache entry for {video_id}.")
        return payload

    def save(
        self,
        video_id: str,
        segments: list[Segment],
        language: str,
        chunks: list[Chunk],
        chapters: list[Chapter],
        claims: list[KeyClaim],
        embedding_model: str,
        chunk_seconds: int,
        overlap_seconds: int,
    ) -> None:
        """Atomically write a complete JSON cache entry."""
        payload = {
            "schema_version": 1,
            "video_id": video_id,
            "language": language,
            "embedding_model": embedding_model,
            "chunk_seconds": chunk_seconds,
            "overlap_seconds": overlap_seconds,
            "segments": [segment.__dict__ for segment in segments],
            "chunks": [chunk.to_dict() for chunk in chunks],
            "chapters": [chapter.__dict__ for chapter in chapters],
            "claims": [claim.__dict__ for claim in claims],
        }
        path = self._path(video_id)
        temporary_path = path.with_suffix(".tmp")
        with temporary_path.open("w", encoding="utf-8") as cache_file:
            json.dump(payload, cache_file, ensure_ascii=False)
        temporary_path.replace(path)

    @staticmethod
    def decode(payload: dict[str, Any]) -> tuple[list[Segment], list[Chunk], list[Chapter], list[KeyClaim]]:
        """Convert a cache record into the pipeline's typed data structures."""
        return (
            [Segment(**item) for item in payload["segments"]],
            [Chunk.from_dict(item) for item in payload["chunks"]],
            [Chapter(**item) for item in payload["chapters"]],
            [KeyClaim(**item) for item in payload["claims"]],
        )
