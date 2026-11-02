"""Typed data structures shared by the processing pipeline."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Segment:
    """A time-aligned portion of a video's transcript."""

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Chunk:
    """A searchable transcript window with its source time interval."""

    video_id: str
    chunk_id: str
    start_time: float
    end_time: float
    text: str
    language: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Chunk":
        return cls(**value)


@dataclass(frozen=True)
class Chapter:
    """A navigable topic heading for a video."""

    title: str
    start_time: float


@dataclass(frozen=True)
class KeyClaim:
    """A concise claim supported by a transcript excerpt."""

    claim: str
    timestamp: float
    quote: str

