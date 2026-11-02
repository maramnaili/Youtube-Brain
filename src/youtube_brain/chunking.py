"""Create overlapping time-window chunks from transcript segments."""

from collections.abc import Sequence

from youtube_brain.models import Chunk, Segment


def chunk_transcript(
    video_id: str,
    segments: Sequence[Segment],
    chunk_seconds: int = 45,
    overlap_seconds: int = 10,
    language: str = "unknown",
) -> list[Chunk]:
    """Group timestamped segments into fixed-duration windows with overlap."""
    if chunk_seconds <= 0 or not 0 <= overlap_seconds < chunk_seconds:
        raise ValueError("Chunk overlap must be non-negative and shorter than the chunk.")
    if not segments:
        return []
    if any(segment.start < 0 or segment.end < segment.start for segment in segments):
        raise ValueError("Transcript segments must have valid non-negative times.")

    ordered = sorted(segments, key=lambda segment: segment.start)
    step = chunk_seconds - overlap_seconds
    chunks: list[Chunk] = []
    start = 0.0
    duration = max(segment.end for segment in ordered)

    while start < duration or not chunks:
        end = start + chunk_seconds
        selected = [segment for segment in ordered if segment.start < end and segment.end > start]
        text = " ".join(segment.text.strip() for segment in selected if segment.text.strip())
        if text:
            actual_end = min(end, max(segment.end for segment in selected))
            chunks.append(
                Chunk(
                    video_id=video_id,
                    chunk_id=f"{video_id}-{len(chunks):04d}",
                    start_time=start,
                    end_time=actual_end,
                    text=text,
                    language=language,
                )
            )
        if end >= duration:
            break
        start += step

    return chunks
