"""Timestamp formatting and video URL utilities."""

import re


def format_timestamp(seconds: float) -> str:
    """Format seconds as a human-readable MM:SS or H:MM:SS timestamp."""
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds_part:02d}"
    return f"{minutes:02d}:{seconds_part:02d}"


def format_citation_timestamp(seconds: float) -> str:
    """Format a timestamp as total minutes and seconds for grounded citations."""
    total_seconds = max(0, int(seconds))
    minutes, seconds_part = divmod(total_seconds, 60)
    return f"{minutes:02d}:{seconds_part:02d}"


def extract_video_id(url: str) -> str:
    """Validate common YouTube URL forms and return the 11-character video ID."""
    patterns = (
        r"(?:https?://)?(?:www\.)?youtube\.com/watch\?(?:[^#\s]*&)?v=([A-Za-z0-9_-]{11})",
        r"(?:https?://)?(?:www\.)?youtu\.be/([A-Za-z0-9_-]{11})(?:[?&#/]|$)",
        r"(?:https?://)?(?:www\.)?youtube\.com/(?:embed|shorts|live)/([A-Za-z0-9_-]{11})(?:[?&#/]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, url.strip(), flags=re.IGNORECASE)
        if match:
            return match.group(1)
    raise ValueError("Enter a valid YouTube video URL.")


def youtube_url(video_id: str, seconds: float) -> str:
    """Build a YouTube link which opens at a given time."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError("Invalid YouTube video ID.")
    return f"https://www.youtube.com/watch?v={video_id}&t={max(0, int(seconds))}s"
