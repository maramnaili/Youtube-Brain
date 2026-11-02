"""YouTube caption retrieval and audio transcription."""

import hashlib
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from youtube_brain.models import Segment
from youtube_brain.timestamps import extract_video_id

ALLOWED_EXTENSIONS = {".mp3", ".mp4", ".wav", ".m4a"}


def upload_cache_id(contents: bytes, filename: str) -> str:
    """Create a stable, non-sensitive cache key for an uploaded media file."""
    digest = hashlib.sha256(contents).hexdigest()
    return f"upload-{digest}"


def validate_upload(filename: str, contents: bytes, max_bytes: int) -> None:
    """Validate an uploaded media file's extension and size."""
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("Upload an MP3, MP4, WAV, or M4A file.")
    if not contents:
        raise ValueError("The uploaded file is empty.")
    if len(contents) > max_bytes:
        raise ValueError(f"The uploaded file exceeds the {max_bytes // (1024 * 1024)} MB limit.")
    signatures = {
        ".wav": contents.startswith(b"RIFF") and contents[8:12] == b"WAVE",
        ".mp4": len(contents) >= 12 and contents[4:8] == b"ftyp",
        ".m4a": len(contents) >= 12 and contents[4:8] == b"ftyp",
        ".mp3": contents.startswith(b"ID3")
        or (len(contents) >= 2 and contents[0] == 0xFF and contents[1] & 0xE0 == 0xE0),
    }
    if not signatures[suffix]:
        raise ValueError("The uploaded file contents do not match the selected audio/video type.")


def _caption_segments(video_id: str, languages: tuple[str, ...]) -> tuple[list[Segment], str]:
    """Fetch manually or automatically generated YouTube captions."""
    from youtube_transcript_api import YouTubeTranscriptApi

    transcript = YouTubeTranscriptApi().fetch(video_id, languages=list(languages))
    segments = [
        Segment(
            start=float(item.start),
            end=float(item.start + item.duration),
            text=str(item.text),
        )
        for item in transcript
    ]
    return segments, str(getattr(transcript, "language_code", "unknown"))


def _transcribe_file(path: str | Path, model_size: str) -> tuple[list[Segment], str]:
    """Transcribe a media file locally with faster-whisper."""
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(path), vad_filter=True)
    result = [
        Segment(start=float(segment.start), end=float(segment.end), text=segment.text.strip())
        for segment in segments
        if segment.text.strip()
    ]
    if not result:
        raise ValueError("No speech could be transcribed from this media.")
    return result, str(info.language or "unknown")


def transcribe_youtube(
    url: str, languages: tuple[str, ...], model_size: str, work_dir: Path
) -> tuple[str, list[Segment], str]:
    """Use captions first, falling back to yt-dlp and local Whisper transcription."""
    video_id = extract_video_id(url)
    caption_error: Exception | None = None
    try:
        segments, language = _caption_segments(video_id, languages)
        if segments:
            return video_id, segments, language
    except Exception as error:
        caption_error = error

    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        import yt_dlp

        with tempfile.TemporaryDirectory(dir=work_dir) as temporary_dir:
            output_template = str(Path(temporary_dir) / "audio.%(ext)s")
            options: dict[str, Any] = {
                "format": "bestaudio/best",
                "outtmpl": output_template,
                "noplaylist": True,
                "quiet": True,
                "no_warnings": True,
                "postprocessors": [
                    {"key": "FFmpegExtractAudio", "preferredcodec": "wav"}
                ],
            }
            with yt_dlp.YoutubeDL(options) as downloader:
                downloader.download([url])
            audio_path = Path(temporary_dir) / "audio.wav"
            if not audio_path.exists():
                raise RuntimeError("Audio download did not produce an audio file.")
            segments, language = _transcribe_file(audio_path, model_size)
            return video_id, segments, language
    except Exception as error:
        raise RuntimeError(
            "Could not obtain captions or download audio. The video may be private, "
            "blocked, or unavailable; try uploading an audio/video file instead. "
            f"Caption retrieval detail: {caption_error or 'no captions were returned'}"
        ) from error


def transcribe_upload(
    contents: bytes, filename: str, model_size: str, work_dir: Path
) -> tuple[str, list[Segment], str]:
    """Transcribe an uploaded file using a temporary file removed after processing."""
    work_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename).suffix.lower()
    with tempfile.NamedTemporaryFile(dir=work_dir, suffix=suffix, delete=False) as media_file:
        media_file.write(contents)
        path = Path(media_file.name)
    try:
        try:
            segments, language = _transcribe_file(path, model_size)
            return upload_cache_id(contents, filename), segments, language
        except Exception as error:
            raise RuntimeError(f"Could not transcribe the uploaded media: {error}") from error
    finally:
        path.unlink(missing_ok=True)
