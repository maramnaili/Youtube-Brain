"""Orchestrate ingestion, indexing, and retrieval for one video at a time."""

from dataclasses import dataclass
from collections.abc import Callable

from youtube_brain.chunking import chunk_transcript
from youtube_brain.config import Settings
from youtube_brain.generation import LanguageModel
from youtube_brain.ingestion import transcribe_upload, transcribe_youtube, validate_upload
from youtube_brain.models import Chapter, Chunk, KeyClaim, Segment
from youtube_brain.retrieval import HybridRetriever, RetrievedChunk
from youtube_brain.storage import CacheStore


@dataclass
class VideoKnowledge:
    video_id: str
    language: str
    segments: list[Segment]
    chunks: list[Chunk]
    chapters: list[Chapter]
    claims: list[KeyClaim]


class VideoPipeline:
    """Process a video, persist its derived data, and answer grounded queries."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cache = CacheStore(settings.cache_dir)
        self.llm = LanguageModel(settings)

    def ingest_youtube(self, url: str) -> tuple[VideoKnowledge, bool]:
        """Retrieve or create a cached transcript and index for a YouTube link."""
        from youtube_brain.timestamps import extract_video_id

        video_id = extract_video_id(url)
        return self._ingest(video_id, lambda: transcribe_youtube(
            url, self.settings.languages, self.settings.whisper_model, self.settings.cache_dir
        ))

    def ingest_upload(
        self, filename: str, contents: bytes
    ) -> tuple[VideoKnowledge, bool]:
        """Validate and retrieve or create a cached index for an uploaded media file."""
        validate_upload(filename, contents, self.settings.max_upload_bytes)
        from youtube_brain.ingestion import upload_cache_id

        video_id = upload_cache_id(contents, filename)
        return self._ingest(video_id, lambda: transcribe_upload(
            contents, filename, self.settings.whisper_model, self.settings.cache_dir
        ))

    def _ingest(
        self,
        video_id: str,
        transcribe: Callable[[], tuple[str, list[Segment], str]],
    ) -> tuple[VideoKnowledge, bool]:
        cached = self.cache.load(video_id)
        if cached:
            segments, cached_chunks, chapters, claims = self.cache.decode(cached)
            language = cached["language"]
            chunks_are_current = (
                cached.get("chunk_seconds") == self.settings.chunk_seconds
                and cached.get("overlap_seconds") == self.settings.overlap_seconds
            )
            if chunks_are_current:
                chunks = cached_chunks
                if (
                    cached.get("embedding_model") == self.settings.embedding_model
                    and chapters
                ):
                    return VideoKnowledge(
                        video_id, language, segments, chunks, chapters, claims
                    ), True
            else:
                chunks = chunk_transcript(
                    video_id,
                    segments,
                    self.settings.chunk_seconds,
                    self.settings.overlap_seconds,
                    language,
                )
        else:
            source_id, segments, language = transcribe()
            if source_id != video_id:
                raise RuntimeError("The transcript source did not match the requested video.")
            chunks = chunk_transcript(
                video_id,
                segments,
                self.settings.chunk_seconds,
                self.settings.overlap_seconds,
                language,
            )
        if not chunks:
            raise RuntimeError("No transcript text was available to index.")
        retrieved = [RetrievedChunk(chunk=chunk, score=1.0) for chunk in chunks]
        self.cache.save(
            video_id, segments, language, chunks, [], [],
            self.settings.embedding_model, self.settings.chunk_seconds,
            self.settings.overlap_seconds,
        )
        retriever = HybridRetriever(chunks, self.settings.embedding_model)
        self.cache.save_embeddings(video_id, retriever.embeddings)
        chapters, claims = self.llm.summarize(retrieved)
        self.cache.save(
            video_id,
            segments,
            language,
            chunks,
            chapters,
            claims,
            self.settings.embedding_model,
            self.settings.chunk_seconds,
            self.settings.overlap_seconds,
        )
        return VideoKnowledge(video_id, language, segments, chunks, chapters, claims), False

    def retriever(self, video: VideoKnowledge) -> HybridRetriever:
        """Create a dense and lexical retriever for an ingested video."""
        embeddings = self.cache.load_embeddings(video.video_id)
        if embeddings is None:
            retriever = HybridRetriever(video.chunks, self.settings.embedding_model)
            self.cache.save_embeddings(video.video_id, retriever.embeddings)
            return retriever
        return HybridRetriever(video.chunks, self.settings.embedding_model, embeddings=embeddings)

    def retrieve(self, retriever: HybridRetriever, query: str) -> list[RetrievedChunk]:
        """Run the configured hybrid retrieval step."""
        return retriever.retrieve(
            query,
            top_k=self.settings.retrieval_candidates,
            method="hybrid",
            rrf_k=self.settings.rrf_k,
        )

    def answer(self, question: str, chunks: list[RetrievedChunk]) -> str:
        """Generate a grounded answer, abstaining when retrieved evidence is weak."""
        from youtube_brain.generation import ABSTENTION

        if not chunks or max(item.score for item in chunks) < self.settings.minimum_relevance:
            return ABSTENTION
        return self.llm.answer(question, chunks)
