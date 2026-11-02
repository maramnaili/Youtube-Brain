"""Multilingual dense, BM25, and reciprocal-rank-fusion retrieval."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from youtube_brain.fusion import reciprocal_rank_fusion
from youtube_brain.models import Chunk


def tokenize(text: str) -> list[str]:
    """Split text into case-folded Unicode word tokens."""
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


class HybridRetriever:
    """Rank transcript chunks using normalized multilingual embeddings and BM25."""

    def __init__(
        self, chunks: list[Chunk], model_name: str, embeddings: np.ndarray | None = None
    ) -> None:
        if not chunks:
            raise ValueError("Cannot create a retriever without transcript chunks.")
        self.chunks = chunks
        try:
            from sentence_transformers import SentenceTransformer
        except (ImportError, OSError) as error:
            raise RuntimeError(
                "Could not initialize sentence-transformers. Check that the project "
                "dependencies are installed correctly and that the computer has enough "
                f"available memory. Details: {error}"
            ) from error
        try:
            self.model = SentenceTransformer(model_name)
        except OSError as error:
            if getattr(error, "winerror", None) == 1455 or error.errno == 1455:
                raise RuntimeError(
                    f"Could not load the embedding model '{model_name}' because Windows "
                    "does not have enough virtual memory (error 1455). Increase the "
                    "Windows paging file size, make sure the drive has free disk space, "
                    "and restart the computer before retrying."
                ) from error
            raise RuntimeError(
                f"Could not load the embedding model '{model_name}'. Check that this "
                "computer can access huggingface.co and retry. If the download was "
                "interrupted, retrying will let the model download again. If Hugging "
                "Face is blocked on your network, download the model on an accessible "
                "network and set pipeline.embedding_model in config.yaml to its local "
                f"folder. Details: {error}"
            ) from error
        if embeddings is None:
            embeddings = self.model.encode(
                [f"passage: {chunk.text}" for chunk in chunks],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        if len(embeddings) != len(chunks):
            raise ValueError("Cached embedding count does not match transcript chunk count.")
        self.embeddings = embeddings
        self.bm25 = BM25Okapi([tokenize(chunk.text) for chunk in chunks])

    def retrieve(
        self, query: str, top_k: int = 5, method: str = "hybrid", rrf_k: int = 60
    ) -> list[RetrievedChunk]:
        """Retrieve the best chunks with hybrid, dense-only, or BM25-only ranking."""
        if not query.strip():
            raise ValueError("Enter a question before searching.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        query_embedding = self.model.encode(
            [f"query: {query}"],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        dense_scores = np.asarray(self.embeddings @ query_embedding).reshape(-1)
        dense_ranking = np.argsort(-dense_scores).tolist()
        lexical_scores = self.bm25.get_scores(tokenize(query))
        lexical_ranking = np.argsort(-lexical_scores).tolist()

        if method == "dense":
            ranked = [(index, float(dense_scores[index])) for index in dense_ranking[:top_k]]
        elif method == "bm25":
            ranked = [(index, float(lexical_scores[index])) for index in lexical_ranking[:top_k]]
        elif method == "hybrid":
            fused = reciprocal_rank_fusion(
                [dense_ranking[:top_k * 2], lexical_ranking[:top_k * 2]], k=rrf_k
            )
            ranked = [
                (index, float(dense_scores[index])) for index, _score in fused[:top_k]
            ]
        else:
            raise ValueError(f"Unknown retrieval method: {method}")

        return [RetrievedChunk(self.chunks[index], score) for index, score in ranked]
