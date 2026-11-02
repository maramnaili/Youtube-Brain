import json
import unittest
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

from youtube_brain.chunking import chunk_transcript
from youtube_brain.config import load_settings
from youtube_brain.generation import ABSTENTION, LanguageModel, _post_json
from youtube_brain.models import Chunk
from youtube_brain.ingestion import validate_upload
from youtube_brain.fusion import reciprocal_rank_fusion
from youtube_brain.models import Segment
from youtube_brain.retrieval import HybridRetriever
from youtube_brain.retrieval import RetrievedChunk
from youtube_brain.timestamps import (
    extract_video_id,
    format_citation_timestamp,
    format_timestamp,
    youtube_url,
)


class TimestampTests(unittest.TestCase):
    def test_timestamp_formatting_handles_hours_and_negative_values(self):
        self.assertEqual(format_timestamp(0), "00:00")
        self.assertEqual(format_timestamp(3661), "1:01:01")
        self.assertEqual(format_timestamp(-10), "00:00")

    def test_extract_youtube_video_ids(self):
        video_id = "aircAruvnKk"
        self.assertEqual(
            extract_video_id(f"https://www.youtube.com/watch?v={video_id}&t=30s"),
            video_id,
        )
        self.assertEqual(extract_video_id(f"https://youtu.be/{video_id}"), video_id)
        self.assertEqual(extract_video_id(f"https://youtube.com/shorts/{video_id}"), video_id)

    def test_invalid_youtube_url_is_rejected(self):
        with self.assertRaises(ValueError):
            extract_video_id("https://example.com/watch?v=aircAruvnKk")

    def test_timestamp_url_points_to_requested_second(self):
        self.assertEqual(
            youtube_url("aircAruvnKk", 65),
            "https://www.youtube.com/watch?v=aircAruvnKk&t=65s",
        )

    def test_citation_timestamps_remain_mm_ss_after_one_hour(self):
        self.assertEqual(format_citation_timestamp(3661), "61:01")


class UploadValidationTests(unittest.TestCase):
    def test_rejects_mismatched_media_content(self):
        with self.assertRaises(ValueError):
            validate_upload("audio.wav", b"not a wave file", 1024)

    def test_rejects_oversized_upload(self):
        with self.assertRaises(ValueError):
            validate_upload("audio.wav", b"RIFFxxxxWAVE", 4)

    def test_accepts_valid_wave_signature(self):
        validate_upload("audio.wav", b"RIFFxxxxWAVE", 16)


class ChunkingTests(unittest.TestCase):
    def test_chunking_preserves_text_and_overlaps_windows(self):
        segments = [
            Segment(0, 20, "first"),
            Segment(25, 40, "second"),
            Segment(45, 65, "third"),
            Segment(70, 90, "fourth"),
        ]
        chunks = chunk_transcript("video", segments, chunk_seconds=60, overlap_seconds=20)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].text, "first second third")
        self.assertEqual(chunks[1].text, "third fourth")
        self.assertEqual(chunks[0].start_time, 0)
        self.assertEqual(chunks[1].start_time, 40)

    def test_invalid_chunk_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            chunk_transcript("video", [Segment(0, 1, "text")], 10, 10)

    def test_segment_ending_at_window_boundary_is_not_repeated(self):
        chunks = chunk_transcript(
            "video",
            [Segment(0, 40, "first"), Segment(40, 60, "second")],
            chunk_seconds=40,
            overlap_seconds=0,
        )
        self.assertEqual([chunk.text for chunk in chunks], ["first", "second"])


class RetrievalFusionTests(unittest.TestCase):
    def test_rrf_combines_rankings_and_breaks_ties_deterministically(self):
        self.assertEqual(
            reciprocal_rank_fusion([[2, 1], [1, 2]], k=60),
            [(1, 1 / 61 + 1 / 62), (2, 1 / 61 + 1 / 62)],
        )

    def test_rrf_rejects_invalid_k(self):
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([[0]], k=0)

    def test_embedding_model_load_failure_is_actionable(self):
        chunk = Chunk("video", "video-0000", 0, 10, "transcript")
        with patch.dict(
            sys.modules,
            {
                "sentence_transformers": SimpleNamespace(
                    SentenceTransformer=Mock(side_effect=OSError("model files are unavailable"))
                )
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "access huggingface.co"):
                HybridRetriever([chunk], "intfloat/multilingual-e5-small")

    def test_embedding_model_page_file_error_has_windows_guidance(self):
        chunk = Chunk("video", "video-0000", 0, 10, "transcript")

        class PageFileError(OSError):
            winerror = 1455

        with patch.dict(
            sys.modules,
            {
                "sentence_transformers": SimpleNamespace(
                    SentenceTransformer=Mock(
                        side_effect=PageFileError("paging file is too small")
                    )
                )
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "Windows.*virtual memory"):
                HybridRetriever([chunk], "intfloat/multilingual-e5-small")


class LLMRequestTests(unittest.TestCase):
    def test_answer_prompt_prefers_explicit_evidence_and_cites_retrieved_timestamp(self):
        settings = load_settings("config.yaml")
        model = LanguageModel(settings)
        retrieved = [
            RetrievedChunk(
                chunk=Chunk(
                    "video",
                    "chunk-a",
                    140,
                    180,
                    "L'inférence consiste à appliquer un modèle entraîné pour générer "
                    "des prédictions à partir de nouvelles données.",
                ),
                score=1.0,
            )
        ]
        with patch.object(
            model,
            "complete",
            return_value=(
                "L'inférence applique un modèle entraîné pour produire des prédictions "
                "[02:20]"
            ),
        ) as complete:
            answer = model.answer("Que signifie l'inférence en IA ?", retrieved)

        self.assertEqual(
            answer,
            "L'inférence applique un modèle entraîné pour produire des prédictions [02:20]",
        )
        system_prompt = complete.call_args.args[0]
        self.assertIn("If so, state that answer; do not abstain.", system_prompt)
        self.assertIn("copy the supporting passage timestamp exactly", system_prompt)
        self.assertNotIn(ABSTENTION, system_prompt)

    def test_llm_timeout_is_configured_and_passed_to_http_request(self):
        settings = load_settings("config.yaml")
        self.assertIsNone(settings.llm_timeout_seconds)

        with patch("youtube_brain.generation.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value.read.return_value = b'{"ok": true}'
            self.assertEqual(
                _post_json(
                    "http://localhost:11434/api/chat",
                    {"model": "qwen2.5:3b"},
                    timeout_seconds=settings.llm_timeout_seconds,
                ),
                {"ok": True},
            )
        self.assertIsNone(urlopen.call_args.kwargs["timeout"])

    def test_ollama_summary_requests_json_output(self):
        settings = load_settings("config.yaml")
        model = LanguageModel(settings)
        schema = {
            "type": "object",
            "properties": {"timestamp": {"type": "string", "enum": ["00:00"]}},
        }
        with patch("youtube_brain.generation._post_json") as post_json:
            post_json.return_value = {"message": {"content": '{"chapters": [], "claims": []}'}}
            self.assertEqual(model.complete("system", "prompt", json_schema=schema),
                             '{"chapters": [], "claims": []}')
        self.assertEqual(post_json.call_args.args[1]["format"], schema)
        self.assertEqual(
            post_json.call_args.args[1]["options"],
            {
                "temperature": 0.2,
                "seed": 42,
                "repeat_penalty": 1.15,
                "num_predict": 1200,
            },
        )

    def test_summary_schema_limits_timestamps_to_input_chunks(self):
        settings = load_settings("config.yaml")
        model = LanguageModel(settings)
        chunks = [
            RetrievedChunk(
                chunk=Chunk("video", "chunk-a", 0, 10, "transcript"),
                score=1.0,
            ),
            RetrievedChunk(
                chunk=Chunk("video", "chunk-b", 70, 80, "transcript"),
                score=1.0,
            ),
        ]
        valid_response = {
            "chapters": [
                {"title": f"Topic {index}", "timestamp": "00:00"} for index in range(5)
            ],
            "claims": [
                {"claim": f"Claim {index}", "timestamp": "00:00", "quote": "transcript"}
                for index in range(5)
            ],
        }
        with patch.object(model, "complete", return_value=json.dumps(valid_response)) as complete:
            model.summarize(chunks)
        schema = complete.call_args.kwargs["json_schema"]
        chapter_timestamps = schema["properties"]["chapters"]["items"]["properties"]["timestamp"]
        claim_timestamps = schema["properties"]["claims"]["items"]["properties"]["timestamp"]
        self.assertEqual(chapter_timestamps["enum"], ["00:00", "01:10"])
        self.assertEqual(claim_timestamps["enum"], ["00:00", "01:10"])

    def test_summary_accepts_fewer_than_five_transcript_supported_claims(self):
        settings = load_settings("config.yaml")
        model = LanguageModel(settings)
        chunk = Chunk(
            "video",
            "chunk-a",
            0,
            10,
            "L’intelligence artificielle apprend à prédire la suite du texte.",
        )
        response = {
            "chapters": [{"title": "Apprentissage", "timestamp": "00:00"}],
            "claims": [
                {
                    "claim": "Le modèle prédit le texte.",
                    "timestamp": "00:00",
                    "quote": "prédire la suite du texte",
                },
                {
                    "claim": "Une affirmation non fondée.",
                    "timestamp": "00:00",
                    "quote": "Cette phrase ne figure pas dans la transcription.",
                },
            ],
        }
        with patch.object(model, "complete", return_value=json.dumps(response)):
            chapters, claims = model.summarize([RetrievedChunk(chunk=chunk, score=1.0)])

        self.assertEqual(len(chapters), 1)
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].claim, "Le modèle prédit le texte.")

    def test_empty_summary_response_reports_the_actual_problem(self):
        settings = load_settings("config.yaml")
        model = LanguageModel(settings)
        chunk = Chunk("video", "video-0000", 0, 10, "transcript")
        with patch.object(model, "complete", return_value=""):
            with self.assertRaisesRegex(RuntimeError, "empty response"):
                model.summarize([RetrievedChunk(chunk=chunk, score=1.0)])

    def test_summary_rejects_transcription_json_with_actionable_error(self):
        settings = load_settings("config.yaml")
        model = LanguageModel(settings)
        chunk = Chunk("video", "video-0000", 0, 10, "transcript")
        with patch.object(
            model,
            "complete",
            return_value='{"transcription": [{"text": "transcript"}]}',
        ):
            with self.assertRaisesRegex(
                RuntimeError, "expected 'chapters' and 'claims'.*transcription"
            ):
                model.summarize([RetrievedChunk(chunk=chunk, score=1.0)])


if __name__ == "__main__":
    unittest.main()
