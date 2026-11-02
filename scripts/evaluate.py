"""Run timestamped retrieval and abstention metrics on a labeled JSON dataset."""

import argparse
import csv
import json
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from youtube_brain.config import load_settings
from youtube_brain.generation import ABSTENTION
from youtube_brain.pipeline import VideoPipeline
from youtube_brain.timestamps import format_citation_timestamp

METHODS = ("dense", "bm25", "hybrid")


def _citation_seconds(answer: str) -> list[int]:
    """Extract MM:SS citations from an answer as integer seconds."""
    result = []
    for minutes, seconds in re.findall(r"\[(\d+):(\d{2})\]", answer):
        result.append(int(minutes) * 60 + int(seconds))
    return result


def _metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate retrieval, timestamp, abstention, and latency measurements."""
    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]
    true_abstentions = sum(row["abstained"] for row in unanswerable)
    false_abstentions = sum(row["abstained"] for row in answerable)
    missed_abstentions = len(unanswerable) - true_abstentions
    precision_denominator = true_abstentions + false_abstentions
    recall_denominator = true_abstentions + missed_abstentions

    return {
        "questions": float(len(rows)),
        "recall_at_1": _recall_at_k(answerable, 1),
        "recall_at_3": _recall_at_k(answerable, 3),
        "recall_at_5": _recall_at_k(answerable, 5),
        "mrr": statistics.mean([row["reciprocal_rank"] for row in answerable]) if answerable else 0.0,
        "timestamp_accuracy_30s": (
            sum(row["timestamp_correct"] for row in answerable) / len(answerable)
            if answerable else 0.0
        ),
        "abstention_precision": (
            true_abstentions / precision_denominator if precision_denominator else 0.0
        ),
        "abstention_recall": (
            true_abstentions / recall_denominator if recall_denominator else 0.0
        ),
        "mean_latency_seconds": statistics.mean([row["latency_seconds"] for row in rows]) if rows else 0.0,
    }


def _recall_at_k(rows: list[dict[str, Any]], k: int) -> float:
    if not rows:
        return 0.0
    return sum(row["first_relevant_rank"] is not None and row["first_relevant_rank"] <= k
               for row in rows) / len(rows)


def run_evaluation(dataset_path: Path, output_dir: Path) -> None:
    """Evaluate dense, lexical, and fused retrieval on labeled questions."""
    with dataset_path.open("r", encoding="utf-8") as dataset_file:
        dataset = json.load(dataset_file)
    questions = dataset.get("questions", [])
    if not questions:
        raise ValueError(
            f"No labeled questions found in {dataset_path}; add answerable and unanswerable examples."
        )

    pipeline = VideoPipeline(load_settings())
    videos = {}
    retrievers = {}
    rows_by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for question_data in questions:
        video_url = question_data["video_url"]
        if video_url not in videos:
            videos[video_url], _ = pipeline.ingest_youtube(video_url)
            retrievers[video_url] = pipeline.retriever(videos[video_url])
        video = videos[video_url]
        query = str(question_data["question"])
        answerable = bool(question_data["answerable"])
        reference_start = float(question_data.get("reference_start", 0))
        reference_end = float(question_data.get("reference_end", reference_start))

        for method in METHODS:
            started = time.perf_counter()
            retrieved = retrievers[video_url].retrieve(
                query,
                top_k=5,
                method=method,
                rrf_k=pipeline.settings.rrf_k,
            )
            relevant_ranks = [
                rank
                for rank, item in enumerate(retrieved, start=1)
                if item.chunk.start_time <= reference_end and item.chunk.end_time >= reference_start
            ]
            answer = pipeline.answer(query, retrieved)
            elapsed = time.perf_counter() - started
            cited_times = _citation_seconds(answer)
            timestamp_correct = (
                any(reference_start - 30 <= timestamp <= reference_end + 30 for timestamp in cited_times)
                if answerable else False
            )
            rows_by_method[method].append(
                {
                    "video_url": video_url,
                    "question": query,
                    "method": method,
                    "answerable": answerable,
                    "answer": answer,
                    "abstained": answer == ABSTENTION,
                    "first_relevant_rank": min(relevant_ranks) if relevant_ranks else None,
                    "reciprocal_rank": 1 / min(relevant_ranks) if relevant_ranks else 0.0,
                    "timestamp_correct": timestamp_correct,
                    "latency_seconds": round(elapsed, 3),
                    "retrieved_timestamps": ";".join(
                        format_citation_timestamp(item.chunk.start_time) for item in retrieved
                    ),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "evaluation.csv"
    all_rows = [row for method_rows in rows_by_method.values() for row in method_rows]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    report_lines = [
        "# YouTube Brain Evaluation",
        "",
        f"Questions: {len(questions)}",
        "",
        "| Method | Recall@1 | Recall@3 | Recall@5 | MRR | Timestamp ±30s | Abstention precision | Abstention recall | Mean latency (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        metrics = _metrics(rows_by_method[method])
        report_lines.append(
            f"| {method} | {metrics['recall_at_1']:.3f} | {metrics['recall_at_3']:.3f} "
            f"| {metrics['recall_at_5']:.3f} | {metrics['mrr']:.3f} "
            f"| {metrics['timestamp_accuracy_30s']:.3f} | {metrics['abstention_precision']:.3f} "
            f"| {metrics['abstention_recall']:.3f} | {metrics['mean_latency_seconds']:.3f} |"
        )
    report_lines.extend(
        [
            "",
            "Retrieval recall and MRR use overlap with each answer's labeled reference interval. "
            "Timestamp accuracy measures whether any cited time falls within 30 seconds of that interval.",
            "Review sampled answers manually for faithfulness and relevance; these subjective metrics "
            "are not automatically scored by this script.",
            "",
        ]
    )
    (output_dir / "evaluation.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Wrote {csv_path} and {output_dir / 'evaluation.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/evaluation/questions.json"),
        help="Labeled questions JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory for CSV and Markdown reports",
    )
    arguments = parser.parse_args()
    run_evaluation(arguments.dataset, arguments.output_dir)


if __name__ == "__main__":
    main()
