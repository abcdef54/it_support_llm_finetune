from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from data.utils import write_json, write_jsonl


@dataclass(frozen=True)
class TechQAExample:
    id: str
    question: str
    answer: str | None
    original_split: str
    answerable: bool


@dataclass
class ExampleMetrics:
    rouge_l_f1: float
    bertscore_f1: float
    judge_score: int
    judge_reason: str
    abstained: bool


@dataclass
class ExampleResult:
    id: str
    question: str
    reference_answer: str | None
    generated_answer: str
    metrics: ExampleMetrics

    def to_dict(self) -> dict:
        return asdict(self)


def write_results(
    predictions_path: Path,
    metrics_path: Path,
    results: list[ExampleResult],
    summary: dict,
) -> None:
    write_jsonl(predictions_path, (result.to_dict() for result in results))
    write_json(metrics_path, summary)
