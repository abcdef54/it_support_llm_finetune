from __future__ import annotations

import re
from collections.abc import Sequence
from statistics import fmean

from benchmark.common.schemas import ExampleResult

_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_ABSTENTION_PATTERNS = (
    re.compile(r"\bi (?:do not|don't) know\b"),
    re.compile(r"\bi (?:do not|don't) have enough (?:information|context)\b"),
    re.compile(r"\b(?:cannot|can't|unable to) (?:answer|determine|provide)\b"),
    re.compile(r"\b(?:not enough|insufficient) (?:information|context)\b"),
    re.compile(r"\bno (?:relevant )?information (?:is |was )?provided\b"),
)


def rouge_l_f1(prediction: str, reference: str) -> float:
    predicted_tokens = _TOKEN.findall(prediction.casefold())
    reference_tokens = _TOKEN.findall(reference.casefold())
    if not predicted_tokens or not reference_tokens:
        return 0.0

    previous = [0] * (len(reference_tokens) + 1)
    for predicted in predicted_tokens:
        current = [0]
        for index, reference_token in enumerate(reference_tokens, 1):
            if predicted == reference_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current

    lcs = previous[-1]
    precision = lcs / len(predicted_tokens)
    recall = lcs / len(reference_tokens)
    return 2 * precision * recall / (precision + recall) if lcs else 0.0


def is_abstention(answer: str) -> bool:
    normalized = " ".join(answer.casefold().replace("’", "'").split())
    return not normalized or any(pattern.search(normalized) for pattern in _ABSTENTION_PATTERNS)


def bertscore_f1(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    model_type: str,
    num_layers: int | None,
    batch_size: int,
    lang: str,
    idf: bool,
    rescale_with_baseline: bool,
    device: str | None = None,
    scorer_class=None,
) -> list[float]:
    if len(predictions) != len(references):
        raise ValueError("BERTScore predictions and references must have the same length")
    if not predictions:
        return []
    comparable = [
        index
        for index, (prediction, reference) in enumerate(zip(predictions, references, strict=True))
        if prediction.strip() and reference.strip()
    ]
    if not comparable:
        return [0.0] * len(predictions)
    if scorer_class is None:
        try:
            from bert_score import BERTScorer
        except ImportError as exc:
            raise RuntimeError("BERTScore is required; install project requirements first") from exc
        scorer_class = BERTScorer

    scorer = scorer_class(
        model_type=model_type,
        num_layers=num_layers,
        batch_size=batch_size,
        lang=lang,
        idf=idf,
        rescale_with_baseline=rescale_with_baseline,
        device=device,
    )
    _, _, scores = scorer.score(
        [predictions[index] for index in comparable],
        [references[index] for index in comparable],
    )
    values = [float(score) for score in scores.tolist()]
    if len(values) != len(comparable):
        raise ValueError("BERTScore returned an unexpected number of scores")
    result = [0.0] * len(predictions)
    for index, value in zip(comparable, values, strict=True):
        result[index] = value
    return result


def aggregate_metrics(results: Sequence[ExampleResult]) -> dict[str, float | int]:
    if not results:
        raise ValueError("Cannot aggregate an empty benchmark")
    return {
        "rouge_l_f1": fmean(result.metrics.rouge_l_f1 for result in results),
        "bertscore_f1": fmean(result.metrics.bertscore_f1 for result in results),
        "llm_judge_mean": fmean(result.metrics.judge_score for result in results),
        "llm_judge_correctness_percent": fmean(result.metrics.judge_score for result in results) / 4 * 100,
        "abstention_rate": fmean(result.metrics.abstained for result in results) * 100,
        "total_examples": len(results),
    }
