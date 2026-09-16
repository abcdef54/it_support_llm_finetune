from __future__ import annotations

import hashlib
import random
from pathlib import Path
from types import ModuleType
from typing import Callable

from tqdm import tqdm

from benchmark.common.config import (
    BERTSCORE_BATCH_SIZE,
    BERTSCORE_IDF,
    BERTSCORE_LANGUAGE,
    BERTSCORE_MODEL,
    BERTSCORE_NUM_LAYERS,
    BERTSCORE_RESCALE_WITH_BASELINE,
    JUDGE_MAX_NEW_TOKENS,
    JUDGE_RETRY_MAX_NEW_TOKENS,
    JUDGE_MODEL_ID,
    JUDGE_MODEL_REVISION,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_TEMPERATURE,
    JUDGE_USER_TEMPLATE,
    TECHQA_SHA256,
)
from benchmark.common.dataset import load_techqa
from benchmark.common.generation import batches
from benchmark.common.judge import build_judge_messages, parse_or_retry_judge_output
from benchmark.common.metrics import aggregate_metrics, bertscore_f1, is_abstention, rouge_l_f1
from benchmark.common.schemas import ExampleMetrics, ExampleResult, write_results


def _set_seed(seed: int) -> None:
    random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_benchmark(
    project_root: Path,
    *,
    settings: ModuleType,
    load_answer_model: Callable,
    load_judge_model: Callable,
    reuse_answer_as_judge: bool,
    overwrite: bool = False,
    extra_metadata: dict | None = None,
) -> dict:
    if settings.ANSWER_BATCH_SIZE < 1 or settings.JUDGE_BATCH_SIZE < 1:
        raise ValueError("Answer and judge batch sizes must be positive")
    if JUDGE_RETRY_MAX_NEW_TOKENS <= JUDGE_MAX_NEW_TOKENS:
        raise ValueError("Judge retry token limit must exceed the initial limit")
    if reuse_answer_as_judge and (settings.BASE_MODEL_ID, settings.BASE_MODEL_REVISION) != (JUDGE_MODEL_ID, JUDGE_MODEL_REVISION):
        raise ValueError("Cannot reuse the answer model as a different judge model")
    dataset_path = project_root / settings.DATASET_PATH
    predictions_path = project_root / settings.PREDICTIONS_PATH
    metrics_path = project_root / settings.METRICS_PATH
    if not overwrite and (predictions_path.exists() or metrics_path.exists()):
        label = "Base" if reuse_answer_as_judge else "Fine-tuned"
        raise FileExistsError(f"{label} benchmark output already exists; pass --overwrite to replace it")

    examples = load_techqa(dataset_path)
    _set_seed(settings.RANDOM_SEED)

    active_model = load_answer_model()
    try:
        generated_answers = []
        for batch in tqdm(
            batches(examples, settings.ANSWER_BATCH_SIZE),
            total=(len(examples) + settings.ANSWER_BATCH_SIZE - 1) // settings.ANSWER_BATCH_SIZE,
            desc="Generating baseline answers" if reuse_answer_as_judge else "Generating fine-tuned answers",
        ):
            messages = [
                [
                    {"role": "system", "content": settings.ANSWER_SYSTEM_PROMPT},
                    {"role": "user", "content": example.question},
                ]
                for example in batch
            ]
            generated_answers.extend(active_model.generate_batch(messages, settings.MAX_NEW_TOKENS))

        if not reuse_answer_as_judge:
            active_model.close()
            active_model = None
            active_model = load_judge_model()

        decisions = []
        judge_retries = 0
        pairs = list(zip(examples, generated_answers, strict=True))
        for batch in tqdm(
            batches(pairs, settings.JUDGE_BATCH_SIZE),
            total=(len(pairs) + settings.JUDGE_BATCH_SIZE - 1) // settings.JUDGE_BATCH_SIZE,
            desc="Judging answers",
        ):
            judge_messages = [build_judge_messages(example, answer) for example, answer in batch]
            raw_decisions = active_model.generate_batch(judge_messages, JUDGE_MAX_NEW_TOKENS)
            for (example, _), judge_message, raw_decision in zip(batch, judge_messages, raw_decisions, strict=True):
                try:
                    decision, retried = parse_or_retry_judge_output(active_model, judge_message, raw_decision)
                except ValueError as exc:
                    raise ValueError(f"Invalid judge response for TechQA {example.id}: {exc}") from exc
                judge_retries += retried
                decisions.append(decision)
    finally:
        if active_model is not None:
            active_model.close()

    references = [example.answer or "" for example in examples]
    semantic_scores = bertscore_f1(
        generated_answers,
        references,
        model_type=BERTSCORE_MODEL,
        num_layers=BERTSCORE_NUM_LAYERS,
        batch_size=BERTSCORE_BATCH_SIZE,
        lang=BERTSCORE_LANGUAGE,
        idf=BERTSCORE_IDF,
        rescale_with_baseline=BERTSCORE_RESCALE_WITH_BASELINE,
    )

    results = [
        ExampleResult(
            id=example.id,
            question=example.question,
            reference_answer=example.answer,
            generated_answer=generated_answer,
            metrics=ExampleMetrics(
                rouge_l_f1=rouge_l_f1(generated_answer, example.answer or ""),
                bertscore_f1=semantic_score,
                judge_score=decision.score,
                judge_reason=decision.reason,
                abstained=is_abstention(generated_answer),
            ),
        )
        for example, generated_answer, semantic_score, decision in zip(
            examples, generated_answers, semantic_scores, decisions, strict=True
        )
    ]
    summary = aggregate_metrics(results)
    summary["judge_retries"] = judge_retries
    summary["benchmark"] = {"path": settings.DATASET_PATH, "sha256": TECHQA_SHA256}
    summary["configuration"] = {
        "base_model": settings.BASE_MODEL_ID,
        "base_model_revision": settings.BASE_MODEL_REVISION,
        "generation_mode": settings.GENERATION_MODE,
        "generation_temperature": settings.GENERATION_TEMPERATURE,
        "model_dtype": settings.MODEL_DTYPE,
        "max_input_tokens": settings.MAX_INPUT_TOKENS,
        "max_new_tokens": settings.MAX_NEW_TOKENS,
        "answer_batch_size": settings.ANSWER_BATCH_SIZE,
        "judge_batch_size": settings.JUDGE_BATCH_SIZE,
        "random_seed": settings.RANDOM_SEED,
        "thinking_enabled": settings.ENABLE_THINKING,
        "bertscore_model": BERTSCORE_MODEL,
        "bertscore_num_layers": BERTSCORE_NUM_LAYERS,
        "bertscore_batch_size": BERTSCORE_BATCH_SIZE,
        "bertscore_language": BERTSCORE_LANGUAGE,
        "bertscore_idf": BERTSCORE_IDF,
        "bertscore_rescale_with_baseline": BERTSCORE_RESCALE_WITH_BASELINE,
        "judge_model": JUDGE_MODEL_ID,
        "judge_model_revision": JUDGE_MODEL_REVISION,
        "judge_temperature": JUDGE_TEMPERATURE,
        "judge_max_new_tokens": JUDGE_MAX_NEW_TOKENS,
        "judge_retry_max_new_tokens": JUDGE_RETRY_MAX_NEW_TOKENS,
        "judge_prompt_sha256": hashlib.sha256(
            (JUDGE_SYSTEM_PROMPT + "\n" + JUDGE_USER_TEMPLATE).encode("utf-8")
        ).hexdigest(),
    }
    if extra_metadata:
        summary.update(extra_metadata)
    write_results(predictions_path, metrics_path, results, summary)
    return summary
