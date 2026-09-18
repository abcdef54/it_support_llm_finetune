from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Callable

from tqdm import tqdm

from benchmark.common.checkpoint import Checkpoint
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
from benchmark.common.dataset import load_general_it, load_techqa
from benchmark.common.generation import batches
from benchmark.common.judge import JudgeDecision, build_judge_messages, parse_or_retry_judge_output
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
    settings: ModuleType | SimpleNamespace,
    load_answer_model: Callable,
    load_judge_model: Callable,
    reuse_answer_as_judge: bool,
    overwrite: bool = False,
    resume: bool = False,
    extra_metadata: dict | None = None,
    rag_records: dict | None = None,
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

    dataset_kind = getattr(settings, "DATASET_KIND", "techqa")
    if rag_records is not None and dataset_kind != "general_it":
        raise ValueError("TechQA is indexed knowledge and cannot be a held-out RAG benchmark")
    if dataset_kind == "general_it":
        manifest = json.loads((project_root / settings.DATASET_MANIFEST_PATH).read_text(encoding="utf-8"))
        if manifest["benchmark_path"] != settings.DATASET_PATH or manifest["benchmark_count"] != 1000:
            raise ValueError("General-IT benchmark path or count differs from the approved layout")
        benchmark_sha256 = manifest["benchmark_sha256"]
        examples = load_general_it(dataset_path, expected_sha256=benchmark_sha256)
    elif dataset_kind == "techqa":
        examples = load_techqa(dataset_path)
        benchmark_sha256 = TECHQA_SHA256
    else:
        raise ValueError(f"Unknown benchmark dataset kind: {dataset_kind}")
    fingerprint_data = {
        "examples": [asdict(example) for example in examples],
        "settings": {name: getattr(settings, name) for name in dir(settings) if name.isupper()},
        "judge": [JUDGE_MODEL_ID, JUDGE_MODEL_REVISION, JUDGE_MAX_NEW_TOKENS,
                  JUDGE_RETRY_MAX_NEW_TOKENS, JUDGE_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE],
        "bertscore": [BERTSCORE_MODEL, BERTSCORE_NUM_LAYERS, BERTSCORE_BATCH_SIZE,
                      BERTSCORE_LANGUAGE, BERTSCORE_IDF, BERTSCORE_RESCALE_WITH_BASELINE],
        "extra_metadata": extra_metadata,
    }
    if dataset_kind == "general_it":
        fingerprint_data["benchmark_sha256"] = benchmark_sha256
    if rag_records is not None:
        if set(rag_records) != {example.id for example in examples}:
            raise ValueError("RAG contexts do not cover exactly the benchmark IDs")
        fingerprint_data["rag_contexts_sha256"] = hashlib.sha256(
            json.dumps(rag_records, sort_keys=True).encode("utf-8")).hexdigest()
    fingerprint = hashlib.sha256(json.dumps(fingerprint_data, sort_keys=True).encode("utf-8")).hexdigest()
    checkpoint = Checkpoint(predictions_path.parent / "progress.sqlite3", fingerprint, resume=resume)

    active_model = None
    try:
        _set_seed(settings.RANDOM_SEED)
        generated_answers = checkpoint.load("answer")
        if len(generated_answers) > len(examples) or any(not isinstance(answer, str) for answer in generated_answers):
            raise ValueError("Checkpoint contains invalid answers")
        if len(generated_answers) < len(examples):
            active_model = load_answer_model()
        for batch in tqdm(
            batches(examples[len(generated_answers):], settings.ANSWER_BATCH_SIZE),
            total=(len(examples) - len(generated_answers) + settings.ANSWER_BATCH_SIZE - 1) // settings.ANSWER_BATCH_SIZE,
            desc="Generating baseline answers" if reuse_answer_as_judge else "Generating fine-tuned answers",
        ):
            messages = [
                [
                    {"role": "system", "content": settings.ANSWER_SYSTEM_PROMPT},
                    {"role": "user", "content": example.question},
                ]
                for example in batch
            ]
            if rag_records is not None:
                messages = [rag_records[example.id]["messages"] for example in batch]
            answers = active_model.generate_batch(messages, settings.MAX_NEW_TOKENS)
            if len(answers) != len(batch) or any(not isinstance(answer, str) for answer in answers):
                raise ValueError("Answer model returned an unexpected batch")
            checkpoint.save("answer", len(generated_answers), answers)
            generated_answers.extend(answers)

        saved_decisions = checkpoint.load("judge")
        if len(saved_decisions) > len(examples):
            raise ValueError("Checkpoint contains too many judgments")
        decisions = [JudgeDecision(**row["decision"]) for row in saved_decisions]
        judge_retries = sum(row["retried"] for row in saved_decisions)
        pairs = list(zip(examples, generated_answers, strict=True))
        if len(decisions) < len(pairs):
            if active_model is not None and not reuse_answer_as_judge:
                active_model.close()
                active_model = None
            if active_model is None:
                active_model = load_judge_model()
        for batch in tqdm(
            batches(pairs[len(decisions):], settings.JUDGE_BATCH_SIZE),
            total=(len(pairs) - len(decisions) + settings.JUDGE_BATCH_SIZE - 1) // settings.JUDGE_BATCH_SIZE,
            desc="Judging answers",
        ):
            judge_messages = [build_judge_messages(example, answer) for example, answer in batch]
            raw_decisions = active_model.generate_batch(judge_messages, JUDGE_MAX_NEW_TOKENS)
            if len(raw_decisions) != len(batch):
                raise ValueError("Judge returned an unexpected batch")
            completed_batch = []
            for (example, _), judge_message, raw_decision in zip(batch, judge_messages, raw_decisions, strict=True):
                try:
                    decision, retried = parse_or_retry_judge_output(active_model, judge_message, raw_decision)
                except ValueError as exc:
                    raise ValueError(f"Invalid judge response for benchmark item {example.id}: {exc}") from exc
                judge_retries += retried
                decisions.append(decision)
                completed_batch.append({"decision": asdict(decision), "retried": retried})
            checkpoint.save("judge", len(decisions) - len(completed_batch), completed_batch)
    finally:
        try:
            if active_model is not None:
                active_model.close()
        finally:
            checkpoint.close()

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
                judge_recovered=decision.recovered,
            ),
        )
        for example, generated_answer, semantic_score, decision in zip(
            examples, generated_answers, semantic_scores, decisions, strict=True
        )
    ]
    summary = aggregate_metrics(results)
    summary["judge_retries"] = judge_retries
    summary["judge_recovered_scores"] = sum(decision.recovered for decision in decisions)
    summary["benchmark"] = {"path": settings.DATASET_PATH, "sha256": benchmark_sha256,
                            "examples": len(examples)}
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
    if rag_records is not None:
        from collections import Counter
        used = [r for example in examples for r in rag_records[example.id]["used_documents"]]
        summary["retrieval_diagnostics"] = {
            "average_documents_used": len(used) / len(examples),
            "average_context_tokens": sum(rag_records[e.id]["context_tokens"] for e in examples) / len(examples),
            "source_distribution": dict(Counter(r["source"] for r in used)),
        }
        for result in results:
            result.rag = {key: value for key, value in rag_records[result.id].items()
                          if key not in {"id", "question", "messages"}}
    write_results(predictions_path, metrics_path, results, summary)
    checkpoint.path.unlink()
    return summary
