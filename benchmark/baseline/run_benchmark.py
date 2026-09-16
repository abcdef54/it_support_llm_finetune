from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path

from tqdm import tqdm

from benchmark.baseline.config import (
    ANSWER_SYSTEM_PROMPT,
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    BATCH_SIZE,
    DATASET_PATH,
    ENABLE_THINKING,
    GENERATION_MODE,
    GENERATION_TEMPERATURE,
    MAX_INPUT_TOKENS,
    MAX_NEW_TOKENS,
    METRICS_PATH,
    MODEL_DTYPE,
    PREDICTIONS_PATH,
    RANDOM_SEED,
)
from benchmark.baseline.model import QwenGenerator
from benchmark.common.config import (
    BERTSCORE_BATCH_SIZE,
    BERTSCORE_IDF,
    BERTSCORE_LANGUAGE,
    BERTSCORE_MODEL,
    BERTSCORE_NUM_LAYERS,
    BERTSCORE_RESCALE_WITH_BASELINE,
    JUDGE_MAX_NEW_TOKENS,
    JUDGE_MODEL_ID,
    JUDGE_MODEL_REVISION,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_TEMPERATURE,
    JUDGE_USER_TEMPLATE,
    TECHQA_SHA256,
)
from benchmark.common.dataset import load_techqa
from benchmark.common.judge import build_judge_messages, parse_judge_output
from benchmark.common.metrics import aggregate_metrics, bertscore_f1, is_abstention, rouge_l_f1
from benchmark.common.schemas import ExampleMetrics, ExampleResult, write_results


def _set_seed(seed: int) -> None:
    random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run(project_root: Path, overwrite: bool = False) -> dict:
    dataset_path = project_root / DATASET_PATH
    predictions_path = project_root / PREDICTIONS_PATH
    metrics_path = project_root / METRICS_PATH
    if not overwrite and (predictions_path.exists() or metrics_path.exists()):
        raise FileExistsError("Base benchmark output already exists; pass --overwrite to replace it")

    examples = load_techqa(dataset_path)
    _set_seed(RANDOM_SEED)

    answer_model = QwenGenerator.load(BASE_MODEL_ID, BASE_MODEL_REVISION)
    generated_answers = []
    for example in tqdm(examples, desc="Generating baseline answers"):
        messages = [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": example.question},
        ]
        generated_answers.append(answer_model.generate(messages, MAX_NEW_TOKENS))
    answer_model.close()

    judge_model = QwenGenerator.load(JUDGE_MODEL_ID, JUDGE_MODEL_REVISION)
    decisions = []
    for example, generated_answer in tqdm(
        zip(examples, generated_answers, strict=True), total=len(examples), desc="Judging answers"
    ):
        raw_decision = judge_model.generate(
            build_judge_messages(example, generated_answer), JUDGE_MAX_NEW_TOKENS
        )
        decisions.append(parse_judge_output(raw_decision))
    judge_model.close()

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
    summary["benchmark"] = {"path": DATASET_PATH, "sha256": TECHQA_SHA256}
    summary["configuration"] = {
        "base_model": BASE_MODEL_ID,
        "base_model_revision": BASE_MODEL_REVISION,
        "generation_mode": GENERATION_MODE,
        "generation_temperature": GENERATION_TEMPERATURE,
        "model_dtype": MODEL_DTYPE,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "batch_size": BATCH_SIZE,
        "random_seed": RANDOM_SEED,
        "thinking_enabled": ENABLE_THINKING,
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
        "judge_prompt_sha256": hashlib.sha256(
            (JUDGE_SYSTEM_PROMPT + "\n" + JUDGE_USER_TEMPLATE).encode("utf-8")
        ).hexdigest(),
    }
    write_results(predictions_path, metrics_path, results, summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the complete base-model TechQA benchmark.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(run(args.project_root.resolve(), overwrite=args.overwrite))
