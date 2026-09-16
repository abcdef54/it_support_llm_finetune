from __future__ import annotations

TECHQA_RECORDS = 902
TECHQA_SHA256 = "d00913397224f4def4a5435bdf8c7a6f7d5b66d4a72dd4648694f8991f22e7e3"

BERTSCORE_MODEL = "microsoft/deberta-large-mnli"
BERTSCORE_NUM_LAYERS: int | None = None  # Use BERTScore's recommended layer for this model.
BERTSCORE_BATCH_SIZE = 8
BERTSCORE_LANGUAGE = "en"
BERTSCORE_IDF = False
BERTSCORE_RESCALE_WITH_BASELINE = False

# Keep this model and prompt unchanged for every experiment.
JUDGE_MODEL_ID = "Qwen/Qwen3.5-9B"
JUDGE_MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_NEW_TOKENS = 256

JUDGE_SYSTEM_PROMPT = """You are a strict evaluator of IT-support answers.
Judge only technical correctness and relevance using the supplied question,
reference status, reference answer, and generated answer. Return JSON only."""

JUDGE_USER_TEMPLATE = """Score the generated answer on this scale:
0 = Completely incorrect or irrelevant
1 = Mostly incorrect; major technical problems
2 = Partially correct; important information missing or incorrect
3 = Mostly correct; minor problems only
4 = Fully correct and relevant

For an unanswerable benchmark item, an appropriate abstention is fully correct;
inventing a solution is incorrect.

Question:
{question}

Reference status:
{reference_status}

Reference answer:
{reference_answer}

Generated answer:
{generated_answer}

Return exactly one JSON object with this shape:
{{"score": 0, "reason": "short justification"}}"""
