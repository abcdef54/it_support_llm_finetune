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
JUDGE_RETRY_MAX_NEW_TOKENS = 1024

JUDGE_SYSTEM_PROMPT = """You are an evaluator of IT-support answers.

Judge the generated answer based on technical correctness, relevance, and whether
it provides a useful answer to the user's question.

The reference answer is provided as a guide to the expected technical answer.
Do not require the generated answer to match the reference wording, length, or
level of detail.

A concise answer can receive a high score if it contains the essential correct
information. Do not penalize an answer simply for being shorter than the
reference.

Return JSON only."""


JUDGE_USER_TEMPLATE = """Score the generated answer on this scale:

0 = Incorrect, irrelevant, or unsupported

1 = Mostly incorrect; the main conclusion or solution is wrong

2 = Partially correct; some useful information is present, but the main answer
    is incomplete or contains an important technical error

3 = Correct overall; the main answer or solution is correct, with only minor
    omissions or minor technical issues

4 = Fully correct and directly answers the question

Evaluation rules:

- Focus on the main technical conclusion or solution.
- Do not penalize an answer simply because it is shorter than the reference.
- Do not require the generated answer to include every detail from the reference.
- Missing details should reduce the score only when they are necessary to answer
  the question correctly or make the solution useful.
- Additional information is acceptable if it is technically correct and relevant.
- Do not reward verbosity.
- Do not reward wording similarity to the reference.
- If the benchmark item is marked unanswerable, an appropriate abstention such
  as "I do not know" is correct, while inventing an answer is incorrect.
- If the benchmark item is answerable, an abstention is incorrect.

Question:

{question}

Reference status:

{reference_status}

Reference answer:

{reference_answer}

Generated answer:

{generated_answer}

Return exactly one JSON object with this shape:

{{"score": 0, "reason": "short justification"}}
"""
