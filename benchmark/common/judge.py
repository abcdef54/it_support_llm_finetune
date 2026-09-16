from __future__ import annotations

import json
import re
from dataclasses import dataclass

from benchmark.common.config import JUDGE_RETRY_MAX_NEW_TOKENS, JUDGE_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE
from benchmark.common.schemas import TechQAExample


@dataclass(frozen=True)
class JudgeDecision:
    score: int
    reason: str


def build_judge_messages(example: TechQAExample, generated_answer: str) -> list[dict[str, str]]:
    user_prompt = JUDGE_USER_TEMPLATE.format(
        question=example.question,
        reference_status="answerable" if example.answerable else "unanswerable",
        reference_answer=example.answer if example.answerable else "No reference answer exists.",
        generated_answer=generated_answer,
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def parse_judge_output(output: str) -> JudgeDecision:
    candidates = [output]
    candidates.extend(match.group(0) for match in re.finditer(r"\{.*?\}", output, re.DOTALL))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        score, reason = value.get("score"), value.get("reason")
        if isinstance(score, int) and not isinstance(score, bool) and 0 <= score <= 4 and isinstance(reason, str):
            return JudgeDecision(score, reason.strip())
    preview = output if len(output) <= 320 else output[:160] + " ... " + output[-160:]
    raise ValueError(f"Judge returned invalid output ({len(output)} characters): {preview!r}")


def parse_or_retry_judge_output(model, messages: list[dict[str, str]], output: str) -> tuple[JudgeDecision, bool]:
    try:
        return parse_judge_output(output), False
    except ValueError:
        # Same prompt and greedy decoding; only the output ceiling is raised.
        retry_outputs = model.generate_batch([messages], JUDGE_RETRY_MAX_NEW_TOKENS)
        if len(retry_outputs) != 1:
            raise ValueError(f"Judge retry returned {len(retry_outputs)} outputs")
        try:
            return parse_judge_output(retry_outputs[0]), True
        except ValueError as exc:
            raise ValueError(f"Judge retry at {JUDGE_RETRY_MAX_NEW_TOKENS} tokens was invalid: {exc}") from exc
