from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from benchmark.baseline import config as baseline
from benchmark.baseline.model import QwenGenerator
from benchmark.common.config import JUDGE_MAX_NEW_TOKENS, JUDGE_MODEL_ID, JUDGE_MODEL_REVISION, JUDGE_RETRY_MAX_NEW_TOKENS
from benchmark.common.judge import parse_or_retry_judge_output
from benchmark.common.metrics import bertscore_f1
from benchmark.common.schemas import TechQAExample
from benchmark.finetuned import config
from benchmark.finetuned.model import inspect_adapter, load_finetuned_generator
from benchmark.finetuned.run_benchmark import run, smoke


class FineTunedBenchmarkTests(unittest.TestCase):
    def test_bertscore_uses_model_limit_when_tokenizer_limit_is_unknown(self):
        class FakeScorer:
            def __init__(self, **kwargs):
                self._tokenizer = SimpleNamespace(model_max_length=10**30)
                self._model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=512))

            def score(self, predictions, references):
                self_check = self._tokenizer.model_max_length
                if self_check != 512:
                    raise AssertionError(f"Expected model limit 512, got {self_check}")
                return None, None, SimpleNamespace(tolist=lambda: [0.75])

        scores = bertscore_f1(
            ["prediction"], ["reference"], model_type="microsoft/deberta-large-mnli",
            num_layers=None, batch_size=8, lang="en", idf=False,
            rescale_with_baseline=False, scorer_class=FakeScorer,
        )
        self.assertEqual(scores, [0.75])

    def test_controlled_settings_and_separate_results(self):
        for name in (
            "BASE_MODEL_ID", "BASE_MODEL_REVISION", "MODEL_DTYPE", "GENERATION_MODE",
            "GENERATION_TEMPERATURE", "MAX_INPUT_TOKENS", "MAX_NEW_TOKENS", "RANDOM_SEED",
            "ENABLE_THINKING", "DATASET_PATH", "ANSWER_SYSTEM_PROMPT",
        ):
            self.assertEqual(getattr(config, name), getattr(baseline, name), name)
        self.assertEqual((config.ANSWER_BATCH_SIZE, config.JUDGE_BATCH_SIZE), (16, 1))
        self.assertEqual(config.MODEL_DTYPE, "bfloat16")
        self.assertNotEqual(config.PREDICTIONS_PATH, baseline.PREDICTIONS_PATH)
        self.assertNotEqual(config.METRICS_PATH, baseline.METRICS_PATH)
        self.assertTrue((Path(__file__).resolve().parents[2] / config.ADAPTER_PATH / "adapter_config.json").is_file())

    def test_adapter_preflight_checks_revision_version_and_hashes_without_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory)
            (adapter / "adapter_config.json").write_text(json.dumps({
                "base_model_name_or_path": config.BASE_MODEL_ID,
                "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 32,
                "target_modules": ["q_proj"], "peft_version": "0.21.0",
            }), encoding="utf-8")
            (adapter / "run_metadata.json").write_text(json.dumps({
                "base_model_id": config.BASE_MODEL_ID,
                "base_model_revision": config.BASE_MODEL_REVISION,
            }), encoding="utf-8")
            (adapter / "adapter_model.safetensors").write_bytes(b"test weights")
            with patch("benchmark.finetuned.model.version", return_value="0.21.0"):
                info = inspect_adapter(adapter, config.BASE_MODEL_ID, config.BASE_MODEL_REVISION)
            self.assertEqual(info["configuration"]["target_modules"], ["q_proj"])
            self.assertEqual(len(info["artifact_sha256"]["adapter_model.safetensors"]), 64)
            with patch("benchmark.finetuned.model.version", return_value="0.20.0"):
                with self.assertRaisesRegex(RuntimeError, "PEFT >= 0.21.0"):
                    inspect_adapter(adapter, config.BASE_MODEL_ID, config.BASE_MODEL_REVISION)
            with self.assertRaisesRegex(ValueError, "revision"):
                inspect_adapter(adapter, config.BASE_MODEL_ID, "wrong-revision")

    def test_adapter_attaches_to_bf16_base_and_is_active_in_eval_mode(self):
        raw_model = object()
        generator = QwenGenerator(tokenizer=object(), model=raw_model)
        expected_config = {"r": 32, "target_modules": ["q_proj", "v_proj"]}

        class FakePeftModel:
            peft_config = {"default": type("Config", (), {"r": 32, "target_modules": ["q_proj", "v_proj"]})()}
            active_adapters = ["default"]
            training = True
            call = None

            @classmethod
            def from_pretrained(cls, model, path, *, is_trainable, autocast_adapter_dtype):
                cls.call = (model, path, is_trainable, autocast_adapter_dtype)
                return cls()

            def eval(self):
                self.training = False

            def named_modules(self):
                module = type("LoRAModule", (), {"lora_A": {}, "lora_B": {}})()
                yield "base.model.q_proj", module
                yield "base.model.v_proj", module

        with (
            patch("benchmark.finetuned.model.QwenGenerator.load", return_value=generator) as load_base,
            patch("peft.PeftModel", FakePeftModel),
        ):
            result = load_finetuned_generator(
                config.BASE_MODEL_ID, config.BASE_MODEL_REVISION, Path("adapter"), expected_config,
                autocast_adapter_dtype=config.ADAPTER_AUTOCAST_DTYPE,
            )
        load_base.assert_called_once_with(config.BASE_MODEL_ID, config.BASE_MODEL_REVISION)
        self.assertEqual(FakePeftModel.call, (raw_model, "adapter", False, False))
        self.assertIs(result, generator)
        self.assertIsInstance(result.model, FakePeftModel)
        self.assertFalse(result.model.training)

    def test_failed_adapter_load_closes_the_base_model(self):
        generator = QwenGenerator(tokenizer=object(), model=object())
        with (
            patch("benchmark.finetuned.model.QwenGenerator.load", return_value=generator),
            patch("peft.PeftModel.from_pretrained", side_effect=RuntimeError("bad adapter")),
            patch.object(generator, "close") as close,
        ):
            with self.assertRaisesRegex(RuntimeError, "bad adapter"):
                load_finetuned_generator(
                    config.BASE_MODEL_ID, config.BASE_MODEL_REVISION, Path("adapter"),
                    {"r": 32, "target_modules": ["q_proj"]}, autocast_adapter_dtype=False,
                )
        close.assert_called_once()

    def test_runner_uses_adapter_only_for_answers_and_fresh_base_for_judging(self):
        examples = [TechQAExample(f"id-{i}", f"Q{i}", f"reference {i}", "dev", True) for i in range(18)]
        events = []

        class FakeGenerator:
            def __init__(self, stage):
                self.stage = stage

            def generate_batch(self, messages, max_new_tokens):
                events.append((self.stage, len(messages), max_new_tokens))
                if self.stage == "answer":
                    self.test_messages = messages
                    return [f"answer {item[-1]['content']}" for item in messages]
                return [json.dumps({"score": int(re.search(r"Question:\nQ(\d+)", item[-1]["content"]).group(1)) % 5,
                                    "reason": "checked"}) for item in messages]

            def close(self):
                events.append(("close", self.stage))

        answer = FakeGenerator("answer")
        judge = FakeGenerator("judge")
        adapter_info = {"path": "unused", "configuration": {"r": 32, "target_modules": ["q_proj"]}, "artifact_sha256": {"adapter_config.json": "hash"}}

        def load_judge(model_id, revision):
            events.append(("load_judge", model_id, revision))
            return judge

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("benchmark.finetuned.run_benchmark.inspect_adapter", return_value=adapter_info) as inspect,
            patch("benchmark.finetuned.run_benchmark.load_finetuned_generator", return_value=answer) as load_answer,
            patch("benchmark.finetuned.run_benchmark.QwenGenerator.load", side_effect=load_judge),
            patch("benchmark.common.runner.load_techqa", return_value=examples),
            patch("benchmark.common.runner.bertscore_f1", return_value=[0.0] * len(examples)),
        ):
            summary = run(Path(directory))
            predictions = [json.loads(line) for line in (Path(directory) / config.PREDICTIONS_PATH).read_text().splitlines()]
            self.assertFalse((Path(directory) / baseline.PREDICTIONS_PATH).exists())

        inspect.assert_called_once()
        load_answer.assert_called_once_with(config.BASE_MODEL_ID, config.BASE_MODEL_REVISION,
                                            Path(directory) / config.ADAPTER_PATH, adapter_info["configuration"],
                                            autocast_adapter_dtype=False)
        self.assertEqual(events[:2], [("answer", 16, config.MAX_NEW_TOKENS), ("answer", 2, config.MAX_NEW_TOKENS)])
        self.assertEqual(events[2], ("close", "answer"))
        self.assertEqual(events[3], ("load_judge", JUDGE_MODEL_ID, JUDGE_MODEL_REVISION))
        self.assertEqual(events[4:22], [("judge", 1, JUDGE_MAX_NEW_TOKENS)] * 18)
        self.assertEqual(events[-1], ("close", "judge"))
        self.assertEqual(answer.test_messages[0][0]["content"], baseline.ANSWER_SYSTEM_PROMPT)
        self.assertEqual([row["id"] for row in predictions], [example.id for example in examples])
        self.assertEqual([row["metrics"]["judge_score"] for row in predictions], [i % 5 for i in range(18)])
        self.assertEqual(summary["configuration"]["answer_batch_size"], 16)
        self.assertEqual(summary["configuration"]["judge_batch_size"], 1)
        self.assertEqual(summary["judge_retries"], 0)
        self.assertEqual(summary["adapter"]["path"], config.ADAPTER_PATH)
        self.assertFalse(summary["adapter"]["autocast_adapter_dtype"])
        self.assertIn("peft", summary["library_versions"])

    def test_invalid_judge_json_retries_same_prompt_with_larger_limit(self):
        example = TechQAExample("id-0", "Q0", "reference", "dev", True)
        answer = MagicMock()
        answer.generate_batch.return_value = ["answer"]
        judge = MagicMock()
        judge.generate_batch.side_effect = [
            ['{"score": 2, "reason": "unfinished'],
            ['{"score": 2, "reason": "partially correct"}'],
        ]
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("benchmark.finetuned.run_benchmark.inspect_adapter", return_value={"configuration": {"r": 32}}),
            patch("benchmark.finetuned.run_benchmark.load_finetuned_generator", return_value=answer),
            patch("benchmark.finetuned.run_benchmark.QwenGenerator.load", return_value=judge),
            patch("benchmark.common.runner.load_techqa", return_value=[example]),
            patch("benchmark.common.runner.bertscore_f1", return_value=[0.0]),
        ):
            summary = run(Path(directory))
            prediction = json.loads((Path(directory) / config.PREDICTIONS_PATH).read_text().splitlines()[0])
        self.assertEqual(summary["judge_retries"], 1)
        self.assertEqual(summary["configuration"]["judge_retry_max_new_tokens"], JUDGE_RETRY_MAX_NEW_TOKENS)
        self.assertEqual(prediction["metrics"]["judge_score"], 2)
        self.assertEqual(judge.generate_batch.call_args_list[0].args[1], JUDGE_MAX_NEW_TOKENS)
        self.assertEqual(judge.generate_batch.call_args_list[1].args[1], JUDGE_RETRY_MAX_NEW_TOKENS)
        self.assertEqual(judge.generate_batch.call_args_list[0].args[0], judge.generate_batch.call_args_list[1].args[0])

    def test_failed_judge_retry_reports_output_tail(self):
        judge = MagicMock()
        judge.generate_batch.return_value = ['{"score": 2, "reason": "' + "x" * 400 + "TRUNCATED"]
        with self.assertRaisesRegex(ValueError, "TRUNCATED"):
            parse_or_retry_judge_output(judge, [{"role": "user", "content": "Judge this"}], "not JSON")
        self.assertEqual(judge.generate_batch.call_args.args[1], JUDGE_RETRY_MAX_NEW_TOKENS)

    def test_incomplete_explanation_recovers_only_matching_scores(self):
        first = '{"score": 2, "reason": "The answer partly works but ' + "loops " * 100
        retry = '{"score": 2, "reason": "Different wording, still ' + "loops " * 400
        judge = MagicMock()
        judge.generate_batch.return_value = [retry]
        decision, retried = parse_or_retry_judge_output(judge, [{"role": "user", "content": "Judge"}], first)
        self.assertEqual(decision.score, 2)
        self.assertTrue(decision.recovered and retried)
        self.assertIn("incomplete", decision.reason)
        judge.generate_batch.return_value = [retry.replace('"score": 2', '"score": 3')]
        with self.assertRaises(ValueError):
            parse_or_retry_judge_output(judge, [{"role": "user", "content": "Judge"}], first)

    def test_recovered_score_is_visible_in_prediction_and_summary(self):
        example = TechQAExample("id-0", "Q0", "reference", "dev", True)
        answer = MagicMock()
        answer.generate_batch.return_value = ["answer"]
        judge = MagicMock()
        judge.generate_batch.side_effect = [
            ['{"score": 2, "reason": "unfinished'],
            ['{"score": 2, "reason": "still unfinished'],
        ]
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("benchmark.finetuned.run_benchmark.inspect_adapter", return_value={"configuration": {"r": 32}}),
            patch("benchmark.finetuned.run_benchmark.load_finetuned_generator", return_value=answer),
            patch("benchmark.finetuned.run_benchmark.QwenGenerator.load", return_value=judge),
            patch("benchmark.common.runner.load_techqa", return_value=[example]),
            patch("benchmark.common.runner.bertscore_f1", return_value=[0.0]),
        ):
            summary = run(Path(directory))
            prediction = json.loads((Path(directory) / config.PREDICTIONS_PATH).read_text().splitlines()[0])
        self.assertEqual(summary["judge_recovered_scores"], 1)
        self.assertEqual(summary["judge_retries"], 1)
        self.assertEqual(prediction["metrics"]["judge_score"], 2)
        self.assertTrue(prediction["metrics"]["judge_recovered"])

    def test_resume_skips_saved_answers_and_judgments(self):
        from benchmark.baseline.run_benchmark import run as run_baseline

        examples = [TechQAExample(f"id-{i}", f"Q{i}", "reference", "dev", True) for i in range(3)]
        class FakeGenerator:
            def __init__(self):
                self.fail_once = True
                self.answer_calls = 0
                self.judge_calls = 0

            def generate_batch(self, messages, limit):
                if limit == baseline.MAX_NEW_TOKENS:
                    self.answer_calls += 1
                    return ["answer"] * len(messages)
                self.judge_calls += 1
                if self.judge_calls == 2 and self.fail_once:
                    self.fail_once = False
                    raise RuntimeError("interrupted")
                return ['{"score": 2, "reason": "checked"}'] * len(messages)

            def close(self):
                pass

        fake = FakeGenerator()
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("benchmark.common.runner.load_techqa", return_value=examples),
            patch("benchmark.baseline.run_benchmark.QwenGenerator.load", return_value=fake),
            patch("benchmark.common.runner.bertscore_f1", return_value=[0.0] * len(examples)),
        ):
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                run_baseline(root)
            progress = root / "results/base/progress.sqlite3"
            self.assertTrue(progress.exists())
            with self.assertRaises(FileExistsError):
                run_baseline(root)
            with patch("benchmark.common.runner.load_techqa", return_value=list(reversed(examples))):
                with self.assertRaisesRegex(ValueError, "changed"):
                    run_baseline(root, resume=True)
            summary = run_baseline(root, resume=True)
            self.assertEqual((fake.answer_calls, fake.judge_calls), (1, 4))
            self.assertEqual(summary["total_examples"], 3)
            self.assertFalse(progress.exists())

    def test_resume_after_metric_failure_does_not_reload_qwen(self):
        from benchmark.baseline.run_benchmark import run as run_baseline

        example = TechQAExample("id-0", "Q0", "reference", "dev", True)
        model = MagicMock()
        model.generate_batch.side_effect = [["answer"], ['{"score": 4, "reason": "correct"}']]
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("benchmark.common.runner.load_techqa", return_value=[example]),
            patch("benchmark.baseline.run_benchmark.QwenGenerator.load", return_value=model) as load_model,
            patch("benchmark.common.runner.bertscore_f1", side_effect=RuntimeError("metric failed")) as score,
        ):
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "metric failed"):
                run_baseline(root)
            self.assertTrue((root / "results/base/progress.sqlite3").exists())
            load_model.reset_mock()
            load_model.side_effect = AssertionError("Qwen should not reload")
            score.side_effect = None
            score.return_value = [0.75]
            summary = run_baseline(root, resume=True)
            load_model.assert_not_called()
            self.assertEqual(summary["total_examples"], 1)
            self.assertFalse((root / "results/base/progress.sqlite3").exists())

    def test_synthetic_smoke_keeps_techqa_and_results_untouched(self):
        class FakeGenerator:
            def __init__(self, response):
                self.response = response
                self.closed = False
                self.calls = []

            def generate_batch(self, messages, max_new_tokens):
                self.calls.append((messages, max_new_tokens))
                return [self.response] * len(messages)

            def close(self):
                self.closed = True

        answer = FakeGenerator("Use systemctl status SERVICE.")
        judge = FakeGenerator('{"score": 4, "reason": "correct"}')
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("benchmark.finetuned.run_benchmark.inspect_adapter", return_value={"configuration": {"r": 32}}),
            patch("benchmark.finetuned.run_benchmark.load_finetuned_generator", return_value=answer),
            patch("benchmark.finetuned.run_benchmark.QwenGenerator.load", return_value=judge) as load_judge,
            patch("benchmark.common.runner.load_techqa", side_effect=AssertionError("TechQA must not be read")),
        ):
            result = smoke(Path(directory))
            self.assertFalse((Path(directory) / "results").exists())
        self.assertEqual(result["synthetic_answers"], config.ANSWER_BATCH_SIZE)
        self.assertEqual(len(answer.calls[0][0]), 16)
        self.assertEqual(len(judge.calls[0][0]), 1)
        self.assertTrue(answer.closed and judge.closed)
        load_judge.assert_called_once_with(JUDGE_MODEL_ID, JUDGE_MODEL_REVISION)


if __name__ == "__main__":
    unittest.main()
