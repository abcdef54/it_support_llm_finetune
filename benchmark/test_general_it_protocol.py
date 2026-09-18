from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark.baseline import config as baseline
from benchmark.common.config import TECHQA_SHA256
from benchmark.common.dataset import file_sha256, load_general_it, load_techqa
from benchmark.common.runner import run_benchmark
from benchmark.common.schemas import BenchmarkExample
from benchmark.finetuned import config as finetuned
from benchmark.finetuned.run_benchmark import run as run_finetuned
from data.preprocess_finetune_v2 import normalized, prompt_key
from data.split_dex_validation import ORIGINAL_VALIDATION_SHA256, TRAIN_SHA256, split_records
from data.utils import read_jsonl, write_json, write_jsonl
from finetune import config as training_config
from finetune.train import check


ROOT = Path(__file__).resolve().parents[1]  # Repository fixtures; no model weights are loaded.


class GeneralITProtocolTests(unittest.TestCase):
    def test_archived_split_and_hashes_are_reproducible(self):
        processed = ROOT / "data/processed/finetune_v2"
        benchmark_path = ROOT / "data/processed/benchmark/dex_general_it_benchmark.jsonl"
        original_path = processed / "validation_original_3000.jsonl"
        manifest = json.loads((processed / "manifest.json").read_text())
        self.assertEqual(file_sha256(processed / "train.jsonl"), TRAIN_SHA256)
        self.assertEqual(file_sha256(original_path), ORIGINAL_VALIDATION_SHA256)
        original = read_jsonl(original_path)
        validation, benchmark, topics = split_records(original)
        self.assertEqual((len(original), len(validation), len(benchmark)), (3000, 2000, 1000))
        self.assertEqual((validation, benchmark), split_records(original)[0:2])
        self.assertEqual(validation, read_jsonl(processed / "validation.jsonl"))
        self.assertEqual(benchmark, read_jsonl(benchmark_path))
        self.assertEqual(manifest["split_topic_distribution"], topics)
        self.assertEqual(manifest["validation_sha256"], file_sha256(processed / "validation.jsonl"))
        self.assertEqual(manifest["benchmark_sha256"], file_sha256(benchmark_path))
        self.assertEqual(manifest["revised_evaluation_layout"], {"train": 27000, "validation": 2000, "benchmark": 1000})
        self.assertTrue(all(not any(pair.values()) for pair in manifest["split_overlap_checks"].values()))

    def test_real_splits_have_no_shared_prompts_and_benchmark_is_answerable(self):
        processed = ROOT / "data/processed/finetune_v2"
        splits = [read_jsonl(processed / name) for name in ("train.jsonl", "validation.jsonl")]
        benchmark_path = ROOT / "data/processed/benchmark/dex_general_it_benchmark.jsonl"
        examples = load_general_it(benchmark_path, expected_sha256=file_sha256(benchmark_path))
        self.assertEqual(len(examples), 1000)
        self.assertTrue(all(example.question.strip() and example.answer and example.answer.strip()
                            and example.answerable for example in examples))
        keys = [{prompt_key(row["prompt"]) for row in split} for split in splits]
        keys.append({prompt_key([{"role": "user", "content": example.question}]) for example in examples})
        self.assertTrue(all(len(keys[index]) == len(splits[index]) for index in (0, 1)))
        self.assertEqual(len(keys[2]), 1000)
        self.assertFalse(keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])
        pairs = [{(prompt_key(row["prompt"]), normalized(row["completion"][0]["content"])) for row in split}
                 for split in splits]
        pairs.append({(prompt_key([{"role": "user", "content": example.question}]), normalized(example.answer))
                      for example in examples})
        self.assertFalse(pairs[0] & pairs[1] or pairs[0] & pairs[2] or pairs[1] & pairs[2])

    def test_general_loader_rejects_missing_reference_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmark.jsonl"
            row = {"id": "DEX_GENERAL_IT_000001", "question": "Why does VPN fail?",
                   "reference_answer": "Check VPN logs.", "metadata": {"topic": "network_vpn"}}
            write_jsonl(path, [row])
            self.assertEqual(len(load_general_it(path, expected_count=1)), 1)
            write_jsonl(path, [{**row, "reference_answer": ""}])
            with self.assertRaisesRegex(ValueError, "missing question/reference"):
                load_general_it(path, expected_count=1)
            write_jsonl(path, [row, row])
            with self.assertRaisesRegex(ValueError, "duplicate ID"):
                load_general_it(path, expected_count=2)

    def test_training_and_benchmarks_use_separate_controlled_paths(self):
        base = baseline.for_benchmark("general_it")
        tuned = finetuned.for_experiment("dex_v2")
        self.assertEqual(baseline.for_benchmark().DATASET_PATH, base.DATASET_PATH)
        self.assertEqual(finetuned.for_experiment().DATASET_PATH, tuned.DATASET_PATH)
        self.assertEqual(training_config.as_dict()["validation_dataset"],
                         "data/processed/finetune_v2/validation.jsonl")
        self.assertEqual(base.DATASET_PATH, tuned.DATASET_PATH)
        self.assertEqual(base.DATASET_KIND, tuned.DATASET_KIND)
        for name in ("BASE_MODEL_ID", "BASE_MODEL_REVISION", "MODEL_DTYPE", "GENERATION_MODE",
                     "GENERATION_TEMPERATURE", "MAX_NEW_TOKENS", "MAX_INPUT_TOKENS", "RANDOM_SEED",
                     "ENABLE_THINKING", "ANSWER_SYSTEM_PROMPT", "JUDGE_BATCH_SIZE"):
            self.assertEqual(getattr(base, name), getattr(tuned, name), name)
        self.assertEqual((base.ANSWER_BATCH_SIZE, tuned.ANSWER_BATCH_SIZE), (12, 16))
        self.assertEqual(baseline.for_benchmark(answer_batch_size=7).ANSWER_BATCH_SIZE, 7)
        self.assertEqual(finetuned.for_experiment(answer_batch_size=7).ANSWER_BATCH_SIZE, 7)
        with self.assertRaisesRegex(ValueError, "positive"):
            baseline.for_benchmark(answer_batch_size=0)
        with self.assertRaisesRegex(ValueError, "positive"):
            finetuned.for_experiment(answer_batch_size=0)
        self.assertNotEqual(base.PREDICTIONS_PATH, tuned.PREDICTIONS_PATH)
        self.assertEqual(baseline.for_benchmark("techqa").DATASET_PATH, baseline.DATASET_PATH)
        self.assertEqual(len(load_techqa(ROOT / baseline.DATASET_PATH)), 902)
        self.assertEqual(file_sha256(ROOT / baseline.DATASET_PATH), TECHQA_SHA256)
        with patch("benchmark.common.dataset.load_general_it", side_effect=AssertionError("benchmark loaded by training")):
            ready = check(ROOT, "dex_v2")
        self.assertEqual((ready["train_examples"], ready["validation_examples"]), (27000, 2000))
        self.assertNotEqual(ready["training_configuration"]["validation_dataset"], tuned.DATASET_PATH)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "adapter artifact"):
                run_finetuned(Path(directory), experiment="dex_v2")

    def test_general_benchmark_resume_rejects_dataset_hash_change(self):
        settings = baseline.for_benchmark("general_it")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / settings.DATASET_PATH
            dataset.parent.mkdir(parents=True)
            dataset.write_bytes(b"version one")
            manifest_path = root / settings.DATASET_MANIFEST_PATH
            write_json(manifest_path, {"benchmark_path": settings.DATASET_PATH,
                                       "benchmark_count": 1000, "benchmark_sha256": file_sha256(dataset)})
            example = BenchmarkExample("example", "VPN fails", "Check VPN logs", "heldout", True)

            class FakeModel:
                def generate_batch(self, messages, max_new_tokens):
                    return ["answer"] if max_new_tokens == settings.MAX_NEW_TOKENS else ["invalid"]

                def close(self):
                    pass

            with (patch("benchmark.common.runner.load_general_it", return_value=[example]),
                  patch("benchmark.common.runner._set_seed"),
                  patch("benchmark.common.runner.tqdm", side_effect=lambda iterable, **kwargs: iterable)):
                with self.assertRaisesRegex(ValueError, "Invalid judge response"):
                    run_benchmark(root, settings=settings, load_answer_model=FakeModel,
                                  load_judge_model=FakeModel, reuse_answer_as_judge=True)
                dataset.write_bytes(b"version two")
                write_json(manifest_path, {"benchmark_path": settings.DATASET_PATH,
                                           "benchmark_count": 1000, "benchmark_sha256": file_sha256(dataset)})
                with self.assertRaisesRegex(ValueError, "cannot safely resume"):
                    run_benchmark(root, settings=settings, load_answer_model=FakeModel,
                                  load_judge_model=FakeModel, reuse_answer_as_judge=True, resume=True)


if __name__ == "__main__":
    unittest.main()
