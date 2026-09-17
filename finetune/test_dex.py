from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark.common.dataset import file_sha256
from benchmark.finetuned import config as benchmark_config
from benchmark.finetuned.model import inspect_adapter
from benchmark.finetuned.run_benchmark import run
from finetune import config
from finetune.dataset import load_datasets, load_split, inspect_chat_format
from finetune.train import dataset_provenance


class DexSetupTests(unittest.TestCase):
    def test_only_training_paths_change(self):
        v1, dex = config.as_dict("v1"), config.as_dict("dex")
        self.assertEqual({k for k in v1 if v1[k] != dex[k]}, {"train_dataset", "validation_dataset", "output_dir"})
        self.assertIn("finetune_v2", dex["train_dataset"])
        self.assertIn("dex-qlora", dex["output_dir"])
        self.assertEqual(dex["effective_batch_size"], dex["per_device_train_batch_size"] * dex["gradient_accumulation_steps"])

    def test_only_benchmark_paths_change(self):
        v1, dex = vars(benchmark_config.for_experiment("v1")), vars(benchmark_config.for_experiment("dex"))
        self.assertEqual({k for k in v1 if v1[k] != dex[k]}, {"ADAPTER_PATH", "PREDICTIONS_PATH", "METRICS_PATH"})
        self.assertIn("finetuned_dex", dex["PREDICTIONS_PATH"])
        with tempfile.TemporaryDirectory() as directory, patch("benchmark.baseline.model.QwenGenerator.load") as load:
            with self.assertRaisesRegex(FileNotFoundError, "adapter artifact"):
                run(Path(directory), experiment="dex")
            load.assert_not_called()

    def test_multiturn_training_schema_and_full_prompt_overlap(self):
        def row(question):
            return {"prompt": [
                {"role": "system", "content": "IT support"},
                {"role": "user", "content": "My network fails"},
                {"role": "assistant", "content": "Check DNS settings"},
                {"role": "user", "content": question}],
                "completion": [{"role": "assistant", "content": "Run nslookup example.com"}]}
        with tempfile.TemporaryDirectory() as directory:
            train, val = Path(directory) / "train.jsonl", Path(directory) / "val.jsonl"
            train.write_text(json.dumps(row("What next?")) + "\n")
            val.write_text(json.dumps(row("Which command?")) + "\n")
            first, second = load_datasets(train, val)
            self.assertEqual((len(first), len(second)), (1, 1))
            self.assertEqual(len(first[0]["completion"]), 1)
            class Processor:
                def apply_chat_template(self, messages, tokenize, **kwargs):
                    text = "\n".join(m["content"] for m in messages)
                    return text.split() if tokenize else text
            self.assertEqual(inspect_chat_format(Processor(), load_split(train), 1024)["over_length_examples"], 0)
            val.write_text(train.read_text())
            with self.assertRaisesRegex(ValueError, "overlapping"):
                load_datasets(train, val)

    def test_dex_manifest_integrity_checked_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train.jsonl").write_text("train data")
            (root / "validation.jsonl").write_text("validation data")
            settings = {"train_dataset": "train.jsonl", "validation_dataset": "validation.jsonl"}
            manifest = {"audit": {"passed": True}, "source_revision": "test",
                        "train_sha256": file_sha256(root / "train.jsonl"),
                        "validation_sha256": file_sha256(root / "validation.jsonl")}
            (root / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(dataset_provenance(root, settings, "dex")["source_revision"], "test")
            (root / "train.jsonl").write_text("changed")
            with self.assertRaisesRegex(ValueError, "differs"):
                dataset_provenance(root, settings, "dex")

    def test_future_adapter_metadata_when_available(self):
        settings = benchmark_config.for_experiment("dex")
        path = Path(__file__).resolve().parents[1] / settings.ADAPTER_PATH
        if not path.exists():
            self.skipTest("DEX adapter will exist after the separately launched training run")
        inspect_adapter(path, settings.BASE_MODEL_ID, settings.BASE_MODEL_REVISION, expected_experiment="dex")

    def test_dex_benchmark_rejects_v1_adapter_metadata(self):
        settings = benchmark_config.for_experiment("dex")
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory)
            (adapter / "adapter_config.json").write_text(json.dumps({
                "base_model_name_or_path": settings.BASE_MODEL_ID,
                "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 32,
                "target_modules": ["q_proj"],
            }))
            (adapter / "adapter_model.safetensors").write_bytes(b"test weights")
            (adapter / "run_metadata.json").write_text(json.dumps({
                "base_model_id": settings.BASE_MODEL_ID,
                "base_model_revision": settings.BASE_MODEL_REVISION,
                "experiment": "v1",
            }))
            with self.assertRaisesRegex(ValueError, "dex training experiment"):
                inspect_adapter(adapter, settings.BASE_MODEL_ID, settings.BASE_MODEL_REVISION,
                                expected_experiment="dex")


if __name__ == "__main__":
    unittest.main()
