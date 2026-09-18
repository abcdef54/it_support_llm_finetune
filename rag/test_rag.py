from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from benchmark.baseline.run_benchmark import run as run_base
from benchmark.baseline.config import for_benchmark
from benchmark.common.runner import run_benchmark
from benchmark.common.schemas import BenchmarkExample
from benchmark.common.dataset import file_sha256
from benchmark.finetuned.run_benchmark import run as run_dex
from data.utils import read_jsonl, write_json, write_jsonl
from rag import config as C
from rag.build_corpus import build, clean_html, normalize_stackoverflow, validate_corpus
from rag.context import build_context
from rag.index import build_index, inspect_index
from rag.prepare_benchmark import load_artifact, prepare, retrieval_configuration
from rag.retrieve import Retriever


class FakeEmbedder:
    dimension = 3
    truncated_documents = 0
    truncated_queries = 0

    def encode(self, texts, query=False):
        values = np.array([[1.0, (len(text) % 7) + 1.0, 2.0] for text in texts], dtype=np.float32)
        return values / np.linalg.norm(values, axis=1, keepdims=True)

    def close(self):
        pass


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return list(text)

    def apply_chat_template(self, messages, **kwargs):
        tokens = list("".join(m["content"] for m in messages))
        return tokens if kwargs.get("return_dict") is False else {"input_ids": tokens, "attention_mask": [1] * len(tokens)}


class RAGTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def corpus(self):
        for name in (C.BENCHMARK_PATH, C.BENCHMARK_MANIFEST, C.TECHQA_PATH):
            destination = self.root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(C.PROJECT_ROOT / name, destination)
        overlap = read_jsonl(self.root / C.BENCHMARK_PATH)[0]["question"]
        rows = [self.so(), self.so(), self.so(question_id=2, question_score=-1),
                self.so(question_id=3, question_body=overlap),
                self.so(question_id=4, question_body="Configure nginx HTTPS server on Ubuntu.", tags=["nginx"])]
        write_jsonl(self.root / C.STACKOVERFLOW_RAW, rows)
        return build(self.root, max_records=10)

    @staticmethod
    def so(**kwargs):
        return {"question_id": 1, "answer_id": 10, "title": "SSH permissions",
                "question_body": "SSH authentication fails after copying my private key.",
                "answer_body": "<p>Set key permissions:</p><pre>sudo chmod 600 ~/.ssh/id_rsa</pre>",
                "tags": ["ssh", "linux"], "question_score": 0, "answer_score": 1, **kwargs}

    def test_normalization_preserves_commands_code_and_stable_ids(self):
        row, error = normalize_stackoverflow(self.so())
        self.assertIsNone(error)
        self.assertEqual(row["id"], "stackoverflow:1")
        self.assertIn("sudo chmod 600 ~/.ssh/id_rsa", row["answer"])
        self.assertNotIn(row["answer"], row["retrieval_text"])
        self.assertEqual(clean_html("<pre>if x &lt; 3:\n    run('/tmp/a')</pre>"), "if x < 3:\n    run('/tmp/a')")
        self.assertEqual(clean_html("echo <filename> > /tmp/log"), "echo <filename> > /tmp/log")
        self.assertEqual(normalize_stackoverflow(self.so(tags="['ssh', 'linux']"))[0]["metadata"]["tags"], ["ssh", "linux"])
        self.assertEqual(normalize_stackoverflow(self.so(answer_body=""))[1], "missing_question_or_accepted_answer")
        self.assertEqual(normalize_stackoverflow(self.so(title="How to sort a list?", tags=["python"],
                                                        question_body="Sort these integers."))[1], "irrelevant_topic")

    def test_corpus_counts_dedup_firewall_and_historical_preservation(self):
        original = file_sha256(C.PROJECT_ROOT / C.TECHQA_PATH)
        report = self.corpus()
        records, manifest = validate_corpus(self.root)
        self.assertEqual(report["filter_stats"]["techqa"]["raw"], 902)
        self.assertEqual(report["filter_stats"]["techqa"]["answerable"], 602)
        self.assertEqual(report["filter_stats"]["techqa"]["excluded_null_or_unanswerable"], 300)
        self.assertEqual(report["filter_stats"]["stackoverflow"]["duplicates_removed"], 1)
        self.assertEqual(report["filter_stats"]["stackoverflow"]["benchmark_leakage_removed"], 1)
        self.assertEqual(set(r["source"] for r in records), {"techqa", "stackoverflow"})
        self.assertTrue(all(r["answer"] for r in records))
        self.assertIn("techqa:TRAIN_Q046", {r["id"] for r in records})
        self.assertEqual(len({r["id"] for r in records}), len(records))
        self.assertEqual(file_sha256(self.root / C.TECHQA_PATH), original)
        self.assertEqual(file_sha256(self.root / C.BENCHMARK_PATH), file_sha256(C.PROJECT_ROOT / C.BENCHMARK_PATH))
        self.assertEqual(build(self.root, max_records=10)["files"], report["files"])
        self.assertEqual(manifest["leakage_checks"]["exact_prompt_overlap"], 0)

    def test_real_chroma_idempotence_structured_retrieval_source_and_stale_guard(self):
        self.corpus()
        first = build_index(self.root, batch_size=128, embedder=FakeEmbedder())
        second = build_index(self.root, batch_size=128, embedder=FakeEmbedder())
        self.assertEqual(first, second)
        (self.root / C.INDEX_PATH / "manifest.json").unlink()
        resumed = build_index(self.root, batch_size=128, embedder=FakeEmbedder())
        self.assertEqual(resumed["configuration"], first["configuration"])
        _, collection, manifest = inspect_index(self.root)
        self.assertEqual(collection.count(), manifest["configuration"]["documents"])
        retriever = Retriever(self.root, embedder=FakeEmbedder())
        result = retriever.retrieve("Linux permissions", top_k=2, source="stackoverflow")
        self.assertTrue(all(r["source"] == "stackoverflow" and r["answer"] and "score" in r for r in result))
        self.assertEqual(len(result), 2)
        retriever.close()
        wrong_dimension = FakeEmbedder()
        wrong_dimension.dimension = 4
        with self.assertRaisesRegex(ValueError, "dimension differs"):
            Retriever(self.root, embedder=wrong_dimension)
        with patch.object(C, "EMBEDDING_REVISION", "changed"):
            with self.assertRaisesRegex(ValueError, "stale"):
                inspect_index(self.root)
        with (self.root / C.CORPUS_PATH).open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "hash changed"):
            inspect_index(self.root)

    def test_context_budget_rank_order_and_no_question_truncation(self):
        records = [{"id": str(i), "source": "stackoverflow", "score": 1 - i / 10,
                    "context_text": "Use chmod 600 on the private key."} for i in range(5)]
        context = build_context("SSH fails", records, CharacterTokenizer(), max_context_tokens=140, max_input_tokens=1000)
        self.assertLessEqual(context["context_tokens"], 140)
        self.assertEqual([r["id"] for r in context["used_documents"]], ["0", "1"])
        self.assertTrue(context["messages"][-1]["content"].endswith("SSH fails"))
        self.assertLessEqual(context["input_tokens"], 1000)
        with self.assertRaisesRegex(ValueError, "not be silently truncated"):
            build_context("q" * 2000, records, CharacterTokenizer(), max_input_tokens=1000)

    def test_both_variants_share_contexts_metrics_and_fingerprint_metadata(self):
        artifact = {"question": {"messages": [{"role": "user", "content": "same context"}]}}
        meta = {"enabled": True, "index_fingerprint": "index", "retrieval_artifact_sha256": "contexts"}
        with (patch("rag.prepare_benchmark.load_artifact", return_value=(artifact, meta)),
              patch("benchmark.baseline.run_benchmark.run_benchmark", return_value={}) as base_runner,
              patch("benchmark.finetuned.run_benchmark.run_benchmark", return_value={}) as dex_runner,
              patch("benchmark.finetuned.run_benchmark.inspect_adapter", return_value={"configuration": {}})):
            run_base(self.root, rag=True)
            run_dex(self.root, rag=True)
        base, dex = base_runner.call_args.kwargs, dex_runner.call_args.kwargs
        self.assertIs(base["rag_records"], dex["rag_records"])
        self.assertEqual(base["extra_metadata"]["rag"], dex["extra_metadata"]["rag"])
        self.assertNotEqual(base["settings"].PREDICTIONS_PATH, dex["settings"].PREDICTIONS_PATH)
        for name in ("DATASET_PATH", "MAX_NEW_TOKENS", "RANDOM_SEED", "GENERATION_TEMPERATURE", "JUDGE_BATCH_SIZE"):
            self.assertEqual(getattr(base["settings"], name), getattr(dex["settings"], name))
        with self.assertRaisesRegex(ValueError, "TechQA is indexed"):
            run_base(self.root, benchmark="techqa", rag=True)
        with self.assertRaisesRegex(ValueError, "dex_v2"):
            run_dex(self.root, experiment="dex", rag=True)

    def test_retrieval_identity_covers_required_configuration(self):
        self.corpus()
        manifest = build_index(self.root, embedder=FakeEmbedder(), batch_size=128)
        identity = retrieval_configuration(self.root, manifest)
        self.assertTrue({"enabled", "corpus_sha256", "index_fingerprint", "embedding_model", "embedding_revision",
                         "benchmark_sha256", "top_k", "max_context_tokens", "rag_prompt_sha256"} <= identity.keys())
        self.assertEqual(identity["top_k"], 3)
        with patch.object(C, "RAG_TOP_K", 5):
            self.assertNotEqual(identity, retrieval_configuration(self.root, manifest))

    def test_shared_artifact_reuse_integrity_and_no_reference_answers(self):
        self.corpus()
        build_index(self.root, embedder=FakeEmbedder(), batch_size=128)
        examples = [BenchmarkExample("sample", "Why does SSH authentication fail?",
                                     "REFERENCE_MUST_NOT_ENTER_CONTEXT", "heldout", True)]
        with (patch("rag.prepare_benchmark.benchmark_examples", return_value=examples),
              patch("rag.prepare_benchmark.load_tokenizer", return_value=CharacterTokenizer()),
              patch("rag.prepare_benchmark.Retriever", side_effect=lambda root, **kwargs: Retriever(root, embedder=FakeEmbedder()))):
            prepare(self.root)
            records, metadata = load_artifact(self.root)
            self.assertEqual(set(records), {"sample"})
            self.assertEqual(metadata["retrieval_artifact_sha256"], file_sha256(self.root / C.RETRIEVAL_ARTIFACT))
            with patch("rag.prepare_benchmark.Retriever", side_effect=AssertionError("cached retrieval recomputed")):
                prepare(self.root)
            raw = (self.root / C.RETRIEVAL_ARTIFACT).read_text()
            self.assertNotIn(examples[0].answer, raw)
            with patch.object(C, "RAG_TOP_K", 99):
                with self.assertRaisesRegex(ValueError, "stale or changed"):
                    load_artifact(self.root)
            artifact = json.loads(raw)
            artifact["records"][0]["context"] = "tampered"
            write_json(self.root / C.RETRIEVAL_ARTIFACT, artifact)
            with self.assertRaisesRegex(ValueError, "stale or changed"):
                load_artifact(self.root)

    def test_rag_resume_rejects_changed_context_and_keeps_judging_separate(self):
        self.corpus()
        settings = for_benchmark()
        example = BenchmarkExample("example", "Why does SSH fail?", "Check permissions", "heldout", True)
        records = {example.id: {"id": example.id, "question": example.question,
                               "messages": [{"role": "system", "content": "RAG test"},
                                            {"role": "user", "content": "SSH knowledge"}],
                               "used_documents": [{"id": "stackoverflow:1", "source": "stackoverflow"}],
                               "context_tokens": 5, "context": "SSH knowledge"}}

        class FakeModel:
            fail = True
            answer_calls = 0

            def generate_batch(self, messages, limit):
                if messages[0][0]["content"] == "RAG test":
                    self.answer_calls += 1
                    return ["Check key permissions"]
                self.assert_judge = "SSH knowledge" not in str(messages)
                return ["invalid" if self.fail else '{"score": 4, "reason": "Correct"}']

            def close(self):
                pass

        model = FakeModel()
        metadata = {"rag": {"retrieval_artifact_sha256": "one", "top_k": 3}}
        with (patch("benchmark.common.runner.load_general_it", return_value=[example]),
              patch("benchmark.common.runner._set_seed"),
              patch("benchmark.common.runner.bertscore_f1", return_value=[0.8])):
            kwargs = dict(settings=settings, load_answer_model=lambda: model, load_judge_model=lambda: model,
                          reuse_answer_as_judge=True, rag_records=records)
            with self.assertRaisesRegex(ValueError, "Invalid judge response"):
                run_benchmark(self.root, extra_metadata=metadata, **kwargs)
            with self.assertRaisesRegex(ValueError, "cannot safely resume"):
                run_benchmark(self.root, resume=True, extra_metadata={"rag": {"retrieval_artifact_sha256": "two"}}, **kwargs)
            model.fail = False
            result = run_benchmark(self.root, resume=True, extra_metadata=metadata, **kwargs)
        self.assertEqual(model.answer_calls, 1)
        self.assertTrue(model.assert_judge)
        self.assertEqual(result["retrieval_diagnostics"]["average_documents_used"], 1)
        prediction = read_jsonl(self.root / settings.PREDICTIONS_PATH)[0]
        self.assertEqual(prediction["rag"]["context"], "SSH knowledge")


if __name__ == "__main__":
    unittest.main()
