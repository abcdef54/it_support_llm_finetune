"""Real embedding/Chroma/context checks using source facts, never benchmark answers or Qwen weights."""
import argparse
import json
from pathlib import Path

from data.utils import write_json
from rag import config as C
from rag.context import build_context
from rag.prepare_benchmark import load_tokenizer
from rag.retrieve import Retriever


TECHQA_CASES = [
    ("Which ITM version supports the Linux KVM CANDLEDATA function?", "techqa:TRAIN_Q046"),
    ("WebSphere on Linux recommended ulimit settings", "techqa:TRAIN_Q012"),
    ("Can a TIP 2.2 fix pack be applied directly to TIP 2.1?", "techqa:TRAIN_Q013"),
    ("SPSS Statistics Subscription installer unsupported on Mac OS X 10.9.5", "techqa:TRAIN_Q154"),
]
STACKOVERFLOW_CASES = [
    ("SSH permission denied publickey authentication fails", ("ssh",)),
    ("Linux file permissions chmod access denied", ("permission", "chmod")),
    ("DNS resolution fails hostname lookup", ("dns", "hostname")),
    ("PostgreSQL connection refused cannot connect", ("postgres",)),
    ("Python package import error module not found", ("python", "import")),
    ("Git authentication fails permission denied", ("git",)),
    ("Nginx web server configuration", ("nginx",)),
]


def run(root, device=None):
    retriever = Retriever(root, device=device)
    checks = []
    try:
        tokenizer = load_tokenizer(root)
        for query, expected in TECHQA_CASES + STACKOVERFLOW_CASES:
            records = retriever.retrieve(query)
            context = build_context(query, records, tokenizer)
            if isinstance(expected, str):
                passed = expected in [r["id"] for r in records]
            else:
                passed = any(r["source"] == "stackoverflow" and any(
                    term in (r["question"] + " " + (r["title"] or "")).casefold() for term in expected)
                             for r in records)
            checks.append({"query": query, "passed": passed, "expected": expected,
                           "context_tokens": context["context_tokens"], "input_tokens": context["input_tokens"],
                           "documents_used": len(context["used_documents"]),
                           "results": [{k: r[k] for k in ("id", "source", "score", "title", "question")}
                                       for r in records]})
        report = {"passed": all(x["passed"] for x in checks), "index_fingerprint": retriever.manifest["fingerprint"],
                  "note": "Source-based retrieval checks only; not a held-out model benchmark", "checks": checks}
        write_json(root / C.CORPUS_DIR / "retrieval_smoke_report.json", report)
        return report
    finally:
        retriever.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=C.PROJECT_ROOT)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    report = run(args.project_root.resolve(), args.device)
    print(json.dumps({"passed": report["passed"], "checks": [
        {"query": x["query"], "passed": x["passed"], "documents_used": x["documents_used"]}
        for x in report["checks"]]}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
