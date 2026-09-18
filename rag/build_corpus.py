"""Deterministic whole-Q&A corpus; no model, training, or benchmark inference."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import heapq
from html import unescape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys

from benchmark.common.dataset import file_sha256, load_general_it, load_techqa
from data.utils import read_jsonl, write_json, write_jsonl
from rag import config as C


TOPICS = {
    "linux_shell": ("linux", "ubuntu", "bash", "shell", "unix", "permissions"),
    "windows_powershell": ("windows", "powershell", "registry", "active-directory"),
    "networking": ("ssh", "dns", "networking", "http", "https", "ssl", "tls", "tcp", "proxy"),
    "web_servers": ("nginx", "apache", "iis", "tomcat"),
    "databases": ("postgresql", "mysql", "sql", "sql-server", "sqlite", "database", "mongodb"),
    "python_packages": ("pip", "virtualenv", "conda", "python-import", "python-packaging", "pythonpath", "importerror", "module-not-found"),
    "developer_tools": ("git", "svn", "maven", "gradle", "npm", "ide", "visual-studio"),
    "servers_deployment": ("server", "deployment", "configuration", "docker", "kubernetes", "systemd"),
}
TOPIC_PATTERNS = {topic: re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, terms)) + r")(?!\w)", re.I)
                  for topic, terms in TOPICS.items()}


def whitespace(value) -> str:
    # Preserve indentation, punctuation, and literal backslashes in technical text.
    return re.sub(r"\n{3,}", "\n\n", str(value or "").replace("\r\n", "\n")).strip()


def key(value) -> str:
    return " ".join(whitespace(value).casefold().split())


class TechnicalHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1
        elif tag in {"p", "div", "pre", "br", "li", "blockquote", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)
        elif tag in {"p", "div", "pre", "li", "blockquote", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def clean_html(value) -> str:
    value = whitespace(value)
    # Already-clean source text may contain shell redirection or generic types.
    if re.search(r"</?(?:p|div|pre|code|br|a|ul|ol|li|blockquote|strong|em|script|style)(?:\s|>|/)", value, re.I):
        parser = TechnicalHTML()
        parser.feed(value)
        return whitespace("".join(parser.parts))
    return whitespace(unescape(value))


def topic_for(tags, text):
    if any(tag == "python" or tag.startswith("python-") for tag in tags) and re.search(
            r"\b(install|installation|import|importing|package|packages|pip|virtualenv|environment|dependency|dependencies|module)\b", text, re.I):
        return "python_packages"
    for topic, terms in TOPICS.items():
        if any(tag == term or tag.startswith(term + "-") for tag in tags for term in terms):
            return topic
    for topic, pattern in TOPIC_PATTERNS.items():
        if pattern.search(text):
            return topic
    return None


def document(source, source_id, title, question, answer, metadata):
    heading = f"Title: {title}\n\n" if title else ""
    retrieval = heading + "Question:\n" + question
    label = "Accepted answer" if source == "stackoverflow" else "IBM support answer"
    return {"id": f"{source}:{source_id}", "source": source, "title": title,
            "question": question, "answer": answer, "retrieval_text": retrieval,
            "context_text": retrieval + f"\n\n{label}:\n" + answer, "metadata": metadata}


def normalize_stackoverflow(row, min_question_score=C.MIN_QUESTION_SCORE, min_answer_score=C.MIN_ANSWER_SCORE):
    title, question, answer = (clean_html(row.get(field)) for field in ("title", "question_body", "answer_body"))
    if not row.get("question_id") or not question or not answer:
        return None, "missing_question_or_accepted_answer"
    scores = {}
    for field, minimum in (("question_score", min_question_score), ("answer_score", min_answer_score)):
        if row.get(field) is not None:
            try:
                scores[field] = int(row[field])
            except (ValueError, TypeError):
                return None, "invalid_score"
            if scores[field] < minimum:
                return None, "low_score"
    tags = row.get("tags") or []
    if isinstance(tags, str):
        tags = re.findall(r"[\w.+#-]+", tags.casefold())
    tags = [str(tag).casefold() for tag in tags]
    topic = topic_for(tags, title)  # Full code bodies mention unrelated platforms incidentally.
    if topic is None:
        return None, "irrelevant_topic"
    if len(answer) < 10 or not re.search(r"\w", answer):
        return None, "answer_too_short_or_noise"
    metadata = {"tags": tags, "topic": topic, **scores,
                "answer_id": row.get("answer_id"), "source_revision": C.STACKOVERFLOW_REVISION,
                "url": f"https://stackoverflow.com/questions/{row['question_id']}",
                "question_creation_date": row.get("question_creation_date")}
    return document("stackoverflow", row["question_id"], title or None, question, answer, metadata), None


def benchmark_keys(root):
    manifest = json.loads((root / C.BENCHMARK_MANIFEST).read_text())
    examples = load_general_it(root / C.BENCHMARK_PATH, expected_sha256=manifest["benchmark_sha256"])
    return {key(x.question) for x in examples}, {key(x.answer) for x in examples}


def overlaps(record, questions, answers):
    fields = {key(record["question"]), key(record["title"]),
              key((record["title"] or "") + "\n\n" + record["question"])} - {""}
    return bool(fields & questions or key(record["answer"]) in answers)


def validate_corpus(root):
    path = root / C.CORPUS_PATH
    if not path.exists():
        raise FileNotFoundError("RAG corpus missing; run python -m rag.build_corpus")
    manifest = json.loads((root / C.CORPUS_DIR / "manifest.json").read_text())
    if file_sha256(path) != manifest["files"]["combined_corpus.jsonl"]["sha256"]:
        raise ValueError("RAG corpus hash changed; rerun python -m rag.build_corpus")
    if file_sha256(root / C.BENCHMARK_PATH) != manifest["leakage_checks"]["benchmark_sha256"]:
        raise ValueError("Benchmark changed since corpus leakage audit; rebuild the corpus")
    questions, answers = benchmark_keys(root)
    records = read_jsonl(path)
    ids, prompts = set(), set()
    for record in records:
        if (record.get("source") not in {"stackoverflow", "techqa"}
                or not all(isinstance(record.get(f), str) and record[f].strip()
                           for f in ("id", "question", "answer", "retrieval_text", "context_text"))
                or not record["id"].startswith(record["source"] + ":")
                or not isinstance(record.get("metadata"), dict)):
            raise ValueError("Invalid RAG document schema")
        if record["id"] in ids or key(record["question"]) in prompts:
            raise ValueError("Duplicate RAG document")
        if overlaps(record, questions, answers):
            raise ValueError("General-IT benchmark overlaps RAG corpus")
        ids.add(record["id"])
        prompts.add(key(record["question"]))
    counts = dict(Counter(x["source"] for x in records))
    if counts != manifest["source_counts"] or set(counts) != {"stackoverflow", "techqa"}:
        raise ValueError("Corpus counts or sources differ from manifest")
    return records, manifest


def build(root: Path, *, download=False, max_records=C.MAX_STACKOVERFLOW_RECORDS,
          min_question_score=C.MIN_QUESTION_SCORE, min_answer_score=C.MIN_ANSWER_SCORE):
    if max_records < 1:
        raise ValueError("max_records must be positive")
    source = root / C.STACKOVERFLOW_RAW
    if download:
        from huggingface_hub import hf_hub_download
        hf_hub_download(C.STACKOVERFLOW_SOURCE, "documents.jsonl", repo_type="dataset",
                        revision=C.STACKOVERFLOW_REVISION, local_dir=source.parent)
    if not source.exists():
        raise FileNotFoundError("Stack Overflow source missing; run python -m rag.build_corpus --download")
    questions, answers = benchmark_keys(root)
    stats = {
        "stackoverflow": Counter({name: 0 for name in (
            "raw", "missing_question_or_accepted_answer", "invalid_score", "low_score",
            "irrelevant_topic", "answer_too_short_or_noise", "benchmark_leakage_removed", "duplicates_removed")}),
        "techqa": Counter({name: 0 for name in (
            "raw", "answerable", "excluded_null_or_unanswerable", "benchmark_leakage_removed", "duplicates_removed")}),
    }
    seen_ids, seen_questions = set(), set()

    def accept(record, source_name):
        if overlaps(record, questions, answers):
            stats[source_name]["benchmark_leakage_removed"] += 1
            return False
        qkey = hashlib.sha256(key(record["question"]).encode()).digest()
        if record["id"] in seen_ids or qkey in seen_questions:
            stats[source_name]["duplicates_removed"] += 1
            return False
        seen_ids.add(record["id"])
        seen_questions.add(qkey)
        return True

    techqa = []
    for example in load_techqa(root / C.TECHQA_PATH):
        stats["techqa"]["raw"] += 1
        if not example.answerable or not whitespace(example.answer):
            stats["techqa"]["excluded_null_or_unanswerable"] += 1
            continue
        stats["techqa"]["answerable"] += 1
        record = document("techqa", example.id, None, whitespace(example.question), whitespace(example.answer),
                          {"product_family": "IBM", "original_split": example.original_split})
        if accept(record, "techqa"):
            techqa.append(record)

    # Bounded hash samples per topic, then round-robin. Full source is scanned;
    # source date/order does not bias the selection toward the oldest records.
    buckets = defaultdict(list)
    per_topic_cap = (max_records + len(TOPICS) - 1) // len(TOPICS)
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            stats["stackoverflow"]["raw"] += 1
            if stats["stackoverflow"]["raw"] % 100_000 == 0:
                print(f"Scanned {stats['stackoverflow']['raw']:,} Stack Overflow rows", file=sys.stderr)
            record, reason = normalize_stackoverflow(json.loads(line), min_question_score, min_answer_score)
            if reason:
                stats["stackoverflow"][reason] += 1
                continue
            if not accept(record, "stackoverflow"):
                continue
            stats["stackoverflow"]["eligible_unique"] += 1
            rank = int(hashlib.sha256(record["id"].encode()).hexdigest(), 16)
            heap = buckets[record["metadata"]["topic"]]
            item = (-rank, record["id"], record)
            if len(heap) < per_topic_cap:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
    ordered = {topic: iter([item[2] for item in sorted(heap, reverse=True)]) for topic, heap in sorted(buckets.items())}
    stackoverflow = []
    while ordered and len(stackoverflow) < max_records:
        for topic in list(ordered):
            record = next(ordered[topic], None)
            if record is None:
                del ordered[topic]
            else:
                stackoverflow.append(record)
            if len(stackoverflow) == max_records:
                break
    if not stackoverflow or not techqa:
        raise ValueError("Both knowledge sources must contain usable records")
    stats["stackoverflow"]["not_selected_by_diversity_cap"] = stats["stackoverflow"]["eligible_unique"] - len(stackoverflow)
    stats["stackoverflow"]["selected"] = len(stackoverflow)
    stats["techqa"]["selected"] = len(techqa)
    output = root / C.CORPUS_DIR
    files = {}
    for name, records in (("stackoverflow_it_support.jsonl", stackoverflow), ("techqa_ibm.jsonl", techqa),
                          ("combined_corpus.jsonl", stackoverflow + techqa)):
        write_jsonl(output / name, records)
        files[name] = {"records": len(records), "sha256": file_sha256(output / name)}
    manifest = {
        "schema_version": 1,
        "sources": {"stackoverflow": {"dataset": C.STACKOVERFLOW_SOURCE, "revision": C.STACKOVERFLOW_REVISION,
                                       "path": C.STACKOVERFLOW_RAW, "sha256": file_sha256(source)},
                    "techqa": {"path": C.TECHQA_PATH, "sha256": file_sha256(root / C.TECHQA_PATH)}},
        "source_counts": {"stackoverflow": len(stackoverflow), "techqa": len(techqa)},
        "total_records": len(stackoverflow) + len(techqa), "filter_stats": stats,
        "selection": {"method": "sha256_id_sample_per_topic_then_round_robin", "full_source_scanned": True,
                      "max_stackoverflow_records": max_records, "min_question_score": min_question_score,
                      "per_topic_cap": per_topic_cap,
                      "min_answer_score": min_answer_score, "topics": TOPICS, "date_cutoff": None},
        "topic_distribution": dict(sorted(Counter(x["metadata"]["topic"] for x in stackoverflow).items())),
        "tag_distribution": dict(sorted(Counter(t for x in stackoverflow for t in x["metadata"]["tags"]).items())),
        "leakage_checks": {"benchmark_path": C.BENCHMARK_PATH, "benchmark_sha256": file_sha256(root / C.BENCHMARK_PATH),
                           "exact_prompt_overlap": 0, "exact_reference_answer_overlap": 0,
                           "method": "casefold_whitespace_exact_fields; exclusion only, no topic-based benchmark filtering",
                           "techqa_intentionally_allowed": True},
        "files": files,
    }
    write_json(output / "manifest.json", manifest)
    write_json(output / "filter_stats.json", stats)
    validate_corpus(root)
    return {"source_counts": manifest["source_counts"], "filter_stats": stats, "files": files}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=C.PROJECT_ROOT)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--validate", action="store_true", help="Validate existing corpus and benchmark firewall only")
    parser.add_argument("--max-records", type=int, default=C.MAX_STACKOVERFLOW_RECORDS)
    parser.add_argument("--min-question-score", type=int, default=C.MIN_QUESTION_SCORE)
    parser.add_argument("--min-answer-score", type=int, default=C.MIN_ANSWER_SCORE)
    args = parser.parse_args()
    if args.validate:
        records, manifest = validate_corpus(args.project_root.resolve())
        result = {"passed": True, "documents": len(records), "source_counts": manifest["source_counts"],
                  "leakage_checks": manifest["leakage_checks"]}
    else:
        result = build(args.project_root.resolve(), download=args.download, max_records=args.max_records,
                       min_question_score=args.min_question_score, min_answer_score=args.min_answer_score)
    print(json.dumps(result, indent=2))
