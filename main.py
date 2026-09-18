"""Ask the fine-tuned DEX model a question from the terminal."""
from __future__ import annotations

import argparse
import sys

from rag import config as rag_config
from rag.query import generate


def print_result(result: dict, *, sources: bool) -> None:
    print(result["answer"])
    if not sources:
        return
    if not result["rag"]:
        print("\nSources: none (RAG disabled)")
        return

    used = {row["id"]: row["citation"] for row in result["context_used"]["used_documents"]}
    print("\nSources ([n] = supplied to DEX; retrieved-only passages were not supplied):")
    if not result["retrieved_sources"]:
        print("  None retrieved")
    for record in result["retrieved_sources"]:
        label = f"[{used[record['id']]}]" if record["id"] in used else "[retrieved only]"
        title = " ".join((record.get("title") or record["question"]).split())
        print(f"  {label} {record['source']}: {title} ({record['id']})")
        if url := (record.get("metadata") or {}).get("url"):
            print(f"      {url}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", help="Ask once and exit; omit for an interactive prompt")
    parser.add_argument("--rag", action="store_true", help="Retrieve technical context before answering")
    parser.add_argument("--sources", action="store_true", help="Show retrieved sources and which reached DEX")
    args = parser.parse_args(argv)
    if args.question is not None and not args.question.strip():
        parser.error("--question must not be empty")

    def ask(question: str) -> None:
        result = generate(rag_config.PROJECT_ROOT, question, model="dex", rag=args.rag)
        print_result(result, sources=args.sources)

    try:
        if args.question is not None:
            ask(args.question)
        else:
            print("DEX terminal. Type 'quit' or 'exit' (or press Ctrl-D) to leave.")
            while True:
                try:
                    question = input("You> ").strip()
                except EOFError:
                    print()
                    break
                if question.casefold() in {"quit", "exit"}:
                    break
                if question:
                    ask(question)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
