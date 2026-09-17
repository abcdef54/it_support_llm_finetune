"""Curate the pinned IT_Support_V2 source for DEX without loading model weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from statistics import mean, median

from data.utils import write_json, write_jsonl

SOURCE_DATASET = "benjaminmacklin/IT_Support_V2"
SOURCE_REVISION = "bbaf7b99593a9492594d5be414619c1ad33e1730"
ASSISTANT_NAME = "DEX"
ASSISTANT_EXPANSION = "Diagnostic EXpert"
DEVELOPER_DESCRIPTION = "Huy, an AI student"
SOURCE_ASSISTANT_NAME = "Mack"
IDENTITY = (f"I am {ASSISTANT_NAME}, a {ASSISTANT_EXPANSION} SLM developed by {DEVELOPER_DESCRIPTION}. "
            "My purpose is to provide practical and technically useful IT support and troubleshooting assistance.")


@dataclass(frozen=True)
class Settings:
    seed: int = 42
    max_total_examples: int = 30_000
    validation_ratio: float = 0.1
    max_identity_examples: int = 8
    max_repeated_response: int = 5
    min_answer_words: int = 3
    max_formatted_tokens: int = 1024
    audit_size: int = 100


MACK = re.compile(r"\bmack\b", re.I)
COMPANY = re.compile(r"\bMBGOC\b", re.I)
SOURCE_ALIASES = ("Axero", "Ax", "Axleshoot", "Ax0", "Axo", "Axxio", "Mackn", "Axpio", "Ax.compliance",
                  "AxMack", "Ax0io", "Ax3io", "Ax.computer", "Axelling", "Axleshoots", "Axxo", "Ax0x",
                  "Mackerms", "Axlo", "Ax3", "Axalo", "Ax0ler", "Ax.com", "Ax.0x00000024", "Axool", "Axelys")
OTHER_PERSONA = re.compile(r"\b(?:" + "|".join(re.escape(s) for s in SOURCE_ALIASES) + r")\b|I am Azure AI Assistant", re.I)
LABEL = re.compile(r"[a-z]+(?:[_-][a-z]+)+")
DOMAIN_RESTRICTION = re.compile(
    r"only (?:provide|answer|assist|handle|support|respond to).{0,50}(?:IT|enterprise|technical)|"
    r"outside (?:my|the) (?:defined )?(?:scope|domain)|rephrase your request to focus on IT|"
    r"cannot (?:generate|provide|perform|assist|help|translate).{0,100}(?:creative|personal|non-IT|historical)|"
    r"cannot[^.!?]{0,100}[.!?]\s*(?:Mack|DEX|I) (?:provides?|offers?) enterprise IT (?:assistance|support)", re.I)
COMPANY_INTRO = re.compile(
    r"^I am (?:Mack|DEX),[^\n]{0,180}?\bMBGOC(?:\s+Pvt\.?\s+Ltd\.?)?\.?\s*", re.I)
IDENTITY_QUESTION = re.compile(
    r"^(?:who (?:are you|(?:developed|created|built) (?:you|Mack|DEX))|"
    r"what (?:is|are) (?:your (?:name|role|purpose|primary objective)|"
    r"(?:the )?(?:Mack|DEX)(?: model)?(?:'s|’s)?(?: (?:primary )?(?:role|purpose|objective))?)|"
    r"what does (?:Mack|DEX) stand for)[?.!]*$", re.I)
VERBS = ("check verify inspect analyze compare evaluate test review monitor scan update restart reset repair "
         "configure enable disable install reinstall uninstall run use open clear remove restore validate "
         "examine ensure confirm troubleshoot replace reconnect renew reboot resolve identify isolate query "
         "audit measure track deploy enforce define capture assess integrate schedule establish collect "
         "adjust set flush reconfigure rebuild investigate look select navigate connect detach mount back "
         "disconnect reattach clean reseat re-seat increase reduce optimize synchronize sync apply grant "
         "revoke export import execute change move stop start switch upgrade delete create perform inspect "
         "detect determine locate flag find").split()
ACTION = re.compile(r"\b(?:" + "|".join(re.escape(v) for v in VERBS) + r")\b", re.I)
THIRD_PERSON = {v + "s": v for v in VERBS}
THIRD_PERSON.update({"verifies": "verify", "identifies": "identify", "queries": "query", "applies": "apply",
                     "analyzes": "analyze", "watches": "watch", "fixes": "fix", "flags": "flag",
                     "assesses": "assess", "searches": "search", "establishes": "establish", "logs": "log"})
THIRD_VERB = re.compile(r"(^|,\s*|;\s*|\band\s+|\bthen\s+)(" + "|".join(THIRD_PERSON) + r")\b", re.I)
EXPLANATION = re.compile(r"\b(?:because|caused by|indicates|means|refers to|used to|enables|allows|requires|"
                         r"prevents|can be|is (?:a|an|the)|are (?:a|the))\b", re.I)
PROMISE = re.compile(r"\b(?:we|I) (?:will|would) (?:send|contact|provide|investigate|get back|follow up)|"
                     r"follow-up email|thank you for reaching out", re.I)
COMMAND = re.compile(r"\b(?:sfc\s+/|dism\s+/|ssh\s+-|pip\s+install|Get-[A-Z]|systemctl\s|ipconfig\s|nslookup\s)")
SECURITY_BYPASS = re.compile(r"\bdisable (?:the |your )?(?:firewalls?|antivirus|security software|SmartScreen|certificate validation)\b", re.I)
TOPICS = {
    "linux": r"linux|ubuntu|debian|centos|systemctl|sudo|/etc/|bash|ssh",
    "databases": r"sql|database|postgres|mysql|mongodb|oracle|redis",
    "cloud_servers": r"azure|aws|cloud|server|kubernetes|docker|vmware|hyper-v|vcenter|lambda",
    "network_vpn": r"network|dns|dhcp|vpn|router|routing|subnet|ethernet|wi-fi|wireless|tcp|firewall",
    "security_identity": r"security|malware|ransom|antivirus|authentication|password|certificate|encryption|"
                         r"bitlocker|kerberos|permission|credential|exploit|attack|threat|phishing|exfiltration",
    "email_outlook": r"outlook|email|exchange|smtp|mailbox|imap",
    "teams": r"\bteams\b",
    "office": r"m365|office|excel|word|sharepoint|onedrive|powerpoint",
    "hardware_drivers": r"driver|printer|touchpad|touchscreen|touch screen|usb|bluetooth|camera|keyboard|"
                        r"mouse|monitor|display|dock|battery|audio|microphone|hardware|fan|bios|firmware|gpu",
    "recovery_storage": r"backup|restore|recovery|disk|storage|partition|ntfs|file system|filesystem|raid",
    "developer_tools": r"python|pip|git|npm|api|script|code|powershell|compiler|sdk|java|visual studio",
    "windows": r"windows|registry|gpo|group policy|device manager|dism|sfc|win32",
    "performance": r"performance|slow|latency|cpu|memory|ram|resource",
    "software": r"software|application|install|package|service|process|client|browser|license|system",
    "enterprise_apps": r"sccm|iis|active directory|intune|endpoint|wsus|netapp|citrix|domain controller",
}
TOPIC_PATTERNS = {k: re.compile(r"\b(?:" + v + r")\b", re.I) for k, v in TOPICS.items()}
TECHNICAL = re.compile(r"\b(?:" + "|".join(TOPICS.values()) + r")\b", re.I)


def normalized(text: str) -> str:
    """Comparison only: never change whitespace inside stored commands/code."""
    return " ".join(text.casefold().split())


def prompt_key(messages: list[dict]) -> str:
    return json.dumps([(m["role"], normalized(m["content"])) for m in messages], ensure_ascii=False)


def parse_messages(row: dict) -> list[dict]:
    messages = row.get("messages")
    if isinstance(messages, str):
        messages = json.loads(messages)
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("Expected a conversation with user and assistant turns")
    result = []
    for i, message in enumerate(messages):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError("Invalid message structure")
        role, content = message["role"], message["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Empty or non-string message")
        expected = "user" if not result or result[-1]["role"] in {"system", "assistant"} else "assistant"
        if role != expected and not (i == 0 and role == "system"):
            raise ValueError("Invalid conversation roles/order")
        result.append({"role": role, "content": content.replace("\r\n", "\n").strip("\n")})
    if result[-1]["role"] != "assistant" or not any(m["role"] == "user" for m in result):
        raise ValueError("Missing final assistant target")
    return result


def is_intent_label_response(text: str) -> bool:
    return bool(LABEL.fullmatch(text.strip())) or text.strip().casefold() == "performance"


def is_domain_restriction(text: str) -> bool:
    if DOMAIN_RESTRICTION.search(text) or re.search(r"(?:capabilities|scope|expertise|function).{0,30}(?:limited to|enterprise IT)", text, re.I):
        return True
    refusal = re.search(r"\bI (?:cannot|can't|am unable to) (?:create|generate|perform|design|write|translate|produce)\b", text, re.I)
    safety = re.search(r"unauthori[sz]ed|without permission|malware|steal|phishing|illegal|harm|bypass|exploit", text, re.I)
    return bool(refusal and not safety)


def has_noise(text: str) -> bool:
    # Protect code blocks, paths, filenames and URLs; inspect unexpected prose only.
    prose = re.sub(r"```.*?```|https?://\S+|(?:[A-Za-z]:\\|/)[^\s`]+|[^\s`]+\.[a-zA-Z]{1,6}\b", "", text, flags=re.S)
    return bool(re.search(r"[\u3400-\u9fff]{2,}|\ufffd", prose))


def known_technical_error(messages: list[dict]) -> bool:
    text = "\n".join(m["content"] for m in messages)
    if re.search(r"\bOutlook 2020\b", text, re.I):
        return True
    return "tempdb" in text.casefold() and any(
        re.search(r"restore.*\bbackup\b|back up tempdb", m["content"], re.I)
        and not re.search(r"cannot|not allowed|do not|don't", m["content"], re.I)
        for m in messages if m["role"] == "assistant")


def transform(messages: list[dict]) -> tuple[list[dict], list[str], str | None]:
    flags = []
    text = "\n".join(m["content"] for m in messages)
    protected = re.findall(r"```.*?```|`[^`]+`|https?://\S+|(?:[A-Za-z]:\\|/)[^\s`]+", text, re.S)
    if any(MACK.search(fragment) or COMPANY.search(fragment) for fragment in protected):
        return messages, flags, "source_identity_in_code_or_path"
    had_mack, had_company = bool(MACK.search(text)), bool(COMPANY.search(text))
    final_question = messages[-2]["content"]
    identity = len(messages) == 2 and bool(IDENTITY_QUESTION.fullmatch(final_question.strip()))
    if identity:
        messages = [{"role": "user", "content": MACK.sub(ASSISTANT_NAME, final_question)},
                    {"role": "assistant", "content": IDENTITY}]
        flags.append("identity_rewritten")
    else:
        changed = []
        for message in messages:
            content = message["content"]
            if message["role"] == "assistant":
                content = COMPANY_INTRO.sub("", content).strip("\n")
            content = MACK.sub(ASSISTANT_NAME, content)
            if COMPANY.search(content) or re.search(r"\b(?:I|DEX).{0,90}(?:developed|created).{0,40}Dev team", content, re.I):
                return messages, flags, "complex_company_persona"
            if message["role"] == "assistant" and content.startswith(ASSISTANT_NAME + " "):
                body = content[len(ASSISTANT_NAME) + 1:]
                first = body.split(maxsplit=1)[0].casefold()
                if first in THIRD_PERSON:
                    content = "Suggested diagnostic approach: " + THIRD_VERB.sub(
                        lambda m: m[1] + THIRD_PERSON[m[2].casefold()], body)
                    flags.append("capability_genericized")
                else:
                    return messages, flags, "complex_company_persona"
            changed.append({"role": message["role"], "content": content})
        messages = changed
    if had_mack:
        flags.append("mack_converted")
    if had_company:
        flags.append("company_rewritten")
    return messages, sorted(set(flags)), None


def topic_for(text: str) -> str:
    return next((topic for topic, pattern in TOPIC_PATTERNS.items() if pattern.search(text)), "other_it")


def length_stats(texts: list[str]) -> dict:
    def describe(values):
        ordered = sorted(values)
        return {"min": min(ordered), "median": median(ordered), "mean": round(mean(ordered), 2),
                "p95": ordered[min(len(ordered) - 1, int(len(ordered) * .95))], "max": max(ordered)} if ordered else {}
    return {"characters": describe([len(t) for t in texts]), "words": describe([len(t.split()) for t in texts])}


def top_counts(texts: list[str], limit: int = 15) -> list[dict]:
    return [{"text": text, "count": count} for text, count in Counter(map(normalized, texts)).most_common(limit)]


def inspect_rows(rows: list[dict]) -> dict:
    valid, malformed = [], 0
    for row in rows:
        try:
            valid.append(parse_messages(row))
        except (ValueError, TypeError, AttributeError):
            malformed += 1
    answers = [m[-1]["content"] for m in valid]
    texts = [json.dumps(row, ensure_ascii=False) for row in rows]
    return {
        "raw_records": len(rows), "valid_conversations": len(valid), "malformed_records": malformed,
        "native_messages_arrays": sum(isinstance(r.get("messages"), list) for r in rows),
        "json_string_messages": sum(isinstance(r.get("messages"), str) for r in rows),
        "role_sequences": dict(Counter("/".join(m["role"] for m in messages) for messages in valid)),
        "multi_turn_records": sum(len(m) > 2 for m in valid), "category_metadata": "None in source schema",
        "mack_records": sum(bool(MACK.search(t)) for t in texts),
        "mack_occurrences": sum(len(MACK.findall(t)) for t in texts),
        "company_records": sum(bool(COMPANY.search(t)) for t in texts),
        "discovered_source_companies": ["MBGOC", "MBGOC Pvt. Ltd."],
        "inconsistent_persona_aliases": list(SOURCE_ALIASES) + ["Azure AI Assistant"],
        "inconsistent_persona_records": sum(bool(OTHER_PERSONA.search(t)) for t in texts),
        "classification_labels": dict(Counter(a for a in answers if is_intent_label_response(a))),
        "response_lengths": length_stats(answers), "top_responses": top_counts(answers),
        "top_prompts": top_counts([m[-2]["content"] for m in valid]),
    }


def prepare_rows(rows: list[dict], *, settings: Settings = Settings(), forbidden: set[str] | None = None,
                 tokenizer=None) -> tuple[list[dict], list[dict], dict, list[dict], list[dict]]:
    if settings.max_total_examples < 2 or not 0 < settings.validation_ratio < 1:
        raise ValueError("Invalid corpus size or validation ratio")
    if settings.max_repeated_response < 1 or settings.max_identity_examples < 0:
        raise ValueError("Invalid repetition/identity caps")
    forbidden = forbidden or set()
    rejected, counts, conversions, candidates = [], Counter(), Counter(), []
    reject_examples = Counter()

    def reject(reason, index, row):
        counts[reason] += 1
        if reject_examples[reason] < 3:
            rejected.append({"source_index": index, "reason": reason, "source_record": row})
            reject_examples[reason] += 1

    for index, row in enumerate(rows):
        try:
            messages = parse_messages(row)
        except (ValueError, TypeError, AttributeError):
            reject("malformed_or_empty", index, row)
            continue
        assistants = [m["content"] for m in messages if m["role"] == "assistant"]
        if any(is_intent_label_response(a) for a in assistants):
            reject("classification", index, row)
            continue
        if any(is_domain_restriction(a) for a in assistants):
            reject("domain_restriction", index, row)
            continue
        if any(SECURITY_BYPASS.search(a) for a in assistants):
            reject("blanket_security_bypass", index, row)
            continue
        if any(has_noise(m["content"]) for m in messages):
            reject("language_noise", index, row)
            continue
        if any(OTHER_PERSONA.search(m["content"]) for m in messages):
            reject("mixed_source_persona", index, row)
            continue
        if known_technical_error(messages):
            reject("confirmed_technical_artifact", index, row)
            continue
        # Exact comparisons only. No benchmark content influences quality rules.
        if any(normalized(m["content"]) in forbidden for m in messages):
            reject("benchmark_exact_overlap", index, row)
            continue
        messages, flags, reason = transform(messages)
        if reason:
            reject(reason, index, row)
            continue
        answer = messages[-1]["content"]
        if any(normalized(m["content"]) in forbidden for m in messages):
            reject("benchmark_exact_overlap", index, row)
            continue
        if "identity_rewritten" not in flags:
            bad_answer = None
            for message in messages:
                if message["role"] != "assistant":
                    continue
                content = message["content"]
                command = bool(COMMAND.search(content))
                actionable = bool(ACTION.search(content)) or command
                if PROMISE.search(content) and not re.search(
                    r"(?:^|[.!?]\s+|\n)(?:\d+[.)]\s*)?(?:" + "|".join(VERBS) + r")\b", content, re.I):
                    bad_answer = "generic_non_answer"
                elif ((len(content.split()) < settings.min_answer_words and not command) or not TECHNICAL.search(
                        " ".join(m["content"] for m in messages)) or not (actionable or EXPLANATION.search(content))):
                    bad_answer = "no_technical_guidance"
                if bad_answer:
                    break
            if bad_answer:
                reject(bad_answer, index, row)
                continue
        conversions.update(flags)
        topic = "identity" if "identity_rewritten" in flags else topic_for(messages[-2]["content"])
        candidates.append({"source_index": index, "topic": topic, "transformations": flags,
                           "prompt": messages[:-1], "completion": messages[-1:]})

    after_filtering = len(candidates)
    # Prefer concrete actions and bounded information content; source order breaks ties.
    candidates.sort(key=lambda c: (-len(set(ACTION.findall(c["completion"][0]["content"].lower()))),
                                  -min(len(c["completion"][0]["content"].split()), 120), c["source_index"]))
    unique, seen_pairs, seen_prompts = [], set(), set()
    for c in candidates:
        prompt = prompt_key(c["prompt"])
        pair = (prompt, normalized(c["completion"][0]["content"]))
        reason = "duplicate_conversation" if pair in seen_pairs else "duplicate_prompt" if prompt in seen_prompts else None
        seen_pairs.add(pair)
        if reason:
            reject(reason, c["source_index"], rows[c["source_index"]])
            continue
        seen_prompts.add(prompt)
        unique.append(c)

    if tokenizer is not None:
        token_checked = []
        for start in range(0, len(unique), 256):
            batch = unique[start:start + 256]
            encodings = tokenizer.apply_chat_template([c["prompt"] + c["completion"] for c in batch],
                                                       tokenize=True, enable_thinking=False)
            if isinstance(encodings, Mapping):
                encodings = encodings["input_ids"]
            for c, ids in zip(batch, encodings, strict=True):
                c["formatted_tokens"] = len(ids)
                if len(ids) > settings.max_formatted_tokens:
                    reject("over_token_limit", c["source_index"], rows[c["source_index"]])
                else:
                    token_checked.append(c)
        unique = token_checked

    repeated, buckets, identity_count = Counter(), defaultdict(list), 0
    for c in unique:
        answer_key = normalized(c["completion"][0]["content"])
        if c["topic"] == "identity":
            if identity_count >= settings.max_identity_examples:
                reject("identity_cap", c["source_index"], rows[c["source_index"]])
                continue
            identity_count += 1
        elif repeated[answer_key] >= settings.max_repeated_response:
            reject("repeated_response_cap", c["source_index"], rows[c["source_index"]])
            continue
        repeated[answer_key] += 1
        buckets[c["topic"]].append(c)
    candidate_count = sum(map(len, buckets.values()))
    candidate_topics = {topic: len(items) for topic, items in sorted(buckets.items())}
    rng = random.Random(settings.seed)
    for topic in sorted(buckets):
        rng.shuffle(buckets[topic])
    selected = []
    while len(selected) < settings.max_total_examples and any(buckets.values()):
        for topic in sorted(buckets):
            if buckets[topic] and len(selected) < settings.max_total_examples:
                selected.append(buckets[topic].pop())
    counts["selection_cap"] = candidate_count - len(selected)
    if len(selected) < 2:
        raise ValueError("Fewer than two valid examples survived; inspect filters/source")
    rng.shuffle(selected)
    validation_size = max(1, min(len(selected) - 1, round(len(selected) * settings.validation_ratio)))
    to_sft = lambda c: {"prompt": c["prompt"], "completion": c["completion"]}
    validation = [to_sft(c) for c in selected[:validation_size]]
    train = [to_sft(c) for c in selected[validation_size:]]
    audit, audit_ids = [], set()
    # Include each available transformation/topic before sampling the remainder.
    for tag in sorted(set(c["topic"] for c in selected) | set(f for c in selected for f in c["transformations"])):
        match = next(c for c in selected if c["topic"] == tag or tag in c["transformations"])
        if match["source_index"] not in audit_ids:
            audit.append(match)
            audit_ids.add(match["source_index"])
    for c in selected:
        if len(audit) >= settings.audit_size:
            break
        if c["source_index"] not in audit_ids:
            audit.append(c)
            audit_ids.add(c["source_index"])
    text = "\n".join(m["content"] for c in selected for m in c["prompt"] + c["completion"])
    selected_ids = {c["source_index"] for c in selected}
    stats = {
        "removals_by_reason": dict(sorted(counts.items())), "transformations_after_quality_filter": dict(conversions),
        "transformations_in_final_corpus": dict(Counter(f for c in selected for f in c["transformations"])),
        "candidate_count_after_quality_filter": after_filtering, "candidate_count_before_selection": candidate_count,
        "final_selected_count": len(selected), "train_count": len(train), "validation_count": len(validation),
        "topic_distribution_candidates": candidate_topics,
        "topic_distribution_final": dict(sorted(Counter(c["topic"] for c in selected).items())),
        "response_lengths_after": length_stats([c["completion"][0]["content"] for c in selected]),
        "top_retained_responses": top_counts([c["completion"][0]["content"] for c in selected]),
        "remaining_mack_occurrences": len(MACK.findall(text)), "remaining_company_occurrences": len(COMPANY.findall(text)),
        "remaining_other_source_persona_occurrences": len(OTHER_PERSONA.findall(text)),
        "dex_occurrences": len(re.findall(r"\bDEX\b", text, re.I)),
        "formatted_token_limit": settings.max_formatted_tokens,
        "max_formatted_tokens": max((c.get("formatted_tokens", 0) for c in selected), default=0) if tokenizer else None,
        "multi_turn_retained": sum(len(c["prompt"]) > 1 for c in selected),
        "source_company_records_removed": sum(bool(COMPANY.search(json.dumps(row))) and i not in selected_ids
                                               for i, row in enumerate(rows)),
        "near_duplicate_policy": f"Normalized exact prompts/pairs deduplicated; identical normalized responses capped at {settings.max_repeated_response} (identity capped separately). No semantic pairwise comparison.",
        "topic_method": "Deterministic keyword buckets from user questions; proxy categories, not source labels. Seeded round-robin selection.",
        "technical_quality_limit": "Deterministic content screening does not certify factual accuracy of synthetic troubleshooting advice; review the audit before training.",
    }
    assert sum(counts.values()) + len(selected) == len(rows), "Removal accounting mismatch"
    return train, validation, stats, audit[:settings.audit_size], rejected


def audit_corpus(train: list[dict], validation: list[dict], forbidden: set[str]) -> dict:
    keys, issues = [], []
    for split in (train, validation):
        split_keys = []
        for row in split:
            messages = parse_messages({"messages": row["prompt"] + row["completion"]})
            split_keys.append(prompt_key(row["prompt"]))
            for m in messages:
                text = m["content"]
                if MACK.search(text) or COMPANY.search(text) or OTHER_PERSONA.search(text) or normalized(text) in forbidden or has_noise(text):
                    issues.append("identity/noise/overlap")
                if m["role"] == "assistant" and (is_intent_label_response(text) or is_domain_restriction(text)):
                    issues.append("label/domain_restriction")
        if len(split_keys) != len(set(split_keys)):
            issues.append("duplicate_prompt_within_split")
        keys.append(set(split_keys))
    if keys[0] & keys[1]:
        issues.append("train_validation_overlap")
    if issues:
        raise ValueError(f"Final corpus audit failed: {Counter(issues)}")
    return {"passed": True, "train_validation_prompt_overlap": 0, "exact_benchmark_overlap": 0,
            "remaining_source_identity": 0, "remaining_labels": 0, "remaining_detected_noise": 0}


def run(root: Path, *, skip_download: bool = False, settings: Settings = Settings()) -> dict:
    from datasets import get_dataset_config_names, load_dataset, load_from_disk
    from transformers import AutoTokenizer
    from benchmark.baseline.config import BASE_MODEL_ID, BASE_MODEL_REVISION
    from benchmark.common.dataset import file_sha256, load_techqa

    raw = root / "data/raw/finetune_v2"
    output = root / "data/processed/finetune_v2"
    if skip_download:
        source = load_from_disk(str(raw / "dataset"))
        provenance = json.loads((raw / "source_manifest.json").read_text())
        if provenance["dataset"] != SOURCE_DATASET or provenance["revision"] != SOURCE_REVISION:
            raise ValueError("Cached source revision mismatch; download the pinned source again")
        configs = provenance["configurations"]
    else:
        configs = get_dataset_config_names(SOURCE_DATASET, revision=SOURCE_REVISION)
        if configs != ["default"]:
            raise ValueError(f"Unexpected source configurations: {configs}")
        source = load_dataset(SOURCE_DATASET, "default", revision=SOURCE_REVISION,
                              cache_dir=str(raw / ".datasets_cache"))
        source.save_to_disk(str(raw / "dataset"))
    rows = [row for split in source.values() for row in split]
    digest = hashlib.sha256()
    for row in rows:
        digest.update((json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode())
    raw_sha256 = digest.hexdigest()
    if skip_download and provenance["records_sha256"] != raw_sha256:
        raise ValueError("Cached source records changed")
    if not skip_download:
        write_json(raw / "source_manifest.json", {"dataset": SOURCE_DATASET, "revision": SOURCE_REVISION,
            "configurations": configs, "records_sha256": raw_sha256})
    inspection = inspect_rows(rows)
    write_json(output / "source_inspection.json", inspection)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, revision=BASE_MODEL_REVISION,
        cache_dir=str(raw / "tokenizer_cache"), local_files_only=skip_download)
    benchmark = load_techqa(root / "data/processed/evaluation/techqa_benchmark.jsonl")
    forbidden = {normalized(t) for example in benchmark for t in (example.question, example.answer) if t}
    train, validation, stats, audit, rejected = prepare_rows(rows, settings=settings, forbidden=forbidden, tokenizer=tokenizer)
    validation_report = audit_corpus(train, validation, forbidden)
    # Corpus/answer length checks use the tokenizer only, never a model.
    before_answers = []
    for row in rows:
        try:
            before_answers.append(parse_messages(row)[-1]["content"])
        except (ValueError, TypeError, AttributeError):
            pass
    for label, texts in (("before", before_answers),
                         ("after", [r["completion"][0]["content"] for r in train + validation])):
        lengths = []
        for start in range(0, len(texts), 512):
            lengths.extend(map(len, tokenizer(texts[start:start + 512], add_special_tokens=False)["input_ids"]))
        stats[f"response_tokens_{label}"] = {"mean": round(mean(lengths), 2), "median": median(lengths), "max": max(lengths)}
    stats["response_lengths_before"] = inspection["response_lengths"]
    stats["raw_classification_labels"] = inspection["classification_labels"]
    stats["top_raw_responses"] = inspection["top_responses"]
    stats["top_raw_prompts"] = inspection["top_prompts"]
    stats["audit"] = validation_report
    stats["technical_rule_references"] = {
        "tempdb_backup_restore": "https://learn.microsoft.com/en-us/sql/relational-databases/databases/tempdb-database",
        "outlook_release_names": "https://support.microsoft.com/en-us/outlook/getstarted/what-version-of-outlook-do-i-have",
    }
    stats["retained_boilerplate_phrase_counts"] = {
        phrase: sum(phrase.casefold() in r["completion"][0]["content"].casefold() for r in train + validation)
        for phrase in ("Thank you for reaching out", "We will provide", "follow-up email")}
    stats["large_removal_rules_for_review"] = {reason: count for reason, count in stats["removals_by_reason"].items()
                                               if count > len(rows) / 2}
    if not skip_download:
        from huggingface_hub import hf_hub_download
        hf_hub_download(SOURCE_DATASET, "License", repo_type="dataset", revision=SOURCE_REVISION, local_dir=raw)
    shutil.copyfile(raw / "License", output / "SOURCE_LICENSE.txt")
    write_jsonl(output / "train.jsonl", train)
    write_jsonl(output / "validation.jsonl", validation)
    write_jsonl(output / "audit_sample.jsonl", audit)
    write_jsonl(output / "rejected_sample.jsonl", rejected)
    write_json(output / "filter_stats.json", stats)
    manifest = {
        "source_dataset": SOURCE_DATASET, "source_revision": SOURCE_REVISION,
        "source_configuration": "default", "available_configurations": configs,
        "available_splits": {name: len(split) for name, split in source.items()},
        "source_schema": {name: split.features.to_dict() for name, split in source.items()},
        "source_format": "JSONL with native messages arrays", "source_license": "MIT",
        "raw_record_count": len(rows), "parsed_record_count": inspection["valid_conversations"],
        "raw_records_sha256": raw_sha256,
        "mack_records_found": inspection["mack_records"], "mack_occurrences_before": inspection["mack_occurrences"],
        "source_company_records_found": inspection["company_records"],
        "assistant_name": ASSISTANT_NAME, "assistant_expansion": ASSISTANT_EXPANSION,
        "developer_description": DEVELOPER_DESCRIPTION, "settings": asdict(settings),
        "tokenizer": {"model": BASE_MODEL_ID, "revision": BASE_MODEL_REVISION, "thinking_enabled": False},
        "train_sha256": file_sha256(output / "train.jsonl"), "validation_sha256": file_sha256(output / "validation.jsonl"),
        "source_file_sha256": {name: file_sha256(raw / name) for name in ("Mack_training_data .jsonl", "train.parquet") if (raw / name).is_file()},
        "library_versions": {p: version(p) for p in ("datasets", "huggingface_hub", "transformers", "tokenizers")},
        "output_files": [str((output / f).relative_to(root)) for f in
                         ("train.jsonl", "validation.jsonl", "manifest.json", "filter_stats.json", "source_inspection.json", "audit_sample.jsonl", "rejected_sample.jsonl", "SOURCE_LICENSE.txt")],
        **stats,
    }
    write_json(output / "manifest.json", manifest)
    from data.split_dex_validation import run as split_dex_validation

    split_dex_validation(root)
    return json.loads((output / "manifest.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--max-total-examples", type=int, default=Settings.max_total_examples)
    args = parser.parse_args()
    result = run(args.project_root.resolve(), skip_download=args.skip_download,
                 settings=Settings(max_total_examples=args.max_total_examples))
    print(json.dumps({k: result[k] for k in ("source_revision", "raw_record_count", "final_selected_count", "train_count", "validation_count", "audit")}, indent=2))
