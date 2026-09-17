from __future__ import annotations

import json
import unittest

from data.preprocess_finetune_v2 import (
    IDENTITY, Settings, audit_corpus, has_noise, is_intent_label_response,
    normalized, parse_messages, prepare_rows, transform, is_domain_restriction,
)


def record(question, answer):
    return {"messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]}


def fixtures():
    return [record("Linux SSH authentication failed", "Check the authorized_keys file and verify SSH permissions."),
            record("SQL database connections time out", "Inspect database connection limits and test server connectivity.")]


class DexPreprocessingTests(unittest.TestCase):
    def process(self, extra, **kwargs):
        return prepare_rows(fixtures() + extra, **kwargs)

    def test_normalization_accepts_native_json_and_multiturn_but_rejects_malformed(self):
        messages = fixtures()[0]["messages"] + record("How can I check permissions?", "Run ls -l on the SSH key file.")["messages"]
        self.assertEqual(parse_messages({"messages": json.dumps(messages)}), messages)
        train, val, stats, _, _ = self.process([{"messages": messages}])
        self.assertEqual(stats["multi_turn_retained"], 1)
        self.assertTrue(any(len(r["prompt"]) == 3 for r in train + val))
        for row in ({"messages": "not json"}, {"messages": []}, record("Question", ""),
                    {"messages": [{"role": "tool", "content": "x"}, {"role": "assistant", "content": "y"}]}):
            with self.subTest(row=row), self.assertRaises(ValueError):
                parse_messages(row)

    def test_labels_removed_but_concise_guidance_retained(self):
        for label in ("teams_issue", "network-problem", "password_reset", "performance"):
            self.assertTrue(is_intent_label_response(label))
        self.assertFalse(is_intent_label_response("Get-WindowsCapability"))
        train, val, stats, _, _ = self.process([record("Router fails", "Restart the router and renew the DHCP lease."),
                                               record("Teams fails", "teams_issue")])
        self.assertEqual(stats["removals_by_reason"]["classification"], 1)
        self.assertTrue(any("renew the DHCP" in r["completion"][0]["content"] for r in train + val))

    def test_mack_renaming_and_technical_capabilities(self):
        for question in ("Mack, my touchpad is not responding.", "Hey Mack, Outlook keeps crashing."):
            messages, flags, error = transform(record(question, "Update the driver and check Device Manager.")["messages"])
            self.assertIsNone(error)
            self.assertIn("DEX", messages[0]["content"])
            self.assertIn("mack_converted", flags)
        messages, flags, error = transform(record("Explain how Mack detects network issues.",
                                                 "Mack checks DNS records, inspects routing tables, and tests connectivity.")["messages"])
        self.assertIsNone(error)
        self.assertIn("DEX", messages[0]["content"])
        self.assertEqual(messages[1]["content"], "Suggested diagnostic approach: check DNS records, inspect routing tables, and test connectivity.")
        self.assertIn("capability_genericized", flags)
        messages, _, error = transform(record("How does Mack monitor Active Directory health?",
                                              "Mack checks replication status, logs event viewer errors, and verifies services.")["messages"])
        self.assertIsNone(error)
        self.assertEqual(messages[1]["content"],
                         "Suggested diagnostic approach: check replication status, log event viewer errors, and verify services.")

    def test_identity_and_incidental_company_intro_are_handled_separately(self):
        messages, flags, error = transform(record("What is Mack's primary objective?",
            "I am Mack, an SLM developed by the Dev team at MBGOC Pvt. Ltd.")["messages"])
        self.assertIsNone(error)
        self.assertEqual(messages[-1]["content"], IDENTITY)
        self.assertIn("company_rewritten", flags)
        messages, _, error = transform(record("How to install Windows updates?",
            "I am Mack, an SLM developed by the Dev team at MBGOC Pvt. Ltd. Check Windows Update logs.")["messages"])
        self.assertIsNone(error)
        self.assertEqual(messages[-1]["content"], "Check Windows Update logs.")
        _, _, error = transform(record("Describe MBGOC's proprietary architecture.", "MBGOC uses a private design.")["messages"])
        self.assertEqual(error, "complex_company_persona")

    def test_nonanswers_domain_restrictions_and_noise_are_removed(self):
        _, _, stats, _, _ = self.process([
            record("Write a play", "I cannot generate creative content. I only provide enterprise IT support."),
            record("Network problem", "Thank you for reaching out. We will send documentation in a follow-up email."),
            record("Windows update fails", "Open Windows Update疑难解答."),
            record("Windows update fails", ""),
        ])
        for reason in ("domain_restriction", "generic_non_answer", "language_noise", "malformed_or_empty"):
            self.assertEqual(stats["removals_by_reason"][reason], 1)
        self.assertTrue(is_domain_restriction("I cannot perform database design. Please contact our IT support."))
        self.assertFalse(is_domain_restriction("I cannot create malware. Use an authorized security testing environment."))

    def test_commands_indentation_and_legitimate_unicode_preserved(self):
        answer = 'Run this Python script:\n```python\nif True:\n    print("✓")\n```\nThen check /etc/resolv.conf and /tmp/测试.txt.'
        self.assertFalse(has_noise(answer))
        train, val, _, _, _ = self.process([record("Check a Python diagnostic script", answer)])
        self.assertIn(answer, [r["completion"][0]["content"] for r in train + val])
        _, _, error = transform(record("How to check this file?", "Run cat /tmp/Mack/config.txt")["messages"])
        self.assertEqual(error, "source_identity_in_code_or_path")

    def test_discovered_artifacts_removed_but_safety_refusal_kept(self):
        train, val, stats, _, _ = self.process([
            record("Explain how Axero checks DNS", "Mack checks DNS configuration and tests connectivity."),
            record("Fix Outlook 2020", "Check the Outlook profile and restart Outlook."),
            record("Fix SQL tempdb", "Restore from backup if necessary."),
            record("Create malware", "I cannot create malware. Use an authorized security testing environment."),
        ])
        self.assertEqual(stats["removals_by_reason"]["mixed_source_persona"], 1)
        self.assertEqual(stats["removals_by_reason"]["confirmed_technical_artifact"], 2)
        self.assertTrue(any("cannot create malware" in r["completion"][0]["content"] for r in train + val))

    def test_tokenizer_mapping_and_sequence_limit(self):
        class Tokenizer:
            def apply_chat_template(self, conversations, **kwargs):
                return {"input_ids": [[1] * (1100 if 'oversized' in c[0]['content'] else 20) for c in conversations]}
        _, _, stats, _, _ = self.process([record("Windows oversized prompt", "Check Windows Update logs and restart the service.")], tokenizer=Tokenizer())
        self.assertEqual(stats["removals_by_reason"]["over_token_limit"], 1)
        self.assertEqual(stats["max_formatted_tokens"], 20)

    def test_deduplication_after_rename_and_benchmark_firewall(self):
        extra = [record("Mack, my touchpad fails", "Update the touchpad driver and check Device Manager."),
                 record("DEX, my touchpad fails", "Update the touchpad driver and check Device Manager."),
                 record("Copied benchmark question", "Restart the router and check DNS."),
                 record("Other Windows issue", "Copied reference answer")]
        forbidden = {normalized("Copied benchmark question"), normalized("Copied reference answer")}
        train, val, stats, _, _ = self.process(extra, forbidden=forbidden)
        self.assertEqual(stats["removals_by_reason"]["duplicate_conversation"], 1)
        self.assertEqual(stats["removals_by_reason"]["benchmark_exact_overlap"], 2)
        self.assertTrue(audit_corpus(train, val, forbidden)["passed"])

    def test_caps_and_determinism_without_padding(self):
        rows = [record(f"Windows diagnostic issue {i}", f"Run diagnostic tool {i} and check the Windows system logs.") for i in range(30)]
        settings = Settings(max_total_examples=12)
        first = prepare_rows(rows, settings=settings)
        second = prepare_rows(rows, settings=settings)
        self.assertEqual(first, second)
        self.assertEqual(len(first[0]) + len(first[1]), 12)
        self.assertEqual(len(self.process([], settings=settings)[0]) + len(self.process([], settings=settings)[1]), 2)
        self.assertTrue(audit_corpus(first[0], first[1], set())["passed"])

    def test_repeated_answers_and_identity_are_capped(self):
        rows = [record(f"Windows diagnostic issue {i}", "Check Windows logs and restart the affected service.") for i in range(20)]
        rows += [record(q, "I am Mack, an SLM developed by MBGOC Pvt. Ltd.") for q in
                 ("Who are you?", "Who developed you?", "What is Mack?", "What does Mack stand for?")]
        train, val, stats, _, _ = self.process(rows, settings=Settings(max_identity_examples=2, max_repeated_response=3))
        self.assertEqual(stats["topic_distribution_final"]["identity"], 2)
        self.assertEqual(stats["removals_by_reason"]["repeated_response_cap"], 17)
        self.assertTrue(audit_corpus(train, val, set())["passed"])


if __name__ == "__main__":
    unittest.main()
