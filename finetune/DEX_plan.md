We need to replace the fine-tuning dataset used by the `it_support_llm_finetune` project with a cleaner IT troubleshooting dataset.

The new source dataset is:

```text
benjaminmacklin/IT_Support_V2
```

The assistant/model persona for this project is now:

```text
DEX = Diagnostic EXpert
```

DEX is our IT-support assistant.

The source dataset frequently refers to an assistant named:

```text
Mack
```

Useful Mack examples should generally be converted to DEX rather than discarded.

This task should:

1. Inspect and download the new dataset.
2. Build a robust preprocessing/filtering pipeline.
3. Convert useful Mack references to DEX.
4. Remove or rewrite source-company identity contamination.
5. Remove classification/junk/low-quality samples.
6. Deduplicate and select a high-quality SFT corpus.
7. Create train/validation datasets.
8. Validate/audit the final corpus.
9. Update the fine-tuning configuration to use the new dataset.
10. Once preprocessing is complete, update/refactor the fine-tuned benchmarking code if necessary so it can benchmark the future DEX adapter cleanly.
11. Add/update tests.

Do NOT start a full QLoRA training run automatically.

Do NOT run the full 902-example benchmark automatically.

Prepare everything needed so those can be launched afterward.

---

# 1. Why We Are Changing Dataset

The retired fine-tuning source produced overly generic support replies.

After training and inspecting its responses, we found that many assistant completions look like:

```text
"Thank you for reaching out..."

"We are pleased to assist..."

"We will provide detailed information..."

"We will send documentation in a follow-up email..."

"Please let us know if you need anything else."
```

Many answers are short customer-service acknowledgements or promises to provide help later rather than actual technical solutions.

This risks teaching the LoRA:

```text
polite support-email style
generic acknowledgements
future follow-up promises
non-answers
```

instead of the desired behavior:

```text
technical diagnosis
troubleshooting steps
concrete remediation
IT reasoning
useful concise answers
```

The new `benjaminmacklin/IT_Support_V2` dataset contains substantially more examples of the desired pattern:

```text
technical problem
→ diagnostic checks
→ concrete troubleshooting steps
→ possible resolution
```

However, the raw dataset also contains classification samples, Mack persona material, source-company references, synthetic noise, duplicates, and other undesirable examples.

Therefore:

```text
DO NOT TRAIN DIRECTLY ON THE RAW DATASET.
```

Build a curated dataset first.

---

# 2. Keep DEX Artifacts Separate

The older fine-tuning experiment has been retired. DEX uses its own processed
data and adapter directory:

```text
data/processed/finetune_v2/
models/qwen3.5-9b-it-support-dex-qlora/
```

The DEX source is `benjaminmacklin/IT_Support_V2`.

---

# 3. Dataset Loading and Reproducibility

Load:

```text
benjaminmacklin/IT_Support_V2
```

through Hugging Face `datasets`.

Before implementing filters:

1. Inspect the actual dataset schema.
2. Inspect all available splits/configurations.
3. Count raw records.
4. Inspect representative samples.
5. Inspect whether conversations are native arrays, JSON strings, or multiple formats.
6. Resolve the exact Hugging Face dataset revision/commit when possible.
7. Record the dataset revision in the manifest.

Do not assume the schema purely from examples we manually inspected.

Record:

```text
dataset name
dataset revision
raw record count
available splits
datasets library version
source format
```

Do not commit the entire Hugging Face cache into Git.

---

# 4. Canonical DEX Identity

Use the following project persona:

```text
Name:
DEX

Meaning:
Diagnostic EXpert

Role:
IT troubleshooting / technical-support assistant
```

Where simple identity training examples are retained, use a consistent identity such as:

```text
I am DEX, a Diagnostic EXpert SLM developed by Huy, an AI student.
My purpose is to provide practical and technically useful IT support
and troubleshooting assistance.
```

Do not create many different descriptions of who created DEX.

Keep identity wording reasonably consistent.

Add configurable constants rather than scattering hardcoded text:

```python
ASSISTANT_NAME = "DEX"
ASSISTANT_EXPANSION = "Diagnostic EXpert"
DEVELOPER_DESCRIPTION = "Huy, an AI student"
SOURCE_ASSISTANT_NAME = "Mack"
```

Use these only where useful.

The primary training objective remains IT troubleshooting, not persona memorization.

---

# 5. Mack → DEX Policy

Do NOT remove every sample containing `Mack`.

Many Mack samples contain valuable technical supervision.

Instead classify them conceptually.

## A. Conversational Mack reference

Example:

```text
Mack, my touchpad is not responding.
```

Convert to:

```text
DEX, my touchpad is not responding.
```

Keep the answer.

Likewise:

```text
Hey Mack, Outlook keeps crashing.
```

becomes:

```text
Hey DEX, Outlook keeps crashing.
```

---

## B. Technical Mack reference

Example:

```text
Explain how Mack detects data exfiltration through zero-day exploits.
```

Convert:

```text
Explain how DEX detects data exfiltration through zero-day exploits.
```

and correspondingly:

```text
Mack scans for...
```

becomes:

```text
DEX scans for...
```

However, do not invent new technical capabilities that are absent from the source.

The transformation should primarily be a controlled persona rename.

If the resulting statement would obviously give DEX unrealistic autonomous capabilities or becomes semantically nonsensical, prefer either:

```text
simple genericization
```

or:

```text
drop the sample
```

rather than inventing substantial new content.

---

## C. Simple Mack identity sample

Example:

```text
User:
What is Mack's primary objective?

Assistant:
I am Mack, an SLM developed by the Dev team at MBGOC Pvt. Ltd.
My primary objective is...
```

This may be converted to:

```text
User:
What is DEX's primary objective?

Assistant:
I am DEX, a Diagnostic EXpert SLM developed by Huy, an AI student.
My primary objective is to provide practical and technically useful
IT support and troubleshooting assistance.
```

This type of rewrite is acceptable because it is simple and deterministic.

Likewise simple questions such as:

```text
Who are you?
What is Mack?
What is Mack's role?
```

may be rewritten into a consistent DEX identity if this can be done safely.

Do not preserve references to the original company.

---

## D. Complicated source-specific Mack/company sample

If a record contains complicated information about:

```text
Mack's development history
MBGOC-specific internal architecture
the original development team
company-specific objectives
company systems
company proprietary facts
```

and rewriting it would require inventing substantial new content:

```text
DROP THE SAMPLE.
```

Do not spend excessive complexity trying to salvage every source-persona record.

Quality matters more than retaining all rows.

---

# 6. Company-Name Contamination

The final training dataset should contain no unwanted source-company identity.

Known example:

```text
MBGOC
MBGOC Pvt. Ltd.
```

Search the dataset for company/developer references discovered during inspection.

For samples containing source-company names:

```text
simple identity sample
→ convert to DEX / Huy identity if straightforward

useful technical sample where company reference is incidental
→ remove/replace only the source identity while keeping technical content

complex company-specific sample
→ drop
```

Do not blindly replace every company string with `Huy`.

For example, if the source text says:

```text
MBGOC's enterprise authentication infrastructure...
```

changing that to:

```text
Huy's enterprise authentication infrastructure...
```

would be nonsense.

Drop or genericize such samples instead.

After preprocessing, audit case-insensitively for:

```text
MBGOC
Mack
```

The expected result should be approximately:

```text
Mack occurrences = 0
MBGOC occurrences = 0
```

because useful Mack references should have become `DEX`.

Report any remaining occurrences.

---

# 7. Do Not Overtrain the Persona

DEX identity examples are allowed, but they should represent only a tiny portion of training data.

The dataset exists primarily to teach:

```text
IT diagnosis
troubleshooting
technical support
procedural reasoning
```

not:

```text
"What is your name?"
"Who developed you?"
"What does DEX stand for?"
```

If the raw dataset contains hundreds or thousands of near-identical Mack identity questions, deduplicate/aggressively cap them.

A small number of consistent DEX identity examples is enough.

Do not let identity samples materially dominate the corpus.

---

# 8. Remove Classification-Only Samples

We observed examples such as:

```json
[
  {
    "role": "user",
    "content": "Teams chat history not syncing across devices."
  },
  {
    "role": "assistant",
    "content": "teams_issue"
  }
]
```

These are intent-classification samples rather than generative IT support.

Remove them.

Other examples may look like:

```text
outlook_issue
network_problem
printer_issue
password_reset
vpn_error
```

Implement a conservative deterministic detector such as:

```python
is_intent_label_response(...)
```

Potential evidence:

```text
very short
snake_case
kebab-case
no sentence structure
frequently repeated as a category
```

Do NOT remove every short response.

Example:

```text
Restart the router and renew the DHCP lease.
```

is concise but valid.

Report:

```text
classification samples removed
top detected labels
```

---

# 9. Remove Artificial Domain-Restriction Examples

We observed examples like:

```text
User:
Write a script for a play.

Assistant:
I cannot generate creative content.
I only provide enterprise IT support.
```

Remove this type of training sample.

We want:

```text
Qwen + stronger IT-support behavior
```

not:

```text
Qwen artificially refusing everything outside IT
```

Remove examples whose main purpose is teaching DEX/Mack to say:

```text
I only answer IT questions.
I cannot perform non-IT tasks.
I only provide enterprise IT support.
```

Do not broadly remove legitimate safety refusals solely because they are refusals.

The target is artificial domain/persona restriction.

---

# 10. Normalize Conversation Format

Inspect actual records and normalize them into a canonical message format:

```python
[
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
]
```

Support valid multi-turn conversations.

If the raw dataset stores conversations as strings containing JSON, parse safely with:

```python
json.loads(...)
```

Do NOT use:

```python
eval(...)
```

Reject malformed structures.

Validate:

```text
known roles
string content
nonempty content
valid conversation
final target message is assistant
```

Do not silently invent missing turns.

---

# 11. Final SFT Schema

Keep compatibility with the project's existing completion-only training pipeline.

Single-turn example:

```json
{
  "prompt": [
    {
      "role": "user",
      "content": "DEX, SSH authentication fails with Permission denied (publickey)."
    }
  ],
  "completion": [
    {
      "role": "assistant",
      "content": "Verify that the correct private key is being used..."
    }
  ]
}
```

For a valid multi-turn conversation:

```text
all messages before final assistant
→ prompt

final assistant message
→ completion
```

Do not change the fundamental SFT objective.

---

# 12. Preserve Useful Technical Information

Do not destroy:

```text
commands
code
PowerShell
Bash
SQL
registry paths
file paths
error codes
stack traces
ports
configuration snippets
package names
API names
URLs used as technical references
```

Examples of valuable content include:

```text
DISM /Online /Cleanup-Image
Get-WindowsCapability
0x0000001A
/etc/resolv.conf
pip install
ssh -v
```

Whitespace normalization must not break code or commands.

---

# 13. Remove Malformed or Noisy Samples

We observed noise such as:

```text
Open Windows Update疑难解答.
```

The dataset is intended to teach English IT support.

Reject clearly corrupted/mixed-generation artifacts.

Be conservative.

Do not reject legitimate Unicode merely because it is non-ASCII.

Allowed examples include:

```text
filenames
symbols
code
technical notation
URLs
usernames
```

Potential suspicious pattern:

```text
significant unexpected CJK/non-English text
inside otherwise English generated content
```

Do not add a heavyweight language-detection system unless genuinely needed.

Track counts.

---

# 14. Filter Generic Non-Answers

We are replacing the old dataset specifically because of responses that provide no actual assistance.

Reject assistant responses that are predominantly:

```text
generic acknowledgement
future follow-up promise
"we will send documentation"
"we will contact you later"
pure politeness with no technical content
meaningless filler
```

Do not reject an answer merely because it begins politely.

Example:

```text
Sure. First verify the DNS configuration...
```

is good.

The filter should identify:

```text
non-answer
```

not:

```text
friendly tone
```

---

# 15. Technical Quality Filter

A retained assistant response should generally provide at least one of:

```text
diagnostic action
technical explanation
troubleshooting step
configuration advice
specific remediation
useful procedure
```

Do not require very long answers.

Examples such as:

```text
Update the touchpad driver, check Device Manager for errors,
and confirm that the touchpad is enabled.
```

are useful despite being short.

Inspect actual response-length distributions before choosing minimum thresholds.

Thresholds should be configurable.

Report before/after:

```text
characters
words/tokens
```

Do not throw away concise but useful answers.

---

# 16. Deduplicate After DEX Conversion

Perform deduplication AFTER persona replacement/sanitization.

This is important because:

```text
Mack, my touchpad is not responding.
```

may become identical to an existing:

```text
DEX, my touchpad is not responding.
```

Normalize comparison text using stable rules such as:

```text
trim
whitespace normalization
case normalization for hashes only
```

Do not lowercase the actual stored training content.

At minimum deduplicate:

```text
exact normalized conversations
exact normalized user prompts
repeated prompt/response pairs
```

If the same prompt has multiple candidate responses, choose deterministically.

Prefer:

```text
valid technical response
non-label response
more informative response within reasonable bounds
stable source ordering as final tie-breaker
```

---

# 17. Inspect Near-Duplicate Synthetic Families

The dataset is synthetic, so inspect whether there are large groups of near-identical samples.

Examples could resemble:

```text
touchpad not responding
touchpad stopped working
touchpad not working after restore
```

Some variation is useful.

Hundreds of effectively identical samples are not.

Do NOT implement O(N²) semantic comparison.

If necessary use a lightweight scalable method such as:

```text
normalized signatures
token fingerprints
repeated-response counts
MinHash/SimHash only if justified
```

Do not introduce major complexity or dependencies unnecessarily.

At minimum report:

```text
most repeated normalized prompts
most repeated assistant responses
```

---

# 18. Dataset Size

We do NOT need all ~100K+ raw samples.

Use quality before quantity.

Target:

```text
approximately 20,000–40,000 good examples
```

Recommended first cap:

```python
MAX_TOTAL_EXAMPLES = 30_000
```

This is a CAP, not a quota.

If:

```text
23,418
```

high-quality records survive:

```text
use 23,418
```

Do NOT loosen filters to reach 30k.

If:

```text
65,000
```

good candidates survive, select a deterministic and reasonably diverse 30k.

---

# 19. Topic Diversity

Inspect any category/intent/domain metadata available in the source.

Try to retain broad IT-support coverage such as:

```text
Windows
Linux
networking
hardware
drivers
email
Outlook
Teams
Office
authentication
VPN
databases
servers
cloud
security
software installation
system recovery
performance
permissions
developer tooling
enterprise applications
```

Avoid a corpus where one synthetic issue category dominates.

Do not build an elaborate topic classifier if useful metadata is unavailable.

Simple deterministic balancing is sufficient.

---

# 20. TechQA Must Stay Evaluation-Only

TechQA remains benchmark-only.

Current benchmark:

```text
data/processed/evaluation/techqa_benchmark.jsonl
```

Use it only for a lightweight exact-contamination firewall.

Compare normalized exact hashes against training prompts/reference answers where appropriate.

If an exact overlap exists:

```text
remove it
```

Do NOT use TechQA to:

```text
rank samples
choose filters
tune thresholds
select topics
perform semantic filtering
rewrite training data
```

No benchmark leakage.

Report exact overlap count.

---

# 21. Determinism

Use:

```python
RANDOM_SEED = 42
```

for selection and splitting.

Given:

```text
same source revision
same preprocessing configuration
same seed
```

the output should be identical.

Generate stable SHA256 hashes for final files.

---

# 22. Train / Validation Split

After all:

```text
parsing
DEX transformation
filtering
deduplication
quality selection
```

perform:

```text
90% train
10% validation
```

Do not split before deduplication.

Ensure normalized duplicate prompts cannot appear in both sets.

Record exact counts.

---

# 23. Output Files

Create:

```text
data/processed/finetune_v2/train.jsonl
data/processed/finetune_v2/validation.jsonl
data/processed/finetune_v2/manifest.json
data/processed/finetune_v2/filter_stats.json
data/processed/finetune_v2/audit_sample.jsonl
```

Optionally create a small rejected-sample audit file containing examples and rejection reasons.

Do not unnecessarily store every rejected source row.

---

# 24. Manifest

`manifest.json` should contain at least:

```text
source dataset
source revision
source raw count

assistant identity = DEX
assistant expansion = Diagnostic EXpert

seed

parsed record count
malformed count

classification records removed

Mack references encountered
Mack → DEX records converted

simple identity records rewritten
source-company records removed
source-company records rewritten
complex persona records dropped

domain-restriction records removed
language/noise records removed
generic non-answer records removed

duplicates removed
near-duplicate policy

TechQA exact-overlap count

candidate count after filtering
final selected count

train count
validation count

filter thresholds

train SHA256
validation SHA256

library versions
```

---

# 25. Filtering Statistics

Create useful statistics including:

```text
response lengths before filtering
response lengths after filtering

top repeated raw assistant responses
top repeated retained responses

detected classification labels

number of Mack occurrences before processing
number of Mack occurrences after processing

number of DEX occurrences after processing

number of MBGOC/company-name occurrences after processing

removal counts by reason

topic/category distribution if available
```

---

# 26. Final Corpus Audit

Before finishing, search final train + validation data case-insensitively for:

```text
Mack
MBGOC
```

Expected ideally:

```text
Mack = 0
MBGOC = 0
```

Search for suspicious label-like completions:

```text
*_issue
*_problem
*_error
```

Inspect remaining matches rather than blindly deleting them.

Search for old-style boilerplate:

```text
Thank you for reaching out
We will provide
follow-up email
```

Review whether those samples contain real troubleshooting.

Also inspect for malformed/mixed-language artifacts.

---

# 27. Human Audit Sample

Produce a deterministic:

```text
~100-example audit sample
```

at:

```text
data/processed/finetune_v2/audit_sample.jsonl
```

Include diverse samples.

Ideally include some records demonstrating:

```text
ordinary technical question
Mack → DEX conversion
DEX identity example
multi-turn example
Windows issue
Linux/network issue
security issue
hardware issue
database/server issue
```

This is for manual inspection before spending GPU time.

---

# 28. Tests

Add lightweight unit tests using synthetic fixtures.

Do NOT download the full Hugging Face dataset in normal tests.

Test at least:

```text
valid technical sample retained

classification label removed

"Mack, my touchpad..." → "DEX, my touchpad..."

"Hey Mack..." → "Hey DEX..."

technical "Mack detects X" → equivalent DEX sample

simple Mack identity sample → DEX identity

MBGOC identity sample → DEX/Huy identity when trivial

complex company-specific sample → removed

company names do not remain

domain-restriction sample removed

useful concise answer retained

generic non-answer removed

empty assistant response removed

malformed JSON rejected

technical commands preserved

noise handling

duplicates removed after DEX conversion

duplicate prompts cannot cross train/validation

MAX_TOTAL_EXAMPLES respected

fewer good examples are not padded with junk

TechQA exact overlap rejected

deterministic output
```

---

# 29. Inspect Before Implementing Aggressive Rules

Do not blindly implement every rule from our handful of manually viewed examples.

First measure:

```text
schema variants
Mack frequency
MBGOC/company frequency
classification-response frequency
response-length distribution
common assistant responses
duplicate frequency
multi-turn frequency
categories/topics
```

If a filter unexpectedly removes a huge fraction of data, investigate.

Example:

```text
a new rule removes 60–70% of all records
```

should trigger review rather than silently proceeding.

Keep filtering explainable.

Do NOT use another LLM to judge all 100K rows.

Prefer deterministic:

```text
rules
regex
normalization
deduplication
frequency analysis
simple quality checks
```

---

# 30. Update Fine-Tuning Configuration After Dataset Preparation

Once preprocessing/tests/audit have completed successfully, update the training configuration so the NEXT QLoRA run uses:

```text
data/processed/finetune_v2/train.jsonl
data/processed/finetune_v2/validation.jsonl
```

and writes to a NEW directory such as:

```text
models/qwen3.5-9b-it-support-dex-qlora/
```

Preserve the existing hyperparameters initially unless code compatibility requires otherwise.

Do not silently change:

```text
base model
LoRA rank
LoRA alpha
LoRA targets
learning rate
optimizer
quantization
epochs
batch size
gradient accumulation
```

The purpose of the next experiment is primarily to measure the effect of the improved dataset.

Use an explicit DEX experiment configuration.

---

# 31. DEX Training System Prompt

Inspect how the current training data is formatted.

If a system prompt/persona is used for SFT, make it consistent with DEX.

A reasonable minimal form is:

```text
You are DEX, a Diagnostic EXpert IT support assistant.
Provide concise, technically useful troubleshooting and diagnostic guidance.
```

Do not make the prompt excessively restrictive.

Do not say:

```text
You can only answer IT questions.
```

The training prompt should encourage useful IT behavior without artificially destroying general capabilities.

If the existing pipeline does not use a system prompt, do not redesign the entire training format merely to add one.

Keep changes minimal and experimentally defensible.

---

# 32. Benchmarking Code May Be Updated After Processing

After the dataset pipeline is complete and validated, you ARE allowed to modify/refactor the benchmarking code if necessary to support the future DEX fine-tuned adapter cleanly.

This permission is intentional: complete related setup work in one task instead of requiring another agent pass.

However, preserve experimental correctness.

The four-system experiment remains conceptually:

```text
Base Qwen3.5-9B

Fine-tuned DEX Qwen3.5-9B

Base Qwen3.5-9B + RAG

Fine-tuned DEX Qwen3.5-9B + RAG
```

Current TechQA benchmark rules must remain unchanged unless a change is necessary purely for model-loading/configuration support.

Preserve:

```text
TechQA 902 examples
fixed TechQA hash

same base model revision
same system prompt used for evaluation
same deterministic greedy generation
same token limits
same random seed
thinking disabled

same ROUGE-L
same BERTScore
same base-Qwen judge
same judge prompt/rubric
same abstention metric
same batching semantics
same checkpoint/resume behavior
```

Do NOT change metrics or benchmark prompts merely to make DEX perform better.

---

# 33. Benchmark Adapter Loading

The future fine-tuned benchmark should load:

```text
exact same BF16 base Qwen3.5-9B
+
DEX LoRA adapter
```

Do not benchmark the new adapter using 4-bit base inference merely because training used QLoRA.

The controlled comparison should remain:

```text
Base:
BF16 Qwen

Fine-tuned:
same BF16 Qwen + LoRA
```

so quantization is not a confounding variable.

The judge remains a completely untouched base Qwen model.

Do not accidentally judge DEX using DEX itself.

---

# 34. Refactor Benchmark Paths If Helpful

Keep adapter paths/configuration clear.

For example:

```python
ADAPTER_PATH = "models/qwen3.5-9b-it-support-dex-qlora"
```

or use a reusable fine-tuned benchmark configuration.

Do not delete old benchmark results.

Keep future DEX results in a clearly separate location, for example:

```text
results/finetuned_dex/
```

or another structure consistent with the existing repository.

---

# 35. Do Not Run Benchmark Before Adapter Exists

The new DEX adapter will not exist until training is performed.

Therefore benchmark-code preparation may be completed now, but:

```text
do not run the full fine-tuned TechQA benchmark
```

until the DEX adapter exists.

Unit tests and tiny loader/config smoke tests that do not require the future adapter are fine.

If some test necessarily requires the adapter, make it gracefully skip when the adapter directory does not yet exist.

---

# 36. Do Not Modify RAG Yet

The RAG phase is separate.

Do NOT implement or modify the StackOverflow/Chroma pipeline as part of this task unless an existing shared interface must be minimally adjusted for benchmark compatibility.

Future architecture:

```text
SFT
curated benjaminmacklin/IT_Support_V2
        ↓
teaches troubleshooting behavior and DEX persona


RAG
curated StackOverflow accepted-answer corpus
        ↓
provides external technical knowledge


TechQA
902 examples
        ↓
evaluation only
```

Keep these responsibilities separated.

---

# 37. Scope Summary

This task SHOULD complete:

```text
inspect IT_Support_V2
download/load dataset
build preprocessing
Mack → DEX conversion
company/persona handling
classification filtering
non-answer filtering
noise filtering
deduplication
quality filtering
diversity-aware selection
TechQA exact-overlap firewall
train/validation split
manifest
statistics
audit sample
unit tests

update training config for V2/DEX
prepare new adapter output path

update/refactor fine-tuned benchmark code if necessary
prepare future DEX benchmark configuration
preserve benchmark experimental controls
```

This task SHOULD NOT:

```text
start the expensive QLoRA training run
run the full 902-example DEX benchmark
change TechQA
tune anything against TechQA
implement the StackOverflow RAG corpus
```

---

# 38. Completion Report

When finished, provide a concise but complete report containing:

1. Exact Hugging Face source revision.
2. Raw source row count.
3. Actual schema discovered.
4. Valid parsed conversation count.
5. Classification examples removed.
6. Mack-containing records found.
7. Mack → DEX records successfully converted.
8. Simple identity records rewritten to DEX.
9. Source-company records rewritten.
10. Source-company/persona records dropped.
11. Remaining `Mack` occurrences.
12. Remaining `MBGOC`/source-company occurrences.
13. DEX occurrences in final corpus.
14. Domain-restriction examples removed.
15. Malformed/noisy examples removed.
16. Generic non-answers removed.
17. Exact duplicates removed.
18. Near-duplicate policy/results.
19. TechQA exact overlaps found.
20. Candidate count after filtering.
21. Final selected count.
22. Train count.
23. Validation count.
24. Train SHA256.
25. Validation SHA256.
26. Response-length statistics before/after.
27. Most common retained assistant responses.
28. Topic/category distribution if available.
29. Generated file paths.
30. Audit-sample path.
31. Unit-test results.
32. Fine-tuning config changes.
33. Future DEX adapter output directory.
34. Benchmark files changed/refactored.
35. Confirmation that benchmark metrics/prompts/evaluation behavior were not altered.
36. Any remaining concern that should be reviewed before launching QLoRA.

Stop after all preparation, configuration, benchmark plumbing, validation, and tests are complete.

Do not launch the expensive training run automatically.
