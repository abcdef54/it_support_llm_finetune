# Fine-Tuning Phase — Qwen3.5-9B IT Support QLoRA

## Objective

Fine-tune the exact same Qwen3.5-9B base checkpoint used by the baseline benchmark on the prepared IT-support training dataset using QLoRA.

The purpose of this phase is to teach the model better IT-support behavior, troubleshooting style, and response patterns.

This phase does **not** include RAG and does **not** use TechQA for training, tuning, prompt design, or model selection.

After this phase, the resulting fine-tuned model will later be benchmarked on the same fixed TechQA benchmark used for the base model.

## Read Before Starting

Before modifying code:

1. Inspect the current repository structure.
2. Read the existing data-preparation code and processed dataset schema.
3. Read the baseline benchmark configuration so the fine-tuning run starts from the exact same base model checkpoint/revision.
4. Inspect existing tests and shared utilities before adding new code.
5. Reuse existing project utilities when appropriate instead of duplicating them.

Do not rewrite working code unnecessarily.

## Experimental Isolation

Dataset roles are fixed.

### Fine-tuning data

Use only:

```text
data/processed/finetune/train.jsonl
data/processed/finetune/validation.jsonl
```

These were prepared from:

```text
Tobi-Bueck/customer-support-tickets
```

The validation split may be used for training-time evaluation and monitoring.

### TechQA

TechQA is **benchmark-only**.

Do not use TechQA for:

- training
- validation during fine-tuning
- hyperparameter tuning
- prompt tuning
- early stopping
- model selection
- RAG
- qualitative development examples

Do not read TechQA answers as part of this phase unless needed only to verify that no accidental dependency exists.

### RAG corpus

Do not use the RAG corpus during fine-tuning.

RAG is a later phase.

## Base Model

Use the exact same base model and pinned revision as the baseline benchmark.

Expected model:

```text
Qwen/Qwen3.5-9B
```

Do not silently change the model revision.

If the baseline benchmark already defines the model ID and revision in reusable configuration, prefer reusing the authoritative values rather than creating inconsistent duplicates.

## Fine-Tuning Method

Use **QLoRA**.

The base model should be loaded in 4-bit and remain frozen while LoRA adapters are trained.

Use these initial training settings unless the current codebase or installed library API requires a justified adjustment:

```text
Quantization:               4-bit
Quantization type:          NF4
Double/nested quantization: enabled
Compute dtype:              bfloat16

LoRA rank (r):              32
LoRA alpha:                 64
LoRA dropout:               0.05

Per-device train batch:     2
Gradient accumulation:      4
Effective batch size:       8
Learning rate:              1e-4
Max sequence length:        1024
Gradient checkpointing:     enabled
```

Choose the correct LoRA target modules for Qwen3.5 after inspecting the actual model architecture.

Do not guess target module names from another model family.

Keep all important hyperparameters centralized in configuration instead of scattering constants throughout the code.

Reasonable additional settings such as epoch count, scheduler, warmup, logging frequency, evaluation frequency, save frequency, optimizer, and checkpoint retention should also be explicit and reproducible.

## Training Format

Inspect the processed JSONL files before implementing the trainer.

The training pipeline must consume the existing processed schema correctly rather than changing the dataset to fit the training code.

For supervised fine-tuning, preserve the structured conversation semantics and apply the **Qwen3.5 chat template** through its tokenizer/processor.

Do not manually serialize examples using Llama, ChatML, or another model family's special tokens.

Verify several formatted examples before training.

Check that:

- user content appears in the user turn
- assistant content appears in the assistant turn
- special tokens are produced by the Qwen tokenizer/processor
- no old `[INST]`, Llama, or incompatible chat-template tokens appear
- truncation does not routinely remove the assistant answer
- labels/loss masking match the intended SFT objective

If completion-only or assistant-only loss is supported cleanly by the selected training stack, it may be used, but do not introduce fragile custom masking merely for complexity's sake. Whatever objective is chosen must be documented.

## Model Loading

Implement a clear QLoRA loading path using the currently installed Hugging Face ecosystem.

Expected components will likely include:

```text
transformers
peft
trl
bitsandbytes
accelerate
```

Use the APIs that actually exist in the installed versions.

Do not blindly copy old tutorials if their APIs are outdated.

The model must be prepared correctly for k-bit training before attaching LoRA adapters.

After LoRA is attached, report:

- total parameter count
- trainable parameter count
- percentage of parameters being trained

This should confirm that only a small adapter parameter subset is trainable.

## Training

Implement a modular training entry point under the existing `finetune/` area.

A sensible structure might be:

```text
finetune/
├── config.py
├── dataset.py
├── model.py
├── train.py
└── utils.py
```

This is only a suggestion.

Use fewer files if the code remains clean.

Do not create abstractions that do not solve a real problem.

The training pipeline should:

1. Load and validate train/validation data.
2. Load the pinned base Qwen3.5-9B model in 4-bit.
3. Configure QLoRA.
4. Apply the correct Qwen chat formatting.
5. Train on the training split.
6. Evaluate training loss on the validation split at reasonable intervals.
7. Save checkpoints as configured.
8. Save the final LoRA adapter and tokenizer/processor-related artifacts needed for reliable reload.
9. Save training configuration and training metrics.

Do not merge the adapter into the base model by default.

Keeping the PEFT adapter separate is preferred because it is smaller and preserves a clear relationship to the pinned base checkpoint.

## Output

Use a clear output location such as:

```text
models/qwen3.5-9b-it-support-qlora/
```

The final saved artifact should contain everything required to reload:

```text
base Qwen3.5-9B revision
+
trained LoRA adapter
```

Also save useful run metadata, including at least:

- base model ID
- base model revision
- training dataset paths
- dataset counts
- random seed
- quantization configuration
- LoRA configuration
- optimizer
- learning rate
- scheduler
- warmup
- batch size
- gradient accumulation
- effective batch size
- max sequence length
- epoch count / max steps
- validation strategy
- final train loss
- final validation loss if available
- library versions when practical

Do not copy large base-model weights into the repository.

## Checkpoint and Resume Support

The training run may be expensive.

Support resuming from a trainer checkpoint if the selected training framework provides this cleanly.

Do not build a custom checkpoint system if the framework already handles it.

A failed or interrupted rented-GPU session should not unnecessarily force training to restart from zero.

## Reproducibility

Set and record a fixed random seed.

Use deterministic behavior where practical, but do not enable extremely expensive deterministic CUDA behavior unless it is needed.

Record enough configuration that the run can be reproduced later.

Do not rely on undocumented defaults for important training parameters.

## Validation and Safety Checks

Before launching a full training run, perform lightweight checks.

At minimum verify:

- train file loads correctly
- validation file loads correctly
- required fields exist
- no empty prompts/completions
- train and validation do not overlap
- TechQA is not referenced by the training pipeline
- model ID/revision matches the baseline model
- Qwen chat template produces sensible text/tokens
- QLoRA target modules exist in the actual model
- only intended LoRA parameters are trainable
- a tiny batch can be collated correctly
- configuration can be serialized

If the current machine cannot safely load Qwen3.5-9B QLoRA, do not force a GPU smoke test locally.

Use lightweight mocks/config/schema tests instead.

## Post-Training Sanity Check

After a real training run on the RTX 5090, perform only a small qualitative sanity check using a few examples that are **not TechQA benchmark examples**.

Before inference, explicitly ensure:

```python
model.eval()
```

and use:

```python
torch.inference_mode()
```

If gradient checkpointing was enabled during training, disable it for inference when appropriate and restore cache usage if the model supports it.

This is important: `torch.no_grad()` or `torch.inference_mode()` does **not** replace `model.eval()`.

Do not accidentally evaluate with LoRA dropout still active.

The sanity check is only for confirming that the trained model loads and produces coherent output. It is not the official benchmark.

## Fine-Tuned Benchmark

Do **not** implement or run the full fine-tuned TechQA benchmark as part of this phase unless explicitly instructed later.

The next benchmark phase will evaluate the trained model on the exact same fixed TechQA benchmark and with the same generation/evaluation methodology used for the base model.

The fine-tuned benchmark should ultimately differ from the baseline only in the model being evaluated:

```text
Base:
Qwen3.5-9B

Fine-tuned:
Qwen3.5-9B + trained LoRA adapter
```

The TechQA benchmark, answer prompt, generation settings, BERTScore configuration, judge rubric, and aggregate metrics must remain comparable.

## Do Not Do Yet

Do not:

- implement RAG
- index documents
- embed the RAG corpus
- modify the TechQA benchmark
- train on TechQA
- optimize hyperparameters using TechQA
- begin the Base + RAG benchmark
- begin the Fine-tuned + RAG benchmark
- merge adapters unless there is a concrete reason
- add unnecessary frameworks or abstractions

Stay focused on producing a correct, reproducible QLoRA fine-tuning pipeline.

## Execution Policy

If the current machine is the local RTX 4050 environment, prepare the implementation and run only lightweight tests.

Do not launch the full Qwen3.5-9B QLoRA training run locally.

If the environment is the intended RTX 5090 32 GB machine and the user explicitly instructs you to launch training, run the real fine-tuning job using the finalized configuration.

Before launching an expensive training run, print or report the effective configuration so it can be reviewed.

## Completion

When the fine-tuning implementation is ready:

1. Run all appropriate lightweight tests.
2. Report files created or modified.
3. Summarize the final QLoRA configuration.
4. Report train/validation example counts.
5. Explain how the dataset is formatted for Qwen3.5.
6. Report the trainable parameter count/percentage if it could be verified.
7. Report any assumptions or unresolved issues.
8. Provide the exact command for launching or resuming training on the RTX 5090.
9. Stop.

Do not automatically continue into RAG or the next benchmark phase.
