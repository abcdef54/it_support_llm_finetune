# DEX — IT Support Assistant

DEX (Diagnostic EXpert) is a terminal-based IT support assistant built from
Qwen3.5-9B and a fine-tuned QLoRA adapter. It can answer technical support questions directly or use retrieval-augmented generation (RAG) to bring relevant, verified technical documentation from Stack Overflow and IBM TechQA into the conversation.

## Architecture

- **Answer model:** Qwen3.5-9B with the DEX IT-support QLoRA adapter.
- **Optional RAG:** BGE embeddings (`BAAI/bge-base-en-v1.5`) search a persistent Chroma vector index of accepted technical answers and official IBM documentation. Relevant passages are supplied to DEX as context.
- **Interface:** `main.py` provides an interactive terminal or one-shot command-line questions.

---

## Benchmarking & Results

DEX was evaluated against **1,000 unseen, held-out general IT support questions** (`data/processed/benchmark/`):

- **Specialized IT Support:** Fine-tuned Qwen3.5-9B on 27,000 IT-support conversations using QLoRA to specialize the model for practical troubleshooting and technical support.
- **Lexical and Semantic Gains:** Improved performance on 1,000 unseen IT-support questions from **0.076 to 0.461 ROUGE-L F1** and from **0.477 to 0.744 BERTScore F1** over the base model.
- **LLM Judge Evaluation:** Achieved **77.95%** on an LLM-based answer-quality evaluation of DEX responses.
- **Knowledge Augmentation:** Built a ChromaDB RAG pipeline using Stack Overflow and IBM TechQA knowledge to augment DEX with external technical information at inference time.

### Benchmark Results Summary

| Model | ROUGE-L F1 | BERTScore F1 | LLM Judge Quality |
| :--- | :---: | :---: | :---: |
| **Qwen3.5-9B (Base)** | 0.076 | 0.477 | - |
| **DEX (Fine-Tuned QLoRA)** | **0.461** | **0.744** | **77.95%** |

### Running Benchmarks

```bash
# Evaluate Base Qwen model
.venv/bin/python -m benchmark.baseline.run_benchmark

# Evaluate Fine-Tuned DEX model
.venv/bin/python -m benchmark.finetuned.run_benchmark
```

---

## Getting Started

### 1. Environment Setup

Run on a CUDA-capable machine with sufficient GPU memory for Qwen3.5-9B:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Model Adapter Weights

Download the pre-trained QLoRA adapter checkpoint:
**[Download DEX Adapter Checkpoint (Google Drive)](https://drive.google.com/file/d/11kNFX2W-P-VizW6mfMYsPovLMzNUfRpO/view?usp=sharing)**

Extract or place the checkpoint files into:
```text
models/qwen3.5-9b-it-support-dex-v2-qlora/
```
*(The base Qwen3.5-9B model downloads automatically from Hugging Face on first use).*

### 3. RAG Knowledge Base Setup

To use `--rag`, download the pre-processed RAG knowledge corpus:
**[Download `combined_corpus.jsonl` (Google Drive)](https://drive.google.com/file/d/1I-YpYIG69k_AzbpCm9bcZKV1VlkgINlm/view?usp=sharing)**

1. Place the downloaded file at:
   ```text
   data/processed/rag/combined_corpus.jsonl
   ```

2. Reconstruct the local Chroma vector database:
   ```bash
   .venv/bin/python -m rag.index --project-root . --device cuda --batch-size 32
   ```
   *(Use `--device cpu` if CUDA is not available).*

---

## Running DEX

### Interactive Terminal

```bash
# Standard DEX
.venv/bin/python main.py

# DEX with RAG enabled
.venv/bin/python main.py --rag --sources
```

Type `quit`, `exit`, or press `Ctrl-D` to leave interactive mode.

### Single Question

```bash
.venv/bin/python main.py --question "How do I check a Linux service?"
.venv/bin/python main.py --question "Which ITM version supports CANDLEDATA?" --rag --sources
```

RAG is off by default. `--sources` displays which retrieved passages were supplied to DEX.

---

## Re-Training the Model (Fine-Tuning)

The complete, curated dataset is included directly in this repository:
- **Training Set:** `data/processed/finetune_v2/train.jsonl` (27,000 examples)
- **Validation Set:** `data/processed/finetune_v2/validation.jsonl` (2,000 examples)

### 1. Pre-Training Check
Validate data integrity and configuration readiness without loading model weights:

```bash
.venv/bin/python -m finetune.train --check
```

### 2. Run Fine-Tuning
Train the 4-bit QLoRA adapter (~28 GiB VRAM recommended):

```bash
.venv/bin/python -m finetune.train --project-root .
```

To resume from an existing checkpoint:
```bash
.venv/bin/python -m finetune.train --resume-from-checkpoint <path_to_checkpoint>
```

---

## Demo

Without RAG:

```text
$ .venv/bin/python main.py
DEX terminal. Type 'quit' or 'exit' (or press Ctrl-D) to leave.
You> how to delete a file in linux
Use the rm command with appropriate permissions, e.g., rm -f filename.
You> how to install python numpy package?
Use pip to install numpy by running 'pip install numpy' in the command prompt or terminal.
You> quit
```

With RAG and sources:

```text
$ .venv/bin/python main.py --rag --sources
DEX terminal. Type 'quit' or 'exit' (or press Ctrl-D) to leave.
You> Which ITM version supports CANDLEDATA?
ITM 6.30.05.00 supports CANDLEDATA.

Sources ([n] = supplied to DEX; retrieved-only passages were not supplied):
  [1] techqa: Does Linux KVM monitoring agent support CANDLEDATA function? Does Linux KVM monitoring agent in ITM support CANDLEDATA function? (techqa:TRAIN_Q046)
  [2] techqa: Which version of TBSM support ITM 6.3? What is the TBSM version that is certified with ITM 6.3? (techqa:TRAIN_Q100)
  [3] techqa: Is ITCAM Data Collector for WebSphere 7.2.0.0.14 available? (techqa:TRAIN_Q538)
```
