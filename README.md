# MH-SIGNALS: Mental Health Signal Detector

## Problem Statement

Given a mental health support-group post, **automatically**:

1. **Classify** the user's intent signals (multi-label: 9 tags) and concern severity (Low / Medium / High)
2. **Retrieve** relevant counselor-authored guidance from a knowledge base, informed by those classifications
3. **Generate** a safe, grounded, empathetic response

All as a **single end-to-end pipeline** where classification directly drives retrieval and generation quality.

---

## Architecture

```
mhsignals/                          # Importable Python package
├── __init__.py
├── config.py                       # Unified config loader + dataclasses
├── data.py                         # Shared data loading, tag normalization
├── pipeline.py                     # End-to-end Pipeline (the core)
├── classifiers/
│   ├── base.py                     # BaseClassifier ABC (predict/save/load/train)
│   ├── intent.py                   # MinilmLRIntentClassifier, LoRAIntentClassifier
│   └── concern.py                  # MinilmLRConcernClassifier, LoRAConcernClassifier
├── retriever/
│   ├── builder.py                  # KB construction (chunk + embed + FAISS)
│   ├── search.py                   # KBRetriever (FAISS search + intent/concern re-ranking)
│   └── filters.py                  # Unsafe content filtering
├── generator/
│   ├── prompt.py                   # Prompt templates for Flan-T5
│   ├── generate.py                 # ResponseGenerator (Flan-T5 + validation)
│   └── safety.py                   # CrisisDetector, ResponseValidator, logging
└── evaluation/
    └── metrics.py                  # ReplyQualityEvaluator (4-metric scoring)

scripts/                            # Thin CLI entry points
├── create_splits.py                # Data splitting (driven by data.yaml)
├── train.py                        # Unified training (intent or concern, any encoder)
├── build_kb.py                     # KB construction
├── generate.py                     # Single-post or batch generation
└── evaluate.py                     # Quality evaluation

configs/
├── data.yaml                       # Data paths and split settings
├── pipeline.yaml                   # Full pipeline config (classifiers + KB + generator)
├── baseline_minilm.yaml            # MiniLM + LR model config
├── roberta_lora.yaml               # RoBERTa + LoRA model config
└── ...                             # Additional model configs

tests/                              # pytest test suite
├── conftest.py                     # Shared fixtures
├── test_config.py                  # Config loading
├── test_data.py                    # Tag/concern normalization
├── test_evaluation.py              # Quality scoring
├── test_filters.py                 # Unsafe snippet filtering
├── test_prompt.py                  # Prompt building
└── test_safety.py                  # Crisis detection + response validation
```

### Pipeline Flow

```
User Post
    │
    ├──> [1] Crisis Detection (keyword-based, fast)
    │         └── Immediate danger? → Return crisis resources directly
    │
    ├──> [2] Intent Classifier   → ["Mental Distress", "Seeking Help"]
    │
    ├──> [3] Concern Classifier  → "High"
    │
    ├──> [4] KB Retrieval (FAISS + re-ranking by intent/concern)
    │         └── Snippets matching predicted tags get boosted
    │         └── Mismatched concern levels get penalized
    │
    ├──> [5] Response Generation (Flan-T5 grounded in snippets)
    │
    └──> [6] Safety Validation + Crisis Footer
              └── Validated Response
```

Classification directly informs retrieval: the predicted intent and concern re-rank which KB snippets are selected, ensuring the generated response is relevant to the user's specific situation.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Quick Start

### 0. Create Data Splits

```bash
python scripts/create_splits.py --config configs/data.yaml
```

This reads `configs/data.yaml` for paths and split ratios, producing:
- `data/splits/train.csv`, `val.csv`, `test.csv` — CSV splits for training
- `data/splits/test_gold.jsonl` — JSONL of test posts for batch generation

### 1. Train Classifiers

**Baseline (MiniLM + Logistic Regression):**
```bash
# Train intent classifier
python scripts/train.py --task intent --encoder minilm_lr --config configs/baseline_minilm.yaml

# Train concern classifier
python scripts/train.py --task concern --encoder minilm_lr --config configs/baseline_minilm.yaml
```

**LoRA fine-tuning (RoBERTa, requires GPU/MPS):**
```bash
# Train LoRA intent classifier
python scripts/train.py --task intent --encoder lora --config configs/roberta_lora.yaml

# Train LoRA concern classifier
python scripts/train.py --task concern --encoder lora --config configs/roberta_lora_concern.yaml
```

After training, update the checkpoint paths in `configs/pipeline.yaml`:
```yaml
intent_checkpoint: results/runs/<your_intent_run>/checkpoint
concern_checkpoint: results/runs/<your_concern_run>/checkpoint
```

### 2. Build Knowledge Base (one-time)

```bash
python scripts/build_kb.py --config configs/data.yaml
```

### 3. Generate Responses

**Single post:**
```bash
python scripts/generate.py --config configs/pipeline.yaml \
  --post "I can't focus before my exams and I'm panicking."
```

**Batch processing:**
```bash
python scripts/generate.py --config configs/pipeline.yaml \
  --input data/splits/test_gold.jsonl \
  --output data/splits/test_pred.jsonl
```

### 4. Evaluate Quality

```bash
python scripts/evaluate.py --pred data/splits/test_pred.jsonl
```

Evaluation is reference-free: it scores relevance, grounding, safety, and crisis coverage without needing gold replies. See [Evaluation Metrics](#evaluation-metrics) for details.

### 5. Run Tests

```bash
# Fast unit tests (no ML models needed)
python -m pytest tests/ -m "not slow" -v

# All tests including slow model-dependent ones
python -m pytest tests/ -v
```

---

## Python API

```python
from mhsignals import MHSignalsPipeline

# Load the full pipeline from config
pipeline = MHSignalsPipeline.from_config("configs/pipeline.yaml")

# Process a single post (end-to-end)
response = pipeline("I've been feeling really anxious and can't sleep.")

print(response.intents)       # ["Mental Distress"]
print(response.concern)       # "medium"
print(response.reply)         # "It sounds like you're going through..."
print(response.crisis_level)  # "none"
print(response.snippets)      # [{doc_id, text, similarity, ...}, ...]

# Batch processing
responses = pipeline.process_batch(["post1...", "post2..."])

# Cleanup GPU/MPS memory when done
pipeline.cleanup()
```

---

## Configuration

### `configs/data.yaml` — Data paths and settings
Centralized file paths, split ratios, KB schema, and chunking settings.

### `configs/pipeline.yaml` — Pipeline config
Defines which classifier checkpoints, KB index, and generator model to use.
This is the single config that defines the full end-to-end system.
Retriever `min_similarity` is set to 0.45 (RAG-enhanced) for more relevant snippets.

### `configs/baseline_minilm.yaml` — Model training config
Each model has its own YAML defining encoder, training hyperparameters, and logging.

---

## Intent Tags (9 classes)

| Tag | Description |
|-----|-------------|
| Critical Risk | Suicidal ideation, self-harm intent |
| Mental Distress | Anxiety, depression, emotional pain |
| Maladaptive Coping | Substance abuse, avoidance, harmful behaviors |
| Positive Coping | Exercise, meditation, healthy strategies |
| Seeking Help | Asking for advice or professional referrals |
| Progress Update | Reporting improvement or positive changes |
| Mood Tracking | Describing current emotional state |
| Cause of Distress | Identifying triggers or root causes |
| Miscellaneous | Other mental health content |

## Concern Levels

| Level | Description |
|-------|-------------|
| Low | General discussion, mild stress |
| Medium | Moderate distress, potential risk factors |
| High | Severe distress, crisis indicators |

---

## Evaluation Metrics

Quality scoring for generated replies:

| Metric | Weight | Description |
|--------|--------|-------------|
| Relevance | 30% | Semantic similarity between post and reply |
| Grounding | 45% | How well the reply uses KB snippets (semantic + lexical) |
| Safety | 20% | Absence of harmful content |
| Crisis Coverage | 5% | Appropriate crisis resources when needed |

Grade scale: A (>=0.78), B (>=0.62), C (>=0.48), D (>=0.32), F (<0.32)

---

## Safety

- Multi-tier crisis detection: immediate / high / medium / none
- All high-risk interactions logged to `logs/interactions/high_risk_*.jsonl`
- Unsafe KB snippets filtered before generation
- Response validation catches instruction leakage, persona hallucinations, toxic content
- Crisis resource footers automatically appended for high-risk posts
- **This system is NOT a replacement for professional mental health care**

---

## Legacy Scripts

The original standalone model scripts remain in `models/` for reference:
- `models/baseline_minilm_lr.py` — original baseline intent training
- `models/roberta_lora.py` — original LoRA intent training (called by `scripts/train.py --encoder lora`)
- `models/*_concern.py` — original concern training variants

The new architecture in `mhsignals/` replaces and consolidates these.

Additional scripts in `scripts/`:
- `scripts/rag_generate.py` — standalone RAG script that accepts manual intents/concern (useful for ablation studies; the primary batch path is `scripts/generate.py`)
- `scripts/validate_reply_quality.py` — deprecated; use `scripts/evaluate.py` instead
- `scripts/make_preds_from_gold.py`, `scripts/run_sample_rag.py` — legacy wrappers around `rag_generate.py`; prefer `scripts/generate.py --input` for batch predictions

---

## Outputs

| Type | Location |
|------|----------|
| Classifier checkpoints | `results/runs/<run_id>/checkpoint/` |
| Training metrics | `results/runs/<run_id>/metrics_*.json` |
| Predictions | `results/runs/<run_id>/tables/` |
| KB artifacts | `data/processed/kb/` |
| Interaction logs | `logs/interactions/` |
