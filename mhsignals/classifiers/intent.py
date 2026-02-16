"""
Intent classifiers: predict multi-label intent tags from a post.

Implementations:
  - MinilmLRIntentClassifier: SentenceTransformer embeddings + sklearn LogisticRegression
  - LoRAIntentClassifier:     HuggingFace transformer + LoRA adapter (multi-label head)
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sentence_transformers import SentenceTransformer

from .base import BaseClassifier
from ..data import (
    CANON_KEYS,
    ensure_dir,
    prob_to_tags,
    read_intent_split,
    set_seed,
)
from ..config import load_yaml


# ---------------------------------------------------------------------------
# Shared encoding helper
# ---------------------------------------------------------------------------

def _encode_texts(embedder: SentenceTransformer, texts: List[str],
                  batch_size: int = 128) -> np.ndarray:
    """Encode a list of texts into normalized embeddings."""
    parts = []
    for i in range(0, len(texts), batch_size):
        parts.append(
            embedder.encode(
                texts[i:i + batch_size],
                show_progress_bar=False,
                normalize_embeddings=True,
            )
        )
    return np.vstack(parts)


# ============================================================================
# MiniLM + Logistic Regression (baseline)
# ============================================================================

class MinilmLRIntentClassifier(BaseClassifier):
    """
    Multi-label intent classifier using SentenceTransformer (MiniLM)
    embeddings + OneVsRestClassifier(LogisticRegression).
    """

    def __init__(
        self,
        embedder: SentenceTransformer,
        clf: OneVsRestClassifier,
        label_names: List[str],
        threshold: float = 0.5,
        embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self._embedder = embedder
        self._clf = clf
        self._label_names = label_names
        self._threshold = threshold
        self._embedder_name = embedder_name

    # -- predict -----------------------------------------------------------

    def predict(self, text: str) -> List[str]:
        """Predict intent tags for a single post."""
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: List[str]) -> List[List[str]]:
        """Predict intent tags for a batch of posts."""
        X = _encode_texts(self._embedder, texts)
        P = self._clf.predict_proba(X)
        return [prob_to_tags(row, self._threshold, self._label_names) for row in P]

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Return raw probability matrix (n_samples x n_labels)."""
        X = _encode_texts(self._embedder, texts)
        return self._clf.predict_proba(X)

    # -- persistence --------------------------------------------------------

    def save(self, path: str) -> None:
        """Save classifier to directory: sklearn model + metadata."""
        out = ensure_dir(Path(path))
        joblib.dump(self._clf, out / "clf.joblib")
        with open(out / "meta.json", "w") as f:
            json.dump({
                "type": "MinilmLRIntentClassifier",
                "embedder_name": self._embedder_name,
                "label_names": self._label_names,
                "threshold": self._threshold,
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "MinilmLRIntentClassifier":
        """Load a saved MinilmLRIntentClassifier from a directory."""
        p = Path(path)
        with open(p / "meta.json") as f:
            meta = json.load(f)
        clf = joblib.load(p / "clf.joblib")
        embedder = SentenceTransformer(meta["embedder_name"])
        return cls(
            embedder=embedder,
            clf=clf,
            label_names=meta["label_names"],
            threshold=meta["threshold"],
            embedder_name=meta["embedder_name"],
        )

    # -- training -----------------------------------------------------------

    @classmethod
    def train(cls, config: Dict) -> "MinilmLRIntentClassifier":
        """
        Train from a config dict.

        Expected config keys:
          data.data_cfg:      path to data.yaml
          model.embedder:     SentenceTransformer model name
          training.max_iter, C, class_weight, threshold, seed, n_jobs
          logging.run_name, save_dir
        """
        from ..config import load_yaml as _load_yaml

        data_cfg = _load_yaml(config["data"]["data_cfg"])
        train_cfg = config.get("training", {})
        seed = int(train_cfg.get("seed", 42))
        set_seed(seed)

        splits_dir = Path(data_cfg["paths"]["splits_dir"])
        train_df = read_intent_split(splits_dir / "train.csv")
        val_df = read_intent_split(splits_dir / "val.csv")
        test_df = read_intent_split(splits_dir / "test.csv")

        mlb = MultiLabelBinarizer(classes=CANON_KEYS)
        Y_train = mlb.fit_transform(train_df["TagsList"])
        Y_val = mlb.transform(val_df["TagsList"])
        Y_test = mlb.transform(test_df["TagsList"])
        label_names = list(mlb.classes_)

        embedder_name = config["model"]["embedder"]
        embedder = SentenceTransformer(embedder_name)

        X_train = _encode_texts(embedder, train_df["Post"].tolist())
        X_val = _encode_texts(embedder, val_df["Post"].tolist())
        X_test = _encode_texts(embedder, test_df["Post"].tolist())

        lr = LogisticRegression(
            max_iter=int(train_cfg.get("max_iter", 200)),
            C=float(train_cfg.get("C", 1.0)),
            class_weight=train_cfg.get("class_weight", "balanced"),
            n_jobs=int(train_cfg.get("n_jobs", -1)),
            random_state=seed,
            solver="liblinear",
        )
        ovr = OneVsRestClassifier(lr, n_jobs=int(train_cfg.get("n_jobs", -1)))

        t0 = time.time()
        ovr.fit(X_train, Y_train)
        train_time = time.time() - t0

        threshold = float(train_cfg.get("threshold", 0.5))
        instance = cls(
            embedder=embedder,
            clf=ovr,
            label_names=label_names,
            threshold=threshold,
            embedder_name=embedder_name,
        )

        # Evaluate and save results
        run_name = config["logging"]["run_name"]
        save_root = Path(config["logging"]["save_dir"])
        save_dir = ensure_dir(
            save_root / f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        ensure_dir(save_dir / "tables")

        P_val = ovr.predict_proba(X_val)
        P_test = ovr.predict_proba(X_test)

        def _evaluate(y_true, y_prob, thr):
            y_pred = (y_prob >= thr).astype(int)
            ap_per = []
            for j in range(y_true.shape[1]):
                try:
                    ap_per.append(average_precision_score(y_true[:, j], y_prob[:, j]))
                except ValueError:
                    ap_per.append(0.0)
            per_label = {
                n: float(f1_score(y_true[:, j], y_pred[:, j], zero_division=0))
                for j, n in enumerate(label_names)
            }
            return {
                "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
                "pr_auc_macro": float(np.mean(ap_per)),
                "per_label_f1": per_label,
            }

        metrics_val = _evaluate(Y_val, P_val, threshold)
        metrics_test = _evaluate(Y_test, P_test, threshold)

        import pandas as pd, yaml as _yaml  # noqa: E401

        pd.DataFrame({
            "Post": val_df["Post"],
            "True": [", ".join(t) for t in val_df["TagsList"]],
            "Pred": [", ".join(prob_to_tags(p, threshold, label_names)) for p in P_val],
        }).to_csv(save_dir / "tables" / "val_predictions.csv", index=False)

        pd.DataFrame({
            "Post": test_df["Post"],
            "True": [", ".join(t) for t in test_df["TagsList"]],
            "Pred": [", ".join(prob_to_tags(p, threshold, label_names)) for p in P_test],
        }).to_csv(save_dir / "tables" / "test_predictions.csv", index=False)

        with open(save_dir / "metrics_val.json", "w") as f:
            json.dump(metrics_val, f, indent=2)
        with open(save_dir / "metrics_test.json", "w") as f:
            json.dump(metrics_test, f, indent=2)
        with open(save_dir / "label_names.json", "w") as f:
            json.dump(label_names, f, indent=2)
        with open(save_dir / "used_config.yaml", "w") as f:
            _yaml.safe_dump(config, f)

        # Save the model itself (NEW: this was missing before)
        instance.save(str(save_dir / "checkpoint"))

        print(f"[DONE] Saved run to: {save_dir}")
        print(f"VAL  -> {metrics_val}")
        print(f"TEST -> {metrics_test}")
        print(f"Train time: {train_time:.2f}s")

        return instance


# ============================================================================
# LoRA fine-tuned transformer (multi-label intent)
# ============================================================================

class LoRAIntentClassifier(BaseClassifier):
    """
    Multi-label intent classifier using a HuggingFace transformer
    fine-tuned with LoRA adapters.
    """

    def __init__(self, model, tokenizer, label_names, threshold=0.25,
                 max_length=512, device="cpu"):
        self._model = model
        self._tokenizer = tokenizer
        self._label_names = label_names
        self._threshold = threshold
        self._max_length = max_length
        self._device = device

    def predict(self, text: str) -> List[str]:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: List[str]) -> List[List[str]]:
        import torch

        self._model.eval()
        enc = self._tokenizer(
            texts, padding=True, truncation=True,
            max_length=self._max_length, return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            logits = self._model(**enc).logits
        probs = torch.sigmoid(logits).cpu().numpy()

        results = []
        for row in probs:
            results.append(prob_to_tags(row, self._threshold, self._label_names))
        return results

    def save(self, path: str) -> None:
        out = ensure_dir(Path(path))
        self._model.save_pretrained(str(out / "model"))
        self._tokenizer.save_pretrained(str(out / "tokenizer"))
        with open(out / "meta.json", "w") as f:
            json.dump({
                "type": "LoRAIntentClassifier",
                "label_names": self._label_names,
                "threshold": self._threshold,
                "max_length": self._max_length,
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "LoRAIntentClassifier":
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from peft import PeftModel

        p = Path(path)
        with open(p / "meta.json") as f:
            meta = json.load(f)

        tokenizer = AutoTokenizer.from_pretrained(str(p / "tokenizer"))
        model = AutoModelForSequenceClassification.from_pretrained(
            str(p / "model"),
            num_labels=len(meta["label_names"]),
            problem_type="multi_label_classification",
        )

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        model.to(device)
        model.eval()

        return cls(
            model=model,
            tokenizer=tokenizer,
            label_names=meta["label_names"],
            threshold=meta["threshold"],
            max_length=meta["max_length"],
            device=device,
        )

    @classmethod
    def train(cls, config: Dict) -> "LoRAIntentClassifier":
        """
        Train LoRA intent classifier. Delegates to HuggingFace Trainer.
        See models/roberta_lora.py for the full training logic.
        """
        raise NotImplementedError(
            "Use scripts/train.py --task intent --encoder lora --config <yaml> "
            "for full LoRA training. This method is a placeholder for the interface."
        )
