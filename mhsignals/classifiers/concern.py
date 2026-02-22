"""
Concern-level classifiers: predict Low / Medium / High from a post.

Implementations:
  - MinilmLRConcernClassifier:  SentenceTransformer + sklearn LogisticRegression (3-class)
  - LoRAConcernClassifier:      HuggingFace transformer + LoRA adapter (3-class head)
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer

from .base import BaseClassifier
from .intent import _encode_texts
from ..data import (
    CONCERN_LABELS,
    ensure_dir,
    normalize_concern,
    read_concern_split,
    set_seed,
)
from ..config import load_yaml


# ============================================================================
# MiniLM + Logistic Regression (baseline concern)
# ============================================================================

class MinilmLRConcernClassifier(BaseClassifier):
    """
    3-class concern classifier: Low / Medium / High.
    Uses SentenceTransformer embeddings + LogisticRegression.
    """

    def __init__(
        self,
        embedder: SentenceTransformer,
        clf: LogisticRegression,
        label_encoder: LabelEncoder,
        embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self._embedder = embedder
        self._clf = clf
        self._le = label_encoder
        self._embedder_name = embedder_name

    @property
    def label_names(self) -> List[str]:
        return list(self._le.classes_)

    # -- predict -----------------------------------------------------------

    def predict(self, text: str) -> str:
        """Predict concern level for a single post."""
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: List[str]) -> List[str]:
        """Predict concern level for a batch of posts."""
        X = _encode_texts(self._embedder, texts)
        y_pred = self._clf.predict(X)
        return [self._le.classes_[i] for i in y_pred]

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Return probability matrix (n_samples x 3)."""
        X = _encode_texts(self._embedder, texts)
        return self._clf.predict_proba(X)

    # -- persistence --------------------------------------------------------

    def save(self, path: str) -> None:
        out = ensure_dir(Path(path))
        joblib.dump(self._clf, out / "clf.joblib")
        joblib.dump(self._le, out / "label_encoder.joblib")
        with open(out / "meta.json", "w") as f:
            json.dump({
                "type": "MinilmLRConcernClassifier",
                "embedder_name": self._embedder_name,
                "label_names": list(self._le.classes_),
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "MinilmLRConcernClassifier":
        p = Path(path)
        with open(p / "meta.json") as f:
            meta = json.load(f)
        clf = joblib.load(p / "clf.joblib")
        le = joblib.load(p / "label_encoder.joblib")
        embedder = SentenceTransformer(meta["embedder_name"])
        return cls(
            embedder=embedder,
            clf=clf,
            label_encoder=le,
            embedder_name=meta["embedder_name"],
        )

    # -- training -----------------------------------------------------------

    @classmethod
    def train(cls, config: Dict) -> "MinilmLRConcernClassifier":
        """
        Train from a config dict.

        Expected config keys mirror baseline_minilm.yaml structure.
        """
        data_cfg = load_yaml(config["data"]["data_cfg"])
        train_cfg = config.get("training", {})
        seed = int(train_cfg.get("seed", 42))
        set_seed(seed)

        splits_dir = Path(data_cfg["paths"]["splits_dir"])
        train_df = read_concern_split(splits_dir / "train.csv")
        val_df = read_concern_split(splits_dir / "val.csv")
        test_df = read_concern_split(splits_dir / "test.csv")

        le = LabelEncoder()
        y_train = le.fit_transform(train_df["Concern_Level"])
        y_val = le.transform(val_df["Concern_Level"])
        y_test = le.transform(test_df["Concern_Level"])
        label_names = list(le.classes_)

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
            solver=train_cfg.get("solver", "lbfgs"),
            multi_class=train_cfg.get("multi_class", "auto"),
        )

        t0 = time.time()
        lr.fit(X_train, y_train)
        train_time = time.time() - t0

        instance = cls(
            embedder=embedder,
            clf=lr,
            label_encoder=le,
            embedder_name=embedder_name,
        )

        # Evaluate
        def _evaluate(y_true, y_pred, names):
            per_label = {}
            for i, name in enumerate(names):
                per_label[name] = float(f1_score(
                    (y_true == i).astype(int),
                    (y_pred == i).astype(int),
                    zero_division=0,
                ))
            return {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "per_label_f1": per_label,
                "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
                "label_order": names,
            }

        y_val_pred = lr.predict(X_val)
        y_test_pred = lr.predict(X_test)
        metrics_val = _evaluate(y_val, y_val_pred, label_names)
        metrics_test = _evaluate(y_test, y_test_pred, label_names)

        # Save results
        run_name = config["logging"]["run_name"]
        save_root = Path(config["logging"]["save_dir"])
        save_dir = ensure_dir(
            save_root / f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        ensure_dir(save_dir / "tables")

        import pandas as pd, yaml as _yaml  # noqa: E401

        pd.DataFrame({
            "Post": val_df["Post"],
            "True": [label_names[i] for i in y_val],
            "Pred": [label_names[i] for i in y_val_pred],
        }).to_csv(save_dir / "tables" / "val_predictions.csv", index=False)

        pd.DataFrame({
            "Post": test_df["Post"],
            "True": [label_names[i] for i in y_test],
            "Pred": [label_names[i] for i in y_test_pred],
        }).to_csv(save_dir / "tables" / "test_predictions.csv", index=False)

        with open(save_dir / "metrics_val.json", "w") as f:
            json.dump(metrics_val, f, indent=2)
        with open(save_dir / "metrics_test.json", "w") as f:
            json.dump(metrics_test, f, indent=2)
        with open(save_dir / "label_names.json", "w") as f:
            json.dump(label_names, f, indent=2)
        with open(save_dir / "used_config.yaml", "w") as f:
            _yaml.safe_dump(config, f)

        instance.save(str(save_dir / "checkpoint"))

        print(f"[DONE] Saved run to: {save_dir}")
        print(f"VAL  -> {metrics_val}")
        print(f"TEST -> {metrics_test}")
        print(f"Train time: {train_time:.2f}s")

        return instance


# ============================================================================
# LoRA fine-tuned transformer (concern)
# ============================================================================

class LoRAConcernClassifier(BaseClassifier):
    """
    3-class concern classifier using a HuggingFace transformer + LoRA.
    """

    def __init__(self, model, tokenizer, label_names, max_length=512, device="cpu"):
        self._model = model
        self._tokenizer = tokenizer
        self._label_names = label_names
        self._max_length = max_length
        self._device = device

    def predict(self, text: str) -> str:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: List[str], batch_size: int = 16) -> List[str]:
        import torch

        self._model.eval()
        all_preds = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self._tokenizer(
                batch, padding=True, truncation=True,
                max_length=self._max_length, return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                logits = self._model(**enc).logits
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        return [self._label_names[i] for i in all_preds]

    def save(self, path: str) -> None:
        out = ensure_dir(Path(path))
        self._model.save_pretrained(str(out / "model"))
        self._tokenizer.save_pretrained(str(out / "tokenizer"))
        with open(out / "meta.json", "w") as f:
            json.dump({
                "type": "LoRAConcernClassifier",
                "label_names": self._label_names,
                "max_length": self._max_length,
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "LoRAConcernClassifier":
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        p = Path(path)
        with open(p / "meta.json") as f:
            meta = json.load(f)

        tokenizer = AutoTokenizer.from_pretrained(str(p / "tokenizer"))

        model_dir = p / "model"
        adapter_config = model_dir / "adapter_config.json"

        if adapter_config.exists():
            # PEFT adapter checkpoint: load base model then apply adapter
            import json as _json
            with open(adapter_config) as f:
                acfg = _json.load(f)
            base_name = acfg.get("base_model_name_or_path", "distilroberta-base")
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_name,
                num_labels=len(meta["label_names"]),
            )
            from peft import PeftModel
            model = PeftModel.from_pretrained(base_model, str(model_dir))
            model = model.merge_and_unload()
        else:
            # Full model checkpoint
            model = AutoModelForSequenceClassification.from_pretrained(
                str(model_dir),
                num_labels=len(meta["label_names"]),
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
            max_length=meta["max_length"],
            device=device,
        )

    @classmethod
    def train(cls, config: Dict) -> "LoRAConcernClassifier":
        raise NotImplementedError(
            "Use scripts/train.py --task concern --encoder lora for full LoRA training."
        )
