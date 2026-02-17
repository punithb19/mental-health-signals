"""
Unified configuration loading and validation for MH-SIGNALS.

Loads data.yaml and pipeline.yaml into validated dataclass structures.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class KBConfig:
    """Knowledge base paths and chunking settings."""
    raw_dir: str = "data/raw/kb"
    processed_dir: str = "data/processed/kb"
    corpus_jsonl: str = "data/processed/kb/kb_snippets.jsonl"
    metadata_jsonl: str = "data/processed/kb/kb_meta.jsonl"
    embeddings_npy: str = "data/processed/kb/kb_embeddings.npy"
    faiss_index: str = "data/processed/kb/kb.faiss"
    chunk_enabled: bool = True
    target_tokens: int = 140
    min_chars: int = 20
    schema: Dict[str, str] = field(default_factory=lambda: {
        "input_col": "input",
        "output_col": "output",
        "intent_col": "Final_Intent_Tag",
        "concern_col": "Concern_Level",
        "id_col": "doc_id",
    })


@dataclass
class RetrieverConfig:
    """Retrieval settings (enhanced defaults for better relevance)."""
    encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    topk: int = 50
    keep: int = 5
    min_similarity: float = 0.45  # Stricter threshold for more relevant snippets


@dataclass
class GeneratorConfig:
    """Text generation settings."""
    model_name: str = "google/flan-t5-base"
    device: str = "cpu"
    max_new_tokens: int = 400
    do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.9
    max_prompt_chars: int = 2000


@dataclass
class PipelineConfig:
    """Full end-to-end pipeline configuration."""
    intent_checkpoint: str = ""
    concern_checkpoint: str = ""
    kb: KBConfig = field(default_factory=KBConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    log_dir: str = "logs/interactions"
    seed: int = 42


@dataclass
class DataConfig:
    """Data paths and split settings."""
    raw_csv: str = ""
    processed_csv: str = ""
    splits_dir: str = "data/splits"
    processed_dir: str = "data/processed"
    results_dir: str = "results"
    seed: int = 42
    test_size: float = 0.15
    val_size_from_train: float = 0.1765
    max_len: int = 256
    concern_map: Dict[str, int] = field(default_factory=lambda: {
        "low": 0, "medium": 1, "high": 2
    })
    kb: KBConfig = field(default_factory=KBConfig)


def load_yaml(path: str) -> dict:
    """Load a YAML file and return its contents as a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_data_config(path: str) -> DataConfig:
    """Load and validate data.yaml into a DataConfig."""
    raw = load_yaml(path)
    paths = raw.get("paths", {})
    split = raw.get("split", {})
    text = raw.get("text", {})
    labels = raw.get("labels", {})
    kb_raw = raw.get("kb", {})

    kb_chunking = kb_raw.get("chunking", {})
    kb = KBConfig(
        raw_dir=kb_raw.get("raw_dir", "data/raw/kb"),
        processed_dir=kb_raw.get("processed_dir", "data/processed/kb"),
        corpus_jsonl=kb_raw.get("corpus_jsonl", "data/processed/kb/kb_snippets.jsonl"),
        metadata_jsonl=kb_raw.get("metadata_jsonl", "data/processed/kb/kb_meta.jsonl"),
        embeddings_npy=kb_raw.get("embeddings_npy", "data/processed/kb/kb_embeddings.npy"),
        faiss_index=kb_raw.get("faiss_index", "data/processed/kb/kb.faiss"),
        chunk_enabled=bool(kb_chunking.get("enabled", True)),
        target_tokens=int(kb_chunking.get("target_tokens", 140)),
        min_chars=int(kb_chunking.get("min_chars", 20)),
        schema=kb_raw.get("schema", {}),
    )

    return DataConfig(
        raw_csv=paths.get("raw_csv", ""),
        processed_csv=paths.get("processed_csv", ""),
        splits_dir=paths.get("splits_dir", "data/splits"),
        processed_dir=paths.get("processed_dir", "data/processed"),
        results_dir=paths.get("results_dir", "results"),
        seed=int(raw.get("seed", 42)),
        test_size=float(split.get("test_size", 0.15)),
        val_size_from_train=float(split.get("val_size_from_train", 0.1765)),
        max_len=int(text.get("max_len", 256)),
        concern_map=labels.get("concern_map", {"low": 0, "medium": 1, "high": 2}),
        kb=kb,
    )


def load_pipeline_config(path: str) -> PipelineConfig:
    """Load and validate pipeline.yaml into a PipelineConfig."""
    raw = load_yaml(path)

    kb_raw = raw.get("kb", {})
    kb = KBConfig(
        metadata_jsonl=kb_raw.get("metadata_jsonl", "data/processed/kb/kb_meta.jsonl"),
        faiss_index=kb_raw.get("faiss_index", "data/processed/kb/kb.faiss"),
        corpus_jsonl=kb_raw.get("corpus_jsonl", "data/processed/kb/kb_snippets.jsonl"),
        embeddings_npy=kb_raw.get("embeddings_npy", "data/processed/kb/kb_embeddings.npy"),
        processed_dir=kb_raw.get("processed_dir", "data/processed/kb"),
    )

    ret_raw = raw.get("retriever", {})
    retriever = RetrieverConfig(
        encoder_model=ret_raw.get("encoder_model", "sentence-transformers/all-MiniLM-L6-v2"),
        topk=int(ret_raw.get("topk", 50)),
        keep=int(ret_raw.get("keep", 5)),
        min_similarity=float(ret_raw.get("min_similarity", 0.45)),
    )

    gen_raw = raw.get("generator", {})
    generator = GeneratorConfig(
        model_name=gen_raw.get("model_name", "google/flan-t5-base"),
        device=gen_raw.get("device", "cpu"),
        max_new_tokens=int(gen_raw.get("max_new_tokens", 400)),
        do_sample=bool(gen_raw.get("do_sample", False)),
        temperature=float(gen_raw.get("temperature", 0.7)),
        top_p=float(gen_raw.get("top_p", 0.9)),
        max_prompt_chars=int(gen_raw.get("max_prompt_chars", 2000)),
    )

    return PipelineConfig(
        intent_checkpoint=raw.get("intent_checkpoint", ""),
        concern_checkpoint=raw.get("concern_checkpoint", ""),
        kb=kb,
        retriever=retriever,
        generator=generator,
        log_dir=raw.get("log_dir", "logs/interactions"),
        seed=int(raw.get("seed", 42)),
    )
