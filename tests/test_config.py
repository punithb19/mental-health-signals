"""
Tests for mhsignals.config — YAML loading and config dataclasses.
"""


import yaml

from mhsignals.config import (
    DataConfig,
    GeneratorConfig,
    KBConfig,
    PipelineConfig,
    RetrieverConfig,
    load_data_config,
    load_pipeline_config,
    load_yaml,
)


class TestLoadYaml:
    def test_load_valid_yaml(self, tmp_path):
        f = tmp_path / "test.yaml"
        f.write_text("key: value\nnested:\n  a: 1\n")
        result = load_yaml(str(f))
        assert result["key"] == "value"
        assert result["nested"]["a"] == 1

    def test_load_empty_yaml(self, tmp_path):
        f = tmp_path / "empty.yaml"
        f.write_text("")
        result = load_yaml(str(f))
        assert result == {}


class TestLoadDataConfig:
    def test_loads_with_defaults(self, tmp_path):
        f = tmp_path / "data.yaml"
        f.write_text(yaml.dump({
            "seed": 42,
            "paths": {"raw_csv": "data/raw.csv", "splits_dir": "data/splits"},
            "split": {"test_size": 0.2},
        }))
        cfg = load_data_config(str(f))
        assert isinstance(cfg, DataConfig)
        assert cfg.raw_csv == "data/raw.csv"
        assert cfg.test_size == 0.2
        assert cfg.seed == 42

    def test_kb_defaults(self, tmp_path):
        f = tmp_path / "data.yaml"
        f.write_text(yaml.dump({"seed": 42}))
        cfg = load_data_config(str(f))
        assert isinstance(cfg.kb, KBConfig)


class TestLoadPipelineConfig:
    def test_loads_with_defaults(self, tmp_path):
        f = tmp_path / "pipeline.yaml"
        f.write_text(yaml.dump({
            "seed": 42,
            "intent_checkpoint": "results/runs/test/checkpoint",
            "concern_checkpoint": "results/runs/test/checkpoint",
            "generator": {"model_name": "google/flan-t5-small", "device": "cpu"},
        }))
        cfg = load_pipeline_config(str(f))
        assert isinstance(cfg, PipelineConfig)
        assert cfg.intent_checkpoint == "results/runs/test/checkpoint"
        assert cfg.generator.model_name == "google/flan-t5-small"
        assert cfg.generator.device == "cpu"

    def test_retriever_defaults(self, tmp_path):
        f = tmp_path / "pipeline.yaml"
        f.write_text(yaml.dump({"seed": 42}))
        cfg = load_pipeline_config(str(f))
        assert isinstance(cfg.retriever, RetrieverConfig)
        assert cfg.retriever.topk == 50
        assert cfg.retriever.min_similarity == 0.45


class TestDataclassDefaults:
    def test_generator_config_defaults(self):
        gc = GeneratorConfig()
        assert gc.model_name == "google/flan-t5-base"
        assert gc.max_new_tokens == 400
        assert gc.do_sample is False

    def test_retriever_config_defaults(self):
        rc = RetrieverConfig()
        assert rc.encoder_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert rc.keep == 5

    def test_kb_config_defaults(self):
        kb = KBConfig()
        assert kb.target_tokens == 140
        assert kb.chunk_enabled is True
