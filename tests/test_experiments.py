"""模块用途：验证长训练必须携带完整不可变溯源信息。"""

import pytest

from sinogpt.experiments import validate_run_manifest


def test_run_manifest_requires_pinned_provenance_and_archive_uri() -> None:
    """空 tokenizer 哈希应在启动昂贵训练前被明确拒绝。"""
    raw = {
        "run_name": "zh_100",
        "config_path": "configs/ablation_125m.yaml",
        "tokenizer_sha256": "",
        "dataset_manifest_sha256": "",
        "seed": 17,
        "target_tokens": 300_000_000,
        "checkpoint_uri": "",
    }
    with pytest.raises(ValueError, match="tokenizer_sha256"):
        validate_run_manifest(raw)


def test_run_manifest_requires_positive_token_budget_and_seed() -> None:
    """空预算或非整数种子不能形成可解释的研究运行。"""
    raw = {
        "run_name": "sino_main_350m",
        "config_path": "configs/main_350m.yaml",
        "tokenizer_sha256": "a" * 64,
        "dataset_manifest_sha256": "b" * 64,
        "seed": "17",
        "target_tokens": 0,
        "checkpoint_uri": "s3://example/sino_main_350m",
    }
    with pytest.raises(ValueError, match="seed"):
        validate_run_manifest(raw)


def test_run_manifest_accepts_complete_provenance() -> None:
    """完整的哈希、预算和归档地址应被允许进入训练调度阶段。"""
    validate_run_manifest(
        {
            "run_name": "zh_70_en_20_code_10",
            "config_path": "configs/ablation_125m.yaml",
            "tokenizer_sha256": "a" * 64,
            "dataset_manifest_sha256": "b" * 64,
            "seed": 17,
            "target_tokens": 300_000_000,
            "checkpoint_uri": "s3://example/sinogpt/zh_70_en_20_code_10/",
        }
    )
