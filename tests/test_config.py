"""模块用途：验证模型配置约束与随机种子可复现性。"""

from pathlib import Path
import tomllib

import pytest
import torch

from sinogpt.config import ModelConfig, load_config, load_sft_config
from sinogpt.seed import seed_everything


def test_model_config_rejects_non_divisible_heads() -> None:
    """隐藏维度无法均分注意力头时必须拒绝配置。"""
    with pytest.raises(ValueError, match="n_embd must be divisible"):
        ModelConfig(vocab_size=32, n_layer=6, n_head=7, n_embd=384, block_size=512)


def test_seed_repeats_torch_values() -> None:
    """同一种子应生成完全一致的 PyTorch 随机序列。"""
    seed_everything(17)
    first = torch.rand(3)
    seed_everything(17)
    assert torch.equal(first, torch.rand(3))


def test_pilot_configs_share_model_and_data_but_extend_training() -> None:
    """冒烟阶段与完整阶段应从相同模型和数据配置连续恢复。"""
    stage_model, stage_data, stage_train = load_config("configs/tiny_30m_pilot_stage1.yaml")
    full_model, full_data, full_train = load_config("configs/tiny_30m_pilot.yaml")
    assert (stage_model, stage_data) == (full_model, full_data)
    assert (stage_train.max_steps, stage_train.checkpoint_every) == (100, 100)
    assert (full_train.max_steps, full_train.checkpoint_every) == (1250, 250)


def test_full_pass_config_continues_the_pilot_to_one_dataset_coverage() -> None:
    """完整覆盖运行必须保留同一模型与数据，并把总步数固定为 4,782。"""
    pilot_model, pilot_data, pilot_train = load_config("configs/tiny_30m_pilot.yaml")
    full_model, full_data, full_train = load_config("configs/tiny_30m_pilot_fullpass.yaml")

    assert (full_model, full_data) == (pilot_model, pilot_data)
    assert (full_train.max_steps, full_train.checkpoint_every) == (4782, 250)
    assert full_train.max_steps > pilot_train.max_steps


def test_data_extra_includes_zstandard_for_compressed_hf_shards() -> None:
    """数据导出环境必须能读取 Hugging Face 的 .zst 压缩分片。"""
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert "zstandard>=0.23" in project["project"]["optional-dependencies"]["data"]


def test_load_sft_config_requires_three_manifest_paths_and_a_base_checkpoint(tmp_path: Path) -> None:
    """SFT 必须显式引用三份数据、冻结 tokenizer 和预训练初始化 checkpoint。"""
    config_path = tmp_path / "sft.yaml"
    config_path.write_text(
        """
model:
  vocab_size: 32
  n_layer: 1
  n_head: 4
  n_embd: 16
  block_size: 8
data:
  train_manifest: train.jsonl
  validation_manifest: validation.jsonl
  test_manifest: test.jsonl
  tokenizer_path: tokenizer.json
  cache_dir: cache
train:
  seed: 17
  batch_size: 1
  gradient_accumulation_steps: 1
  learning_rate: 0.00005
  max_epochs: 3
  checkpoint_every_epoch: 1
  use_bf16: false
  base_checkpoint: pretrain.pt
  output_dir: artifacts/sft
""".strip(),
        encoding="utf-8",
    )

    model, data, train = load_sft_config(config_path)

    assert model.block_size == 8
    assert (data.train_manifest, data.validation_manifest, data.test_manifest) == (
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
    )
    assert (train.max_epochs, train.base_checkpoint) == (3, "pretrain.pt")
