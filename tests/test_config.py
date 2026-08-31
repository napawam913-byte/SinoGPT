"""模块用途：验证模型配置约束与随机种子可复现性。"""

import pytest
import torch

from sinogpt.config import ModelConfig
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
