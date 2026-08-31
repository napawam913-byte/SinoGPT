"""模块用途：验证 token 流可被无重排地切成因果训练样本。"""

import torch

from sinogpt.data.prepare import pack_training_sequences


def test_pack_training_sequences_keeps_next_token_target_boundary() -> None:
    """每个样本须有 block_size 个输入和一个右移标签 token。"""
    sequences = pack_training_sequences([1, 2, 3, 4, 5], block_size=2)
    assert torch.equal(sequences, torch.tensor([[1, 2, 3], [3, 4, 5]]))
