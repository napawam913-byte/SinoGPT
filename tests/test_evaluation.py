"""模块用途：验证 SFT 验证/测试指标只统计非忽略标签。"""

import math

import pytest
import torch
from torch import Tensor, nn

from sinogpt.training.evaluation import evaluate_causal_lm


class FixedLogitModel(nn.Module):
    """用固定 logits 构造可预测的交叉熵评估夹具。"""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, input_ids: Tensor) -> tuple[Tensor, None]:
        logits = torch.tensor([0.0, 2.0, -1.0], device=input_ids.device)
        return logits.expand(*input_ids.shape, 3) + self.anchor, None


def test_evaluate_causal_lm_counts_only_nonignored_labels_and_restores_training_state() -> None:
    """`-100` 不能影响损失/计数，评估后模型应回到调用前的训练状态。"""
    model = FixedLogitModel().train()

    result = evaluate_causal_lm(
        model,
        torch.tensor([[0, 1], [0, 1]]),
        torch.tensor([[1, -100], [1, 2]]),
        batch_size=1,
    )

    assert result.supervised_tokens == 3
    assert result.loss > 0.0
    assert result.perplexity == pytest.approx(math.exp(result.loss))
    assert model.training


def test_evaluate_causal_lm_rejects_a_split_without_supervised_labels() -> None:
    """没有 assistant 标签的 split 不能被误报为低损失。"""
    with pytest.raises(ValueError, match="no supervised tokens"):
        evaluate_causal_lm(
            FixedLogitModel(),
            torch.tensor([[0, 1]]),
            torch.tensor([[-100, -100]]),
            batch_size=1,
        )
