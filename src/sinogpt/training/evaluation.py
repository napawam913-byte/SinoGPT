"""模块用途：在不更新权重的情况下计算带 -100 掩码的因果语言模型指标。"""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class EvaluationResult:
    """按有效监督 token 加权的交叉熵与困惑度。"""

    loss: float
    perplexity: float
    supervised_tokens: int


def _validate_inputs(input_ids: Tensor, labels: Tensor, batch_size: int) -> None:
    """拒绝无法和语言模型 logits 一一对应的评估输入。"""
    if input_ids.ndim != 2 or labels.ndim != 2 or input_ids.shape != labels.shape:
        raise ValueError("input_ids and labels must be equal rank-2 tensors")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")


@torch.inference_mode()
def evaluate_causal_lm(
    model: nn.Module, input_ids: Tensor, labels: Tensor, batch_size: int
) -> EvaluationResult:
    """以 token 加权方式评估，仅统计 labels 中不等于 -100 的位置。"""
    _validate_inputs(input_ids, labels, batch_size)
    try:
        device = next(model.parameters()).device
    except StopIteration as error:
        raise ValueError("model must have at least one parameter") from error
    was_training = model.training
    total_nll = 0.0
    supervised_tokens = 0
    model.eval()
    try:
        for start in range(0, input_ids.size(0), batch_size):
            ids = input_ids[start : start + batch_size].to(device)
            targets = labels[start : start + batch_size].to(device)
            logits, _ = model(ids)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            total_nll += float(nll)
            supervised_tokens += int((targets != -100).sum())
    finally:
        model.train(was_training)
    if supervised_tokens == 0:
        raise ValueError("evaluation split has no supervised tokens")
    loss = total_nll / supervised_tokens
    return EvaluationResult(
        loss=loss,
        perplexity=math.exp(loss),
        supervised_tokens=supervised_tokens,
    )
