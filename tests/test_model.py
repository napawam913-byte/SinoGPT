"""模块用途：验证 GPT 解码器的形状、因果性与梯度传播。"""

import torch

from sinogpt.config import ModelConfig
from sinogpt.model.gpt import GPTLanguageModel


def build_model() -> GPTLanguageModel:
    """构造便于 CPU 单元测试的小型解码器。"""
    return GPTLanguageModel(ModelConfig(vocab_size=32, n_layer=2, n_head=4, n_embd=16, block_size=8))


def test_logits_and_loss_shape() -> None:
    """模型应输出每个位置的词表 logits 与标量交叉熵。"""
    logits, loss = build_model()(torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]]))
    assert logits.shape == (1, 3, 32)
    assert loss is not None and loss.ndim == 0


def test_future_tokens_do_not_change_earlier_logits() -> None:
    """改变未来 token 不得改变较早位置的 logits。"""
    model = build_model().eval()
    with torch.no_grad():
        first, _ = model(torch.tensor([[1, 2, 3, 4]]))
        second, _ = model(torch.tensor([[1, 2, 9, 9]]))
    assert torch.allclose(first[:, :2], second[:, :2], atol=1e-5)


def test_backward_populates_finite_gelu_gradient() -> None:
    """交叉熵反向传播应到达 GELU 前线性层，且梯度为有限值。"""
    model = build_model()
    _, loss = model(torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]]))
    assert loss is not None
    loss.backward()
    gradient = model.blocks[0].mlp.up.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
