"""模块用途：验证优化器反向更新和检查点可恢复性。"""

from pathlib import Path

import torch

from sinogpt.config import ModelConfig
from sinogpt.model.gpt import GPTLanguageModel
from sinogpt.training.checkpoint import load_checkpoint, save_checkpoint
from sinogpt.training.trainer import Trainer


def build_model() -> GPTLanguageModel:
    """构造用于 CPU 训练器测试的极小模型。"""
    return GPTLanguageModel(ModelConfig(vocab_size=32, n_layer=1, n_head=4, n_embd=16, block_size=8))


def test_train_step_returns_finite_loss_and_gradient_norm() -> None:
    """一个训练步应完成 AdamW 更新并返回可记录指标。"""
    trainer = Trainer(build_model(), learning_rate=1e-3)
    metrics = trainer.train_step(torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]]))
    assert metrics["loss"] > 0.0
    assert metrics["global_grad_norm"] >= 0.0


def test_checkpoint_round_trip_preserves_global_step(tmp_path: Path) -> None:
    """原子保存后应能读取训练游标和优化器状态。"""
    model = build_model()
    trainer = Trainer(model, learning_rate=1e-3)
    trainer.train_step(torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]]))
    checkpoint_path = tmp_path / "step_7.pt"
    save_checkpoint(
        checkpoint_path,
        {"global_step": 7, "model": model.state_dict(), "optimizer": trainer.optimizer.state_dict()},
    )
    restored = load_checkpoint(checkpoint_path)
    assert restored["global_step"] == 7
    assert restored["optimizer"]["state"]
