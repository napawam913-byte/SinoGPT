"""模块用途：加载预训练底座并管理 SFT 最佳验证 checkpoint。"""

from dataclasses import asdict
import math
from pathlib import Path
import random
from typing import Any

import torch

from sinogpt.config import ModelConfig, SFTDataConfig, SFTTrainConfig
from sinogpt.model.gpt import GPTLanguageModel
from sinogpt.training.checkpoint import load_checkpoint, save_checkpoint
from sinogpt.training.trainer import Trainer


def load_sft_base_model(
    checkpoint_path: Path, expected_config: ModelConfig, device: torch.device
) -> GPTLanguageModel:
    """以预训练 checkpoint 初始化模型，并拒绝任何结构不兼容的权重。"""
    state = load_checkpoint(checkpoint_path)
    if state.get("model_config") != asdict(expected_config):
        raise ValueError("base checkpoint model_config differs from SFT config")
    model = GPTLanguageModel(expected_config).to(device)
    model.load_state_dict(state["model"])
    return model


class BestCheckpointSelector:
    """只在验证损失严格下降时原子写入 `best.pt`。"""

    def __init__(self, checkpoint_dir: Path, best_validation_loss: float = math.inf) -> None:
        if not math.isfinite(best_validation_loss) and best_validation_loss != math.inf:
            raise ValueError("best_validation_loss must be finite or infinity")
        self.checkpoint_dir = checkpoint_dir
        self.best_validation_loss = best_validation_loss

    def consider(self, validation_loss: float, state: dict[str, Any]) -> bool:
        """保存更优状态并返回是否改写了 `best.pt`。"""
        if not math.isfinite(validation_loss):
            raise ValueError("validation_loss must be finite")
        if validation_loss >= self.best_validation_loss:
            return False
        save_checkpoint(self.checkpoint_dir / "best.pt", state)
        self.best_validation_loss = validation_loss
        return True


def capture_rng_state() -> dict[str, Any]:
    """保存 Python、CPU 及可用 CUDA 的随机状态，以支持 epoch 边界恢复。"""
    state: dict[str, Any] = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """恢复 SFT checkpoint 中的随机状态。"""
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def sft_checkpoint_state(
    model: GPTLanguageModel,
    trainer: Trainer,
    model_config: ModelConfig,
    data_config: SFTDataConfig,
    train_config: SFTTrainConfig,
    *,
    completed_epoch: int,
    global_step: int,
    tokens_seen: int,
    best_validation_loss: float,
) -> dict[str, Any]:
    """收集 resume、最佳选择与论文复现实验所需的 SFT 状态。"""
    return {
        "kind": "sft",
        "model": model.state_dict(),
        "trainer": trainer.state_dict(),
        "model_config": asdict(model_config),
        "sft_data_config": asdict(data_config),
        "sft_train_config": asdict(train_config),
        "completed_epoch": completed_epoch,
        "global_step": global_step,
        "tokens_seen": tokens_seen,
        "best_validation_loss": best_validation_loss,
        "rng_state": capture_rng_state(),
    }
