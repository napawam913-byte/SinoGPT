"""模块用途：以原子替换方式保存和读取可恢复训练状态。"""

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    """先保存临时文件，再原子替换目标文件，避免中断留下半个 checkpoint。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    """将 checkpoint 读到 CPU，交由调用者迁移模型和优化器状态。"""
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise ValueError("checkpoint root must be a dictionary")
    return state
