"""模块用途：统一控制训练随机性；不创建模型、不读写 checkpoint。"""

import random

import torch


def seed_everything(seed: int) -> None:
    """设置 Python 与 PyTorch CPU/GPU 的随机种子。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
