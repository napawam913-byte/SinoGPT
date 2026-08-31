"""模块用途：封装单次语言模型反向传播、AdamW 更新和梯度裁剪。"""

from contextlib import nullcontext
from typing import Any

import torch
from torch import Tensor, nn


class Trainer:
    """以可选梯度累积训练 decoder-only 语言模型。"""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        use_bf16: bool = False,
    ) -> None:
        if gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        self.model = model
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.use_bf16 = use_bf16
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.0)

    @property
    def device(self) -> torch.device:
        """返回模型参数所在设备，确保数据与模型放在同一设备。"""
        return next(self.model.parameters()).device

    def train_step(self, input_ids: Tensor, targets: Tensor) -> dict[str, float]:
        """完成一次优化器更新，返回损失和裁剪前全局梯度范数。"""
        if input_ids.shape != targets.shape:
            raise ValueError("input_ids and targets must have the same shape")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        chunks = input_ids.tensor_split(self.gradient_accumulation_steps, dim=0)
        target_chunks = targets.tensor_split(self.gradient_accumulation_steps, dim=0)
        if any(chunk.size(0) == 0 for chunk in chunks):
            raise ValueError("batch is smaller than gradient_accumulation_steps")

        detached_loss = 0.0
        for ids_chunk, targets_chunk in zip(chunks, target_chunks, strict=True):
            autocast_context = (
                torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
                if self.use_bf16
                else nullcontext()
            )
            with autocast_context:
                _, loss = self.model(ids_chunk.to(self.device), targets_chunk.to(self.device))
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("non-finite causal language-model loss")
            (loss / len(chunks)).backward()
            detached_loss += float(loss.detach()) / len(chunks)

        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        return {"loss": detached_loss, "global_grad_norm": float(grad_norm)}

    def state_dict(self) -> dict[str, Any]:
        """导出优化器和调度器状态，以便训练可断点恢复。"""
        return {"optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """恢复优化器和调度器状态，不直接恢复模型权重。"""
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
