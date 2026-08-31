"""模块用途：实现因果自注意力和 GELU 前馈网络两个 Transformer 子层。"""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """带下三角掩码的多头自注意力，禁止当前位置读取未来 token。"""

    def __init__(self, n_embd: int, n_head: int) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x: Tensor) -> Tensor:
        """计算 QKᵀ 缩放分数、因果 Softmax 权重及其对 V 的加权和。"""
        batch, time, channels = x.shape
        q, k, v = self.qkv(x).split(channels, dim=-1)
        q = q.view(batch, time, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, time, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, time, self.n_head, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.ones(time, time, device=x.device, dtype=torch.bool).tril()
        weights = F.softmax(scores.masked_fill(~causal_mask, float("-inf")), dim=-1)
        attended = weights @ v
        merged = attended.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.proj(merged)


class MLP(nn.Module):
    """GPT 的两层前馈网络：四倍扩展、GELU 激活、再投影。"""

    def __init__(self, n_embd: int) -> None:
        super().__init__()
        self.up = nn.Linear(n_embd, 4 * n_embd)
        self.down = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x: Tensor) -> Tensor:
        """执行 Linear → GELU → Linear。"""
        return self.down(F.gelu(self.up(x)))
