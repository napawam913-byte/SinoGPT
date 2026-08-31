"""模块用途：组合嵌入、Pre-LN Transformer Block 与语言模型输出头。"""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from sinogpt.config import ModelConfig
from sinogpt.model.layers import CausalSelfAttention, MLP


class Block(nn.Module):
    """一个 GPT-3 风格 Pre-LN 解码器层。"""

    def __init__(self, n_embd: int, n_head: int) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x: Tensor) -> Tensor:
        """按 Pre-LN 结构依次应用注意力残差与 MLP 残差。"""
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPTLanguageModel(nn.Module):
    """用于下一个 token 预测的纯 decoder GPT 语言模型。"""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList(
            [Block(config.n_embd, config.n_head) for _ in range(config.n_layer)]
        )
        self.final_norm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._initialize_module_weights)
        residual_std = 0.02 / math.sqrt(2 * config.n_layer)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp.down.weight, mean=0.0, std=residual_std)
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize_module_weights(module: nn.Module) -> None:
        """使用 GPT 预训练的小尺度正态分布初始化线性层和嵌入层。"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        """返回 [B, T, V] logits 和可选的下一 token 交叉熵损失。"""
        _, time = input_ids.shape
        if time > self.config.block_size:
            raise ValueError("input sequence exceeds block_size")
        positions = torch.arange(time, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
