"""模块用途：验证聊天模板编码和 assistant 自回归生成的边界。"""

from types import SimpleNamespace

import torch
from torch import Tensor, nn

from sinogpt.inference import build_chat_prompt, generate_assistant_ids
from sinogpt.tokenizer import load_tokenizer, train_bpe


class EosModel(nn.Module):
    """始终让 EOS 概率最大的最小生成模型。"""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(block_size=8)

    def forward(self, input_ids: Tensor) -> tuple[Tensor, None]:
        logits = torch.tensor([0.0, 0.0, 4.0], device=input_ids.device)
        return logits.expand(*input_ids.shape, 3) + self.anchor, None


def build_tokenizer(tmp_path):
    """创建保留聊天控制 token 的最小 BPE tokenizer。"""
    path = tmp_path / "tokenizer.json"
    train_bpe(["系统 什么是梯度下降"], vocab_size=64, output_path=path)
    return load_tokenizer(path)


def test_build_chat_prompt_places_assistant_marker_last(tmp_path) -> None:
    """推理提示词应在最后放置 assistant 标记，供模型开始预测回答。"""
    tokenizer = build_tokenizer(tmp_path)

    prompt_ids = build_chat_prompt("系统", "什么是梯度下降？", tokenizer)

    assert prompt_ids[-1] == tokenizer.token_to_id("<|assistant|>")
    assert tokenizer.token_to_id("<|system|>") in prompt_ids
    assert tokenizer.token_to_id("<|user|>") in prompt_ids


def test_generate_assistant_ids_stops_when_eos_is_sampled() -> None:
    """模型预测结束符后，生成不能继续填充无意义 token。"""
    generated = generate_assistant_ids(
        EosModel(),
        [0, 1],
        eos_id=2,
        max_new_tokens=8,
        temperature=1.0,
        top_k=0,
    )

    assert generated == [2]
