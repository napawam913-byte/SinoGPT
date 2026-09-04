"""模块用途：验证聊天模板编码和 assistant 自回归生成的边界。"""

from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

import sinogpt.inference as inference
from sinogpt.inference import (
    build_chat_prompt,
    generate_assistant_ids,
    sample_next_token,
)
from sinogpt.tokenizer import load_tokenizer, train_bpe


class EosModel(nn.Module):
    """始终让 EOS 概率最大的最小生成模型。"""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(block_size=8)

    def forward(self, input_ids: Tensor) -> tuple[Tensor, None]:
        logits = torch.tensor([float("-inf"), float("-inf"), 0.0], device=input_ids.device)
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


def test_build_chat_prompt_includes_history_before_current_question(tmp_path) -> None:
    tokenizer = build_tokenizer(tmp_path)

    prompt_ids = build_chat_prompt(
        "系统",
        "新问题",
        tokenizer,
        history=[("旧问题", "旧回答")],
    )

    expected = tokenizer.encode(
        "<|system|>系统<eos><|user|>旧问题<eos><|assistant|>旧回答<eos>"
        "<|user|>新问题<eos><|assistant|>"
    ).ids
    assert prompt_ids == expected


def test_build_chat_prompt_rejects_empty_history_turn(tmp_path) -> None:
    tokenizer = build_tokenizer(tmp_path)

    with pytest.raises(ValueError, match="history turns must not be empty"):
        build_chat_prompt("系统", "问题", tokenizer, history=[(" ", "回答")])


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


def test_apply_repetition_penalty_reduces_seen_positive_and_negative_logits() -> None:
    """已生成 token 的正负 logits 应按标准重复惩罚方向调整。"""
    adjusted = inference.apply_repetition_penalty(
        torch.tensor([2.0, -2.0, 1.0]),
        generated_ids=[0, 1, 1],
        repetition_penalty=2.0,
    )

    assert torch.equal(adjusted, torch.tensor([1.0, -4.0, 1.0]))


def test_sample_next_token_top_p_excludes_lower_probability_tokens() -> None:
    """top-p 只保留累计概率覆盖阈值所需的最高概率候选。"""
    torch.manual_seed(17)

    next_id = sample_next_token(
        torch.tensor([3.0, 1.0, 0.0]),
        temperature=1.0,
        top_k=0,
        top_p=0.5,
    )

    assert next_id == 0
