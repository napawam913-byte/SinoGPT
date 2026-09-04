"""模块用途：验证无界面双模型聊天服务的公共行为与边界。"""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

from sinogpt.demo.chat_service import (
    ChatResponse,
    DualModelChatService,
    ModelBundle,
    SamplingSettings,
)
from sinogpt.config import ModelConfig
from sinogpt.model.gpt import GPTLanguageModel
from sinogpt.tokenizer import load_tokenizer, train_bpe
from sinogpt.training.checkpoint import save_checkpoint


class EosModel(nn.Module):
    """始终选择 EOS 的最小可训练状态模型。"""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(block_size=8)

    def forward(self, input_ids: Tensor) -> tuple[Tensor, None]:
        logits = torch.tensor([float("-inf"), float("-inf"), 0.0], device=input_ids.device)
        return logits.expand(*input_ids.shape, 3) + self.anchor, None


def build_service(tmp_path) -> tuple[DualModelChatService, EosModel]:
    """创建使用真实临时 tokenizer 的单模型服务。"""
    tokenizer_path = tmp_path / "tokenizer.json"
    train_bpe(["系统 你好"], vocab_size=64, output_path=tokenizer_path)
    tokenizer = load_tokenizer(tokenizer_path)
    model = EosModel()
    bundle = ModelBundle(
        label="v2 SFT（聊天候选）",
        checkpoint_path=Path("sft.pt"),
        model=model,
        tokenizer=tokenizer,
        eos_id=tokenizer.token_to_id("<eos>"),
    )
    return DualModelChatService({bundle.label: bundle}, "系统"), model


def test_respond_returns_empty_text_when_sft_generates_eos(tmp_path) -> None:
    service, _ = build_service(tmp_path)

    response = service.respond("v2 SFT（聊天候选）", "你好", [], SamplingSettings())

    assert isinstance(response, ChatResponse)
    assert response.model_label == "v2 SFT（聊天候选）"
    assert response.generated_tokens == 1
    assert response.stopped_by_eos is True
    assert response.text == ""


def test_rejects_empty_message_and_invalid_sampling_settings(tmp_path) -> None:
    service, _ = build_service(tmp_path)

    with pytest.raises(ValueError, match="message must not be empty"):
        service.respond("v2 SFT（聊天候选）", " ", [], SamplingSettings())
    with pytest.raises(ValueError, match="sampling settings are invalid"):
        SamplingSettings(top_p=0)


def test_rejects_unknown_model_label(tmp_path) -> None:
    service, _ = build_service(tmp_path)

    with pytest.raises(ValueError, match="unknown model label: missing"):
        service.respond("missing", "你好", [], SamplingSettings())


def test_respond_restores_model_training_mode(tmp_path) -> None:
    service, model = build_service(tmp_path)
    model.train()

    service.respond("v2 SFT（聊天候选）", "你好", [], SamplingSettings())

    assert model.training is True


def test_loader_falls_back_to_data_config_when_sft_config_is_unusable(tmp_path) -> None:
    """SFT 元数据不含 tokenizer 时，加载器应使用可用的基础数据配置。"""
    tokenizer_path = tmp_path / "tokenizer.json"
    train_bpe(["系统 你好"], vocab_size=64, output_path=tokenizer_path)
    tokenizer = load_tokenizer(tokenizer_path)
    config = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(),
        n_layer=1,
        n_head=1,
        n_embd=8,
        block_size=8,
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        {
            "model": GPTLanguageModel(config).state_dict(),
            "model_config": asdict(config),
            "sft_data_config": {},
            "data_config": {"tokenizer_path": str(tokenizer_path)},
        },
    )

    bundle = DualModelChatService._load_bundle(
        "test", checkpoint_path, torch.device("cpu")
    )

    assert bundle.tokenizer.get_vocab_size() == config.vocab_size
