"""模块用途：封装只读双模型聊天推理，不负责界面、CLI 或 checkpoint 训练。"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from tokenizers import Tokenizer

from sinogpt.config import ModelConfig
from sinogpt.inference import build_chat_prompt, generate_assistant_ids
from sinogpt.model.gpt import GPTLanguageModel
from sinogpt.tokenizer import load_tokenizer
from sinogpt.training.checkpoint import load_checkpoint


@dataclass(frozen=True)
class SamplingSettings:
    """聊天采样的不可变参数。"""

    max_new_tokens: int = 128
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.9
    repetition_penalty: float = 1.1

    def __post_init__(self) -> None:
        if (
            self.max_new_tokens < 1
            or self.temperature <= 0
            or self.top_k < 0
            or not 0 < self.top_p <= 1
            or self.repetition_penalty < 1
        ):
            raise ValueError("sampling settings are invalid")


@dataclass(frozen=True)
class ModelBundle:
    """一个已加载模型及其冻结 tokenizer 的只读引用。"""

    label: str
    checkpoint_path: Path
    model: nn.Module
    tokenizer: Tokenizer
    eos_id: int


@dataclass(frozen=True)
class ChatResponse:
    """一次生成的展示与诊断结果。"""

    text: str
    model_label: str
    checkpoint_name: str
    generated_tokens: int
    stopped_by_eos: bool
    elapsed_seconds: float


class DualModelChatService:
    """在内存中选择预训练或 SFT 模型进行无状态聊天推理。"""

    def __init__(self, bundles: Mapping[str, ModelBundle], system_prompt: str) -> None:
        self._bundles = dict(bundles)
        self._system_prompt = system_prompt
        self.model_labels = tuple(self._bundles)

    def respond(
        self,
        model_label: str,
        message: str,
        history: list[tuple[str, str]] | None,
        settings: SamplingSettings,
    ) -> ChatResponse:
        """用所选模型生成一次回答，并恢复调用前的训练状态。"""
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must not be empty")
        if model_label not in self._bundles:
            raise ValueError(f"unknown model label: {model_label}")
        bundle = self._bundles[model_label]
        prompt_ids = build_chat_prompt(
            self._system_prompt, message, bundle.tokenizer, history
        )
        was_training = bundle.model.training
        started_at = perf_counter()
        try:
            generated_ids = generate_assistant_ids(
                bundle.model,
                prompt_ids,
                eos_id=bundle.eos_id,
                max_new_tokens=settings.max_new_tokens,
                temperature=settings.temperature,
                top_k=settings.top_k,
                top_p=settings.top_p,
                repetition_penalty=settings.repetition_penalty,
            )
        finally:
            bundle.model.train(was_training)
        stopped_by_eos = bool(generated_ids and generated_ids[-1] == bundle.eos_id)
        visible_ids = generated_ids[:-1] if stopped_by_eos else generated_ids
        return ChatResponse(
            text=bundle.tokenizer.decode(visible_ids),
            model_label=bundle.label,
            checkpoint_name=bundle.checkpoint_path.name,
            generated_tokens=len(generated_ids),
            stopped_by_eos=stopped_by_eos,
            elapsed_seconds=perf_counter() - started_at,
        )

    @classmethod
    def from_checkpoints(
        cls,
        pretrain_checkpoint: Path,
        sft_checkpoint: Path,
        device: torch.device,
    ) -> "DualModelChatService":
        """从两个兼容 checkpoint 建立固定标签的只读聊天服务。"""
        pretrain = cls._load_bundle(
            "v2 预训练（续写基线）", pretrain_checkpoint, device
        )
        sft = cls._load_bundle("v2 SFT（聊天候选）", sft_checkpoint, device)
        return cls({pretrain.label: pretrain, sft.label: sft}, "你是有帮助的中文助手。")

    @staticmethod
    def _load_bundle(label: str, checkpoint_path: Path, device: torch.device) -> ModelBundle:
        """校验一个 checkpoint 并仅加载一次其模型状态。"""
        state = load_checkpoint(checkpoint_path)
        model_config_raw = state.get("model_config")
        model_state = state.get("model")
        if not isinstance(model_config_raw, Mapping) or not isinstance(model_state, Mapping):
            raise ValueError("checkpoint model_config and model must be mappings")
        data_config = state.get("sft_data_config", state.get("data_config"))
        if not isinstance(data_config, Mapping) or not isinstance(
            data_config.get("tokenizer_path"), str
        ):
            raise ValueError("checkpoint data config must provide tokenizer_path")
        model_config = ModelConfig(**model_config_raw)
        tokenizer = load_tokenizer(Path(data_config["tokenizer_path"]))
        if tokenizer.get_vocab_size() != model_config.vocab_size:
            raise ValueError("tokenizer vocab size must match model_config")
        eos_id = tokenizer.token_to_id("<eos>")
        if eos_id is None:
            raise ValueError("tokenizer must contain <eos>")
        model = GPTLanguageModel(model_config)
        model.load_state_dict(model_state)
        model.to(device)
        return ModelBundle(label, checkpoint_path, model, tokenizer, eos_id)
