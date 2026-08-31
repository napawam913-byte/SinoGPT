"""模块用途：验证 SFT 从兼容预训练权重初始化并仅按验证损失选 best checkpoint。"""

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from sinogpt.config import ModelConfig
from sinogpt.model.gpt import GPTLanguageModel
from sinogpt.training.checkpoint import load_checkpoint, save_checkpoint
from sinogpt.cli.train_sft import _load_resume_state, iter_epoch_batches
from sinogpt.config import SFTDataConfig
from sinogpt.training.sft import (
    BestCheckpointSelector,
    capture_rng_state,
    load_sft_base_model,
)
from sinogpt.training.trainer import Trainer


def model_config() -> ModelConfig:
    """构造 CPU 单元测试使用的小模型配置。"""
    return ModelConfig(vocab_size=32, n_layer=1, n_head=4, n_embd=16, block_size=8)


def write_pretrain_checkpoint(path: Path, config: ModelConfig) -> GPTLanguageModel:
    """写入只包含模型权重和结构的最小预训练 checkpoint。"""
    model = GPTLanguageModel(config)
    save_checkpoint(path, {"model": model.state_dict(), "model_config": asdict(config)})
    return model


def test_load_sft_base_model_preserves_compatible_pretraining_weights(tmp_path: Path) -> None:
    """SFT 必须从 pretrain 权重开始，而非重新随机初始化。"""
    config = model_config()
    expected = write_pretrain_checkpoint(tmp_path / "pretrain.pt", config)

    loaded = load_sft_base_model(tmp_path / "pretrain.pt", config, torch.device("cpu"))

    assert torch.equal(loaded.token_embedding.weight, expected.token_embedding.weight)


def test_load_sft_base_model_rejects_an_incompatible_checkpoint(tmp_path: Path) -> None:
    """词表或层数不同的 checkpoint 不能被错误地用于 SFT。"""
    write_pretrain_checkpoint(
        tmp_path / "pretrain.pt",
        ModelConfig(vocab_size=33, n_layer=1, n_head=1, n_embd=16, block_size=8),
    )

    with pytest.raises(ValueError, match="model_config"):
        load_sft_base_model(tmp_path / "pretrain.pt", model_config(), torch.device("cpu"))


def test_best_checkpoint_selector_writes_only_for_lower_validation_loss(tmp_path: Path) -> None:
    """测试集不参与选择；验证损失变差时不得覆盖 best.pt。"""
    selector = BestCheckpointSelector(tmp_path)

    assert selector.consider(2.0, {"global_step": 1})
    assert not selector.consider(2.1, {"global_step": 2})

    assert load_checkpoint(tmp_path / "best.pt")["global_step"] == 1


def test_iter_epoch_batches_is_seeded_and_keeps_fixed_effective_batch_size() -> None:
    """最后一个不足的 SFT batch 应循环补齐，而不是导致梯度累积失败。"""
    input_ids = torch.arange(10).reshape(5, 2)
    labels = input_ids.clone()

    first = list(iter_epoch_batches(input_ids, labels, effective_batch_size=4, seed=17, epoch=1))
    second = list(iter_epoch_batches(input_ids, labels, effective_batch_size=4, seed=17, epoch=1))

    assert len(first) == 2
    assert all(ids.shape == labels.shape == (4, 2) for ids, labels in first)
    assert all(torch.equal(left[0], right[0]) for left, right in zip(first, second, strict=True))


def test_resume_rebinds_optimizer_to_the_restored_model_parameters(tmp_path: Path) -> None:
    """恢复时优化器若仍指向临时模型，后续 SFT 会看似训练却不更新输出模型。"""
    config = model_config()
    data_config = SFTDataConfig("train", "validation", "test", "tokenizer", "cache")
    original = GPTLanguageModel(config)
    original_trainer = Trainer(original, learning_rate=1e-3)
    state = {
        "kind": "sft",
        "model": original.state_dict(),
        "trainer": original_trainer.state_dict(),
        "model_config": asdict(config),
        "sft_data_config": asdict(data_config),
        "completed_epoch": 1,
        "global_step": 3,
        "tokens_seen": 10,
        "best_validation_loss": 2.0,
        "rng_state": capture_rng_state(),
    }
    checkpoint_path = tmp_path / "resume.pt"
    save_checkpoint(checkpoint_path, state)
    restored, restored_trainer, *_ = _load_resume_state(
        checkpoint_path,
        config,
        torch.device("cpu"),
        data_config,
        learning_rate=1e-3,
        gradient_accumulation_steps=1,
        use_bf16=False,
    )

    assert restored_trainer.optimizer.param_groups[0]["params"][0] is next(restored.parameters())
