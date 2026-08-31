"""模块用途：从预训练 checkpoint 进行 assistant-only SFT，并以验证损失选择最佳权重。"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch
from torch import Tensor

from sinogpt.config import ModelConfig, load_sft_config
from sinogpt.data.sft import load_prepared_sft_split
from sinogpt.model.gpt import GPTLanguageModel
from sinogpt.seed import seed_everything
from sinogpt.tokenizer import load_tokenizer
from sinogpt.training.checkpoint import load_checkpoint, save_checkpoint
from sinogpt.training.evaluation import evaluate_causal_lm
from sinogpt.training.sft import (
    BestCheckpointSelector,
    load_sft_base_model,
    restore_rng_state,
    sft_checkpoint_state,
)
from sinogpt.training.trainer import Trainer


def build_parser() -> argparse.ArgumentParser:
    """构建显式配置、恢复路径和设备选择的 SFT 训练命令。"""
    parser = argparse.ArgumentParser(description="从 SinoGPT 预训练 checkpoint 执行 assistant-only SFT")
    parser.add_argument("--config", required=True, type=Path, help="SFT YAML 配置路径")
    parser.add_argument("--resume", type=Path, help="从完成的 SFT epoch checkpoint 恢复")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser


def resolve_device(requested: str) -> torch.device:
    """将用户请求映射为实际可用的训练设备。"""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def require_bf16_support(use_bf16: bool, device: torch.device) -> None:
    """拒绝硬件不支持的 bf16 配置，避免隐式退化成不可记录的精度。"""
    if use_bf16 and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("this config requires CUDA bf16 support")


def iter_epoch_batches(
    input_ids: Tensor,
    labels: Tensor,
    *,
    effective_batch_size: int,
    seed: int,
    epoch: int,
):
    """按 epoch 种子打乱样本；末批循环补齐以满足梯度累积的固定批大小。"""
    if effective_batch_size < 1:
        raise ValueError("effective_batch_size must be positive")
    if input_ids.ndim != 2 or labels.shape != input_ids.shape or input_ids.size(0) < 1:
        raise ValueError("input_ids and labels must be equal nonempty rank-2 tensors")
    generator = torch.Generator(device="cpu").manual_seed(seed + epoch)
    permutation = torch.randperm(input_ids.size(0), generator=generator)
    offset = torch.arange(effective_batch_size)
    for start in range(0, input_ids.size(0), effective_batch_size):
        indices = permutation[(offset + start) % input_ids.size(0)]
        yield input_ids[indices], labels[indices]


def _load_resume_state(
    resume_path: Path,
    model_config: ModelConfig,
    device: torch.device,
    data_config: object,
    *,
    learning_rate: float,
    gradient_accumulation_steps: int,
    use_bf16: bool,
) -> tuple[GPTLanguageModel, Trainer, int, int, int, float]:
    """恢复一个完整 epoch 后的 SFT 状态，并校验数据和架构没有变化。"""
    state = load_checkpoint(resume_path)
    if state.get("kind") != "sft":
        raise ValueError("resume checkpoint is not an SFT checkpoint")
    if state.get("model_config") != asdict(model_config):
        raise ValueError("resume checkpoint model_config differs from SFT config")
    if state.get("sft_data_config") != asdict(data_config):
        raise ValueError("resume checkpoint data config differs from SFT config")
    model = GPTLanguageModel(model_config).to(device)
    model.load_state_dict(state["model"])
    trainer = Trainer(
        model,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_bf16=use_bf16,
    )
    trainer.load_state_dict(state["trainer"])
    restore_rng_state(state["rng_state"])
    return (
        model,
        trainer,
        int(state["completed_epoch"]),
        int(state["global_step"]),
        int(state["tokens_seen"]),
        float(state["best_validation_loss"]),
    )


def _validate_tokenizer_vocab(tokenizer_path: Path, vocab_size: int) -> None:
    """防止把不属于预训练底座的 tokenizer 用于 SFT。"""
    tokenizer = load_tokenizer(tokenizer_path)
    if tokenizer.get_vocab_size() != vocab_size:
        raise ValueError("tokenizer vocabulary size differs from model config")


def main() -> None:
    """训练至配置的 epoch 上限，并在每个 epoch 后记录验证指标和 checkpoint。"""
    args = build_parser().parse_args()
    model_config, data_config, train_config = load_sft_config(args.config)
    device = resolve_device(args.device)
    require_bf16_support(train_config.use_bf16, device)
    _validate_tokenizer_vocab(Path(data_config.tokenizer_path), model_config.vocab_size)
    seed_everything(train_config.seed)

    train_input_ids, train_labels = load_prepared_sft_split(
        Path(data_config.cache_dir), "train", model_config.block_size
    )
    validation_input_ids, validation_labels = load_prepared_sft_split(
        Path(data_config.cache_dir), "validation", model_config.block_size
    )
    effective_batch_size = train_config.batch_size * train_config.gradient_accumulation_steps

    if args.resume is None:
        model = load_sft_base_model(Path(train_config.base_checkpoint), model_config, device)
        trainer = Trainer(
            model,
            learning_rate=train_config.learning_rate,
            gradient_accumulation_steps=train_config.gradient_accumulation_steps,
            use_bf16=train_config.use_bf16,
        )
        completed_epoch = global_step = tokens_seen = 0
        best_validation_loss = float("inf")
    else:
        model, trainer, completed_epoch, global_step, tokens_seen, best_validation_loss = _load_resume_state(
            args.resume,
            model_config,
            device,
            data_config,
            learning_rate=train_config.learning_rate,
            gradient_accumulation_steps=train_config.gradient_accumulation_steps,
            use_bf16=train_config.use_bf16,
        )
        print(f"resumed_from={args.resume} completed_epoch={completed_epoch}")

    output_dir = Path(train_config.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    metrics_path = output_dir / "metrics.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    selector = BestCheckpointSelector(checkpoint_dir, best_validation_loss)

    for epoch in range(completed_epoch + 1, train_config.max_epochs + 1):
        epoch_loss = epoch_grad_norm = 0.0
        update_count = 0
        for batch_input_ids, batch_labels in iter_epoch_batches(
            train_input_ids,
            train_labels,
            effective_batch_size=effective_batch_size,
            seed=train_config.seed,
            epoch=epoch,
        ):
            metrics = trainer.train_step(batch_input_ids, batch_labels)
            global_step += 1
            update_count += 1
            epoch_loss += metrics["loss"]
            epoch_grad_norm += metrics["global_grad_norm"]
            tokens_seen += int((batch_labels != -100).sum())

        validation = evaluate_causal_lm(
            model,
            validation_input_ids,
            validation_labels,
            batch_size=effective_batch_size,
        )
        next_best_loss = min(selector.best_validation_loss, validation.loss)
        state = sft_checkpoint_state(
            model,
            trainer,
            model_config,
            data_config,
            train_config,
            completed_epoch=epoch,
            global_step=global_step,
            tokens_seen=tokens_seen,
            best_validation_loss=next_best_loss,
        )
        save_checkpoint(checkpoint_dir / f"epoch_{epoch:03d}.pt", state)
        is_best = selector.consider(validation.loss, state)
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": epoch_loss / update_count,
            "train_global_grad_norm": epoch_grad_norm / update_count,
            "validation_loss": validation.loss,
            "validation_perplexity": validation.perplexity,
            "validation_supervised_tokens": validation.supervised_tokens,
            "tokens_seen": tokens_seen,
            "is_best": is_best,
        }
        with metrics_path.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(json.dumps(record) + "\n")
        print(json.dumps(record))


if __name__ == "__main__":
    main()
