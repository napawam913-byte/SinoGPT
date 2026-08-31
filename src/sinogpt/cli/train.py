"""模块用途：编排可恢复的 GPT 因果语言模型训练，不实现网络数学。"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time
from typing import Any

import torch

from sinogpt.config import DataConfig, ModelConfig, TrainConfig, load_config
from sinogpt.data.manifest import validate_manifest
from sinogpt.model.gpt import GPTLanguageModel
from sinogpt.seed import seed_everything
from sinogpt.training.checkpoint import load_checkpoint, save_checkpoint
from sinogpt.training.trainer import Trainer


def build_parser() -> argparse.ArgumentParser:
    """构造训练 CLI；配置路径始终显式，恢复操作必须显式选择 checkpoint。"""
    parser = argparse.ArgumentParser(description="训练从零实现的 SinoGPT 因果语言模型")
    parser.add_argument("--config", required=True, type=Path, help="YAML 训练配置路径")
    parser.add_argument("--resume", type=Path, help="从指定 checkpoint 恢复训练")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="训练设备；auto 优先使用 CUDA",
    )
    return parser


def resolve_device(requested: str) -> torch.device:
    """根据用户显式选择或 CUDA 可用性确定训练设备。"""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def require_bf16_support(use_bf16: bool, device: torch.device) -> None:
    """拒绝配置要求 bf16、但硬件或设备无法提供 bf16 的运行。"""
    if use_bf16 and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("this config requires CUDA bf16 support")


def load_prepared_sequences(path: Path, block_size: int) -> torch.Tensor:
    """读取 `prepare_data` 的产物，并验证它保留一个右移标签 token。"""
    sequences = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(sequences, torch.Tensor) or sequences.ndim != 2:
        raise ValueError("prepared data must be a rank-2 tensor")
    if sequences.size(1) != block_size + 1:
        raise ValueError("prepared data does not match configured block_size")
    return sequences.long()


def manifest_revisions(path: Path) -> list[str]:
    """提取 source@revision，作为每次训练日志的最小溯源提示。"""
    return sorted({f"{record.source}@{record.revision}" for record in validate_manifest(path)})


def capture_rng_state() -> dict[str, Any]:
    """收集 Python、CPU 及可用 CUDA 的 RNG 状态，以支持确定性恢复。"""
    state: dict[str, Any] = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """恢复 checkpoint 中存储的随机状态。"""
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def checkpoint_state(
    model: GPTLanguageModel,
    trainer: Trainer,
    model_config: ModelConfig,
    data_config: DataConfig,
    train_config: TrainConfig,
    global_step: int,
    cursor: int,
    tokens_seen: int,
) -> dict[str, Any]:
    """汇集恢复训练所需的模型、优化器、调度器、配置、游标及随机状态。"""
    return {
        "model": model.state_dict(),
        "trainer": trainer.state_dict(),
        "model_config": asdict(model_config),
        "data_config": asdict(data_config),
        "train_config": asdict(train_config),
        "global_step": global_step,
        "cursor": cursor,
        "tokens_seen": tokens_seen,
        "rng_state": capture_rng_state(),
    }


def restore_training_state(
    checkpoint_path: Path,
    model: GPTLanguageModel,
    trainer: Trainer,
    expected_model_config: ModelConfig,
) -> tuple[int, int, int]:
    """检查架构兼容性并恢复模型、优化器、调度器、RNG 与训练游标。"""
    state = load_checkpoint(checkpoint_path)
    if state.get("model_config") != asdict(expected_model_config):
        raise ValueError("checkpoint model_config differs from requested config")
    model.load_state_dict(state["model"])
    trainer.load_state_dict(state["trainer"])
    restore_rng_state(state["rng_state"])
    return int(state["global_step"]), int(state["cursor"]), int(state["tokens_seen"])


def next_batch(sequences: torch.Tensor, cursor: int, batch_size: int) -> tuple[torch.Tensor, int]:
    """以循环游标抽取一个批次，保证 resume 后继续同一数据顺序。"""
    indices = (torch.arange(batch_size) + cursor) % sequences.size(0)
    return sequences[indices], (cursor + batch_size) % sequences.size(0)


def main() -> None:
    """执行训练，并在固定间隔和最终步写入可恢复 checkpoint。"""
    args = build_parser().parse_args()
    model_config, data_config, train_config = load_config(args.config)
    device = resolve_device(args.device)
    require_bf16_support(train_config.use_bf16, device)
    print("frozen_config:")
    print(args.config.read_text(encoding="utf-8"), end="")
    print(f"train_manifest_revisions={manifest_revisions(Path(data_config.train_manifest))}")
    print(f"validation_manifest_revisions={manifest_revisions(Path(data_config.valid_manifest))}")

    seed_everything(train_config.seed)
    sequences = load_prepared_sequences(Path(data_config.cache_dir) / "train.pt", model_config.block_size)
    model = GPTLanguageModel(model_config).to(device)
    trainer = Trainer(
        model,
        learning_rate=train_config.learning_rate,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        use_bf16=train_config.use_bf16,
    )
    global_step, cursor, tokens_seen = 0, 0, 0
    if args.resume is not None:
        global_step, cursor, tokens_seen = restore_training_state(args.resume, model, trainer, model_config)
        print(f"resumed_from={args.resume} global_step={global_step}")

    output_dir = Path(train_config.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    metrics_path = output_dir / "metrics.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_size = train_config.batch_size * train_config.gradient_accumulation_steps
    if sequences.size(0) < train_config.gradient_accumulation_steps:
        raise ValueError("prepared dataset has fewer sequences than accumulation steps")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step in range(global_step + 1, train_config.max_steps + 1):
        batch, cursor = next_batch(sequences, cursor, batch_size)
        started_at = time.perf_counter()
        metrics = trainer.train_step(batch[:, :-1], batch[:, 1:])
        elapsed_seconds = max(time.perf_counter() - started_at, 1e-12)
        step_tokens = batch.numel() - batch.size(0)
        tokens_seen += step_tokens
        peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        record = {
            "global_step": step,
            **metrics,
            "tokens_per_second": step_tokens / elapsed_seconds,
            "peak_memory_bytes": peak_memory,
            "tokens_seen": tokens_seen,
        }
        with metrics_path.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(json.dumps(record) + "\n")
        print(json.dumps(record))
        if step % train_config.checkpoint_every == 0 or step == train_config.max_steps:
            state = checkpoint_state(
                model,
                trainer,
                model_config,
                data_config,
                train_config,
                step,
                cursor,
                tokens_seen,
            )
            save_checkpoint(checkpoints_dir / f"step_{step}.pt", state)


if __name__ == "__main__":
    main()
