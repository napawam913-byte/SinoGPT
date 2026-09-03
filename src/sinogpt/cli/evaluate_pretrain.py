"""模块用途：对预训练 checkpoint 的训练或验证缓存做只读因果语言模型评估。"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from sinogpt.config import ModelConfig, load_config
from sinogpt.model.gpt import GPTLanguageModel
from sinogpt.training.checkpoint import load_checkpoint
from sinogpt.training.evaluation import evaluate_causal_lm


def build_parser() -> argparse.ArgumentParser:
    """构建显式指定 checkpoint、配置、缓存与数据切分的评估命令。"""
    parser = argparse.ArgumentParser(description="只读评估 SinoGPT 预训练 checkpoint")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("train", "validation"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser


def resolve_device(requested: str) -> torch.device:
    """将 auto/cuda/cpu 选项转换为可用的 PyTorch 设备。"""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def load_prepared_sequences(path: Path, block_size: int) -> torch.Tensor:
    """读取 `[N, block_size + 1]` 的预训练缓存，并拒绝不兼容的输入。"""
    sequences = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(sequences, torch.Tensor) or sequences.ndim != 2:
        raise ValueError("prepared data must be a rank-2 tensor")
    if sequences.size(1) != block_size + 1:
        raise ValueError("prepared data does not match configured block_size")
    return sequences.long()


def main() -> None:
    """加载指定权重，以右移 token 标签评估缓存；全过程不执行反向传播。"""
    args = build_parser().parse_args()
    model_config, _, _ = load_config(args.config)
    state = load_checkpoint(args.checkpoint)
    if state.get("model_config") != asdict(model_config):
        raise ValueError("checkpoint model_config differs from requested config")

    device = resolve_device(args.device)
    model = GPTLanguageModel(ModelConfig(**state["model_config"])).to(device)
    model.load_state_dict(state["model"])
    sequences = load_prepared_sequences(
        args.cache_dir / f"{args.split}.pt", model_config.block_size
    )
    result = evaluate_causal_lm(
        model,
        sequences[:, :-1],
        sequences[:, 1:],
        args.batch_size,
    )
    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "split": args.split,
                "loss": result.loss,
                "perplexity": result.perplexity,
                "supervised_tokens": result.supervised_tokens,
            }
        )
    )


if __name__ == "__main__":
    main()
