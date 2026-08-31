"""模块用途：对指定 SFT checkpoint 的验证或测试缓存输出可复现实验指标。"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from sinogpt.config import ModelConfig, load_sft_config
from sinogpt.data.sft import load_prepared_sft_split
from sinogpt.model.gpt import GPTLanguageModel
from sinogpt.training.checkpoint import load_checkpoint
from sinogpt.training.evaluation import evaluate_causal_lm


def build_parser() -> argparse.ArgumentParser:
    """构建必须显式指向 checkpoint、配置和 held-out split 的评估命令。"""
    parser = argparse.ArgumentParser(description="评估 SinoGPT SFT checkpoint 的验证或测试集")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("validation", "test"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser


def resolve_device(requested: str) -> torch.device:
    """将 auto/cuda/cpu 选项转换为可用的 PyTorch 设备。"""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def main() -> None:
    """加载兼容 SFT 权重并以 token 加权的方式评估一个 held-out split。"""
    args = build_parser().parse_args()
    model_config, data_config, _ = load_sft_config(args.config)
    state = load_checkpoint(args.checkpoint)
    if state.get("kind") != "sft":
        raise ValueError("checkpoint is not an SFT checkpoint")
    if state.get("model_config") != asdict(model_config):
        raise ValueError("checkpoint model_config differs from requested config")
    device = resolve_device(args.device)
    model = GPTLanguageModel(ModelConfig(**state["model_config"])).to(device)
    model.load_state_dict(state["model"])
    input_ids, labels = load_prepared_sft_split(
        Path(data_config.cache_dir), args.split, model_config.block_size
    )
    result = evaluate_causal_lm(model, input_ids, labels, args.batch_size)
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
