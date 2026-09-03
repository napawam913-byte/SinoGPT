"""模块用途：将 manifest 中的审计文本转换为本地可训练的 token 序列缓存。"""

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from sinogpt.config import load_config
from sinogpt.data.manifest import validate_manifest
from sinogpt.data.prepare import pack_training_sequences, records_to_token_ids
from sinogpt.tokenizer import load_tokenizer


def build_parser() -> argparse.ArgumentParser:
    """构建配置驱动、可为当前数据分片覆盖输入和缓存的命令行参数。"""
    parser = argparse.ArgumentParser(description="把已审计 manifest 打包为因果训练序列")
    parser.add_argument("--config", required=True, type=Path, help="YAML 训练配置路径")
    parser.add_argument("--train-manifest", type=Path, help="仅本次使用的 train manifest 分片")
    parser.add_argument("--cache-dir", type=Path, help="仅本次使用的 token 缓存目录")
    return parser


def prepare_split(
    manifest_path: Path,
    expected_split: str,
    tokenizer_path: Path,
    block_size: int,
    output_path: Path,
) -> int:
    """校验一个数据切分、编码并保存其 `[N,T+1]` 长整型张量。"""
    records = validate_manifest(manifest_path)
    if any(record.split != expected_split for record in records):
        raise ValueError(f"{manifest_path} contains records outside the {expected_split} split")
    token_ids = records_to_token_ids(records, load_tokenizer(tokenizer_path))
    sequences = pack_training_sequences(token_ids, block_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sequences, output_path)
    return int(sequences.size(0))


def main() -> None:
    """读取配置并构建 train/validation 两个可恢复数据缓存。"""
    args = build_parser().parse_args()
    model_config, data_config, _ = load_config(args.config)
    data_config = replace(
        data_config,
        train_manifest=str(args.train_manifest or data_config.train_manifest),
        cache_dir=str(args.cache_dir or data_config.cache_dir),
    )
    cache_dir = Path(data_config.cache_dir)
    train_count = prepare_split(
        Path(data_config.train_manifest),
        "train",
        Path(data_config.tokenizer_path),
        model_config.block_size,
        cache_dir / "train.pt",
    )
    validation_count = prepare_split(
        Path(data_config.valid_manifest),
        "validation",
        Path(data_config.tokenizer_path),
        model_config.block_size,
        cache_dir / "validation.pt",
    )
    print(f"prepared train_sequences={train_count} validation_sequences={validation_count}")


if __name__ == "__main__":
    main()
