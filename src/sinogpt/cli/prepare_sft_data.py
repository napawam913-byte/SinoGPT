"""模块用途：将可审计 COIG SFT JSONL 编码为 input/assistant-label 缓存。"""

import argparse
from pathlib import Path

from sinogpt.config import load_sft_config
from sinogpt.data.sft import (
    load_sft_records,
    prepare_sft_records,
    save_prepared_sft_split,
)
from sinogpt.tokenizer import load_tokenizer


def build_parser() -> argparse.ArgumentParser:
    """构建要求显式 SFT YAML 配置的缓存准备命令。"""
    parser = argparse.ArgumentParser(description="准备 SinoGPT 的 assistant-only SFT 数据缓存")
    parser.add_argument("--config", required=True, type=Path, help="SFT YAML 配置路径")
    return parser


def main() -> None:
    """逐个 split 编码 COIG 问答，绝不跨对话拼接。"""
    args = build_parser().parse_args()
    model_config, data_config, _ = load_sft_config(args.config)
    tokenizer = load_tokenizer(Path(data_config.tokenizer_path))
    manifests = {
        "train": data_config.train_manifest,
        "validation": data_config.validation_manifest,
        "test": data_config.test_manifest,
    }
    counts: dict[str, int] = {}
    for split, manifest_path in manifests.items():
        records = load_sft_records(Path(manifest_path))
        input_ids, labels = prepare_sft_records(records, tokenizer, model_config.block_size)
        save_prepared_sft_split(input_ids, labels, Path(data_config.cache_dir), split)
        counts[split] = input_ids.size(0)
    print(
        "prepared "
        f"train_sequences={counts['train']} "
        f"validation_sequences={counts['validation']} "
        f"test_sequences={counts['test']}"
    )


if __name__ == "__main__":
    main()
