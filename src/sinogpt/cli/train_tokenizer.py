"""模块用途：从经过 provenance 校验的 JSONL 清单训练 BPE tokenizer。"""

import argparse
from pathlib import Path

from sinogpt.data.manifest import validate_manifest
from sinogpt.tokenizer import train_bpe


def build_parser() -> argparse.ArgumentParser:
    """构造不含隐式数据来源的 tokenizer 命令行参数。"""
    parser = argparse.ArgumentParser(description="Train the SinoGPT BPE tokenizer")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--vocab-size", type=int, default=50_000)
    return parser


def main() -> None:
    """读取经验证的文本并写出 tokenizer.json。"""
    args = build_parser().parse_args()
    records = validate_manifest(args.manifest)
    train_bpe([record.text for record in records if record.split == "train"], args.vocab_size, args.output)
    print(f"tokenizer written to {args.output} from {len(records)} manifest records")


if __name__ == "__main__":
    main()
