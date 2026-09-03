"""模块用途：提供低内存 CLI，将大训练 manifest 生成一个确定性混合分片。"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from sinogpt.data.shard import materialize_manifest_shard


def build_parser() -> argparse.ArgumentParser:
    """构建显式声明输入、输出和哈希桶编号的分片命令行接口。"""
    parser = argparse.ArgumentParser(description="按稳定 document hash 物化一个训练 manifest 分片")
    parser.add_argument("--input", required=True, type=Path, help="全量 train manifest JSONL")
    parser.add_argument("--output", required=True, type=Path, help="当前分片 JSONL 输出路径")
    parser.add_argument("--shard-count", required=True, type=int, help="总分片数量")
    parser.add_argument("--shard-index", required=True, type=int, help="当前分片编号，从 0 开始")
    return parser


def main() -> None:
    """流式生成一个分片，并输出可记录到实验日志的统计信息。"""
    args = build_parser().parse_args()
    stats = materialize_manifest_shard(
        args.input,
        args.output,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
    )
    print(json.dumps({"input": str(args.input), "output": str(args.output), **asdict(stats)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
