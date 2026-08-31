"""模块用途：将指定 Hugging Face 数据集切片流式导出为 SinoGPT manifest。"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from sinogpt.data.export import export_records


def build_parser() -> argparse.ArgumentParser:
    """构建要求显式数据来源、revision 和输出路径的命令行接口。"""
    parser = argparse.ArgumentParser(description="导出可审计的 Hugging Face 文本数据集")
    parser.add_argument("--dataset", required=True, help="Hugging Face 数据集 ID")
    parser.add_argument("--subset", required=True, help="数据集配置或切片名称")
    parser.add_argument("--revision", required=True, help="数据集分支、标签或 commit")
    parser.add_argument("--license-note", required=True, help="写入 manifest 的许可证说明")
    parser.add_argument("--train-output", required=True, type=Path)
    parser.add_argument("--validation-output", required=True, type=Path)
    parser.add_argument("--limit", required=True, type=int, help="最多导出多少条唯一文档")
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--max-chars", type=int, default=12_000)
    parser.add_argument("--validation-percent", type=int, default=1)
    parser.add_argument("--language", default="zh")
    return parser


def main() -> None:
    """解析不可变 revision 后流式读取远端数据，并输出可保存的统计 JSON。"""
    args = build_parser().parse_args()
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ModuleNotFoundError as error:
        raise RuntimeError('请先执行 python -m pip install -e ".[data]"') from error

    resolved_revision = HfApi().dataset_info(args.dataset, revision=args.revision).sha
    if not resolved_revision:
        raise RuntimeError("无法解析数据集的 immutable revision")
    records = load_dataset(
        args.dataset,
        args.subset,
        split="train",
        streaming=True,
        revision=resolved_revision,
    )
    stats = export_records(
        records,
        args.train_output,
        args.validation_output,
        source=f"{args.dataset}/{args.subset}",
        revision=resolved_revision,
        license_note=args.license_note,
        language=args.language,
        minimum_characters=args.min_chars,
        maximum_characters=args.max_chars,
        validation_percent=args.validation_percent,
        limit=args.limit,
    )
    print(
        json.dumps(
            {
                "source": f"{args.dataset}/{args.subset}",
                "resolved_revision": resolved_revision,
                **asdict(stats),
                "train_output": str(args.train_output),
                "validation_output": str(args.validation_output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
