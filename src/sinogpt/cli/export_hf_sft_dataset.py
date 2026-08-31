"""模块用途：流式导出带不可变版本和许可证说明的 COIG SFT 问答数据。"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from sinogpt.data.sft import DEFAULT_SYSTEM_PROMPT, export_coig_records


def build_parser() -> argparse.ArgumentParser:
    """构建要求显式 revision、许可证说明和精确切分数的导出命令。"""
    parser = argparse.ArgumentParser(description="导出可审计的 COIG 中文 SFT 问答数据")
    parser.add_argument("--dataset", default="BAAI/COIG", help="Hugging Face 数据集 ID")
    parser.add_argument("--split", default="train", help="读取的数据集 split")
    parser.add_argument("--revision", required=True, help="数据集分支、标签或 commit")
    parser.add_argument("--license-note", required=True, help="写入每条记录的许可证说明")
    parser.add_argument("--output-dir", required=True, type=Path, help="三个 SFT manifest 的目录")
    parser.add_argument("--limit", type=int, default=5_000, help="导出的唯一问答对总数")
    parser.add_argument("--train-count", type=int, default=4_000, help="训练集问答对数量")
    parser.add_argument("--validation-count", type=int, default=500, help="验证集问答对数量")
    parser.add_argument("--test-count", type=int, default=500, help="测试集问答对数量")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="固定的系统提示词")
    return parser


def main() -> None:
    """解析 SHA 后流式读取 COIG，并输出完整可审计的导出统计。"""
    args = build_parser().parse_args()
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ModuleNotFoundError as error:
        raise RuntimeError('请先执行 python -m pip install -e ".[data]"') from error

    resolved_revision = HfApi().dataset_info(args.dataset, revision=args.revision).sha
    if not resolved_revision:
        raise RuntimeError("无法解析数据集的 immutable revision")
    rows = load_dataset(
        args.dataset,
        split=args.split,
        streaming=True,
        revision=resolved_revision,
    )
    stats = export_coig_records(
        rows,
        args.output_dir,
        source=f"{args.dataset}/{args.split}",
        revision=resolved_revision,
        license_note=args.license_note,
        system_prompt=args.system_prompt,
        limit=args.limit,
        train_count=args.train_count,
        validation_count=args.validation_count,
        test_count=args.test_count,
    )
    print(
        json.dumps(
            {
                "source": f"{args.dataset}/{args.split}",
                "resolved_revision": resolved_revision,
                **asdict(stats),
                "train_output": str(args.output_dir / "coig_sft_train.jsonl"),
                "validation_output": str(args.output_dir / "coig_sft_validation.jsonl"),
                "test_output": str(args.output_dir / "coig_sft_test.jsonl"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
