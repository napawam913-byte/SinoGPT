"""模块用途：验证中文语料导出时的清洗、去重和确定性切分。"""

import json
from pathlib import Path

from sinogpt.data.export import export_records


def test_export_records_filters_duplicates_and_creates_disjoint_splits(tmp_path: Path) -> None:
    """短文本与重复文档不应进入训练或验证 manifest。"""
    records = [
        {"text": "短"},
        {"text": "甲" * 12},
        {"text": "甲" * 12},
        {"text": "乙" * 12},
    ]
    train_output = tmp_path / "train.jsonl"
    validation_output = tmp_path / "validation.jsonl"
    stats = export_records(
        records,
        train_output,
        validation_output,
        source="owner/dataset/subset",
        revision="a" * 40,
        license_note="ODC-By 1.0",
        language="zh",
        minimum_characters=10,
        maximum_characters=20,
        validation_percent=50,
        limit=2,
    )
    rows = [
        json.loads(line)
        for path in (train_output, validation_output)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]

    assert stats.exported == 2
    assert stats.too_short == 1
    assert stats.duplicates == 1
    assert {row["split"] for row in rows} <= {"train", "validation"}
    assert len({row["document_hash"] for row in rows}) == 2
