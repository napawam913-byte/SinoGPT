"""模块用途：验证 COIG SFT 记录的来源、去重、切分和字段过滤。"""

from pathlib import Path

import pytest

from sinogpt.data.sft import DEFAULT_SYSTEM_PROMPT, export_coig_records, load_sft_records


def export_fixture(rows: list[dict[str, object]], output_dir: Path):
    """用小型 COIG 风格数据执行固定 3/1/1 切分。"""
    return export_coig_records(
        rows,
        output_dir,
        source="BAAI/COIG",
        revision="a" * 40,
        license_note="COIG research pilot",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        limit=5,
        train_count=3,
        validation_count=1,
        test_count=1,
    )


def test_export_coig_records_deduplicates_and_uses_exact_split_counts(tmp_path: Path) -> None:
    """相同问答只能保留一次，且导出必须精确写入三个已声明的 split。"""
    rows = [
        {"conversations": [{"question": "问题一", "answer": "回答一"}]},
        {"conversations": [{"question": "问题二", "answer": "回答二"}]},
        {"conversations": [{"question": "问题二", "answer": "回答二"}]},
        {"conversations": [{"question": "问题三", "answer": "回答三"}]},
        {"conversations": [{"question": "问题四", "answer": "回答四"}]},
        {"conversations": [{"question": "问题五", "answer": "回答五"}]},
    ]

    stats = export_fixture(rows, tmp_path)
    train = load_sft_records(tmp_path / "coig_sft_train.jsonl")
    validation = load_sft_records(tmp_path / "coig_sft_validation.jsonl")
    test = load_sft_records(tmp_path / "coig_sft_test.jsonl")

    assert (stats.exported, stats.duplicates, stats.train, stats.validation, stats.test) == (5, 1, 3, 1, 1)
    assert {record.split for record in train} == {"train"}
    assert {record.split for record in validation} == {"validation"}
    assert {record.split for record in test} == {"test"}
    assert len({record.record_hash for record in [*train, *validation, *test]}) == 5
    assert all(record.revision == "a" * 40 for record in train)


def test_export_coig_records_reports_empty_and_malformed_turns(tmp_path: Path) -> None:
    """空字段和非字符串字段不得进入数据集，且必须体现在统计中。"""
    rows = [
        {
            "conversations": [
                {"question": "   ", "answer": "回答"},
                {"question": "问题", "answer": ["不是字符串"]},
                {"question": "有效问题", "answer": "有效回答"},
            ]
        }
    ]

    stats = export_coig_records(
        rows,
        tmp_path,
        source="BAAI/COIG",
        revision="a" * 40,
        license_note="COIG research pilot",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        limit=1,
        train_count=1,
        validation_count=0,
        test_count=0,
    )

    assert (stats.exported, stats.empty_turns, stats.malformed_turns) == (1, 1, 1)


def test_export_coig_records_rejects_counts_that_do_not_sum_to_limit(tmp_path: Path) -> None:
    """配置错误不能悄悄产生与论文约定不一致的数据切分。"""
    with pytest.raises(ValueError, match="must sum to limit"):
        export_coig_records(
            [],
            tmp_path,
            source="BAAI/COIG",
            revision="a" * 40,
            license_note="COIG research pilot",
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            limit=5,
            train_count=3,
            validation_count=1,
            test_count=0,
        )
