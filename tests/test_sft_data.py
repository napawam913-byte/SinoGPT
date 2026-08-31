"""模块用途：验证 COIG SFT 记录的来源、去重、切分和字段过滤。"""

from pathlib import Path

import pytest
import torch

from sinogpt.data.sft import (
    DEFAULT_SYSTEM_PROMPT,
    SFTRecord,
    encode_sft_record,
    export_coig_records,
    load_prepared_sft_split,
    load_sft_records,
    prepare_sft_records,
    save_prepared_sft_split,
)
from sinogpt.tokenizer import load_tokenizer, train_bpe


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


def build_tokenizer(tmp_path: Path):
    """训练一个包含项目聊天控制 token 的小型测试 tokenizer。"""
    tokenizer_path = tmp_path / "tokenizer.json"
    train_bpe(["系统 问题 回答 很长"], vocab_size=64, output_path=tokenizer_path)
    return load_tokenizer(tokenizer_path)


def sft_record() -> SFTRecord:
    """构造适合验证聊天模板标签的最小记录。"""
    return SFTRecord(
        system="系统",
        user="问题",
        assistant="回答",
        source="BAAI/COIG",
        revision="a" * 40,
        license_note="COIG research pilot",
        language="zh",
        split="train",
        record_hash="159b152936e455f41f218d70b1ef312e1fe7d2dfbd3950b11dac0e0d88febbec",
    )


def test_encode_sft_record_masks_prompt_and_keeps_answer_eos(tmp_path: Path) -> None:
    """只有 assistant 正文与结束 EOS 能作为语言模型的监督目标。"""
    tokenizer = build_tokenizer(tmp_path)
    record = sft_record()

    input_ids, labels = encode_sft_record(record, tokenizer, block_size=32)
    assistant_id = tokenizer.token_to_id("<|assistant|>")
    answer_id = tokenizer.encode("回答").ids[0]
    eos_id = tokenizer.token_to_id("<eos>")
    marker_index = input_ids.tolist().index(assistant_id)
    supervised_positions = (labels != -100).nonzero().flatten()

    assert torch.equal(labels[:marker_index], torch.full_like(labels[:marker_index], -100))
    assert labels[marker_index].item() == answer_id
    assert labels[supervised_positions[-1]].item() == eos_id
    assert torch.equal(labels[supervised_positions[-1] + 1 :], torch.full_like(labels[supervised_positions[-1] + 1 :], -100))


def test_encode_sft_record_rejects_a_dialogue_that_exceeds_context(tmp_path: Path) -> None:
    """不能静默截断 assistant 答案，否则监督目标会被破坏。"""
    tokenizer = build_tokenizer(tmp_path)
    record = SFTRecord(
        system="系统",
        user="很长" * 200,
        assistant="回答",
        source="BAAI/COIG",
        revision="a" * 40,
        license_note="COIG research pilot",
        language="zh",
        split="train",
        record_hash="0" * 64,
    )

    with pytest.raises(ValueError, match="block_size"):
        encode_sft_record(record, tokenizer, block_size=8)


def test_prepared_sft_split_round_trips_inputs_and_labels(tmp_path: Path) -> None:
    """缓存必须保留一条记录一行及标签中的 -100 掩码。"""
    tokenizer = build_tokenizer(tmp_path)
    inputs, labels = prepare_sft_records([sft_record()], tokenizer, block_size=32)

    save_prepared_sft_split(inputs, labels, tmp_path, "train")
    restored_inputs, restored_labels = load_prepared_sft_split(tmp_path, "train", block_size=32)

    assert torch.equal(restored_inputs, inputs)
    assert torch.equal(restored_labels, labels)


def test_export_coig_records_skips_samples_that_exceed_the_frozen_context(tmp_path: Path) -> None:
    """导出阶段应依据预训练 tokenizer 过滤过长对话，而不是让后续准备阶段崩溃。"""
    tokenizer = build_tokenizer(tmp_path)
    rows = [
        {"conversations": [{"question": "很长" * 200, "answer": "回答"}]},
        {"conversations": [{"question": "问", "answer": "答"}]},
    ]

    stats = export_coig_records(
        rows,
        tmp_path,
        source="BAAI/COIG",
        revision="a" * 40,
        license_note="COIG research pilot",
        system_prompt="系",
        limit=1,
        train_count=1,
        validation_count=0,
        test_count=0,
        tokenizer=tokenizer,
        block_size=64,
    )

    assert (stats.exported, stats.too_long) == (1, 1)
