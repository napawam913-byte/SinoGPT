"""模块用途：过滤、去重并导出带来源信息的流式文本语料。"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path


@dataclass(frozen=True)
class ExportStats:
    """记录导出过程的数量证据，便于写入实验日志。"""

    seen: int
    exported: int
    too_short: int
    too_long: int
    duplicates: int
    train: int
    validation: int


def normalized_text(raw: object) -> str:
    """折叠空白字符，并拒绝无法作为文本训练的字段。"""
    if not isinstance(raw, str):
        raise ValueError("text must be a string")
    return " ".join(raw.split())


def split_for_hash(document_hash: str, validation_percent: int) -> str:
    """依据内容哈希做稳定且不依赖运行顺序的训练集切分。"""
    if not 0 < validation_percent < 100:
        raise ValueError("validation_percent must be between 1 and 99")
    return "validation" if int(document_hash[:8], 16) % 100 < validation_percent else "train"


def _validate_export_arguments(
    train_output: Path,
    validation_output: Path,
    minimum_characters: int,
    maximum_characters: int,
    validation_percent: int,
    limit: int,
) -> None:
    """在创建任何文件前校验导出范围，避免无意处理整份大数据集。"""
    if train_output.resolve() == validation_output.resolve():
        raise ValueError("train_output and validation_output must differ")
    if minimum_characters < 1:
        raise ValueError("minimum_characters must be positive")
    if maximum_characters < minimum_characters:
        raise ValueError("maximum_characters must be at least minimum_characters")
    split_for_hash("0" * 64, validation_percent)
    if limit < 1:
        raise ValueError("limit must be positive")


def export_records(
    records: Iterable[Mapping[str, object]],
    train_output: Path,
    validation_output: Path,
    *,
    source: str,
    revision: str,
    license_note: str,
    language: str,
    minimum_characters: int,
    maximum_characters: int,
    validation_percent: int,
    limit: int,
) -> ExportStats:
    """导出唯一文本为现有 manifest 格式，且不在内存累积原始语料。"""
    if not all((source, revision, license_note, language)):
        raise ValueError("source, revision, license_note and language must not be empty")
    _validate_export_arguments(
        train_output,
        validation_output,
        minimum_characters,
        maximum_characters,
        validation_percent,
        limit,
    )
    train_output.parent.mkdir(parents=True, exist_ok=True)
    validation_output.parent.mkdir(parents=True, exist_ok=True)
    seen_hashes: set[str] = set()
    seen = exported = too_short = too_long = duplicates = train = validation = 0

    with train_output.open("w", encoding="utf-8") as train_file, validation_output.open(
        "w", encoding="utf-8"
    ) as validation_file:
        for raw_record in records:
            if exported >= limit:
                break
            seen += 1
            text = normalized_text(raw_record["text"])
            if len(text) < minimum_characters:
                too_short += 1
                continue
            if len(text) > maximum_characters:
                too_long += 1
                continue
            document_hash = sha256(text.encode("utf-8")).hexdigest()
            if document_hash in seen_hashes:
                duplicates += 1
                continue
            seen_hashes.add(document_hash)
            split = split_for_hash(document_hash, validation_percent)
            row = {
                "text": text,
                "source": source,
                "revision": revision,
                "license_note": license_note,
                "language": language,
                "split": split,
                "document_hash": document_hash,
            }
            destination = validation_file if split == "validation" else train_file
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
            exported += 1
            if split == "validation":
                validation += 1
            else:
                train += 1

    return ExportStats(
        seen=seen,
        exported=exported,
        too_short=too_short,
        too_long=too_long,
        duplicates=duplicates,
        train=train,
        validation=validation,
    )
