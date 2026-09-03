"""模块用途：按稳定文档哈希把大型 JSONL manifest 物化为单个可审计训练分片。"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from sinogpt.data.manifest import ManifestRecord


@dataclass(frozen=True)
class ShardStats:
    """单次分片物化的输入与输出记录计数。"""

    input_records: int
    output_records: int


def shard_index_for_document(document_hash: str, shard_count: int) -> int:
    """由不可变 document hash 计算稳定分片编号。"""
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    digest = hashlib.sha256(document_hash.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % shard_count


def materialize_manifest_shard(
    input_path: Path,
    output_path: Path,
    *,
    shard_count: int,
    shard_index: int,
) -> ShardStats:
    """流式读取 manifest，只写出指定哈希桶，避免将全量清单读入内存。"""
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    input_records = 0
    output_records = 0
    try:
        with input_path.open("r", encoding="utf-8") as source, temporary_path.open("w", encoding="utf-8") as target:
            for line in source:
                if not line.strip():
                    continue
                raw_record = json.loads(line)
                record = ManifestRecord.from_dict(raw_record)
                input_records += 1
                if shard_index_for_document(record.document_hash, shard_count) != shard_index:
                    continue
                target.write(json.dumps(raw_record, ensure_ascii=False, sort_keys=True) + "\n")
                output_records += 1
        temporary_path.replace(output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return ShardStats(input_records=input_records, output_records=output_records)
