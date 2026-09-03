"""模块用途：验证大规模 manifest 的确定性分片，不读取或训练模型。"""

import json
from pathlib import Path

from sinogpt.data.shard import materialize_manifest_shard


def _record(document_hash: str) -> dict[str, str]:
    """构造最小可审计训练记录。"""
    return {
        "text": f"文档 {document_hash}",
        "source": "demo/source",
        "revision": "revision-1",
        "license_note": "research-only",
        "language": "zh",
        "split": "train",
        "document_hash": document_hash,
    }


def test_materialized_shards_are_disjoint_complete_and_repeatable(tmp_path: Path) -> None:
    """同一哈希分片必须不重不漏，并在再次运行时产生相同结果。"""
    source = tmp_path / "train.jsonl"
    records = [_record(f"hash-{index}") for index in range(12)]
    source.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    observed_hashes: set[str] = set()
    for shard_index in range(3):
        output = tmp_path / f"shard-{shard_index}.jsonl"
        stats = materialize_manifest_shard(source, output, shard_count=3, shard_index=shard_index)
        selected = [json.loads(line)["document_hash"] for line in output.read_text(encoding="utf-8").splitlines()]
        assert stats.input_records == len(records)
        assert stats.output_records == len(selected)
        assert observed_hashes.isdisjoint(selected)
        observed_hashes.update(selected)

    repeated = tmp_path / "shard-0-repeat.jsonl"
    materialize_manifest_shard(source, repeated, shard_count=3, shard_index=0)
    assert repeated.read_text(encoding="utf-8") == (tmp_path / "shard-0.jsonl").read_text(encoding="utf-8")
    assert observed_hashes == {record["document_hash"] for record in records}
