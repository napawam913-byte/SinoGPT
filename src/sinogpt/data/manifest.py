"""模块用途：校验带有来源和许可证信息的 JSONL 训练语料清单。"""

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class ManifestRecord:
    """单篇可追溯训练文档的最小字段集合。"""

    text: str
    source: str
    revision: str
    license_note: str
    language: str
    split: str
    document_hash: str

    @classmethod
    def from_dict(cls, raw: dict[str, str]) -> "ManifestRecord":
        """从 JSON 对象建立记录，并拒绝无来源或无许可证字段。"""
        required_fields = (
            "text",
            "source",
            "revision",
            "license_note",
            "language",
            "split",
            "document_hash",
        )
        missing = [field for field in required_fields if not raw.get(field)]
        if missing:
            raise ValueError(", ".join(missing))
        if raw["split"] not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        return cls(**{field: raw[field] for field in required_fields})


def validate_manifest(path: Path) -> list[ManifestRecord]:
    """读取 JSONL 清单并拒绝同一清单中的重复文档哈希。"""
    records = [
        ManifestRecord.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    hashes = [record.document_hash for record in records]
    if len(hashes) != len(set(hashes)):
        raise ValueError("duplicate document_hash in manifest")
    return records
