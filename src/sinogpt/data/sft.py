"""模块用途：导出并校验带来源证据的单轮中文 SFT 问答记录。"""

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

import torch
from torch import Tensor
from tokenizers import Tokenizer


DEFAULT_SYSTEM_PROMPT = "你是一个简洁、诚实的中文助手。"
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class SFTRecord:
    """单条可追溯的 system/user/assistant 监督样本。"""

    system: str
    user: str
    assistant: str
    source: str
    revision: str
    license_note: str
    language: str
    split: str
    record_hash: str

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "SFTRecord":
        """读取 JSONL 对象，并拒绝缺失或不合法的审计字段。"""
        required_fields = (
            "system",
            "user",
            "assistant",
            "source",
            "revision",
            "license_note",
            "language",
            "split",
            "record_hash",
        )
        missing = [field for field in required_fields if not isinstance(raw.get(field), str) or not raw[field]]
        if missing:
            raise ValueError(", ".join(missing))
        if raw["split"] not in SPLITS:
            raise ValueError("split must be train, validation or test")
        record = cls(**{field: str(raw[field]) for field in required_fields})
        if record.record_hash != canonical_record_hash(record.system, record.user, record.assistant):
            raise ValueError("record_hash does not match SFT content")
        return record


@dataclass(frozen=True)
class SFTExportStats:
    """记录 COIG 导出的完整计数，避免过滤行为不可见。"""

    source_rows: int
    turns: int
    exported: int
    empty_turns: int
    malformed_turns: int
    duplicates: int
    train: int
    validation: int
    test: int


def normalized_sft_text(raw: object) -> str | None:
    """规范化字符串字段；非字符串返回 None，空白字符串返回空字符串。"""
    if not isinstance(raw, str):
        return None
    return " ".join(raw.split())


def canonical_record_hash(system: str, user: str, assistant: str) -> str:
    """以稳定 JSON 表示计算问答内容哈希，不将来源字段误作内容去重键。"""
    payload = json.dumps(
        {"assistant": assistant, "system": system, "user": user},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _validate_export_options(
    source: str,
    revision: str,
    license_note: str,
    system_prompt: str,
    limit: int,
    train_count: int,
    validation_count: int,
    test_count: int,
) -> None:
    """在访问输入数据前锁定来源及精确三分切分约束。"""
    if not all(isinstance(value, str) and value.strip() for value in (source, revision, license_note, system_prompt)):
        raise ValueError("source, revision, license_note and system_prompt must not be empty")
    if len(revision) != 40:
        raise ValueError("revision must be a 40-character commit SHA")
    if limit < 1 or any(count < 0 for count in (train_count, validation_count, test_count)):
        raise ValueError("limit must be positive and split counts must be nonnegative")
    if train_count + validation_count + test_count != limit:
        raise ValueError("train_count, validation_count and test_count must sum to limit")


def _split_for_position(position: int, train_count: int, validation_count: int) -> str:
    """按冻结源顺序把已接收的样本划入精确数量的三个 split。"""
    if position < train_count:
        return "train"
    if position < train_count + validation_count:
        return "validation"
    return "test"


def _write_records(records: list[SFTRecord], output_dir: Path) -> None:
    """在成功获得完整固定样本后一次性写出三个 JSONL 清单。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        destination = output_dir / f"coig_sft_{split}.jsonl"
        selected = (record for record in records if record.split == split)
        with destination.open("w", encoding="utf-8") as output:
            for record in selected:
                output.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def export_coig_records(
    rows: Iterable[Mapping[str, object]],
    output_dir: Path,
    *,
    source: str,
    revision: str,
    license_note: str,
    system_prompt: str,
    limit: int,
    train_count: int,
    validation_count: int,
    test_count: int,
    language: str = "zh",
) -> SFTExportStats:
    """从 COIG 行流中提取唯一问答，并写入精确 4k/500/500 式清单。"""
    _validate_export_options(
        source,
        revision,
        license_note,
        system_prompt,
        limit,
        train_count,
        validation_count,
        test_count,
    )
    if not isinstance(language, str) or not language.strip():
        raise ValueError("language must not be empty")

    accepted: list[SFTRecord] = []
    seen_hashes: set[str] = set()
    source_rows = turns = empty_turns = malformed_turns = duplicates = 0
    for row in rows:
        if len(accepted) == limit:
            break
        source_rows += 1
        conversations = row.get("conversations") if isinstance(row, Mapping) else None
        if not isinstance(conversations, list):
            malformed_turns += 1
            continue
        for turn in conversations:
            if len(accepted) == limit:
                break
            turns += 1
            if not isinstance(turn, Mapping):
                malformed_turns += 1
                continue
            question = normalized_sft_text(turn.get("question"))
            answer = normalized_sft_text(turn.get("answer"))
            if question is None or answer is None:
                malformed_turns += 1
                continue
            if not question or not answer:
                empty_turns += 1
                continue
            record_hash = canonical_record_hash(system_prompt, question, answer)
            if record_hash in seen_hashes:
                duplicates += 1
                continue
            seen_hashes.add(record_hash)
            split = _split_for_position(len(accepted), train_count, validation_count)
            accepted.append(
                SFTRecord(
                    system=system_prompt,
                    user=question,
                    assistant=answer,
                    source=source,
                    revision=revision,
                    license_note=license_note,
                    language=language,
                    split=split,
                    record_hash=record_hash,
                )
            )

    if len(accepted) != limit:
        raise ValueError(f"only exported {len(accepted)} unique SFT records; requested {limit}")
    _write_records(accepted, output_dir)
    return SFTExportStats(
        source_rows=source_rows,
        turns=turns,
        exported=len(accepted),
        empty_turns=empty_turns,
        malformed_turns=malformed_turns,
        duplicates=duplicates,
        train=train_count,
        validation=validation_count,
        test=test_count,
    )


def load_sft_records(path: Path) -> list[SFTRecord]:
    """加载一份 SFT JSONL，并拒绝重复内容或 split 与文件名不一致。"""
    expected_split = path.stem.removeprefix("coig_sft_")
    if expected_split not in SPLITS:
        raise ValueError("SFT manifest filename must begin with coig_sft_")
    records = [
        SFTRecord.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("SFT manifest must contain at least one record")
    if any(record.split != expected_split for record in records):
        raise ValueError("SFT manifest split does not match filename")
    hashes = [record.record_hash for record in records]
    if len(hashes) != len(set(hashes)):
        raise ValueError("duplicate record_hash in SFT manifest")
    return records


def _token_id(tokenizer: Tokenizer, token: str) -> int:
    """读取必需的控制 token ID，拒绝后续 SFT 改动词表。"""
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        raise ValueError(f"tokenizer must contain {token}")
    return token_id


def _chat_token_ids(record: SFTRecord, tokenizer: Tokenizer) -> tuple[list[int], int]:
    """按项目固定模板编码完整对话，并返回 assistant 标记的下标。"""
    assistant_id = _token_id(tokenizer, "<|assistant|>")
    for token in ("<pad>", "<eos>", "<|system|>", "<|user|>"):
        _token_id(tokenizer, token)
    text = (
        f"<|system|>{record.system}<eos>"
        f"<|user|>{record.user}<eos>"
        f"<|assistant|>{record.assistant}<eos>"
    )
    token_ids = tokenizer.encode(text).ids
    try:
        return token_ids, token_ids.index(assistant_id)
    except ValueError as error:
        raise ValueError("encoded SFT record is missing <|assistant|>") from error


def encode_sft_record(record: SFTRecord, tokenizer: Tokenizer, block_size: int) -> tuple[Tensor, Tensor]:
    """构造定长 input/label；仅 assistant 正文和 EOS 保留标签。"""
    if block_size < 1:
        raise ValueError("block_size must be positive")
    token_ids, assistant_index = _chat_token_ids(record, tokenizer)
    if len(token_ids) < 2 or len(token_ids) > block_size + 1:
        raise ValueError("SFT record does not fit block_size")
    pad_id = _token_id(tokenizer, "<pad>")
    input_ids = torch.full((block_size,), pad_id, dtype=torch.long)
    labels = torch.full((block_size,), -100, dtype=torch.long)
    input_length = len(token_ids) - 1
    input_ids[:input_length] = torch.tensor(token_ids[:-1], dtype=torch.long)
    targets = torch.tensor(token_ids[1:], dtype=torch.long)
    labels[assistant_index:input_length] = targets[assistant_index:]
    if not bool((labels != -100).any()):
        raise ValueError("SFT record has no assistant supervision tokens")
    return input_ids, labels


def prepare_sft_records(
    records: list[SFTRecord], tokenizer: Tokenizer, block_size: int
) -> tuple[Tensor, Tensor]:
    """逐条编码 SFT 记录，不跨对话拼接或重排 token。"""
    if not records:
        raise ValueError("SFT records must not be empty")
    pairs = [encode_sft_record(record, tokenizer, block_size) for record in records]
    return torch.stack([pair[0] for pair in pairs]), torch.stack([pair[1] for pair in pairs])


def _validate_prepared_sft_tensors(input_ids: Tensor, labels: Tensor, block_size: int) -> None:
    """验证磁盘缓存仍保持同形状、整型和至少一个监督 token。"""
    if input_ids.ndim != 2 or labels.ndim != 2 or input_ids.shape != labels.shape:
        raise ValueError("prepared SFT inputs and labels must be equal rank-2 tensors")
    if input_ids.size(0) < 1 or input_ids.size(1) != block_size:
        raise ValueError("prepared SFT tensors do not match block_size")
    if input_ids.dtype != torch.long or labels.dtype != torch.long:
        raise ValueError("prepared SFT tensors must use torch.long")
    if not bool((labels != -100).any()):
        raise ValueError("prepared SFT labels have no supervised tokens")


def save_prepared_sft_split(input_ids: Tensor, labels: Tensor, cache_dir: Path, split: str) -> None:
    """保存一个 split 的输入及标签张量，文件名保持可读且明确。"""
    if split not in SPLITS:
        raise ValueError("split must be train, validation or test")
    _validate_prepared_sft_tensors(input_ids, labels, input_ids.size(1))
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(input_ids, cache_dir / f"{split}_input_ids.pt")
    torch.save(labels, cache_dir / f"{split}_labels.pt")


def load_prepared_sft_split(cache_dir: Path, split: str, block_size: int) -> tuple[Tensor, Tensor]:
    """加载一个缓存 split，并在使用前再次验证其形状和标签掩码。"""
    if split not in SPLITS:
        raise ValueError("split must be train, validation or test")
    input_ids = torch.load(cache_dir / f"{split}_input_ids.pt", map_location="cpu", weights_only=True)
    labels = torch.load(cache_dir / f"{split}_labels.pt", map_location="cpu", weights_only=True)
    if not isinstance(input_ids, Tensor) or not isinstance(labels, Tensor):
        raise ValueError("prepared SFT cache must contain tensors")
    input_ids = input_ids.long()
    labels = labels.long()
    _validate_prepared_sft_tensors(input_ids, labels, block_size)
    return input_ids, labels
