"""模块用途：在启动长训练前校验不可变数据与归档溯源信息。"""

from string import hexdigits


def _require_nonempty_string(raw: dict[str, object], key: str) -> str:
    """读取一个必填非空字符串，并给出可定位的错误字段名。"""
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must not be empty")
    return value


def _require_sha256(raw: dict[str, object], key: str) -> None:
    """验证哈希是完整的 64 位十六进制 SHA-256，而非任意标签。"""
    value = _require_nonempty_string(raw, key)
    if len(value) != 64 or any(character not in hexdigits for character in value):
        raise ValueError(f"{key} must be a 64-character hexadecimal SHA-256")


def validate_run_manifest(raw: dict[str, object]) -> None:
    """拒绝缺少 provenance、恢复归档或明确 token 预算的长训练请求。"""
    _require_nonempty_string(raw, "run_name")
    _require_nonempty_string(raw, "config_path")
    _require_sha256(raw, "tokenizer_sha256")
    _require_sha256(raw, "dataset_manifest_sha256")
    _require_nonempty_string(raw, "checkpoint_uri")
    if type(raw.get("seed")) is not int:
        raise ValueError("seed must be an integer")
    if type(raw.get("target_tokens")) is not int or raw["target_tokens"] < 1:
        raise ValueError("target_tokens must be a positive integer")
