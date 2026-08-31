"""模块用途：验证训练数据的来源字段、切分规则与 token 分片边界。"""

import pytest

from sinogpt.data.manifest import ManifestRecord
from sinogpt.data.pack import pack_token_ids


def test_manifest_requires_license_note() -> None:
    """缺少许可证说明的样本不能进入训练清单。"""
    with pytest.raises(ValueError, match="license_note"):
        ManifestRecord.from_dict(
            {
                "text": "你好",
                "source": "demo",
                "revision": "r1",
                "language": "zh",
                "split": "train",
                "document_hash": "a",
            }
        )


def test_pack_preserves_order_at_shard_boundary() -> None:
    """分片不能改变 token 顺序或丢失最后一个不满分片。"""
    assert pack_token_ids([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
