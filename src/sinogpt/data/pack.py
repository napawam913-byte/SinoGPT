"""模块用途：按顺序将 token ID 切分为固定上限的训练分片。"""


def pack_token_ids(token_ids: list[int], shard_size: int) -> list[list[int]]:
    """保序分片，最后一片允许小于 shard_size。"""
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    return [token_ids[index : index + shard_size] for index in range(0, len(token_ids), shard_size)]
