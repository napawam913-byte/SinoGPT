"""模块用途：将经审计文档编码并打包为因果语言模型训练样本。"""

from collections.abc import Iterable

import torch
from tokenizers import Tokenizer

from sinogpt.data.manifest import ManifestRecord


def records_to_token_ids(records: Iterable[ManifestRecord], tokenizer: Tokenizer) -> list[int]:
    """按 manifest 顺序编码文档，并在每篇后追加保留的 EOS token。"""
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        raise ValueError("tokenizer must contain the <eos> special token")
    token_ids: list[int] = []
    for record in records:
        token_ids.extend(tokenizer.encode(record.text).ids)
        token_ids.append(eos_id)
    return token_ids


def pack_training_sequences(token_ids: list[int], block_size: int) -> torch.Tensor:
    """将连续 token 流切成不重排的 `[N, block_size + 1]` 因果样本。"""
    if block_size < 1:
        raise ValueError("block_size must be positive")
    tokens = torch.tensor(token_ids, dtype=torch.long)
    if tokens.numel() < block_size + 1:
        raise ValueError("not enough tokens for one input-target sequence")
    usable_length = ((tokens.numel() - 1) // block_size) * block_size + 1
    return tokens[:usable_length].unfold(0, block_size + 1, block_size).clone()
