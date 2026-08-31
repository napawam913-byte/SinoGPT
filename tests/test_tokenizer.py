"""模块用途：验证 BPE 分词器可保留中英文文本并保存后重新加载。"""

from pathlib import Path

from sinogpt.tokenizer import SPECIAL_TOKENS, load_tokenizer, train_bpe


def test_bpe_round_trips_bilingual_text(tmp_path: Path) -> None:
    """中英文训练样例编码后应可还原为原始文本。"""
    tokenizer_path = tmp_path / "tokenizer.json"
    train_bpe(["你好 world", "world 你好"], vocab_size=64, output_path=tokenizer_path)
    tokenizer = load_tokenizer(tokenizer_path)
    assert tokenizer.decode(tokenizer.encode("你好 world").ids) == "你好 world"


def test_bpe_reserves_chat_control_tokens(tmp_path: Path) -> None:
    """正式预训练的 tokenizer 必须在起始词表中固定聊天控制 token。"""
    tokenizer_path = tmp_path / "tokenizer.json"
    train_bpe(["你好"], vocab_size=64, output_path=tokenizer_path)
    tokenizer = load_tokenizer(tokenizer_path)
    assert SPECIAL_TOKENS[-3:] == ["<|system|>", "<|user|>", "<|assistant|>"]
    assert [tokenizer.token_to_id(token) for token in SPECIAL_TOKENS] == list(range(len(SPECIAL_TOKENS)))
