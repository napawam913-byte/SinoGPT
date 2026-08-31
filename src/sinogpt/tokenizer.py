"""模块用途：训练并加载项目唯一使用的 BPE 分词器。"""

from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def train_bpe(texts: list[str], vocab_size: int, output_path: Path) -> None:
    """使用给定文本训练含四个保留 token 的 ByteLevel BPE。"""
    if vocab_size < len(SPECIAL_TOKENS):
        raise ValueError("vocab_size must reserve all special tokens")
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)
    tokenizer.train_from_iterator(texts, trainer=trainer)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))


def load_tokenizer(path: Path) -> Tokenizer:
    """从冻结的 tokenizer.json 加载 tokenizer。"""
    return Tokenizer.from_file(str(path))
