"""模块用途：定义并校验模型、数据与训练配置，不执行训练或数据访问。"""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelConfig:
    """GPT decoder 的结构参数。"""

    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int

    def __post_init__(self) -> None:
        if self.vocab_size < 4:
            raise ValueError("vocab_size must reserve at least four special tokens")
        if self.n_layer < 1:
            raise ValueError("n_layer must be positive")
        if self.n_head < 1:
            raise ValueError("n_head must be positive")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if self.block_size < 2:
            raise ValueError("block_size must be at least two tokens")

    @property
    def mlp_size(self) -> int:
        """返回 GPT MLP 的四倍隐藏层宽度。"""
        return 4 * self.n_embd


@dataclass(frozen=True)
class DataConfig:
    """训练与验证数据的不可变路径引用。"""

    train_manifest: str
    valid_manifest: str
    tokenizer_path: str
    cache_dir: str


@dataclass(frozen=True)
class TrainConfig:
    """优化、精度与输出目录参数。"""

    seed: int
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    max_steps: int
    checkpoint_every: int
    use_bf16: bool
    output_dir: str

    def __post_init__(self) -> None:
        if self.batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("batch_size and gradient_accumulation_steps must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.max_steps < 1 or self.checkpoint_every < 1:
            raise ValueError("max_steps and checkpoint_every must be positive")


@dataclass(frozen=True)
class SFTDataConfig:
    """COIG SFT 三分清单、冻结 tokenizer 与缓存目录。"""

    train_manifest: str
    validation_manifest: str
    test_manifest: str
    tokenizer_path: str
    cache_dir: str

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value.strip()
            for value in (
                self.train_manifest,
                self.validation_manifest,
                self.test_manifest,
                self.tokenizer_path,
                self.cache_dir,
            )
        ):
            raise ValueError("all SFT data paths must be nonempty strings")


@dataclass(frozen=True)
class SFTTrainConfig:
    """SFT 的优化、checkpoint 选择和预训练初始化参数。"""

    seed: int
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    max_epochs: int
    checkpoint_every_epoch: int
    use_bf16: bool
    base_checkpoint: str
    output_dir: str

    def __post_init__(self) -> None:
        if self.batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("batch_size and gradient_accumulation_steps must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be positive")
        if self.checkpoint_every_epoch != 1:
            raise ValueError("COIG SFT pilot must validate and checkpoint every epoch")
        if not isinstance(self.base_checkpoint, str) or not self.base_checkpoint.strip():
            raise ValueError("base_checkpoint must be a nonempty string")
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ValueError("output_dir must be a nonempty string")


def load_config(path: str | Path) -> tuple[ModelConfig, DataConfig, TrainConfig]:
    """从 YAML 文件加载三类已校验配置。"""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    return ModelConfig(**raw["model"]), DataConfig(**raw["data"]), TrainConfig(**raw["train"])


def load_sft_config(path: str | Path) -> tuple[ModelConfig, SFTDataConfig, SFTTrainConfig]:
    """从 YAML 加载 SFT 专属配置，避免误用预训练的二分数据配置。"""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    return ModelConfig(**raw["model"]), SFTDataConfig(**raw["data"]), SFTTrainConfig(**raw["train"])
