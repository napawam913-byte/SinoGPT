# SinoGPT Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` (recommended) or `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a from-scratch GPT-3-style training foundation that trains the 25M tutorial model on the cloud GPU and produces the evidence for the Markdown tutorial.

**Architecture:** Python modules isolate configuration, provenance-bearing data, tokenizer, decoder-only model, trainer, checkpoints and CLI. A 25M model is the executable reference; 125M and 350M only reuse its tested pipeline after data and storage are pinned.

**Tech Stack:** Python 3.12, PyTorch 2.3+, Hugging Face tokenizers, PyYAML, safetensors, TensorBoard, pytest, ruff, Mermaid Markdown.

## Global Constraints

- Use decoder-only causal language modeling only; SFT and RLHF are out of scope.
- Use Pre-LayerNorm, learned token/position embeddings, causal attention, GELU MLPs, residual connections and AdamW.
- Tiny configuration is 6 layers, 384 hidden, 6 heads, MLP 1536 and context 512.
- Main configuration is 24 layers, 1024 hidden, 16 heads, MLP 4096 and context 1024 unless profiling proves a different safe setting.
- Every corpus record must keep source, revision, license note, language, hash and split.
- bf16 must fail with an actionable message when the selected GPU lacks bf16 support.
- Checkpoints contain model, optimizer, scheduler, RNG state, dataset cursor and global step.
- The 100GB cloud disk holds only code, token cache and two newest checkpoints; raw corpora and archives live in object storage or a mounted data disk.
- All documentation calls the work a constrained-resource reproduction and data-mixture study, never GPT-3 175B or SOTA reproduction.

---

## Planned File Structure

```text
pyproject.toml
README.md
configs/tiny_25m.yaml
configs/ablation_125m.yaml
configs/main_350m.yaml
src/sinogpt/config.py
src/sinogpt/seed.py
src/sinogpt/tokenizer.py
src/sinogpt/data/manifest.py
src/sinogpt/data/pack.py
src/sinogpt/data/dataset.py
src/sinogpt/model/layers.py
src/sinogpt/model/gpt.py
src/sinogpt/training/checkpoint.py
src/sinogpt/training/trainer.py
src/sinogpt/cli/train_tokenizer.py
src/sinogpt/cli/prepare_data.py
src/sinogpt/cli/train.py
src/sinogpt/cli/sample.py
tests/test_config.py
tests/test_data.py
tests/test_tokenizer.py
tests/test_model.py
tests/test_training.py
tests/test_cli.py
docs/tutorials/00-项目总览.md through docs/tutorials/09-论文式报告.md
docs/experiments/mixture-ablation.md
docs/experiments/run-manifest-template.json
```

## Task 1: Initialize a testable package and exact model configurations

**Files:**
- Create: `pyproject.toml`, `src/sinogpt/__init__.py`, `src/sinogpt/config.py`, `src/sinogpt/seed.py`
- Create: `configs/tiny_25m.yaml`, `configs/ablation_125m.yaml`, `configs/main_350m.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces `ModelConfig`, `DataConfig`, `TrainConfig`, `load_config(path: str)`, and `seed_everything(seed: int)`.
- `GPTLanguageModel` consumes `ModelConfig`; the trainer consumes all three configs.

- [ ] **Step 1: Write the failing tests**

```python
import pytest
import torch
from sinogpt.config import ModelConfig
from sinogpt.seed import seed_everything


def test_model_config_rejects_non_divisible_heads():
    with pytest.raises(ValueError, match="n_embd must be divisible"):
        ModelConfig(vocab_size=32, n_layer=6, n_head=7, n_embd=384, block_size=512)


def test_seed_repeats_torch_values():
    seed_everything(17)
    first = torch.rand(3)
    seed_everything(17)
    assert torch.equal(first, torch.rand(3))
```

- [ ] **Step 2: Verify tests fail before implementation**

Run: `python -m pytest tests/test_config.py -v`

Expected: `ModuleNotFoundError: No module named 'sinogpt'`.

- [ ] **Step 3: Add package configuration and typed configuration code**

```toml
[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[project]
name = "sinogpt"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["torch>=2.3", "tokenizers>=0.19", "PyYAML>=6.0", "safetensors>=0.4", "tensorboard>=2.16"]

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.6"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

```python
# src/sinogpt/config.py
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")

    @property
    def mlp_size(self) -> int:
        return 4 * self.n_embd


@dataclass(frozen=True)
class DataConfig:
    train_manifest: str
    valid_manifest: str
    tokenizer_path: str
    cache_dir: str


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    max_steps: int
    checkpoint_every: int
    use_bf16: bool
    output_dir: str


def load_config(path: str) -> tuple[ModelConfig, DataConfig, TrainConfig]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ModelConfig(**raw["model"]), DataConfig(**raw["data"]), TrainConfig(**raw["train"])
```

```python
# src/sinogpt/seed.py
import random
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

Use the following exact tiny-model YAML and alter only stated fields for the 125M/350M files:

```yaml
model: {vocab_size: 50000, n_layer: 6, n_head: 6, n_embd: 384, block_size: 512}
data: {train_manifest: data/manifests/train.jsonl, valid_manifest: data/manifests/validation.jsonl, tokenizer_path: artifacts/tokenizer.json, cache_dir: data/cache}
train: {seed: 17, batch_size: 4, gradient_accumulation_steps: 8, learning_rate: 0.0003, max_steps: 1000, checkpoint_every: 250, use_bf16: true, output_dir: artifacts/tiny_25m}
```

- [ ] **Step 4: Verify configuration contract**

Run: `python -m pip install -e ".[dev]" && python -m pytest tests/test_config.py -v && ruff check .`

Expected: two tests pass and ruff reports `All checks passed!`.

- [ ] **Step 5: Commit the scaffold**

Run: `git init && git add pyproject.toml configs src/sinogpt tests/test_config.py && git commit -m "chore: initialize reproducible sinogpt package"`

## Task 2: Enforce dataset provenance and train a BPE tokenizer

**Files:**
- Create: `src/sinogpt/data/__init__.py`, `src/sinogpt/data/manifest.py`, `src/sinogpt/data/pack.py`
- Create: `src/sinogpt/tokenizer.py`, `src/sinogpt/cli/train_tokenizer.py`
- Test: `tests/test_data.py`, `tests/test_tokenizer.py`
- Create: `docs/tutorials/02-数据集与数据治理.md`, `docs/tutorials/03-训练分词器.md`

**Interfaces:**
- Produces `ManifestRecord.from_dict(raw)`, `validate_manifest(path)`, `pack_token_ids(ids, shard_size)`, `train_bpe(texts, vocab_size, output_path)`, and `load_tokenizer(path)`.
- The preparation CLI accepts a manifest only after validation and supplies tokenizer IDs to the dataset layer.

- [ ] **Step 1: Write failing provenance and tokenizer tests**

```python
import pytest
from pathlib import Path
from sinogpt.data.manifest import ManifestRecord
from sinogpt.data.pack import pack_token_ids
from sinogpt.tokenizer import train_bpe, load_tokenizer


def test_manifest_requires_license_note():
    with pytest.raises(ValueError, match="license_note"):
        ManifestRecord.from_dict({"text": "你好", "source": "demo", "revision": "r1", "language": "zh", "split": "train", "document_hash": "a"})


def test_pack_preserves_order():
    assert pack_token_ids([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]


def test_bpe_round_trips_bilingual_text(tmp_path: Path):
    path = tmp_path / "tokenizer.json"
    train_bpe(["你好 world", "world 你好"], 64, path)
    assert load_tokenizer(path).decode(load_tokenizer(path).encode("你好 world").ids) == "你好 world"
```

- [ ] **Step 2: Verify tests fail**

Run: `python -m pytest tests/test_data.py tests/test_tokenizer.py -v`

Expected: imports fail because the data and tokenizer modules do not exist.

- [ ] **Step 3: Implement the manifest, packer and BPE contract**

```python
# src/sinogpt/data/manifest.py
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class ManifestRecord:
    text: str
    source: str
    revision: str
    license_note: str
    language: str
    split: str
    document_hash: str

    @classmethod
    def from_dict(cls, raw: dict[str, str]) -> "ManifestRecord":
        fields = ("text", "source", "revision", "license_note", "language", "split", "document_hash")
        missing = [field for field in fields if not raw.get(field)]
        if missing:
            raise ValueError(", ".join(missing))
        if raw["split"] not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        return cls(**{field: raw[field] for field in fields})


def validate_manifest(path: Path) -> list[ManifestRecord]:
    records = [ManifestRecord.from_dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len({record.document_hash for record in records}) != len(records):
        raise ValueError("duplicate document_hash in manifest")
    return records
```

```python
# src/sinogpt/data/pack.py
def pack_token_ids(ids: list[int], shard_size: int) -> list[list[int]]:
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    return [ids[index:index + shard_size] for index in range(0, len(ids), shard_size)]
```

```python
# src/sinogpt/tokenizer.py
from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def train_bpe(texts: list[str], vocab_size: int, output_path: Path) -> None:
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))


def load_tokenizer(path: Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))
```

The data tutorial documents the immutable manifest fields and the 70/20/10 target mix. The tokenizer tutorial includes BPE theory, special tokens, `python -m sinogpt.cli.train_tokenizer --manifest data/manifests/train.jsonl --output artifacts/tokenizer.json`, expected tokenizer output and bilingual round-trip verification.

- [ ] **Step 4: Verify provenance and tokenization**

Run: `python -m pytest tests/test_data.py tests/test_tokenizer.py -v && ruff check src tests`

Expected: all three tests pass; missing license data and duplicate hashes fail validation.

- [ ] **Step 5: Commit data primitives**

Run: `git add src/sinogpt/data src/sinogpt/tokenizer.py src/sinogpt/cli/train_tokenizer.py tests docs/tutorials && git commit -m "feat: add auditable data and BPE tokenizer"`

## Task 3: Implement causal attention, GELU MLP and GPT decoder

**Files:**
- Create: `src/sinogpt/model/__init__.py`, `src/sinogpt/model/layers.py`, `src/sinogpt/model/gpt.py`
- Test: `tests/test_model.py`
- Create: `docs/tutorials/04-GPT3架构.md`, `docs/tutorials/05-前向传播.md`

**Interfaces:**
- Produces `GPTLanguageModel(config)` and `forward(input_ids, targets=None) -> (logits, loss)`.
- Input is `[B, T]`; logits are `[B, T, V]`; loss is scalar causal cross entropy when targets are supplied.

- [ ] **Step 1: Write failing model tests**

```python
import torch
from sinogpt.config import ModelConfig
from sinogpt.model.gpt import GPTLanguageModel


def build_model():
    return GPTLanguageModel(ModelConfig(vocab_size=32, n_layer=2, n_head=4, n_embd=16, block_size=8))


def test_logits_and_loss_shape():
    logits, loss = build_model()(torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]]))
    assert logits.shape == (1, 3, 32)
    assert loss is not None and loss.ndim == 0


def test_future_tokens_do_not_change_earlier_logits():
    model = build_model().eval()
    with torch.no_grad():
        first, _ = model(torch.tensor([[1, 2, 3, 4]]))
        second, _ = model(torch.tensor([[1, 2, 9, 9]]))
    assert torch.allclose(first[:, :2], second[:, :2], atol=1e-5)
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_model.py -v`

Expected: import failure for `sinogpt.model`.

- [ ] **Step 3: Implement the exact forward formulas**

```python
# src/sinogpt/model/layers.py
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int) -> None:
        super().__init__()
        self.n_head, self.head_dim = n_head, n_embd // n_head
        self.qkv, self.proj = nn.Linear(n_embd, 3 * n_embd), nn.Linear(n_embd, n_embd)

    def forward(self, x: Tensor) -> Tensor:
        batch, time, channels = x.shape
        q, k, v = self.qkv(x).split(channels, dim=-1)
        q = q.view(batch, time, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, time, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, time, self.n_head, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.ones(time, time, device=x.device, dtype=torch.bool).tril()
        weights = F.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
        return self.proj((weights @ v).transpose(1, 2).contiguous().view(batch, time, channels))


class MLP(nn.Module):
    def __init__(self, n_embd: int) -> None:
        super().__init__()
        self.up, self.down = nn.Linear(n_embd, 4 * n_embd), nn.Linear(4 * n_embd, n_embd)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.gelu(self.up(x)))
```

```python
# src/sinogpt/model/gpt.py
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from sinogpt.config import ModelConfig
from sinogpt.model.layers import CausalSelfAttention, MLP


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int) -> None:
        super().__init__()
        self.ln_1, self.attn = nn.LayerNorm(n_embd), CausalSelfAttention(n_embd, n_head)
        self.ln_2, self.mlp = nn.LayerNorm(n_embd), MLP(n_embd)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPTLanguageModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config.n_embd, config.n_head) for _ in range(config.n_layer)])
        self.final_norm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: Tensor, targets: Tensor | None = None):
        _, time = input_ids.shape
        if time > self.config.block_size:
            raise ValueError("input sequence exceeds block_size")
        positions = torch.arange(time, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
```

- [ ] **Step 4: Verify causal and GELU behavior**

Run: `python -m pytest tests/test_model.py -v`

Expected: both tests pass. Add a third assertion that `model.blocks[0].mlp.up.weight.grad` is finite after `loss.backward()`.

- [ ] **Step 5: Commit model and forward-propagation tutorial**

Run: `git add src/sinogpt/model tests/test_model.py docs/tutorials/04-GPT3架构.md docs/tutorials/05-前向传播.md && git commit -m "feat: add causal GELU GPT decoder"`

## Task 4: Implement trainer metrics, checkpoint resume and cloud entry points

**Files:**
- Create: `src/sinogpt/training/__init__.py`, `src/sinogpt/training/trainer.py`, `src/sinogpt/training/checkpoint.py`
- Create: `src/sinogpt/cli/prepare_data.py`, `src/sinogpt/cli/train.py`, `src/sinogpt/cli/sample.py`
- Test: `tests/test_training.py`, `tests/test_cli.py`
- Create: `docs/tutorials/01-云端环境.md`, `docs/tutorials/06-反向传播.md`, `docs/tutorials/07-从25M到350M训练.md`

**Interfaces:**
- Produces `Trainer.train_step(input_ids, targets) -> dict[str, float]`, `save_checkpoint(path, state)`, and `load_checkpoint(path)`.
- CLIs require explicit `--config`; `--resume` is the only way to resume a run.

- [ ] **Step 1: Write failing gradient/checkpoint/CLI tests**

```python
import subprocess
import sys
import torch
from sinogpt.config import ModelConfig
from sinogpt.model.gpt import GPTLanguageModel


def test_backward_populates_gelu_gradient():
    model = GPTLanguageModel(ModelConfig(vocab_size=32, n_layer=1, n_head=4, n_embd=16, block_size=8))
    _, loss = model(torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]]))
    loss.backward()
    assert torch.isfinite(model.blocks[0].mlp.up.weight.grad).all()


def test_train_help_requires_explicit_config_contract():
    result = subprocess.run([sys.executable, "-m", "sinogpt.cli.train", "--help"], capture_output=True, text=True)
    assert result.returncode == 0 and "--config" in result.stdout
```

- [ ] **Step 2: Verify current failure**

Run: `python -m pytest tests/test_training.py tests/test_cli.py -v`

Expected: the CLI import fails and checkpoint/trainer helpers are absent.

- [ ] **Step 3: Add explicit optimizer, safety and checkpoint behavior**

```python
# src/sinogpt/training/trainer.py
import torch
from torch import Tensor, nn


class Trainer:
    def __init__(self, model: nn.Module, learning_rate: float) -> None:
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def train_step(self, input_ids: Tensor, targets: Tensor) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        _, loss = self.model(input_ids, targets)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("non-finite causal language-model loss")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return {"loss": float(loss.detach()), "global_grad_norm": float(grad_norm)}
```

```python
# src/sinogpt/training/checkpoint.py
from pathlib import Path
import torch


def save_checkpoint(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path) -> dict[str, object]:
    return torch.load(path, map_location="cpu", weights_only=False)
```

The train CLI must print the frozen config and manifest revision before creating a model, reject unavailable bf16, write metrics containing loss/grad norm/tokens/s/peak memory, and save model+optimizer+scheduler+RNG+cursor+step at each checkpoint.

- [ ] **Step 4: Verify real backward and resume behavior**

Run: `python -m pytest tests/test_training.py tests/test_cli.py -v && python -m sinogpt.cli.train --help`

Expected: finite GELU gradient; help displays `--config` and `--resume`; checkpoint round trip preserves global step and optimizer state.

- [ ] **Step 5: Commit trainer and cloud controls**

Run: `git add src/sinogpt/training src/sinogpt/cli tests docs/tutorials && git commit -m "feat: add resumable causal language model training"`

## Task 5: Complete the 25M acceptance run and Markdown tutorial evidence

**Files:**
- Create: `docs/tutorials/00-项目总览.md`, `docs/tutorials/08-评测与消融实验.md`, `docs/tutorials/09-论文式报告.md`
- Create: `docs/paper/README.md`, `data/manifests/demo_train.jsonl`, `data/manifests/demo_validation.jsonl`
- Modify: `README.md`

**Interfaces:**
- Consumes tested CLI commands and produces a verified 20-step training record, a resume checkpoint and two deterministic samples.
- Every tutorial chapter has headings `学习目标`, `前置条件`, `原理`, `执行命令`, `预期输出`, `常见故障`, `复现实验证据`, `论文对应章节`.

- [ ] **Step 1: Add the failing acceptance assertion**

```python
import json
from pathlib import Path


def test_acceptance_metrics_require_twenty_finite_loss_rows():
    metrics_path = Path("artifacts/tiny_25m/metrics.jsonl")
    assert metrics_path.exists()
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 20
    assert all(row["loss"] > 0.0 for row in rows)
    assert rows[-1]["global_step"] == 20
```

- [ ] **Step 2: Verify it fails before the run exists**

Run: `python -m pytest tests/test_acceptance.py -v`

Expected: FAIL because `artifacts/tiny_25m/metrics.jsonl` has not been produced.

- [ ] **Step 3: Execute a licensed demo-only run and capture facts**

Run: `python -m sinogpt.cli.train_tokenizer --manifest data/manifests/demo_train.jsonl --output artifacts/tokenizer.json && python -m sinogpt.cli.prepare_data --config configs/tiny_25m.yaml && python -m sinogpt.cli.train --config configs/tiny_25m.yaml && python -m sinogpt.cli.sample --checkpoint artifacts/tiny_25m/checkpoints/step_20.pt --prompt "人工智能" --tokens 32 --seed 17`

Expected: 20 finite loss rows, a checkpoint at step 10 and 20, a successful resume record, and a fixed-seed sample. Fluency is not required at 20 steps.

- [ ] **Step 4: Write tutorials from actual output, not invented results**

Copy the approved Mermaid forward/backward figures into `05-前向传播.md` and `06-反向传播.md`; identify GELU as the MLP activation, identify attention/output Softmax separately, and label LayerNorm/residual as non-activation operations. Put each actual acceptance command and its observed metrics path in the appropriate chapter.

- [ ] **Step 5: Verify docs and commit the runnable lesson**

Run: `python -m pytest -v && ruff check .`

Expected: all tests pass and every tutorial command corresponds to a completed output file.

Run: `git add README.md docs data/manifests tests/test_acceptance.py && git commit -m "docs: add verified tiny model training tutorial"`

## Task 6: Gate the 125M/350M research experiments on immutable provenance

**Files:**
- Create: `src/sinogpt/experiments.py`, `tests/test_experiments.py`
- Create: `docs/experiments/mixture-ablation.md`, `docs/experiments/run-manifest-template.json`
- Modify: `configs/ablation_125m.yaml`, `configs/main_350m.yaml`, `docs/tutorials/07-从25M到350M训练.md`

**Interfaces:**
- Produces `validate_run_manifest(raw: dict[str, object]) -> None`.
- The 125M runs are `zh_70_en_20_code_10`, `zh_100`, and `en_100`, each at 300M tokens; the 350M run is `sino_main_350m` at 1B tokens.

- [ ] **Step 1: Write a failing run-manifest test**

```python
import pytest
from sinogpt.experiments import validate_run_manifest


def test_run_manifest_requires_pinned_provenance_and_archive_uri():
    raw = {"run_name": "zh_100", "config_path": "configs/ablation_125m.yaml", "tokenizer_sha256": "", "dataset_manifest_sha256": "", "seed": 17, "target_tokens": 300_000_000, "checkpoint_uri": ""}
    with pytest.raises(ValueError, match="tokenizer_sha256"):
        validate_run_manifest(raw)
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_experiments.py -v`

Expected: import failure because the experiment gate does not yet exist.

- [ ] **Step 3: Implement the no-provenance/no-run gate**

```python
def validate_run_manifest(raw: dict[str, object]) -> None:
    for key in ("run_name", "config_path", "tokenizer_sha256", "dataset_manifest_sha256", "checkpoint_uri"):
        if not isinstance(raw.get(key), str) or not raw[key]:
            raise ValueError(f"{key} must not be empty")
    if not isinstance(raw.get("target_tokens"), int) or raw["target_tokens"] < 1:
        raise ValueError("target_tokens must be a positive integer")
```

`mixture-ablation.md` fixes the sole variable (language mixture), validation splits, fixed hyperparameters, storage URI requirement, stop conditions, metrics, failure reporting and the no-SOTA claim language.

- [ ] **Step 4: Verify expensive runs remain blocked until data/storage are pinned**

Run: `python -m pytest tests/test_experiments.py -v && ruff check .`

Expected: blank checksums and missing checkpoint URI fail; fully populated manifests pass.

- [ ] **Step 5: Commit the research gate**

Run: `git add src/sinogpt/experiments.py tests/test_experiments.py configs docs/experiments docs/tutorials/07-从25M到350M训练.md && git commit -m "feat: gate research runs on immutable provenance"`

## Plan Self-Review

| Specification requirement | Plan coverage |
|---|---|
| 25M teaching loop and tutorial evidence | Tasks 1–5 |
| Pre-LN causal decoder, GELU, Softmax and residuals | Task 3 |
| Backward gradients, AdamW and resumable checkpoints | Task 4 |
| Manifest-led licensing, hash and split separation | Task 2 |
| 100GB cloud storage and bf16 constraints | Tasks 1, 4 and 6 |
| 125M controlled mixture study and 350M main run | Task 6 |
| Markdown tutorial and paper correspondence | Tasks 2–5 |

No placeholder markers, ambiguous interfaces or unowned specification requirements remain. The 350M job is deliberately prepared but not launched until exact dataset revisions and object-storage URI are supplied; this prevents an expensive untraceable run.

## Execution Handoff

Plan saved to `docs/superpowers/plans/2026-08-31-sinogpt-foundation.md`.

1. **Subagent-Driven** — fresh worker per task and review gates.
2. **Inline Execution** — execute tasks in this session, task by task, with checkpoints.
