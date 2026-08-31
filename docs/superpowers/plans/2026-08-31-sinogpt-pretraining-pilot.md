# SinoGPT-30M 中文预训练先导实验 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (\`- [ ]\`) syntax for tracking.

**Goal:** 为单张 RTX 4090 建立可审计的中文语料导入、固定聊天控制词表和约 20M token 的 SinoGPT-30M 预训练先导实验。

**Architecture:** 数据导出模块以流式 \`datasets\` 迭代器读取显式指定的 Hugging Face 数据集切片，过滤空/过短/过长文本并按内容 SHA-256 去重。它将每条合格记录写为现有 \`ManifestRecord\` 格式的 train 或 validation JSONL。Tokenizer 在任何正式训练之前固定聊天控制 token；现有 prepare/train CLI 保持不变，使用两个新 YAML 配置完成 100-step 冒烟和 1,250-step 连续训练。

**Tech Stack:** Python 3.12、PyTorch、Hugging Face \`datasets\`、Hugging Face \`tokenizers\`、PyYAML、pytest、ruff。

## Global Constraints

- 先导数据仅用于学习与论文实验，导出记录必须保留数据集 ID、解析后的 immutable revision、ODC-By 许可证说明和内容哈希；这不是商用语料批准。
- 仅在训练 tokenizer 前加入 \`<|system|>\`、\`<|user|>\`、\`<|assistant|>\`；已有 tokenizer 或 checkpoint 不得改写。
- 使用现有 6 层、384 hidden、6 heads、512 context 的约 30M decoder-only 模型。
- 先运行 100 步（1,638,400 token）确认 GPU 训练，再从其 checkpoint 恢复至 1,250 步（20,480,000 token）。
- 数据导出与缓存不进入 Git；\`.gitignore\` 继续覆盖 \`artifacts/\` 和 \`data/cache/\`。
- 所有新增 Python 模块包含简短中文模块用途注释；新增功能先写失败测试。

---

## Planned File Structure

\`\`\`text
pyproject.toml                                      # 新增可选 data 依赖
src/sinogpt/tokenizer.py                            # 固定聊天控制 token
src/sinogpt/data/export.py                          # 流式记录过滤、哈希、切分和 JSONL 导出
src/sinogpt/cli/export_hf_dataset.py                # 指定 HF 数据集与 revision 的导出入口
configs/tiny_30m_pilot_stage1.yaml                  # 100 step GPU 冒烟配置
configs/tiny_30m_pilot.yaml                         # 1,250 step 预训练配置
tests/test_tokenizer.py                             # 聊天 token 稳定性
tests/test_data_export.py                            # 不依赖网络的导出契约
docs/tutorials/10-中文预训练先导实验.md               # 云端逐条命令、日志与停止条件
README.md                                           # 连接新教程
\`\`\`

### Task 1: 固定聊天特殊 token

**Files:**
- Modify: \`src/sinogpt/tokenizer.py\`
- Modify: \`tests/test_tokenizer.py\`

**Interfaces:**
- Produces: \`SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", "<|system|>", "<|user|>", "<|assistant|>"]\`.
- Consumes: existing \`train_bpe(texts, vocab_size, output_path)\` and \`load_tokenizer(path)\`.

- [ ] **Step 1: Write the failing tokenizer contract test**

\`\`\`python
from sinogpt.tokenizer import SPECIAL_TOKENS, load_tokenizer, train_bpe


def test_bpe_reserves_chat_control_tokens(tmp_path: Path) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    train_bpe(["你好"], vocab_size=64, output_path=tokenizer_path)
    tokenizer = load_tokenizer(tokenizer_path)
    assert SPECIAL_TOKENS[-3:] == ["<|system|>", "<|user|>", "<|assistant|>"]
    assert [tokenizer.token_to_id(token) for token in SPECIAL_TOKENS] == list(range(len(SPECIAL_TOKENS)))
\`\`\`

- [ ] **Step 2: Run test to verify it fails**

Run: \`python -m pytest tests/test_tokenizer.py::test_bpe_reserves_chat_control_tokens -v\`

Expected: FAIL because the three chat tokens are absent from \`SPECIAL_TOKENS\`.

- [ ] **Step 3: Write minimal implementation**

\`\`\`python
SPECIAL_TOKENS = [
    "<pad>", "<bos>", "<eos>", "<unk>",
    "<|system|>", "<|user|>", "<|assistant|>",
]
\`\`\`

No tokenizer file is migrated. The new list only affects tokenizers trained after this commit.

- [ ] **Step 4: Run test to verify it passes**

Run: \`python -m pytest tests/test_tokenizer.py -v && ruff check src/sinogpt/tokenizer.py tests/test_tokenizer.py\`

Expected: both tokenizer tests pass and ruff reports no violations.

- [ ] **Step 5: Commit**

\`\`\`bash
git add src/sinogpt/tokenizer.py tests/test_tokenizer.py
git commit -m "feat: reserve chat tokens before pretraining"
\`\`\`

### Task 2: 导出可审计的流式中文 manifest

**Files:**
- Modify: \`pyproject.toml\`
- Create: \`src/sinogpt/data/export.py\`
- Create: \`src/sinogpt/cli/export_hf_dataset.py\`
- Create: \`tests/test_data_export.py\`

**Interfaces:**
- Produces: \`ExportStats\`, \`export_records(records, train_output, validation_output, source, revision, license_note, language, minimum_characters, maximum_characters, validation_percent, limit) -> ExportStats\`.
- CLI consumes \`--dataset\`, \`--subset\`, \`--revision\`, \`--license-note\`, \`--train-output\`, \`--validation-output\`, \`--limit\`, \`--min-chars\`, \`--max-chars\`, \`--validation-percent\` and obtains the immutable resolved revision before export.

- [ ] **Step 1: Write a failing, network-free export test**

\`\`\`python
import json
from sinogpt.data.export import export_records


def test_export_records_filters_duplicates_and_creates_disjoint_splits(tmp_path: Path) -> None:
    records = [{"text": "短"}, {"text": "甲" * 12}, {"text": "甲" * 12}, {"text": "乙" * 12}]
    stats = export_records(
        records, tmp_path / "train.jsonl", tmp_path / "validation.jsonl",
        source="owner/dataset/subset", revision="a" * 40, license_note="ODC-By 1.0",
        language="zh", minimum_characters=10, maximum_characters=20,
        validation_percent=50, limit=2,
    )
    rows = [
        json.loads(line)
        for path in (tmp_path / "train.jsonl", tmp_path / "validation.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert stats.exported == 2
    assert stats.too_short == 1
    assert stats.duplicates == 1
    assert {row["split"] for row in rows} <= {"train", "validation"}
    assert len({row["document_hash"] for row in rows}) == 2
\`\`\`

- [ ] **Step 2: Run test to verify it fails**

Run: \`python -m pytest tests/test_data_export.py -v\`

Expected: FAIL with \`ModuleNotFoundError: No module named 'sinogpt.data.export'\`.

- [ ] **Step 3: Write minimal implementation**

\`src/sinogpt/data/export.py\` contains a Chinese module-purpose comment, a frozen \`ExportStats\` dataclass with \`seen\`, \`exported\`, \`too_short\`, \`too_long\`, \`duplicates\`, \`train\`, \`validation\`, and these helpers:

\`\`\`python
def split_for_hash(document_hash: str, validation_percent: int) -> str:
    return "validation" if int(document_hash[:8], 16) % 100 < validation_percent else "train"


def normalized_text(raw: object) -> str:
    return " ".join(str(raw).split())
\`\`\`

For each mapping, read only \`raw["text"]\`, normalize whitespace, reject text shorter than \`minimum_characters\` or longer than \`maximum_characters\`, calculate \`sha256(text.encode("utf-8")).hexdigest()\`, reject repeats, deterministically assign a split, and write exactly the seven existing manifest fields. \`limit\` counts accepted unique records across both splits. Reject invalid ranges: \`0 < validation_percent < 100\`, positive minimum and limit, and \`maximum_characters >= minimum_characters\`.

Add the optional dependency:

\`\`\`toml
[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.6"]
data = ["datasets>=3.0", "huggingface_hub>=0.26"]
\`\`\`

The thin CLI resolves \`HfApi().dataset_info(args.dataset, revision=args.revision).sha\`, then calls:

\`\`\`python
records = load_dataset(
    args.dataset, args.subset, split="train",
    streaming=True, revision=resolved_revision,
)
\`\`\`

It prints one JSON object containing the source, resolved revision, stats and output paths. It must not silently fall back from a failed revision, missing subset or unavailable \`text\` field.

- [ ] **Step 4: Run test to verify it passes**

Run: \`python -m pytest tests/test_data_export.py tests/test_data.py -v && python -m sinogpt.cli.export_hf_dataset --help && ruff check src tests\`

Expected: export test passes; help lists \`--dataset\`, \`--subset\`, \`--revision\`, \`--train-output\`, and \`--validation-output\`.

- [ ] **Step 5: Commit**

\`\`\`bash
git add pyproject.toml src/sinogpt/data/export.py src/sinogpt/cli/export_hf_dataset.py tests/test_data_export.py
git commit -m "feat: export auditable streaming datasets"
\`\`\`

### Task 3: 添加可恢复的 30M GPU 先导配置

**Files:**
- Create: \`configs/tiny_30m_pilot_stage1.yaml\`
- Create: \`configs/tiny_30m_pilot.yaml\`
- Modify: \`tests/test_config.py\`

**Interfaces:**
- \`tiny_30m_pilot_stage1.yaml\`: \`max_steps: 100\`, \`checkpoint_every: 100\`, \`output_dir: artifacts/tiny_30m_pilot\`.
- \`tiny_30m_pilot.yaml\`: \`max_steps: 1250\`, \`checkpoint_every: 250\`, same model/data/output fields, so it resumes from \`step_100.pt\`.

- [ ] **Step 1: Write the failing config test**

\`\`\`python
from sinogpt.config import load_config


def test_pilot_configs_share_model_and_data_but_extend_training() -> None:
    stage_model, stage_data, stage_train = load_config("configs/tiny_30m_pilot_stage1.yaml")
    full_model, full_data, full_train = load_config("configs/tiny_30m_pilot.yaml")
    assert (stage_model, stage_data) == (full_model, full_data)
    assert (stage_train.max_steps, stage_train.checkpoint_every) == (100, 100)
    assert (full_train.max_steps, full_train.checkpoint_every) == (1250, 250)
\`\`\`

- [ ] **Step 2: Run test to verify it fails**

Run: \`python -m pytest tests/test_config.py::test_pilot_configs_share_model_and_data_but_extend_training -v\`

Expected: FAIL with \`FileNotFoundError\` because the YAML files do not exist.

- [ ] **Step 3: Write minimal configuration**

Both files use the existing architecture. Their shared data section is:

\`\`\`yaml
data:
  train_manifest: data/manifests/pilot_train.jsonl
  valid_manifest: data/manifests/pilot_validation.jsonl
  tokenizer_path: artifacts/tiny_30m_pilot/tokenizer.json
  cache_dir: data/cache/tiny_30m_pilot
\`\`\`

Their shared train values are \`seed: 17\`, \`batch_size: 4\`, \`gradient_accumulation_steps: 8\`, \`learning_rate: 0.0003\`, and \`use_bf16: true\`. The model is \`vocab_size: 50000\`, \`n_layer: 6\`, \`n_head: 6\`, \`n_embd: 384\`, \`block_size: 512\`.

- [ ] **Step 4: Run test to verify it passes**

Run: \`python -m pytest tests/test_config.py -v && ruff check .\`

Expected: all config tests pass; effective tokens per optimizer update are \`4 × 8 × 512 = 16,384\`.

- [ ] **Step 5: Commit**

\`\`\`bash
git add configs/tiny_30m_pilot_stage1.yaml configs/tiny_30m_pilot.yaml tests/test_config.py
git commit -m "feat: add 30m chinese pretraining pilot configs"
\`\`\`

### Task 4: 写入云端训练教程并做本地验收

**Files:**
- Create: \`docs/tutorials/10-中文预训练先导实验.md\`
- Modify: \`README.md\`
- Modify: \`tests/test_cli.py\`

**Interfaces:**
- The tutorial supplies one-command-at-a-time cloud operations and treats the generated manifests, tokenizer, cache, checkpoints and \`metrics.jsonl\` as evidence.

- [ ] **Step 1: Write a failing documentation-presence test**

\`\`\`python
from pathlib import Path


def test_pretraining_pilot_tutorial_lists_required_commands() -> None:
    text = Path("docs/tutorials/10-中文预训练先导实验.md").read_text(encoding="utf-8")
    assert "sinogpt.cli.export_hf_dataset" in text
    assert "sinogpt.cli.train_tokenizer" in text
    assert "sinogpt.cli.prepare_data" in text
    assert "--resume" in text
\`\`\`

- [ ] **Step 2: Run test to verify it fails**

Run: \`python -m pytest tests/test_cli.py::test_pretraining_pilot_tutorial_lists_required_commands -v\`

Expected: FAIL because the test and tutorial do not exist.

- [ ] **Step 3: Write the tutorial**

Use all standard tutorial headings and this command order:

\`\`\`bash
python -m pip install -e ".[dev,data]"
python -m sinogpt.cli.export_hf_dataset ...
python -m sinogpt.cli.train_tokenizer ...
python -m sinogpt.cli.prepare_data --config configs/tiny_30m_pilot_stage1.yaml
python -m sinogpt.cli.train --config configs/tiny_30m_pilot_stage1.yaml --device cuda
python -m sinogpt.cli.train --config configs/tiny_30m_pilot.yaml --resume artifacts/tiny_30m_pilot/checkpoints/step_100.pt --device cuda
\`\`\`

Use \`agentlans/fineweb2-chinese\`, \`MAINLAND_CHINA\`, \`main\`, an ODC-By research notice, \`--limit 100000\`, \`--min-chars 200\`, \`--max-chars 12000\`, and \`--validation-percent 1\`. Explain that the exporter resolves \`main\` to a commit SHA and that the printed JSON must be retained. Include stop conditions: non-finite loss, CUDA out-of-memory, missing train/validation sequences, loss not decreasing over a comparable window, and cloud storage below 15GB. State that chat UI/SFT begins only after a full checkpoint exists.

- [ ] **Step 4: Run regression suite**

Run: \`python -m pytest -q && ruff check .\`

Expected: all existing tests remain green plus the new tokenizer, exporter, config and tutorial checks.

- [ ] **Step 5: Commit**

\`\`\`bash
git add docs/tutorials/10-中文预训练先导实验.md README.md tests/test_cli.py
git commit -m "docs: add chinese pretraining pilot guide"
\`\`\`

## Plan Self-Review

| Design requirement | Plan task |
|---|---|
| Tokenizer fixed before formal training with chat controls | Task 1 |
| Traceable Chinese source, immutable revision, JSONL manifest | Task 2 |
| 30M model, 100-step smoke then 1,250-step resume | Task 3 |
| Exact cloud route and measurable stop conditions | Task 4 |
| Chat UI and SFT deferred until first real checkpoint | Scope boundary |

Placeholder scan found no unresolved markers or undefined interfaces. The test referenced in Task 4 is added to the existing \`tests/test_cli.py\`.

## Execution Handoff

Plan saved to \`docs/superpowers/plans/2026-08-31-sinogpt-pretraining-pilot.md\`. The user asked to begin training first, so execute inline in this session; do not delegate work to subagents.
