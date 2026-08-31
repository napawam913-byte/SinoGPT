# COIG SFT Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Starting from the completed 30M `step_1250.pt` pretraining checkpoint, build an auditable COIG 5,000-pair supervised fine-tuning pipeline that optimizes only assistant-answer tokens and reports validation and held-out test metrics.

**Architecture:** A new SFT data boundary will export normalized `system/user/assistant` JSONL records from `BAAI/COIG`, retain the resolved Hugging Face revision and license note, and create fixed-size input/label tensors without packing conversations together. A dedicated SFT trainer will load the pretraining checkpoint, select `best.pt` only by validation loss after each epoch, and a separate evaluator will use the test split only after training selection is complete. A small chat-sampling CLI will apply the same template for qualitative pretraining-versus-SFT comparison; the Gradio UI is deliberately deferred.

**Tech Stack:** Python 3.12, PyTorch 2.3+, Hugging Face `datasets` and `huggingface_hub`, `tokenizers`, PyYAML, pytest, Ruff.

## Global Constraints

- Keep the existing 50,000-token BPE tokenizer and 30M `ModelConfig` unchanged; start SFT from `artifacts/tiny_30m_pilot/checkpoints/step_1250.pt`, never from new random weights.
- The initial data source is `BAAI/COIG` only, with a fixed 5,000 unique question-answer pairs split exactly into 4,000 train, 500 validation, and 500 test records.
- Treat COIG as a research/paper pilot only. Persist the source revision and an explicit mixed-license/fair-use warning; do not claim commercial clearance.
- Encode every record with `<|system|>`, `<|user|>`, `<|assistant|>`, and `<eos>`; labels for all non-assistant-answer targets and padding must be `-100`.
- Use validation loss for `best.pt`; never load or report the test split from the training selection loop.
- Keep raw JSONL, tensor caches, checkpoints, metrics, and generated outputs out of Git. Add `data/manifests/coig_sft_*.jsonl` to `.gitignore`.
- Do not add Gradio, RAG, RLHF, reward models, distributed training, or a new tokenizer in this plan.

---

## File Structure

| Path | Responsibility |
|---|---|
| `src/sinogpt/data/sft.py` | SFT record validation, COIG question-answer extraction, deterministic fixed-count export, template encoding, and tensor-cache I/O. |
| `src/sinogpt/cli/export_hf_sft_dataset.py` | Resolve the immutable COIG revision, stream the source split, and print auditable export statistics. |
| `src/sinogpt/cli/prepare_sft_data.py` | Convert the three SFT JSONL manifests into input/label tensors using the frozen pretraining tokenizer. |
| `src/sinogpt/training/evaluation.py` | Compute token-weighted cross-entropy, perplexity, and supervised-token counts without gradients. |
| `src/sinogpt/cli/train_sft.py` | Load the pretraining checkpoint, run up to three epochs, validate every epoch, and save epoch plus best checkpoints. |
| `src/sinogpt/cli/evaluate_sft.py` | Evaluate an already selected checkpoint on exactly one requested prepared split. |
| `src/sinogpt/inference.py` | Encode the chat prompt and perform reproducible top-k autoregressive generation. |
| `src/sinogpt/cli/sample_chat.py` | Load a checkpoint and print only the generated assistant response for a user question. |
| `configs/tiny_30m_coig_sft.yaml` | Frozen model/data/training settings for the first COIG SFT run. |
| `tests/test_sft_data.py` | Unit tests for provenance, extraction, split counts, deduplication, template IDs, masks, and cache round trips. |
| `tests/test_evaluation.py` | Unit tests for masked-token metric accounting and perplexity. |
| `tests/test_sft_training.py` | CPU tests for base-checkpoint compatibility, validation-only checkpoint selection, and resume metadata. |
| `tests/test_inference.py` | Unit tests for chat formatting and deterministic bounded generation. |
| `docs/tutorials/11-COIG监督微调.md` | Cloud commands, expected artifacts, evaluation interpretation, and failure handling. |
| `docs/paper/README.md` | Reproducibility fields and result-table template for the SFT experiment. |

### Task 1: Auditable COIG SFT records and exporter

**Files:**
- Create: `src/sinogpt/data/sft.py`
- Create: `src/sinogpt/cli/export_hf_sft_dataset.py`
- Create: `tests/test_sft_data.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: streamed mappings with COIG `conversations` entries containing `question` and `answer`; a resolved Hugging Face commit SHA.
- Produces: `SFTRecord`, `SFTExportStats`, `export_coig_records(...)`, and three JSONL manifests that later tasks load.

- [ ] **Step 1: Write failing data-export tests**

```python
def test_export_coig_records_deduplicates_and_uses_exact_split_counts(tmp_path: Path) -> None:
    source_rows = [
        {"conversations": [{"question": f"问题{i}", "answer": f"回答{i}"}]}
        for i in range(5)
    ] + [{"conversations": [{"question": "问题1", "answer": "回答1"}]}]
    stats = export_coig_records(
        source_rows, tmp_path, source="BAAI/COIG", revision="a" * 40,
        license_note="research pilot", system_prompt=DEFAULT_SYSTEM_PROMPT,
        limit=5, train_count=3, validation_count=1, test_count=1,
    )
    assert (stats.exported, stats.duplicates, stats.train, stats.validation, stats.test) == (5, 1, 3, 1, 1)
    assert {record.split for record in load_sft_records(tmp_path / "coig_sft_train.jsonl")} == {"train"}


def test_export_rejects_a_nonempty_answer_that_is_not_a_string(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="answer"):
        export_coig_records(
            [{"conversations": [{"question": "问题", "answer": ["不是字符串"]}]}],
            tmp_path, source="BAAI/COIG", revision="a" * 40, license_note="research pilot",
            system_prompt=DEFAULT_SYSTEM_PROMPT, limit=1, train_count=1,
            validation_count=0, test_count=0,
        )
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `python -m pytest tests/test_sft_data.py -q`

Expected: FAIL during collection because `sinogpt.data.sft` does not exist.

- [ ] **Step 3: Implement the normalized record contract and deterministic exporter**

```python
DEFAULT_SYSTEM_PROMPT = "你是一个简洁、诚实的中文助手。"

@dataclass(frozen=True)
class SFTRecord:
    system: str
    user: str
    assistant: str
    source: str
    revision: str
    license_note: str
    language: str
    split: str
    record_hash: str


def canonical_record_hash(system: str, user: str, assistant: str) -> str:
    payload = json.dumps(
        {"system": system, "user": user, "assistant": assistant},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def extract_coig_pairs(raw_row: Mapping[str, object]) -> Iterable[tuple[str, str]]:
    conversations = raw_row.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError("conversations must be a list")
    for turn in conversations:
        if not isinstance(turn, Mapping):
            raise ValueError("conversation turn must be a mapping")
        question = normalized_sft_text(turn.get("question"), "question")
        answer = normalized_sft_text(turn.get("answer"), "answer")
        if question and answer:
            yield question, answer
```

Implement `export_coig_records` to stream rows in the frozen source order, count skipped malformed/empty turns and duplicates, stop after exactly `limit` unique pairs, and assign the first `train_count`, next `validation_count`, and final `test_count` exported records to their named JSONL files. Reject counts whose sum differs from `limit`, empty source/revision/license fields, or a revision that is not a 40-character SHA. Serialize with `ensure_ascii=False`; `load_sft_records` must require every provenance field and only accept the three declared splits.

- [ ] **Step 4: Implement the CLI with immutable revision resolution**

```python
parser.add_argument("--dataset", default="BAAI/COIG")
parser.add_argument("--revision", required=True)
parser.add_argument("--output-dir", required=True, type=Path)
parser.add_argument("--limit", type=int, default=5_000)
parser.add_argument("--train-count", type=int, default=4_000)
parser.add_argument("--validation-count", type=int, default=500)
parser.add_argument("--test-count", type=int, default=500)
parser.add_argument("--license-note", required=True)
```

Use `HfApi().dataset_info(args.dataset, revision=args.revision).sha`; fail if it is empty. Call `load_dataset(args.dataset, split="train", streaming=True, revision=resolved_revision)`, pass the iterable to `export_coig_records`, then print one JSON object containing `source`, `resolved_revision`, each `SFTExportStats` counter, and the three output paths. Convert a missing optional `datasets` dependency into `RuntimeError('请先执行 python -m pip install -e ".[data]"')`.

- [ ] **Step 5: Ignore generated manifests and run focused tests**

Add this line under the existing pilot-manifest rule:

```gitignore
data/manifests/coig_sft_*.jsonl
```

Run: `python -m pytest tests/test_sft_data.py -q`

Expected: PASS with tests covering valid export, duplicate handling, exact 4,000/500/500 boundary logic on a small fixture, malformed turn rejection, and provenance round trip.

- [ ] **Step 6: Commit the audited exporter**

```bash
git add .gitignore src/sinogpt/data/sft.py src/sinogpt/cli/export_hf_sft_dataset.py tests/test_sft_data.py
git commit -m "feat: export auditable COIG SFT records"
```

### Task 2: Assistant-only labels and prepared SFT tensor caches

**Files:**
- Modify: `src/sinogpt/data/sft.py`
- Create: `src/sinogpt/cli/prepare_sft_data.py`
- Modify: `tests/test_sft_data.py`

**Interfaces:**
- Consumes: `list[SFTRecord]`, a frozen `tokenizers.Tokenizer`, and `block_size`.
- Produces: `encode_sft_record(...) -> tuple[Tensor, Tensor]`, `save_prepared_sft_split(...)`, and cache paths `<cache_dir>/<split>_input_ids.pt` plus `<cache_dir>/<split>_labels.pt`.

- [ ] **Step 1: Write failing template and masking tests**

```python
def test_encode_sft_record_masks_everything_before_answer_and_keeps_answer_eos(tokenizer: Tokenizer) -> None:
    record = SFTRecord(system="系统", user="问题", assistant="答案", **provenance("train"))
    input_ids, labels = encode_sft_record(record, tokenizer, block_size=32)
    assistant_id = tokenizer.token_to_id("<|assistant|>")
    answer_id = tokenizer.encode("答案").ids[0]
    marker_index = input_ids.tolist().index(assistant_id)
    assert torch.equal(labels[:marker_index], torch.full_like(labels[:marker_index], -100))
    assert labels[marker_index] == answer_id
    assert labels[(labels != -100).nonzero()[-1]].item() == tokenizer.token_to_id("<eos>")


def test_encode_sft_record_rejects_sequences_longer_than_context(tokenizer: Tokenizer) -> None:
    record = SFTRecord(system="系统", user="很长" * 200, assistant="回答", **provenance("train"))
    with pytest.raises(ValueError, match="block_size"):
        encode_sft_record(record, tokenizer, block_size=8)
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `python -m pytest tests/test_sft_data.py -q`

Expected: FAIL because `encode_sft_record` and prepared-cache functions are absent.

- [ ] **Step 3: Implement a non-packed, right-padded encoder**

```python
def chat_token_ids(record: SFTRecord, tokenizer: Tokenizer) -> tuple[list[int], int]:
    text = (
        f"<|system|>{record.system}<eos>"
        f"<|user|>{record.user}<eos>"
        f"<|assistant|>{record.assistant}<eos>"
    )
    ids = tokenizer.encode(text).ids
    assistant_id = tokenizer.token_to_id("<|assistant|>")
    if assistant_id is None:
        raise ValueError("tokenizer must contain <|assistant|>")
    return ids, ids.index(assistant_id)


def encode_sft_record(record: SFTRecord, tokenizer: Tokenizer, block_size: int) -> tuple[Tensor, Tensor]:
    ids, assistant_index = chat_token_ids(record, tokenizer)
    if len(ids) < 2 or len(ids) > block_size + 1:
        raise ValueError("SFT record does not fit block_size")
    input_ids = torch.full((block_size,), tokenizer.token_to_id("<pad>"), dtype=torch.long)
    labels = torch.full((block_size,), -100, dtype=torch.long)
    input_ids[: len(ids) - 1] = torch.tensor(ids[:-1])
    targets = torch.tensor(ids[1:])
    labels[assistant_index : len(ids) - 1] = targets[assistant_index:]
    return input_ids, labels
```

Do not concatenate or pack records: every stored row remains one complete dialogue. Validate that the tokenizer has every required special token and that every record contributes at least one supervised answer token. Save rank-2 `long` tensors separately for input IDs and labels; load them with `weights_only=True`, verify equal shapes `[N, block_size]`, and reject an empty split.

- [ ] **Step 4: Implement the preparation CLI**

```python
parser.add_argument("--config", required=True, type=Path)
```

Load `SFTDataConfig` from the YAML introduced in Task 4, load each `train`, `validation`, and `test` manifest through `load_sft_records`, encode with `load_tokenizer(Path(config.tokenizer_path))`, save the three cache pairs, and print exactly:

```text
prepared train_sequences=<n> validation_sequences=<n> test_sequences=<n>
```

- [ ] **Step 5: Run data tests**

Run: `python -m pytest tests/test_sft_data.py -q`

Expected: PASS, including assistant-only labels, answer EOS supervision, `-100` padding, oversized-record rejection, and cache round trip.

- [ ] **Step 6: Commit prepared SFT data support**

```bash
git add src/sinogpt/data/sft.py src/sinogpt/cli/prepare_sft_data.py tests/test_sft_data.py
git commit -m "feat: prepare assistant-only SFT batches"
```

### Task 3: Reusable masked-loss evaluation

**Files:**
- Create: `src/sinogpt/training/evaluation.py`
- Create: `tests/test_evaluation.py`
- Create: `src/sinogpt/cli/evaluate_sft.py`

**Interfaces:**
- Consumes: `nn.Module`, rank-2 `input_ids`, rank-2 labels containing token IDs or `-100`, and an evaluation batch size.
- Produces: `EvaluationResult(loss: float, perplexity: float, supervised_tokens: int)` and one JSON result from the CLI.

- [ ] **Step 1: Write failing evaluator tests**

```python
def test_evaluate_causal_lm_counts_only_nonignored_labels() -> None:
    model = FixedLogitModel(vocab_size=3)
    result = evaluate_causal_lm(
        model,
        torch.tensor([[0, 1], [0, 1]]),
        torch.tensor([[1, -100], [1, 2]]),
        batch_size=1,
    )
    assert result.supervised_tokens == 3
    assert result.loss > 0.0
    assert result.perplexity == pytest.approx(math.exp(result.loss))
```

- [ ] **Step 2: Run the test and verify failure**

Run: `python -m pytest tests/test_evaluation.py -q`

Expected: FAIL because `sinogpt.training.evaluation` does not exist.

- [ ] **Step 3: Implement token-weighted evaluation without gradients**

```python
@dataclass(frozen=True)
class EvaluationResult:
    loss: float
    perplexity: float
    supervised_tokens: int


@torch.inference_mode()
def evaluate_causal_lm(model: nn.Module, input_ids: Tensor, labels: Tensor, batch_size: int) -> EvaluationResult:
    total_nll = 0.0
    total_tokens = 0
    was_training = model.training
    model.eval()
    for ids, target in batched(input_ids, labels, batch_size):
        logits, _ = model(ids.to(device))
        total_nll += float(F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), target.to(device).reshape(-1),
            ignore_index=-100, reduction="sum",
        ))
        total_tokens += int((target != -100).sum())
    model.train(was_training)
    if total_tokens == 0:
        raise ValueError("evaluation split has no supervised tokens")
    loss = total_nll / total_tokens
    return EvaluationResult(loss=loss, perplexity=math.exp(loss), supervised_tokens=total_tokens)
```

The function must reject non-rank-2 or shape-mismatched tensors and nonpositive batch sizes. It must restore the model's original training/eval state even when invoked from training code.

- [ ] **Step 4: Implement an explicit test-split CLI**

```python
parser.add_argument("--checkpoint", required=True, type=Path)
parser.add_argument("--config", required=True, type=Path)
parser.add_argument("--split", required=True, choices=("validation", "test"))
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
```

Load a checkpoint only after checking its stored `model_config` equals the YAML model config, load `<split>_input_ids.pt` and `<split>_labels.pt`, call `evaluate_causal_lm`, and print JSON with `checkpoint`, `split`, `loss`, `perplexity`, and `supervised_tokens`. The training CLI will call the library function directly for validation; it must not invoke this test CLI.

- [ ] **Step 5: Run evaluator tests and commit**

Run: `python -m pytest tests/test_evaluation.py -q`

Expected: PASS with masked-token accounting and training-mode restoration covered.

```bash
git add src/sinogpt/training/evaluation.py src/sinogpt/cli/evaluate_sft.py tests/test_evaluation.py
git commit -m "feat: add masked SFT evaluation"
```

### Task 4: Checkpoint-based SFT training and best-validation selection

**Files:**
- Modify: `src/sinogpt/config.py`
- Create: `configs/tiny_30m_coig_sft.yaml`
- Create: `src/sinogpt/cli/train_sft.py`
- Create: `tests/test_sft_training.py`

**Interfaces:**
- Consumes: `load_sft_config(path) -> tuple[ModelConfig, SFTDataConfig, SFTTrainConfig]`, prepared train and validation tensors, and a compatible pretraining checkpoint.
- Produces: `epoch_001.pt` through `epoch_003.pt`, optional `best.pt`, `metrics.jsonl`, and resume checkpoints that state `completed_epoch` and `best_validation_loss`.

- [ ] **Step 1: Write failing configuration and selection tests**

```python
def test_sft_config_requires_exact_three_way_counts(tmp_path: Path) -> None:
    path = write_yaml(tmp_path, train_count=4_000, validation_count=500, test_count=499)
    with pytest.raises(ValueError, match="must sum to limit"):
        load_sft_config(path)


def test_best_checkpoint_changes_only_when_validation_loss_improves(tmp_path: Path) -> None:
    selector = BestCheckpointSelector(tmp_path)
    assert selector.consider(epoch=1, validation_loss=2.0, state={"global_step": 1})
    assert not selector.consider(epoch=2, validation_loss=2.1, state={"global_step": 2})
    assert load_checkpoint(tmp_path / "best.pt")["global_step"] == 1
```

- [ ] **Step 2: Run the training tests and verify failure**

Run: `python -m pytest tests/test_sft_training.py -q`

Expected: FAIL because SFT configuration and checkpoint selector do not exist.

- [ ] **Step 3: Add frozen SFT configuration types and YAML**

```python
@dataclass(frozen=True)
class SFTDataConfig:
    train_manifest: str
    validation_manifest: str
    test_manifest: str
    tokenizer_path: str
    cache_dir: str


@dataclass(frozen=True)
class SFTTrainConfig:
    seed: int
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    max_epochs: int
    checkpoint_every_epoch: int
    use_bf16: bool
    base_checkpoint: str
    output_dir: str
```

Validate positive batch/accumulation/learning-rate/epochs, require `checkpoint_every_epoch == 1` for this pilot, and load this YAML exactly:

```yaml
model:
  vocab_size: 50000
  n_layer: 6
  n_head: 6
  n_embd: 384
  block_size: 512
data:
  train_manifest: data/manifests/coig_sft_train.jsonl
  validation_manifest: data/manifests/coig_sft_validation.jsonl
  test_manifest: data/manifests/coig_sft_test.jsonl
  tokenizer_path: artifacts/tiny_30m_pilot/tokenizer.json
  cache_dir: data/cache/tiny_30m_coig_sft
train:
  seed: 17
  batch_size: 4
  gradient_accumulation_steps: 8
  learning_rate: 0.00005
  max_epochs: 3
  checkpoint_every_epoch: 1
  use_bf16: true
  base_checkpoint: artifacts/tiny_30m_pilot/checkpoints/step_1250.pt
  output_dir: artifacts/tiny_30m_coig_sft
```

- [ ] **Step 4: Implement the SFT training CLI**

```python
def load_base_model(path: Path, expected: ModelConfig, device: torch.device) -> GPTLanguageModel:
    state = load_checkpoint(path)
    if state.get("model_config") != asdict(expected):
        raise ValueError("base checkpoint model_config differs from SFT config")
    model = GPTLanguageModel(expected).to(device)
    model.load_state_dict(state["model"])
    return model


for epoch in range(completed_epoch + 1, train_config.max_epochs + 1):
    for batch in deterministic_epoch_batches(train_input_ids, train_labels, train_config, epoch):
        metrics = trainer.train_step(batch.input_ids, batch.labels)
    validation = evaluate_causal_lm(model, valid_input_ids, valid_labels, batch_size=effective_batch_size)
    state = sft_checkpoint_state(..., completed_epoch=epoch, best_validation_loss=best_loss)
    save_checkpoint(checkpoints_dir / f"epoch_{epoch:03d}.pt", state)
    if validation.loss < best_loss:
        save_checkpoint(checkpoints_dir / "best.pt", state)
```

The CLI accepts `--config`, optional `--resume`, and `--device`. For a new run it loads only `base_checkpoint`; for resume it requires checkpoint type `"sft"`, identical model config, and restores optimizer/scheduler/RNG after a completed epoch. Each metrics JSON line must contain `epoch`, `global_step`, `train_loss`, `validation_loss`, `validation_perplexity`, `validation_supervised_tokens`, and `tokens_seen`. Never open the test cache here. Reject a tokenizer vocabulary size different from `model.vocab_size` before loading weights.

- [ ] **Step 5: Run training tests**

Run: `python -m pytest tests/test_sft_training.py tests/test_training.py -q`

Expected: PASS, including CPU one-epoch SFT update, incompatible-base rejection, best-checkpoint selection only on lower validation loss, and completed-epoch resume metadata.

- [ ] **Step 6: Commit SFT training**

```bash
git add src/sinogpt/config.py configs/tiny_30m_coig_sft.yaml src/sinogpt/cli/train_sft.py tests/test_sft_training.py
git commit -m "feat: train SFT from pretrained checkpoint"
```

### Task 5: Chat-template sampling for qualitative comparison

**Files:**
- Create: `src/sinogpt/inference.py`
- Create: `src/sinogpt/cli/sample_chat.py`
- Create: `tests/test_inference.py`

**Interfaces:**
- Consumes: an SFT-capable checkpoint, frozen tokenizer, user question, system prompt, seed, temperature, top-k, and generation-token limit.
- Produces: only the decoded assistant continuation, stopping at `<eos>` or the requested token limit.

- [ ] **Step 1: Write failing chat-format tests**

```python
def test_build_chat_prompt_places_assistant_marker_last(tokenizer: Tokenizer) -> None:
    ids = build_chat_prompt("系统", "什么是梯度下降？", tokenizer)
    assert ids[-1] == tokenizer.token_to_id("<|assistant|>")
    assert tokenizer.token_to_id("<|user|>") in ids


def test_generate_stops_when_eos_is_sampled() -> None:
    response = generate_assistant_ids(EosModel(), [1, 2], eos_id=3, max_new_tokens=8, temperature=1.0, top_k=0)
    assert response == [3]
```

- [ ] **Step 2: Run the focused test and verify failure**

Run: `python -m pytest tests/test_inference.py -q`

Expected: FAIL because the inference module is absent.

- [ ] **Step 3: Implement the shared template and generator**

```python
def build_chat_prompt(system: str, question: str, tokenizer: Tokenizer) -> list[int]:
    if not system.strip() or not question.strip():
        raise ValueError("system and question must not be empty")
    return tokenizer.encode(
        f"<|system|>{system}<eos><|user|>{question}<eos><|assistant|>"
    ).ids


def generate_assistant_ids(model: nn.Module, prompt_ids: list[int], *, eos_id: int, max_new_tokens: int,
                           temperature: float, top_k: int) -> list[int]:
    generated: list[int] = []
    context_ids = list(prompt_ids)
    for _ in range(max_new_tokens):
        logits, _ = model(torch.tensor([context_ids[-model.config.block_size:]], device=device))
        next_id = sample_next_token(logits[0, -1], temperature=temperature, top_k=top_k)
        generated.append(next_id)
        context_ids.append(next_id)
        if next_id == eos_id:
            break
    return generated
```

`sample_next_token` must reject nonpositive temperature and negative top-k, apply top-k only when `top_k > 0`, and use the caller-seeded PyTorch RNG. The CLI must require `--checkpoint` and `--question`, default the system prompt to `DEFAULT_SYSTEM_PROMPT`, and expose `--tokens`, `--seed`, `--temperature`, `--top-k`, and `--device`. Decode only generated IDs before EOS, so the control tokens and prompt never appear in the printed answer.

- [ ] **Step 4: Run inference tests and commit**

Run: `python -m pytest tests/test_inference.py -q`

Expected: PASS with role-marker order, EOS stopping, argument validation, and seeded deterministic generation covered.

```bash
git add src/sinogpt/inference.py src/sinogpt/cli/sample_chat.py tests/test_inference.py
git commit -m "feat: sample SFT chat responses"
```

### Task 6: Reproducible cloud tutorial, paper evidence, and full verification

**Files:**
- Create: `docs/tutorials/11-COIG监督微调.md`
- Modify: `docs/paper/README.md`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: the four CLIs from Tasks 1–5 and the cloud path `/workspace/novel-agent/inbox/SinoGPT`.
- Produces: a Chinese step-by-step tutorial, a paper-ready SFT evidence checklist, and regression coverage that the tutorial names all required commands.

- [ ] **Step 1: Write the failing tutorial contract test**

```python
def test_coig_sft_tutorial_lists_export_prepare_train_and_test_evaluation() -> None:
    text = Path("docs/tutorials/11-COIG监督微调.md").read_text(encoding="utf-8")
    for command in (
        "sinogpt.cli.export_hf_sft_dataset",
        "sinogpt.cli.prepare_sft_data",
        "sinogpt.cli.train_sft",
        "sinogpt.cli.evaluate_sft",
        "sinogpt.cli.sample_chat",
    ):
        assert command in text
```

- [ ] **Step 2: Run the test and verify failure**

Run: `python -m pytest tests/test_cli.py::test_coig_sft_tutorial_lists_export_prepare_train_and_test_evaluation -q`

Expected: FAIL because the SFT tutorial does not exist.

- [ ] **Step 3: Write the cloud procedure and report template**

The tutorial must include these exact ordered commands, preserve the printed exporter JSON, and state that the test command is run only after `best.pt` exists:

```bash
python -m pip install -e ".[dev,data]"
python -m pytest -q

python -m sinogpt.cli.export_hf_sft_dataset \
  --dataset BAAI/COIG \
  --revision main \
  --license-note "COIG mixed-license research pilot; not cleared for commercial use" \
  --output-dir data/manifests \
  --limit 5000 --train-count 4000 --validation-count 500 --test-count 500

python -m sinogpt.cli.prepare_sft_data --config configs/tiny_30m_coig_sft.yaml
python -m sinogpt.cli.train_sft --config configs/tiny_30m_coig_sft.yaml --device cuda
python -m sinogpt.cli.evaluate_sft --config configs/tiny_30m_coig_sft.yaml \
  --checkpoint artifacts/tiny_30m_coig_sft/checkpoints/best.pt --split test --device cuda
python -m sinogpt.cli.sample_chat --checkpoint artifacts/tiny_30m_coig_sft/checkpoints/best.pt \
  --question "什么是梯度下降？" --seed 17 --temperature 0.7 --top-k 40 --device cuda
```

Explain that validation loss decides the checkpoint, test loss estimates held-out fit, and neither metric measures truthfulness or broad GPT-3-level ability. Add a report table with Git commit, base checkpoint SHA-256, COIG resolved revision, license note, split-manifest SHA-256 values, tokenizer SHA-256, configuration, hardware, epochs, best validation loss/perplexity, test loss/perplexity, and fixed-prompt outputs. Include the COIG mixed-license limitation and a warning not to use medical, legal, financial, or commercial decisions based on this 30M pilot.

- [ ] **Step 4: Run the tutorial test, full suite, and lint**

Run: `python -m pytest -q`

Expected: PASS for all existing and new tests.

Run: `python -m ruff check .`

Expected: `All checks passed!`

- [ ] **Step 5: Commit documentation and tests**

```bash
git add docs/tutorials/11-COIG监督微调.md docs/paper/README.md tests/test_cli.py
git commit -m "docs: add COIG SFT cloud tutorial"
```

## Self-Review

- **Spec coverage:** Task 1 implements frozen COIG provenance, exact split counts, deduplication, and research-only licensing evidence. Task 2 implements the chat template, non-packed records, and assistant-only labels. Task 3 separates token-weighted validation/test evaluation. Task 4 starts from `step_1250.pt`, enforces compatibility, limits training to three epochs, and selects only with validation loss. Task 5 supplies repeatable chat behavior observations without prematurely adding Gradio. Task 6 gives the cloud workflow, paper evidence, and final regression verification.
- **Placeholder scan:** No task contains unfinished placeholder language or an unspecified test command. Every code-changing task names exact files, functions, inputs, outputs, test commands, and a commit.
- **Type consistency:** `SFTRecord` is exported in Task 1, encoded in Task 2, cached as `input_ids`/`labels`, evaluated by `evaluate_causal_lm` in Task 3, trained in Task 4, and sampled through the same role-token format in Task 5. `SFTDataConfig` supplies the paths consumed by Tasks 2–4.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-31-coig-sft-pilot.md`. Two execution options:

1. **Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
