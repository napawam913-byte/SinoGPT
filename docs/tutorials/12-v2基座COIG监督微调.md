# v2 基座的 COIG 监督微调

本章将 v2 预训练 checkpoint `step_38191.pt` 变成初步的中文聊天模型。它学习的是“用户问题 → 助手回答 → `<eos>`”的格式，不会使 30M 模型获得可靠百科知识或 GPT-3 级能力。

> 许可边界：COIG 是混合许可集合，部分内容来自网页 fair-use 场景。本实验用于学习与论文先导，不能据此宣称数据或模型可直接商用。

## 0. 不要误用旧 Pilot 配置

必须使用 `configs/tiny_30m_v2_coig_sft.yaml`。它绑定：

```text
v2 tokenizer: artifacts/tiny_30m_v2/tokenizer.json
v2 base:      artifacts/tiny_30m_v2/checkpoints/step_38191.pt
v2 输出目录:  artifacts/tiny_30m_v2_coig_sft/
```

旧的 `tiny_30m_coig_sft.yaml` 指向 Pilot tokenizer 和 `step_1250.pt`，不能用于本章。

## 1. 更新云端代码

```bash
cd /workspace/novel-agent/inbox/SinoGPT
git pull --ff-only origin master
python -m pip install -e ".[dev,data]"
python -m pytest -q
```

## 2. 用 v2 tokenizer 导出固定的 5,000 条 COIG 对话

```bash
python -m sinogpt.cli.export_hf_sft_dataset \
  --dataset BAAI/COIG \
  --split train \
  --revision refs/convert/parquet \
  --license-note "COIG mixed-license research pilot; not cleared for commercial use" \
  --tokenizer artifacts/tiny_30m_v2/tokenizer.json \
  --block-size 512 \
  --output-dir data/manifests/v2_coig_sft \
  --limit 5000 \
  --train-count 4000 \
  --validation-count 500 \
  --test-count 500
```

成功输出必须显示 `train: 4000`、`validation: 500` 和 `test: 500`。若网络出现 Hugging Face 重试，等待重试完成；不要把 token 或密码贴到终端记录和论文中。

## 3. 编码 SFT 数据

```bash
python -m sinogpt.cli.prepare_sft_data \
  --config configs/tiny_30m_v2_coig_sft.yaml
```

预期结果：

```text
prepared train_sequences=4000 validation_sequences=500 test_sequences=500
```

训练标签只覆盖 assistant 回答及其后的 `<eos>`；system、user、控制 token 和 padding 的标签都是 `-100`，不会参与损失。

## 4. 从 v2 基座开始 SFT

```bash
python -m sinogpt.cli.train_sft \
  --config configs/tiny_30m_v2_coig_sft.yaml \
  --device cuda
```

每个 epoch 后都会计算 validation loss，并保存 `epoch_001.pt` 等 checkpoint。只以 validation loss 最低的 `best.pt` 做最终推理；训练中断时，从最新的 `epoch_*.pt` 恢复，不要从 `best.pt` 恢复。

## 5. 只评估一次测试集

```bash
python -m sinogpt.cli.evaluate_sft \
  --config configs/tiny_30m_v2_coig_sft.yaml \
  --checkpoint artifacts/tiny_30m_v2_coig_sft/checkpoints/best.pt \
  --split test \
  --device cuda
```

## 6. 测试聊天结束行为

```bash
python -m sinogpt.cli.sample_chat \
  --checkpoint artifacts/tiny_30m_v2_coig_sft/checkpoints/best.pt \
  --question "什么是梯度下降？请用简单的中文解释。" \
  --tokens 128 --seed 17 --temperature 0.7 \
  --top-k 40 --top-p 0.9 --repetition-penalty 1.1 \
  --device cuda
```

SFT 后模型更可能在回答结束时生成 `<eos>`；`sample_chat` 会检测该 token 并停止。答案仍可能错误或幻觉，不能作为事实来源。
