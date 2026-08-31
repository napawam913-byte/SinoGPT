# COIG 监督微调：把中文续写器变为初步问答模型

## 本章目标

本章以已经完成的 30M 中文预训练 checkpoint 为底座，使用公开 COIG 的 5,000 个中文问答对完成第一轮 SFT（Supervised Fine-Tuning，监督微调）。训练后模型仍然是约 30M 参数的学习实验，不会获得 GPT-3 级别的事实知识、推理或代码能力；本章的目标是观察“文本续写”如何转变为“按用户问题作答”。

SFT 不是强化学习。每条标准问答仍使用交叉熵训练下一个 token，只是把 system、user、控制 token 和 padding 的 label 设为 `-100`，只让 assistant 回答正文与结束 `<eos>` 参与损失。

## 数据与许可边界

本实验使用 `BAAI/COIG`。其 Hugging Face 数据卡标注 Apache-2.0，但卡中同时说明：只有 BAAI 创作部分遵循 Apache-2.0，集合还含其他宽松许可内容及按 fair-use 使用的网页内容。因此下列命令仅用于学习与论文先导实验，**不能**据此宣称训练数据可直接商用。

导出命令会把解析后的不可变 revision SHA、完整许可证说明和内容哈希写进每条 JSONL 记录。数据集、缓存和 checkpoint 都由 `.gitignore` 排除，不能提交到公共仓库。

## 前置条件

- 云端目录为 `/workspace/novel-agent/inbox/SinoGPT`。
- 已完成教程 10，且存在：

```text
artifacts/tiny_30m_pilot/tokenizer.json
artifacts/tiny_30m_pilot/checkpoints/step_1250.pt
```

- 使用 RTX 4090，`python -c "import torch; print(torch.cuda.is_bf16_supported())"` 输出 `True`。
- 先更新代码并安装依赖：

```bash
git pull --ff-only
python -m pip install -e ".[dev,data]"
python -m pytest -q
```

## 1. 导出固定的 5,000 条 COIG 问答

下面的导出器会先把 `main` 解析成不可变 commit SHA，然后按冻结源顺序抽取 5,000 条去重、非空、可放入 512-token 上下文的问答。最终切分严格为 4,000 条训练、500 条验证和 500 条测试。

```bash
python -m sinogpt.cli.export_hf_sft_dataset \
  --dataset BAAI/COIG \
  --split train \
  --revision main \
  --license-note "COIG mixed-license research pilot; not cleared for commercial use" \
  --tokenizer artifacts/tiny_30m_pilot/tokenizer.json \
  --block-size 512 \
  --output-dir data/manifests \
  --limit 5000 \
  --train-count 4000 \
  --validation-count 500 \
  --test-count 500
```

成功后保留终端输出的一行 JSON。它至少包含：`resolved_revision`、`exported`、`empty_turns`、`malformed_turns`、`too_long`、`duplicates`、`train`、`validation` 与 `test`。这不是普通下载日志，而是论文中数据版本与过滤结果的证据。

产生的本地文件是：

```text
data/manifests/coig_sft_train.jsonl
data/manifests/coig_sft_validation.jsonl
data/manifests/coig_sft_test.jsonl
```

如果 Hugging Face 暂时出现 TLS/EOF 重试，先等待客户端重试完成；不要为了绕过错误而替换为来源不明的镜像，也不要把访问 token 发到聊天中。

## 2. 查看一条样本如何变成监督标签

每个问答被编码为：

```text
<|system|>你是一个简洁、诚实的中文助手。<|eos|>
<|user|>什么是梯度下降？<|eos|>
<|assistant|>梯度下降是一种优化方法。<|eos|>
```

模型输入是整段 token 的右移前序列；目标是右移后一位的序列。到 `<|assistant|>` 之前的目标全部替换为 `-100`，因此它们会被 PyTorch 交叉熵忽略。第一个未被忽略的目标是回答的第一个 token，最后一个是回答后的 `<eos>`。

## 3. 准备三份 SFT tensor 缓存

```bash
python -m sinogpt.cli.prepare_sft_data \
  --config configs/tiny_30m_coig_sft.yaml
```

预期输出：

```text
prepared train_sequences=4000 validation_sequences=500 test_sequences=500
```

缓存位于 `data/cache/tiny_30m_coig_sft/`，每个 split 各有输入 tensor 与 labels tensor。它们是一条对话一个样本，不会把两段不同对话拼接在一起。若命令提示某条记录超过 `block_size`，说明导出器、tokenizer 或配置来自不同版本；删除本次 SFT manifest 与 cache，使用同一个冻结 tokenizer 重做导出。

## 4. 从 `step_1250.pt` 开始 SFT

```bash
python -m sinogpt.cli.train_sft \
  --config configs/tiny_30m_coig_sft.yaml \
  --device cuda
```

配置固定为 batch size `4`、梯度累积 `8`、有效 batch `32` 条问答、学习率 `5e-5`、最多 3 个 epoch。它加载的不是新模型，而是：

```text
artifacts/tiny_30m_pilot/checkpoints/step_1250.pt
```

每个 epoch 完成后会写入一行 `metrics.jsonl`，其中有 `train_loss`、`validation_loss`、`validation_perplexity`、`validation_supervised_tokens` 和 `is_best`。并保存：

```text
artifacts/tiny_30m_coig_sft/checkpoints/epoch_001.pt
artifacts/tiny_30m_coig_sft/checkpoints/epoch_002.pt
artifacts/tiny_30m_coig_sft/checkpoints/epoch_003.pt
artifacts/tiny_30m_coig_sft/checkpoints/best.pt
```

`best.pt` 只会在验证 loss 严格降低时更新。若训练中断，可从已经完整保存的 epoch 恢复，例如：

```bash
python -m sinogpt.cli.train_sft \
  --config configs/tiny_30m_coig_sft.yaml \
  --resume artifacts/tiny_30m_coig_sft/checkpoints/epoch_001.pt \
  --device cuda
```

不要把 `best.pt` 当作“最适合继续训练”的默认恢复点；它用于推理与最终测试。恢复训练优先选择最新的 `epoch_*.pt`。

## 5. 只在模型选择完成后评估测试集

训练期间模型只读取 train 和 validation cache。选定 `best.pt` 后，运行一次 held-out test：

```bash
python -m sinogpt.cli.evaluate_sft \
  --config configs/tiny_30m_coig_sft.yaml \
  --checkpoint artifacts/tiny_30m_coig_sft/checkpoints/best.pt \
  --split test \
  --device cuda
```

输出 JSON 中的 `loss` 是每个有效 assistant token 的平均负对数似然，`perplexity = exp(loss)`，`supervised_tokens` 是实际参与评估的回答 token 数。测试 loss 不能反过来选择 checkpoint；否则测试集不再是独立估计。

## 6. 用相同问题比较预训练与 SFT

固定问题和采样参数，先观察预训练 checkpoint：

```bash
python -m sinogpt.cli.sample_chat \
  --checkpoint artifacts/tiny_30m_pilot/checkpoints/step_1250.pt \
  --question "什么是梯度下降？" \
  --seed 17 --temperature 0.7 --top-k 40 --tokens 128 --device cuda
```

再观察 SFT checkpoint：

```bash
python -m sinogpt.cli.sample_chat \
  --checkpoint artifacts/tiny_30m_coig_sft/checkpoints/best.pt \
  --question "什么是梯度下降？" \
  --seed 17 --temperature 0.7 --top-k 40 --tokens 128 --device cuda
```

预训练模型很可能输出网页式续写、重复或控制 token 附近的异常内容；SFT 后更可能以“回答”形式开头并在 `<eos>` 附近结束。这只是行为对齐的初步证据，不代表答案真实、完整或安全。用相同设置测试至少五个固定问题，并保留原始输出，下一阶段再把这些 checkpoint 接入 Gradio 聊天界面。

## 7. 如何判断是否过拟合

- `train_loss` 和 `validation_loss` 都下降：当前数据上的拟合在改善。
- `train_loss` 持续下降、`validation_loss` 持平或上升：出现过拟合征兆，保留验证 loss 最低的 `best.pt`，不要因为训练 loss 更低而选最后一轮。
- 测试 loss 只用于最后报告；不要在看到测试结果后继续改学习率、epoch 或数据过滤规则。

5,000 条公开问答主要教会模型“问题后应如何回答”，不会给 30M 模型注入大规模可靠知识。遇到医疗、法律、金融或商业决策问题，不应依赖它的生成结果。

## 8. 论文与复现记录

保存：导出 JSON、三个 manifest 的 SHA-256、tokenizer SHA-256、预训练 base checkpoint SHA-256、SFT YAML、Git commit、每轮 metrics、`best.pt` 路径、最终 test JSON 以及固定问题的两组输出。报告 COIG 的混合许可/fair-use 限制，而不是把本实验描述为商用模型训练。
