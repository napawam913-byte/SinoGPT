# SinoGPT 论文式项目报告模板

## 标题

**SinoGPT：面向中文主导双语语料的从零训练 GPT-3 风格解码器研究**

## 摘要

写明研究问题、模型规模、数据混合比例、训练 token 预算、评测协议和可复现工件。不要把 20 步 demo 作为语言能力证据。

## 1. 引言与研究问题

解释为什么中文能力取决于直接中文 token 训练、分词效率与语料覆盖，而不是把中文实时翻译成英文。提出固定架构和 token 预算下的数据配比问题。

## 2. 相关工作

讨论 decoder-only Transformer、GPT-3 缩放、数据质量与 token 预算；只引用实际阅读过的原始论文和官方数据卡。

## 3. 数据与治理

列出每个数据源、revision、许可证、语言、文档哈希、清洗/去重规则和 train/validation 隔离。明确代码语料的许可审计和剔除规则。

## 4. 模型与训练方法

报告 token/position embedding、Pre-LN、因果注意力、GELU MLP、交叉熵、AdamW、梯度裁剪、bf16 与 checkpoint 状态。将 `04`、`05`、`06` 教程中的公式和图转为论文图表。

## 5. 实验设置

报告约 30M 教学模型（历史目录名 `tiny_25m`）、125M 三组消融和 350M 主实验的配置；列 GPU、内存、训练 token 数、随机种子、吞吐和存储策略。

## 6. 结果与讨论

只填入真实运行的评测结果。当前已验证事实是：本机 CPU 上使用 16-token 上下文完成了 20 步架构验收，loss 从 `243.5234` 到 `42.2638`，恢复训练从 step 10 成功持续到 step 20。它不用于比较语言能力。

### COIG SFT 先导实验记录

本实验以 `artifacts/tiny_30m_pilot/checkpoints/step_1250.pt` 作为初始化，不应写成从零 SFT 或 GPT-3 复现。COIG 首轮固定为 5,000 个公开问答，4,000 / 500 / 500 划分。训练只使用 train/validation；以 validation loss 选 `best.pt` 后，才运行一次 test 评估。

| 字段 | 实际值（实验后填写） |
|---|---|
| Git commit |  |
| 基座 checkpoint 路径与 SHA-256 |  |
| COIG resolved revision |  |
| 许可证说明与混合许可/fair-use 限制 |  |
| train / validation / test manifest SHA-256 |  |
| tokenizer SHA-256 与词表大小 |  |
| 模型参数量、上下文长度、有效 batch |  |
| GPU、CUDA/PyTorch、bf16 |  |
| epoch 数、学习率、随机种子 |  |
| 最佳 validation loss / perplexity |  |
| held-out test loss / perplexity / supervised tokens |  |
| 五个固定问题的预训练与 SFT 输出 |  |

解释时应区分：较低的 held-out token loss 表示该数据分布上更好的下一个 token 拟合，不足以证明事实正确性、泛化推理、安全性或商业可用性。

## 7. 局限性与伦理/许可

讨论单卡规模限制、语料偏差、中文/英文覆盖不均衡、生成风险、著作权和代码许可证风险。

## 8. 复现附录

附上命令、Git commit、依赖版本、所有哈希、run manifest、metrics JSONL、checkpoint URI 和随机种子。
