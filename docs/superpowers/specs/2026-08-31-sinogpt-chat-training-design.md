# SinoGPT-30M 从零训练、对话观察与 SFT 设计

## 目标

在单张 RTX 4090（24GB）上训练一个约 30M 参数的中文 decoder-only 语言模型，并让学习者在网页聊天界面中比较不同训练阶段的 checkpoint。项目复现 GPT-3 风格的训练机制，不复刻 175B GPT-3 的参数规模或通用能力。

训练流程必须使学习者能区分：预训练让模型学习下一个 token 的语言分布；SFT 让模型学习对话格式、回答风格和角色约束。首轮 SFT 锁定为从公开 COIG 导出的 5,000 个中文问答对，初始化权重为已完成预训练的 `step_1250.pt`。

## 非目标

- 不把 SinoGPT-30M 宣称为 GPT-3 等级模型、通用助手或可商用聊天产品。
- 第一阶段不接入 RAG、RLHF、分布式训练或现成底模。
- 不把来源或许可不清晰的文本加入正式训练清单。
- 不把 COIG 首轮 SFT 视为商业训练数据批准：COIG 数据卡说明只有 BAAI 创作部分采用 Apache-2.0，数据集中还包含其他宽松许可内容与按 fair-use 使用的网页内容。该轮只能作为学习与论文先导实验；任何商用前必须逐条确认拟用子集、上游权利、用途和合规要求。

## 交付物

1. 可审计中文语料导入命令：导出符合项目 `ManifestRecord` 格式的 train/validation JSONL，并冻结来源、revision、许可证说明、文档哈希与随机种子。
2. 固定词表的 BPE tokenizer：除 `<pad>`、`<bos>`、`<eos>`、`<unk>` 外，训练前加入 `<|system|>`、`<|user|>`、`<|assistant|>`。训练开始后禁止修改词表。
3. 30M 预训练配置与分阶段 checkpoint：使用现有 6 层、384 hidden、6 heads、512 context 的模型；保存可恢复 checkpoint 和 `metrics.jsonl`。
4. 单独的推理与网页聊天模块：加载指定 checkpoint，以同一采样设置生成回复；展示 checkpoint 步数、训练 loss、温度、top-k、最大生成 token 数和随机种子。
5. SFT 数据格式、训练脚本、教程及前后对比：从 COIG 的 `conversations[].question` / `conversations[].answer` 导出恰好 5,000 个去重问答对，并固定为 4,000 / 500 / 500 的训练、验证、测试集。同一聊天 UI 可以切换预训练 checkpoint 与 SFT checkpoint。

## 用户体验与数据流

```mermaid
flowchart LR
    A[带来源/许可证的中文文本] --> B[审计 Manifest JSONL]
    B --> C[BPE tokenizer\n含聊天控制 token]
    B --> D[Token 打包缓存]
    C --> D
    D --> E[从零预训练 SinoGPT-30M]
    E --> F[step_*.pt checkpoint]
    F --> G[聊天界面：续写/聊天模式]
    H[人工审核的对话 JSONL] --> I[SFT]
    F --> I
    I --> J[SFT checkpoint]
    J --> G
    G --> K[同题对比：预训练与 SFT]
```

### 聊天模式

聊天输入统一编码为：

```text
<|system|>你是一个简洁、诚实的中文助手。<|eos|>
<|user|>用户问题<|eos|>
<|assistant|>
```

预训练模型没有经过该格式的监督，它可能只会续写或重复；这是预期的教学现象。SFT 时仅对 assistant 回复 token 计算损失，system/user/padding token 的 label 设为忽略值，避免把提示词当成回答目标。

### checkpoint 比较

聊天界面允许选择多个 checkpoint，但每次生成固定：同一 prompt、同一温度、同一 top-k、同一最大生成长度、同一随机种子。界面保存对比记录，避免把随机采样变化误解为训练改进。

## 首轮 COIG SFT（已确认）

### 数据与切分

- 使用 Hugging Face 的 `BAAI/COIG`，导出器先把用户请求的 revision 解析成不可变 commit SHA，再把它连同完整许可证说明写入每一条 SFT JSONL 记录。
- 每个 COIG 行包含可选 `instruction` 与 `conversations` 列表。首轮仅使用其中具有非空字符串 `question` 和 `answer` 的轮次；每一轮被转换成一条独立的单轮问答样本，不把同一原对话的先前轮次当作模型可见上下文。
- 导出器规范化空白、以规范化后的 `system + user + assistant` 做 SHA-256 去重，跳过空字段、非字符串字段和超过模型 512-token 上限的样本，并逐项报告原因。
- 冻结数据集 revision 后，以固定源顺序收集首批 5,000 条合格且去重的样本，再按 4,000 / 500 / 500 顺序写入 train / validation / test。切分清单和哈希均不提交至公共仓库。

### SFT 目标

每条样本编码为：

```text
<|system|>你是一个简洁、诚实的中文助手。<|eos|>
<|user|>问题<|eos|>
<|assistant|>回答<|eos|>
```

模型输入仍是右移前的 token 序列；labels 中 system、user、控制 token 与 padding 均为 `-100`，仅 assistant 回答正文和其结束 `<eos>` 保留 token ID。PyTorch 交叉熵的默认 `ignore_index=-100` 因而只优化回答。SFT 不是强化学习：没有奖励模型、打分或策略优化，仍是标准交叉熵监督学习。

训练最多 3 个 epoch。每个 epoch 完成后计算 validation loss 和 perplexity；只以 validation loss 选择 `best.pt`，测试集只在训练选择完成后由独立评估命令使用一次。评价报告还包括同一组固定中文问题上，预训练 checkpoint 与 SFT checkpoint 的文本输出；它们是教学证据，不是能力或安全性声明。

## 分阶段训练

| 阶段 | 训练目标 | 预期观察 | 是否可聊天 |
|---|---:|---|---|
| 冒烟验证 | 至少 1M token | loss、吞吐、保存与恢复均正常 | 否，输出通常无意义 |
| 基础预训练 | 约 20M token | 中文词组与短句出现，重复仍明显 | 仅能演示续写 |
| 可观察预训练 | 约 200M token | 短段落续写较连贯，仍缺乏可靠知识与推理 | 可体验但非助手 |
| SFT | COIG 的 5,000 条公开问答对；研究先导，不作商业数据声明 | 更遵循角色、问答和简洁性 | 可以展示简单对话 |

每个阶段的评价至少包括 held-out loss、固定 prompts 的样本、重复率/终止率和人工盲测记录。模型输出不能单独用作能力结论。

## 模块边界

| 模块 | 职责 | 不负责 |
|---|---|---|
| `data` 导入模块 | 读取公开数据、清洗、抽样、哈希、输出 manifest | 训练与网页服务 |
| tokenizer 模块 | 固定词表及聊天控制 token | 数据许可判断 |
| pretrain/SFT trainer | 计算 loss、更新参数、checkpoint 恢复 | 浏览器 UI |
| inference 模块 | 模板编码、采样、终止 token 处理 | 读取或更改训练数据 |
| chat UI | 输入、参数控制、checkpoint 选择、结果对比 | 直接修改模型权重 |

## 验证与安全边界

- 每个新增 CLI 先由 pytest 覆盖，再实现；下载和导出不允许静默丢弃记录，必须报告处理数量与过滤原因。
- tokenizer 特殊 token 必须有 round-trip 和 ID 稳定性测试。
- 生成模块必须验证 checkpoint/config/tokenizer 的词表大小一致，避免加载错误模型。
- SFT 的 label mask 必须由测试证明只对 assistant token 产生损失。
- 验证与测试必须分别保存：训练器不能把 test loss 用于 checkpoint 选择；测试 CLI 必须明确打印数据 split、checkpoint 路径、loss、perplexity 和有效监督 token 数。
- 聊天 UI 默认仅绑定 `127.0.0.1`；云端公开访问由用户显式配置平台端口和访问控制。
- 原始语料和训练产物保持在 `.gitignore` 范围，不提交到公共仓库。

## 成功标准

- 用户能从云端启动网页界面，加载至少一个预训练 checkpoint 并完成一次生成。
- 用户能在同一页面比较至少两个 checkpoint 的固定样例。
- SFT 后的 checkpoint 能正确套用聊天模板，且训练只优化 assistant 回复。
- 所有新命令、数据格式和预期现象均写入 Markdown 教程；测试、lint 和现有训练验收测试全部通过。

## 取舍

选择 Gradio 作为首个界面，原因是它能在同一 Python 项目中实现 GPU 推理与聊天控件，避免在学习阶段引入独立前端、后端和部署栈。先使用 30M 配置而非 125M/350M，以确保单卡预训练、checkpoint 对比和 SFT 能在可控成本内完成；放大实验需要在吞吐、显存和语料规模均有实测记录后再决定。
