# SinoGPT 云端 Gradio 双模型演示与视频教程设计

## 状态与目标

状态：用户已选择“同一页面常驻加载 v2 预训练与 v2 SFT 模型，并可即时切换”。

目标是在云端 RTX 4090 实例运行一个仅用于学习和视频演示的 Gradio 页面。页面让观众用同一个问题比较：

- `v2 预训练`：`artifacts/tiny_30m_v2/checkpoints/step_38191.pt`；
- `v2 SFT`：`artifacts/tiny_30m_v2_coig_sft/checkpoints/best.pt`。

页面和教程必须明确：该项目约 30M 参数，输出可能包含幻觉或错误事实，不能作为事实来源或商用服务。

## 范围

### 包含

- 一个云端 Gradio 聊天页面；
- 下拉框切换“v2 预训练”与“v2 SFT”；
- 两个模型启动时均加载到 GPU，以保证切换无需重新载入 checkpoint；
- 同一聊天模板、EOS 自动结束和既有采样控制；
- 可调参数：最大新 token、temperature、top-k、top-p、重复惩罚；
- 每次回答显示模型名称、实际生成 token 数和耗时；
- 模型/分词器/checkpoint 不匹配、空消息、CUDA 不可用等可读错误；
- 一份启动、录屏和讲解教程。

### 不包含

- Ollama/GGUF 转换；
- 流式逐 token 输出；
- 用户账户、聊天记录持久化、联网检索、RAG、知识库；
- 将模型包装为可靠问答或生产服务；
- 修改模型权重、继续训练或重新做 SFT。

## 页面与交互

页面从上到下包含：

1. 标题与固定声明：`SinoGPT 学习实验：回答可能错误，不可作为事实依据。`
2. 模型选择框：`v2 预训练（续写基线）` 或 `v2 SFT（聊天候选）`。
3. 聊天历史区。切换模型时清空历史，防止不同模型的对话状态混用。
4. 单行/多行用户输入框和“发送”按钮。
5. 折叠的高级采样参数，默认值为：128、0.7、40、0.9、1.1。
6. 结果元信息：所选模型、checkpoint 文件名、生成 token 数、耗时和是否由 EOS 提前结束。

两个模型都使用相同的 system/user/assistant 控制 token 模板。预训练模型不具备真正聊天能力，因此它的结果用于演示 SFT 前的行为，不应被描述为等价的聊天模型。

首次版本不流式输出：用户提交后显示 Gradio 等待状态，生成完成后一次性显示完整回答和元信息。这样能复用已验证的 `generate_assistant_ids`，并避免新建流式状态机导致演示与训练样本不一致。

## 架构与文件边界

```text
浏览器 Gradio 页面
  -> cli/gradio_chat.py：参数解析、页面装配、启动服务
  -> demo/chat_service.py：加载双模型、选择模型、生成回答、返回元信息
  -> inference.py：现有聊天模板、top-k/top-p、重复惩罚、EOS 停止
  -> checkpoint + tokenizer：只读加载
```

- `sinogpt/demo/chat_service.py`：不依赖 Gradio；保存只读模型句柄与 tokenizer，负责验证 checkpoint/tokenizer 兼容性、调用既有推理函数和统计结果。
- `sinogpt/cli/gradio_chat.py`：仅依赖页面框架与 service；支持 `--pretrain-checkpoint`、`--sft-checkpoint`、`--host`、`--port`、`--device`。
- `pyproject.toml`：增加可选依赖组 `demo`，包含 Gradio；核心训练依赖不强制安装 Gradio。
- `docs/tutorials/13-Gradio双模型对话演示与视频录制.md`：包含安装、启动、端口访问、演示脚本和素材清单。

默认启动绑定 `0.0.0.0:7860`，但不自动创建 Gradio 公网 share 链接。用户在云平台公开/映射该端口；若平台无法访问，教程要求检查平台端口规则，而不输出密码、访问 token 或 checkpoint 文件。

## 推理规则与失败处理

- 两个模型均只读加载；服务不创建优化器、不调用反向传播、不写 checkpoint。
- SFT 模型和预训练模型都必须与其 checkpoint 中记录的 tokenizer 词表大小匹配。
- 若 checkpoint、tokenizer 或 CUDA 缺失，服务在启动阶段给出明确路径和原因，而非在首次对话时崩溃。
- 空白输入或无效采样参数在提交阶段拒绝；`temperature > 0`、`top_k >= 0`、`0 < top_p <= 1`、`repetition_penalty >= 1`。
- 模型输出 EOS 时去除控制 token，并在元信息中标记提前结束；达到 `max_new_tokens` 时标记“达到长度上限”。
- 不保存用户消息、回答或浏览器历史；重启页面即清空运行内存中的对话。

## 测试策略

- service 单元测试：模型选择、空输入、无效参数、EOS 元信息和不改变模型训练状态；
- CLI 帮助测试：所有 checkpoint、主机、端口和设备参数可见；
- 页面装配测试：在不启动公网服务的情况下验证组件与默认值；
- 回归：完整 pytest 与 Ruff；
- 云端人工验收：同一问题切换两种模型，确认 SFT 更接近回答格式，预训练输出仅作为对照。

## 视频与教程结构

建议录制 10–14 分钟视频，按以下顺序展示：

1. **目标（0:00–0:45）**：不是复现 GPT-3，而是从零走通中文 GPT 训练链路。
2. **数据与 tokenizer（0:45–2:00）**：FineWeb2 中文、100k 文档 tokenizer 样本、800k 文档训练；说明许可证边界。
3. **预训练（2:00–3:30）**：下一个 token 预测、loss、验证集 `4.921` 与训练集 `4.790`。
4. **失败案例（3:30–5:00）**：网页续写跑题、重复；解释低 loss 不等于事实正确或会聊天。
5. **SFT（5:00–7:00）**：COIG 5,000 条、assistant-only loss、EOS；展示验证 loss `3.614`，并说明不能同预训练 loss 横比。
6. **Gradio 对比（7:00–10:00）**：同一问题切换 v2 预训练和 v2 SFT，演示格式改善与“梯度下降”错误回答。
7. **诚实结论（10:00–12:00）**：30M、网页噪声、短回答 SFT 的限制；下一步是更高质量数据、更大模型、长回答 SFT 与 RAG。
8. **复现入口（结尾）**：展示仓库教程编号、配置文件、固定命令与实验记录位置。

录屏代码展示优先顺序：`configs/tiny_30m_v2.yaml`、`configs/tiny_30m_v2_coig_sft.yaml`、`src/sinogpt/model/gpt.py`、`src/sinogpt/training/trainer.py`、`src/sinogpt/inference.py`、新 Gradio CLI，最后展示 `metrics.jsonl` 与聊天界面。不要展示云端密码、Hugging Face token、终端用户名或本地绝对路径。
