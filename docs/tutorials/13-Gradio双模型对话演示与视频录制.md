# Gradio 双模型对话演示与视频录制

> 模块用途与范围：本教程用于在云端启动一个可录屏的 Gradio 对话演示，对比 SinoGPT 30M 学习实验的 v2 预训练模型与 v2 SFT 模型。它只覆盖演示、录制和复现，不覆盖重新训练、部署生产服务或发布模型。

## 1. 目标与边界（约 1 分钟）

本次演示的目标是展示同一道问题在两个 checkpoint 上的输出差异，并记录 checkpoint、token 上限、耗时和 EOS 行为。这是 30M 学习实验：回答可能错误、不可作为事实依据；本实验不可商用，也不是商用服务。COIG 为混合许可，尚未完成商业清权。FineWeb2 中文数据记录为 ODC-By 1.0；FineWeb2 中文数据的商业适用性尚未审查，且上游 Common Crawl 条款尚未完成商业适用性审查；因此本实验不主张任何商业清权。

## 2. 两个模型与前提（约 1 分钟）

必须同时存在以下 checkpoint；缺失时停止演示，不在录制中临时重训：

```bash
ls -lh artifacts/tiny_30m_v2/checkpoints/step_38191.pt
ls -lh artifacts/tiny_30m_v2_coig_sft/checkpoints/best.pt
```

前者是 **v2 预训练**，后者是基于 COIG SFT 的 **v2 SFT**。还需要可用 CUDA 环境和项目的 `[demo]` 依赖。

## 3. 云端启动命令（约 2 分钟）

在云端终端逐行执行，保持命令和输出可复现：

```bash
cd /workspace/novel-agent/inbox/SinoGPT
git pull --ff-only origin master
python -m pip install -e ".[demo]"
python -m sinogpt.cli.gradio_chat --pretrain-checkpoint artifacts/tiny_30m_v2/checkpoints/step_38191.pt --sft-checkpoint artifacts/tiny_30m_v2_coig_sft/checkpoints/best.pt --host 0.0.0.0 --port 7860 --device cuda
```

云平台使用普通 7860 端口映射访问页面；启动配置明确为 `share=False`，不使用 `share=True`。禁止创建公开隧道。终端运行时页面可用；演示结束在终端按 `Ctrl+C` 停止。

## 4. 页面操作与记录（约 3 分钟）

先说明模型选择器：切换模型会清空历史。选同一题，先选择 v2 预训练并提交，再切换到 v2 SFT、重新提交同一题。录制或旁白记录：模型名称、checkpoint 路径、token 上限、实际耗时、是否提前遇到 EOS，以及可观察到的输出差异。

tokens 是生成上限，EOS 可以让生成提前结束；这些指标不等于事实正确。可选问题应短小、可重复，避免输入个人资料、密钥或未公开内容。

## 5. 10–14 分钟录制流程

建议按以下顺序剪辑或一次录完：

1. 目标与边界：这是 30M 学习实验，结果可能错误。
2. 数据与 tokenizer：说明预训练/SFT 数据角色和 tokenizer 约束。
3. V2 验证指标：展示仓库中已有验证结果，说明指标用途与局限。
4. 糟糕续写/幻觉：展示预训练模型可能出现的不连贯或虚构回答。
5. SFT/EOS：说明监督微调和 EOS 提前结束现象。
6. 同题双模型：用同一问题依次对比 v2 预训练和 v2 SFT，并读出记录的耗时与 token。
7. 限制和下一步：更多清洁预训练、长而高质量的 SFT、RAG；不承诺商用准确率。
8. 仓库复现：回放 checkpoint 检查、安装和完整启动命令。

## 6. 录制前安全检查

发布前隐藏 HF token、云账户或浏览器资料；隐藏 SSH 密码，裁剪终端用户名、本地绝对路径和其他个人信息。不要把 COIG 或 FineWeb2 直接宣传为已商业可用，也不要在画面中展示可登录的公开隧道地址。

## 7. 常见问题

- `gradio` 未安装：确认执行了 `python -m pip install -e ".[demo]"`，不要修改教程命令绕过依赖声明。
- checkpoint 缺失：停止演示并报告缺失路径；不要在录制现场重训。
- 端口无法访问：检查云平台的 7860 映射和防火墙规则；确认进程仍在运行且绑定 `0.0.0.0`。

## 8. 复现收尾

保存录制说明、问题文本、两模型记录和运行日期；在发布前再次核对许可证边界与安全遮挡。该页面仅用于实验演示，任何事实性结论都应由可靠来源另行验证。

## 9. 云端手工验收：会话竞态（约 1 分钟）

先做 **EOS 空答后继续提问**：当助手气泡为空且原因为 EOS 时，再提交一个新问题，确认请求正常返回，不出现“历史为空”或历史状态错误。

再分别验证 **点击发送与 Enter**：对每种提交方式启动一个故意较慢/较长的请求，执行“提交后立即切换模型”，等待旧请求结束；确认聊天仍为空、切换后的模型状态保留，且旧回答不应重新出现。这些检查验证内存中的 session epoch guard；聊天内容不会被保存。
