# SinoGPT

一个从零实现 GPT-3 风格中英双语 decoder-only 语言模型的可复现教程项目。它先完成 25M 教学模型，再为 125M 数据配比消融和 350M 主实验建立严格的溯源闸门；它不是 175B GPT-3 的复现。

## 快速验证

```bash
python -m pip install -e ".[dev]"
python -m pytest -v
python -m sinogpt.cli.train_tokenizer --manifest data/manifests/demo_train.jsonl --output artifacts/tiny_25m_demo/tokenizer.json --vocab-size 50000
python -m sinogpt.cli.prepare_data --config configs/tiny_25m_demo_stage1.yaml
python -m sinogpt.cli.train --config configs/tiny_25m_demo_stage1.yaml --device cpu
python -m sinogpt.cli.train --config configs/tiny_25m_demo.yaml --resume artifacts/tiny_25m_demo/checkpoints/step_10.pt --device cpu
python -m sinogpt.cli.sample --checkpoint artifacts/tiny_25m_demo/checkpoints/step_20.pt --prompt 人工智能 --tokens 32 --seed 17 --device cpu
```

该 demo 是完整训练与恢复链路验收，不是能力评测。云端 RTX 4090 训练请从 `docs/tutorials/01-云端环境.md`、`02-数据集与数据治理.md` 和 `07-从25M到350M训练.md` 开始。

## 教程与报告

按 `docs/tutorials/00-项目总览.md` 至 `09-论文式报告.md` 顺序阅读；论文式报告模板在 `docs/paper/README.md`。正式 125M/350M 训练前，必须固定数据 revision、许可证记录、tokenizer/manifest 哈希、Git commit、目标 token 数和 checkpoint 归档位置。
