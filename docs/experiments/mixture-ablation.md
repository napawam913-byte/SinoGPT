# 中英数据混合消融计划

本文件描述计划，不包含未执行实验的结果。启动任何长训练前，复制 `run-manifest-template.json`，填入真实 tokenizer/manifest SHA-256 与已创建的归档 URI，并调用 `validate_run_manifest`。

| run_name | 目标 token 数 | 中文 | 英文 | 代码 | 目的 |
| --- | ---: | ---: | ---: | ---: | --- |
| `zh_70_en_20_code_10` | 300M | 70% | 20% | 10% | 主混合配比 |
| `zh_100` | 300M | 100% | 0% | 0% | 中文单语基线 |
| `en_100` | 300M | 0% | 100% | 0% | 英文单语基线 |
| `sino_main_350m` | 1B | 70% | 20% | 10% | 350M 主实验 |

所有 125M 消融共享 `configs/ablation_125m.yaml`，且固定 tokenizer、清洗规则、评测集、学习率计划和硬件。若代码许可证审计失败，不能把运行名继续称为 `zh_70_en_20_code_10`；应创建反映实际配比的新运行名并在报告中说明。

建议启动前检查：

```bash
python -c "import json; from pathlib import Path; from sinogpt.experiments import validate_run_manifest; validate_run_manifest(json.loads(Path('run-manifest.json').read_text(encoding='utf-8')))"
```
