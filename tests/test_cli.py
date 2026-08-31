"""模块用途：验证训练命令提供显式配置和恢复参数。"""

import os
from pathlib import Path
import subprocess
import sys


def test_train_help_requires_explicit_config_contract() -> None:
    """命令帮助应展示强制配置和可选恢复 checkpoint。"""
    environment = {**os.environ, "PYTHONPATH": str(Path("src").resolve())}
    result = subprocess.run(
        [sys.executable, "-m", "sinogpt.cli.train", "--help"],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )
    assert result.returncode == 0
    assert "--config" in result.stdout
    assert "--resume" in result.stdout


def test_pretraining_pilot_tutorial_lists_required_commands() -> None:
    """真实中文预训练教程必须覆盖导出、编码、训练与恢复命令。"""
    text = Path("docs/tutorials/10-中文预训练先导实验.md").read_text(encoding="utf-8")
    assert "sinogpt.cli.export_hf_dataset" in text
    assert "sinogpt.cli.train_tokenizer" in text
    assert "sinogpt.cli.prepare_data" in text
    assert "--resume" in text
