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
