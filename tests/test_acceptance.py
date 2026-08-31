"""模块用途：确认 demo 训练产生 20 步有限损失与最终 checkpoint 的验收证据。"""

import json
import math
from pathlib import Path

import pytest


def test_acceptance_metrics_require_twenty_finite_loss_rows() -> None:
    """20 步 demo 运行必须产生连续的正且有限损失记录。"""
    metrics_path = Path("artifacts/tiny_25m_demo/metrics.jsonl")
    if not metrics_path.exists():
        pytest.skip("需要本地生成的 tiny_25m_demo 产物才能执行集成验收")
    assert metrics_path.exists()
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 20
    assert all(row["loss"] > 0.0 and math.isfinite(row["loss"]) for row in rows)
    assert rows[-1]["global_step"] == 20
    assert Path("artifacts/tiny_25m_demo/checkpoints/step_20.pt").exists()


def test_acceptance_skips_when_local_demo_artifacts_are_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """干净克隆没有被忽略的 demo 产物时，验收应显示为跳过而非失败。"""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(pytest.skip.Exception):
        test_acceptance_metrics_require_twenty_finite_loss_rows()
