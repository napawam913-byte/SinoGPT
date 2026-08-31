"""模块用途：确认 demo 训练产生 20 步有限损失与最终 checkpoint 的验收证据。"""

import json
import math
from pathlib import Path


def test_acceptance_metrics_require_twenty_finite_loss_rows() -> None:
    """20 步 demo 运行必须产生连续的正且有限损失记录。"""
    metrics_path = Path("artifacts/tiny_25m_demo/metrics.jsonl")
    assert metrics_path.exists()
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 20
    assert all(row["loss"] > 0.0 and math.isfinite(row["loss"]) for row in rows)
    assert rows[-1]["global_step"] == 20
    assert Path("artifacts/tiny_25m_demo/checkpoints/step_20.pt").exists()
