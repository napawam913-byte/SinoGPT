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


def test_coig_sft_export_help_requires_a_revision_and_license_note() -> None:
    """公开 SFT 导出命令必须暴露不可变版本和许可证说明参数。"""
    environment = {**os.environ, "PYTHONPATH": str(Path("src").resolve())}
    result = subprocess.run(
        [sys.executable, "-m", "sinogpt.cli.export_hf_sft_dataset", "--help"],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )
    assert result.returncode == 0
    assert "--revision" in result.stdout
    assert "--license-note" in result.stdout
    assert "--test-count" in result.stdout


def test_prepare_sft_data_help_requires_a_frozen_config() -> None:
    """SFT 缓存命令必须以显式 YAML 配置连接数据与冻结 tokenizer。"""
    environment = {**os.environ, "PYTHONPATH": str(Path("src").resolve())}
    result = subprocess.run(
        [sys.executable, "-m", "sinogpt.cli.prepare_sft_data", "--help"],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )
    assert result.returncode == 0
    assert "--config" in result.stdout


def test_evaluate_sft_help_requires_checkpoint_config_and_split() -> None:
    """测试集评估必须显式声明被测 checkpoint、配置与数据 split。"""
    environment = {**os.environ, "PYTHONPATH": str(Path("src").resolve())}
    result = subprocess.run(
        [sys.executable, "-m", "sinogpt.cli.evaluate_sft", "--help"],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )
    assert result.returncode == 0
    assert "--checkpoint" in result.stdout
    assert "--config" in result.stdout
    assert "--split" in result.stdout


def test_train_sft_help_requires_config_and_exposes_resume() -> None:
    """SFT 训练须显式指定配置，并允许从 epoch checkpoint 继续。"""
    environment = {**os.environ, "PYTHONPATH": str(Path("src").resolve())}
    result = subprocess.run(
        [sys.executable, "-m", "sinogpt.cli.train_sft", "--help"],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )
    assert result.returncode == 0
    assert "--config" in result.stdout
    assert "--resume" in result.stdout


def test_sample_chat_help_requires_checkpoint_and_question() -> None:
    """聊天采样须显式选择 checkpoint 和用户问题，避免误用随机模型。"""
    environment = {**os.environ, "PYTHONPATH": str(Path("src").resolve())}
    result = subprocess.run(
        [sys.executable, "-m", "sinogpt.cli.sample_chat", "--help"],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )
    assert result.returncode == 0
    assert "--checkpoint" in result.stdout
    assert "--question" in result.stdout
    assert "--top-k" in result.stdout


def test_coig_sft_tutorial_lists_export_prepare_train_and_test_evaluation() -> None:
    """云端教程必须覆盖公开数据导出、标签缓存、训练、测试与聊天采样。"""
    text = Path("docs/tutorials/11-COIG监督微调.md").read_text(encoding="utf-8")
    for command in (
        "sinogpt.cli.export_hf_sft_dataset",
        "sinogpt.cli.prepare_sft_data",
        "sinogpt.cli.train_sft",
        "sinogpt.cli.evaluate_sft",
        "sinogpt.cli.sample_chat",
    ):
        assert command in text


def test_pretraining_pilot_tutorial_lists_required_commands() -> None:
    """真实中文预训练教程必须覆盖导出、编码、训练与恢复命令。"""
    text = Path("docs/tutorials/10-中文预训练先导实验.md").read_text(encoding="utf-8")
    assert "sinogpt.cli.export_hf_dataset" in text
    assert "sinogpt.cli.train_tokenizer" in text
    assert "sinogpt.cli.prepare_data" in text
    assert "--resume" in text
