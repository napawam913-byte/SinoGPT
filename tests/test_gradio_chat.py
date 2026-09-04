"""模块用途：验证 Gradio 双模型聊天页的安全界面契约，不启动服务器。"""

from types import SimpleNamespace

import pytest

from sinogpt.cli.gradio_chat import _extract_complete_history


class FakeService:
    """页面测试使用的无模型服务替身，仅记录页面传入的数据。"""

    model_labels = ("v2 预训练（续写基线）", "v2 SFT（聊天候选）")

    def respond(self, model_label, message, history, settings):
        return SimpleNamespace(
            text="回答",
            model_label=model_label,
            checkpoint_name="sft.pt",
            generated_tokens=3,
            stopped_by_eos=True,
            elapsed_seconds=0.25,
        )


def test_extract_complete_history_ignores_unknown_and_incomplete_messages() -> None:
    """回调只向推理服务传递完整、顺序正确的 user/assistant 回合。"""
    messages = [
        {"role": "user", "content": "第一问"},
        {"role": "assistant", "content": "第一答"},
        {"role": "assistant", "content": "孤立回答"},
        {"role": "system", "content": "忽略"},
        {"role": "user", "content": "未完成问题"},
    ]

    assert _extract_complete_history(messages) == [("第一问", "第一答")]


def test_build_demo_contains_required_warning_and_model_choices() -> None:
    """页面必须呈现固定免责声明与两个中文模型标签。"""
    pytest.importorskip("gradio")
    from sinogpt.cli.gradio_chat import build_demo

    demo = build_demo(FakeService())
    config = demo.get_config_file()

    assert "学习实验：回答可能错误，不可作为事实依据或商用服务。" in str(config)
    assert "v2 预训练（续写基线）" in str(config)
    assert "v2 SFT（聊天候选）" in str(config)
