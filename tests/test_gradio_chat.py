"""模块用途：验证 Gradio 双模型聊天页的安全界面契约，不启动服务器。"""

from types import SimpleNamespace

import pytest

import sinogpt.cli.gradio_chat as gradio_chat
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


def test_extract_complete_history_requires_adjacent_user_assistant_messages() -> None:
    """未知或无效消息必须中断待配对的 user 消息。"""
    messages = [
        {"role": "user", "content": "第一问"},
        {"role": "system", "content": "不应跨越"},
        {"role": "assistant", "content": "第一答"},
    ]

    assert _extract_complete_history(messages) == []


def test_extract_complete_history_rejects_nonmessage_gaps() -> None:
    """非消息对象同样必须中断待配对的 user 消息。"""
    messages = [
        {"role": "user", "content": "第一问"},
        "坏消息对象",
        {"role": "assistant", "content": "第一答"},
    ]

    assert _extract_complete_history(messages) == []


@pytest.mark.parametrize("role", ["user", "assistant"])
@pytest.mark.parametrize("content", ["", " \t\n"])
def test_extract_complete_history_ignores_blank_turn_content(role, content) -> None:
    """问题或回答的空白内容都不能进入下一轮 prompt 历史。"""
    messages = [
        {"role": "user", "content": "第一问"},
        {"role": "assistant", "content": "第一答"},
    ]
    messages[0 if role == "user" else 1]["content"] = content

    assert _extract_complete_history(messages) == []


def test_submit_after_eos_only_response_uses_empty_history() -> None:
    """EOS 空回答后的第二问应继续生成，且服务收到空历史。"""
    service = FakeService()
    recorded_histories = []
    original_respond = service.respond

    def record_respond(model_label, message, history, settings):
        recorded_histories.append(history)
        return original_respond(model_label, message, history, settings)

    service.respond = record_respond

    result = gradio_chat._submit_message(
        service,
        gradio_chat._SessionGenerationEpochs(),
        "session-a",
        "v2 SFT（聊天候选）",
        "第二问",
        [
            {"role": "user", "content": "第一问"},
            {"role": "assistant", "content": ""},
        ],
        128,
        0.7,
        40,
        0.9,
        1.1,
    )

    assert result is not None
    assert recorded_histories == [[]]
    messages, message, status = result
    assert messages[-2:] == [
        {"role": "user", "content": "第二问"},
        {"role": "assistant", "content": "回答"},
    ]
    assert message == ""
    assert "错误" not in status


@pytest.mark.parametrize("service_raises", [False, True])
def test_submit_rejects_generation_invalidated_while_service_responds(service_raises) -> None:
    """模型切换发生在推理期间时，旧请求不能返回聊天输出。"""
    guard = gradio_chat._SessionGenerationEpochs()

    class SwitchingService(FakeService):
        def respond(self, model_label, message, history, settings):
            guard.invalidate("session-a")
            if service_raises:
                raise ValueError("旧模型推理失败")
            return super().respond(model_label, message, history, settings)

    result = gradio_chat._submit_message(
        SwitchingService(),
        guard,
        "session-a",
        "v2 SFT（聊天候选）",
        "问题",
        [],
        128,
        0.7,
        40,
        0.9,
        1.1,
    )

    assert result is None


def test_generation_epochs_are_isolated_between_sessions() -> None:
    """一个页面会话切换模型不能使另一个页面会话的请求失效。"""
    guard = gradio_chat._SessionGenerationEpochs()
    epoch_a = guard.begin("session-a")
    epoch_b = guard.begin("session-b")

    guard.invalidate("session-a")

    assert guard.is_current("session-a", epoch_a) is False
    assert guard.is_current("session-b", epoch_b) is True


@pytest.mark.parametrize("session_hash", [None, ""])
def test_generation_epochs_allow_calls_without_session_hash(session_hash) -> None:
    """缺少页面会话时不阻止普通 Python 调用的结果。"""
    guard = gradio_chat._SessionGenerationEpochs()
    epoch = guard.begin(session_hash)

    guard.invalidate(session_hash)
    guard.forget(session_hash)

    assert guard.is_current(session_hash, epoch) is True


def test_generation_epochs_forget_invalidates_in_flight_request() -> None:
    """页面卸载清理代次后，原请求不能重新写回页面。"""
    guard = gradio_chat._SessionGenerationEpochs()
    epoch = guard.begin("session-a")

    guard.forget("session-a")

    assert guard.is_current("session-a", epoch) is False


def test_build_demo_contains_required_warning_and_model_choices() -> None:
    """页面必须呈现固定免责声明与两个中文模型标签。"""
    pytest.importorskip("gradio")
    from sinogpt.cli.gradio_chat import build_demo

    demo = build_demo(FakeService())
    config = demo.get_config_file()

    assert "学习实验：回答可能错误，不可作为事实依据或商用服务。" in str(config)
    assert "v2 预训练（续写基线）" in str(config)
    assert "v2 SFT（聊天候选）" in str(config)
