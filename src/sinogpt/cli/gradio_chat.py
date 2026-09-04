"""模块用途：提供双 checkpoint 的本地 Gradio 聊天演示，不负责模型训练或持久化聊天。"""

import argparse
from importlib import import_module
from pathlib import Path
from threading import Lock
from typing import Any

import torch

from sinogpt.demo.chat_service import DualModelChatService, SamplingSettings

_GRADIO_INSTALL_HINT = 'Gradio is not installed; run python -m pip install -e ".[demo]"'
_DEFAULT_MODEL_LABEL = "v2 SFT（聊天候选）"
_DISCLAIMER = "学习实验：回答可能错误，不可作为事实依据或商用服务。"


class _SessionGenerationEpochs:
    """线程安全地记录页面会话的生成代次，不保存聊天内容。"""

    def __init__(self) -> None:
        self._epochs: dict[str, int] = {}
        self._epoch_counter = 0
        self._lock = Lock()

    def begin(self, session_hash: str | None) -> int:
        """读取请求所属代次；无页面会话的调用安全退化为无守卫。"""
        if not session_hash:
            return 0
        with self._lock:
            if session_hash not in self._epochs:
                self._epoch_counter += 1
                self._epochs[session_hash] = self._epoch_counter
            return self._epochs[session_hash]

    def invalidate(self, session_hash: str | None) -> None:
        """切换模型时使本会话已开始的请求失效。"""
        if session_hash:
            with self._lock:
                self._epoch_counter += 1
                self._epochs[session_hash] = self._epoch_counter

    def is_current(self, session_hash: str | None, epoch: int) -> bool:
        """检查请求是否仍属于当前代次。"""
        if not session_hash:
            return True
        with self._lock:
            return self._epochs.get(session_hash) == epoch

    def forget(self, session_hash: str | None) -> None:
        """页面卸载后清理其代次，已运行的请求也不再有效。"""
        if session_hash:
            with self._lock:
                self._epochs.pop(session_hash, None)


def build_parser() -> argparse.ArgumentParser:
    """构建启动本地双模型演示所需的 checkpoint、网络和设备参数。"""
    parser = argparse.ArgumentParser(description="启动 SinoGPT 双模型学习演示")
    parser.add_argument("--pretrain-checkpoint", required=True, type=Path)
    parser.add_argument("--sft-checkpoint", required=True, type=Path)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=7860, type=int)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser


def _require_gradio() -> Any:
    """只在构建或运行页面时加载可选 Gradio 依赖。"""
    try:
        return import_module("gradio")
    except ModuleNotFoundError as error:
        raise RuntimeError(_GRADIO_INSTALL_HINT) from error


def _extract_complete_history(messages: list[dict[str, Any]] | None) -> list[tuple[str, str]]:
    """从 messages 聊天记录提取相邻且完整的 user/assistant 回合。"""
    history: list[tuple[str, str]] = []
    pending_user: str | None = None
    for item in messages or []:
        if not isinstance(item, dict):
            pending_user = None
            continue
        role, content = item.get("role"), item.get("content")
        if role == "user" and isinstance(content, str) and content.strip():
            pending_user = content
        elif (
            role == "assistant"
            and isinstance(content, str)
            and content.strip()
            and pending_user is not None
        ):
            history.append((pending_user, content))
            pending_user = None
        else:
            pending_user = None
    return history


def _response_status(response: Any) -> str:
    """将服务诊断信息整理为页面可读的单行状态。"""
    ending = "EOS 结束" if response.stopped_by_eos else "达到长度上限"
    return (
        f"模型：{response.model_label}｜checkpoint：{response.checkpoint_name}｜"
        f"token：{response.generated_tokens}｜耗时：{response.elapsed_seconds:.2f} 秒｜{ending}"
    )


def _submit_message(
    service: DualModelChatService,
    guard: _SessionGenerationEpochs,
    session_hash: str | None,
    model_label: str,
    message: str,
    messages: list[dict[str, Any]] | None,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
) -> tuple[list[dict[str, Any]], str, str] | None:
    """处理一轮提交；返回 None 表示模型已切换，UI 应保留当前状态。"""
    epoch = guard.begin(session_hash)
    current_messages = list(messages or [])
    try:
        response = service.respond(
            model_label,
            message,
            _extract_complete_history(current_messages),
            SamplingSettings(
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_k=int(top_k),
                top_p=float(top_p),
                repetition_penalty=float(repetition_penalty),
            ),
        )
        current_messages.extend(
            ({"role": "user", "content": message}, {"role": "assistant", "content": response.text})
        )
        result = current_messages, "", _response_status(response)
    except Exception as error:
        result = current_messages, message, f"错误：{error}"
    return result if guard.is_current(session_hash, epoch) else None


def build_demo(service: DualModelChatService) -> Any:
    """构建无持久化、可切换模型的 Gradio Blocks 页面。"""
    gr = _require_gradio()
    guard = _SessionGenerationEpochs()

    def respond(
        model_label: str,
        message: str,
        messages: list[dict[str, Any]] | None,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        request: gr.Request = None,
    ) -> tuple[Any, Any, Any]:
        result = _submit_message(
            service,
            guard,
            getattr(request, "session_hash", None),
            model_label,
            message,
            messages,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
        )
        return (gr.skip(), gr.skip(), gr.skip()) if result is None else result

    def switch_model(
        model_label: str, request: gr.Request = None
    ) -> tuple[list[dict[str, Any]], str]:
        guard.invalidate(getattr(request, "session_hash", None))
        return [], f"已切换至：{model_label}；为避免混用上下文，历史已清空。"

    def forget_session(request: gr.Request = None) -> None:
        guard.forget(getattr(request, "session_hash", None))

    with gr.Blocks(title="SinoGPT 双模型学习演示") as demo:
        gr.Markdown("# SinoGPT 双模型学习演示")
        gr.Markdown(_DISCLAIMER)
        model = gr.Dropdown(
            choices=service.model_labels,
            value=_DEFAULT_MODEL_LABEL,
            label="模型",
        )
        chatbot = gr.Chatbot(type="messages", label="对话")
        message = gr.Textbox(label="输入", placeholder="请输入问题")
        with gr.Accordion("高级采样参数", open=False):
            max_new_tokens = gr.Slider(1, 512, value=128, step=1, label="最大 token")
            temperature = gr.Slider(0.1, 2, value=0.7, label="温度")
            top_k = gr.Slider(0, 200, value=40, step=1, label="Top-k")
            top_p = gr.Slider(0.05, 1, value=0.9, label="Top-p")
            repetition_penalty = gr.Slider(1, 2, value=1.1, label="重复惩罚")
        send = gr.Button("发送")
        status = gr.Markdown("等待输入。")
        inputs = [model, message, chatbot, max_new_tokens, temperature, top_k, top_p, repetition_penalty]
        submit_event = send.click(respond, inputs=inputs, outputs=[chatbot, message, status])
        message_submit_event = message.submit(
            respond, inputs=inputs, outputs=[chatbot, message, status]
        )
        model.change(
            switch_model,
            inputs=model,
            outputs=[chatbot, status],
            cancels=[submit_event, message_submit_event],
        )
        demo.unload(forget_session)
    return demo


def _resolve_device(requested: str) -> torch.device:
    """按显式参数或 CUDA 可用性选择模型加载设备。"""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def main() -> None:
    """加载两个 checkpoint 并仅以本地地址启动页面，不创建公开隧道。"""
    args = build_parser().parse_args()
    service = DualModelChatService.from_checkpoints(
        args.pretrain_checkpoint,
        args.sft_checkpoint,
        _resolve_device(args.device),
    )
    build_demo(service).launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
