"""模块用途：以固定角色模板从预训练或 SFT checkpoint 生成助手回答。"""

import argparse
from pathlib import Path

import torch

from sinogpt.config import ModelConfig
from sinogpt.data.sft import DEFAULT_SYSTEM_PROMPT
from sinogpt.inference import build_chat_prompt, generate_assistant_ids
from sinogpt.model.gpt import GPTLanguageModel
from sinogpt.seed import seed_everything
from sinogpt.tokenizer import load_tokenizer
from sinogpt.training.checkpoint import load_checkpoint


def build_parser() -> argparse.ArgumentParser:
    """构建明确指定 checkpoint、问题和采样超参数的聊天命令。"""
    parser = argparse.ArgumentParser(description="从 SinoGPT checkpoint 生成中文聊天回答")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--question", required=True, help="用户问题")
    parser.add_argument("--system", default=DEFAULT_SYSTEM_PROMPT, help="系统提示词")
    parser.add_argument("--tokens", type=int, default=128, help="最多生成 token 数")
    parser.add_argument("--seed", type=int, default=17, help="采样随机种子")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    parser.add_argument("--top-k", type=int, default=40, help="top-k；0 表示不截断")
    parser.add_argument("--top-p", type=float, default=0.9, help="nucleus 采样累计概率阈值")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, help="已生成 token 的重复惩罚；1 表示关闭")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser


def resolve_device(requested: str) -> torch.device:
    """根据显式设备或 CUDA 可用性选择生成设备。"""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def _tokenizer_path_from_checkpoint(state: dict[str, object]) -> Path:
    """兼容预训练和 SFT checkpoint 中各自的 tokenizer 配置字段。"""
    data_config = state.get("sft_data_config", state.get("data_config"))
    if not isinstance(data_config, dict) or not isinstance(data_config.get("tokenizer_path"), str):
        raise ValueError("checkpoint does not contain a tokenizer_path")
    return Path(data_config["tokenizer_path"])


def main() -> None:
    """加载 checkpoint、套用聊天模板，并打印不含控制 token 的 assistant 回答。"""
    args = build_parser().parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    raw_model_config = checkpoint.get("model_config")
    if not isinstance(raw_model_config, dict):
        raise ValueError("checkpoint does not contain model_config")
    model_config = ModelConfig(**raw_model_config)
    tokenizer = load_tokenizer(_tokenizer_path_from_checkpoint(checkpoint))
    if tokenizer.get_vocab_size() != model_config.vocab_size:
        raise ValueError("tokenizer vocabulary size differs from checkpoint model config")
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        raise ValueError("tokenizer must contain <eos>")
    device = resolve_device(args.device)
    seed_everything(args.seed)
    model = GPTLanguageModel(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    prompt_ids = build_chat_prompt(args.system, args.question, tokenizer)
    generated = generate_assistant_ids(
        model,
        prompt_ids,
        eos_id=eos_id,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    if generated and generated[-1] == eos_id:
        generated.pop()
    print(tokenizer.decode(generated))


if __name__ == "__main__":
    main()
