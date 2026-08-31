"""模块用途：从已保存的 GPT checkpoint 以固定随机种子生成下一个 token。"""

import argparse
from pathlib import Path

import torch

from sinogpt.config import ModelConfig
from sinogpt.model.gpt import GPTLanguageModel
from sinogpt.seed import seed_everything
from sinogpt.tokenizer import load_tokenizer
from sinogpt.training.checkpoint import load_checkpoint


def build_parser() -> argparse.ArgumentParser:
    """构建 checkpoint 采样的命令行参数。"""
    parser = argparse.ArgumentParser(description="从 SinoGPT checkpoint 生成文本")
    parser.add_argument("--checkpoint", required=True, type=Path, help="checkpoint 文件路径")
    parser.add_argument("--prompt", required=True, help="生成的起始文本")
    parser.add_argument("--tokens", type=int, default=32, help="追加生成 token 数")
    parser.add_argument("--seed", type=int, default=17, help="采样随机种子")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax 温度")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser


def main() -> None:
    """加载 checkpoint、编码提示词并按自回归方式输出完整文本。"""
    args = build_parser().parse_args()
    if args.tokens < 1 or args.temperature <= 0.0:
        raise ValueError("tokens and temperature must be positive")
    checkpoint = load_checkpoint(args.checkpoint)
    model_config = ModelConfig(**checkpoint["model_config"])
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if args.device == "auto" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    tokenizer_path = Path(checkpoint["data_config"]["tokenizer_path"])
    tokenizer = load_tokenizer(tokenizer_path)
    prompt_ids = tokenizer.encode(args.prompt).ids
    if not prompt_ids:
        raise ValueError("prompt must contain at least one token")

    seed_everything(args.seed)
    model = GPTLanguageModel(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    generated = list(prompt_ids)
    with torch.no_grad():
        for _ in range(args.tokens):
            context = torch.tensor([generated[-model_config.block_size :]], device=device)
            logits, _ = model(context)
            probabilities = torch.softmax(logits[0, -1] / args.temperature, dim=-1)
            generated.append(int(torch.multinomial(probabilities, num_samples=1)))
    print(tokenizer.decode(generated))


if __name__ == "__main__":
    main()
