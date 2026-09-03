"""模块用途：以项目固定聊天模板进行可复现的 decoder-only 自回归采样。"""

import torch
from torch import Tensor, nn
from tokenizers import Tokenizer


def build_chat_prompt(system: str, question: str, tokenizer: Tokenizer) -> list[int]:
    """将系统提示与用户问题编码，并让 assistant 控制 token 成为最后一个输入。"""
    if not isinstance(system, str) or not system.strip():
        raise ValueError("system must not be empty")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must not be empty")
    for token in ("<eos>", "<|system|>", "<|user|>", "<|assistant|>"):
        if tokenizer.token_to_id(token) is None:
            raise ValueError(f"tokenizer must contain {token}")
    return tokenizer.encode(
        f"<|system|>{system}<eos><|user|>{question}<eos><|assistant|>"
    ).ids


def apply_repetition_penalty(
    logits: Tensor, generated_ids: list[int], repetition_penalty: float
) -> Tensor:
    """降低已生成 token 的再次概率，不惩罚提示词中的 token。"""
    if logits.ndim != 1:
        raise ValueError("logits must be rank-1")
    if repetition_penalty < 1.0:
        raise ValueError("repetition_penalty must be at least 1.0")
    if repetition_penalty == 1.0 or not generated_ids:
        return logits
    seen_ids = sorted(set(generated_ids))
    if seen_ids[0] < 0 or seen_ids[-1] >= logits.numel():
        raise ValueError("generated_ids must be valid token IDs")
    indices = torch.tensor(seen_ids, device=logits.device)
    adjusted = logits.clone()
    seen_logits = adjusted[indices]
    adjusted[indices] = torch.where(
        seen_logits < 0,
        seen_logits * repetition_penalty,
        seen_logits / repetition_penalty,
    )
    return adjusted


def sample_next_token(
    logits: Tensor,
    temperature: float,
    top_k: int,
    *,
    top_p: float = 1.0,
    generated_ids: list[int] | None = None,
    repetition_penalty: float = 1.0,
) -> int:
    """按温度、top-k、top-p 与重复惩罚从单一位置 logits 采样一个 token。"""
    if logits.ndim != 1:
        raise ValueError("logits must be rank-1")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if top_k < 0:
        raise ValueError("top_k must be nonnegative")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    filtered = apply_repetition_penalty(
        logits, generated_ids or [], repetition_penalty
    ) / temperature
    if top_k > 0:
        k = min(top_k, filtered.numel())
        values, _ = torch.topk(filtered, k)
        filtered = filtered.masked_fill(filtered < values[-1], float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        cumulative_probabilities = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove_sorted = cumulative_probabilities > top_p
        remove_sorted[1:] = remove_sorted[:-1].clone()
        remove_sorted[0] = False
        filtered = filtered.masked_fill(
            torch.zeros_like(remove_sorted).scatter(0, sorted_indices, remove_sorted),
            float("-inf"),
        )
    probabilities = torch.softmax(filtered, dim=-1)
    return int(torch.multinomial(probabilities, num_samples=1))


@torch.inference_mode()
def generate_assistant_ids(
    model: nn.Module,
    prompt_ids: list[int],
    *,
    eos_id: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
) -> list[int]:
    """从 assistant 标记后按可控采样生成 token，遇 EOS 或达到上限时停止。"""
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if not hasattr(model, "config") or not hasattr(model.config, "block_size"):
        raise ValueError("model must expose config.block_size")
    try:
        device = next(model.parameters()).device
    except StopIteration as error:
        raise ValueError("model must have at least one parameter") from error
    context_ids = list(prompt_ids)
    generated: list[int] = []
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            context = torch.tensor([context_ids[-model.config.block_size :]], device=device)
            logits, _ = model(context)
            next_id = sample_next_token(
                logits[0, -1],
                temperature,
                top_k,
                top_p=top_p,
                generated_ids=generated,
                repetition_penalty=repetition_penalty,
            )
            generated.append(next_id)
            context_ids.append(next_id)
            if next_id == eos_id:
                break
    finally:
        model.train(was_training)
    return generated
