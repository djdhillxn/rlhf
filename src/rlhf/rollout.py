import torch

from .lm_policy import response_label_mask_from_lengths, shifted_token_logprobs
from .ppo_lm import LMRolloutBatch


class GenerationConfig:
    def __init__(
        self,
        max_prompt_length=512,
        max_new_tokens=128,
        min_new_tokens=0,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
    ):
        self.max_prompt_length = max_prompt_length
        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size


def _ensure_pad_token(tokenizer):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return int(tokenizer.pad_token_id)


def _eos_token_ids(tokenizer):
    eos = tokenizer.eos_token_id
    if eos is None:
        return set()
    if isinstance(eos, (list, tuple, set)):
        return {int(x) for x in eos if x is not None}
    return {int(eos)}


def _response_lengths(response_ids, tokenizer):
    """Number of generated tokens to keep per row, including first EOS if present."""
    eos_ids = _eos_token_ids(tokenizer)
    lengths = []
    for row in response_ids.detach().cpu().tolist():
        keep = len(row)
        if eos_ids:
            for idx, token_id in enumerate(row):
                if int(token_id) in eos_ids:
                    keep = idx + 1
                    break
        lengths.append(max(0, keep))
    return torch.tensor(lengths, device=response_ids.device, dtype=torch.long)


def _build_full_attention(prompt_attention, generated, prompt_width, response_lengths):
    full_attention = torch.zeros_like(generated, dtype=torch.long)
    full_attention[:, :prompt_width] = prompt_attention.long()
    if generated.size(1) > prompt_width:
        pos = torch.arange(
            generated.size(1) - prompt_width, device=generated.device
        ).unsqueeze(0)
        full_attention[:, prompt_width:] = (pos < response_lengths.unsqueeze(1)).long()
    return full_attention


@torch.no_grad()
def collect_lm_rollouts(
    policy,
    reference,
    reward_model,
    tokenizer,
    prompts,
    *,
    generation,
    kl_coef,
    device,
    metadata=None,
    reward_clip_min=None,
    reward_clip_max=None,
    length_penalty_coef=0.0,
    missing_eos_penalty=0.0,
    min_response_tokens=0,
    short_response_penalty=0.0,
    group_size=1,
    group_normalize=False,
    group_advantage_eps=1e-6,
):
    """Generate responses and build an on-policy token-level PPO batch."""
    device = torch.device(device)
    pad_id = _ensure_pad_token(tokenizer)
    old_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(generation.max_prompt_length),
    )
    tokenizer.padding_side = old_padding_side
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    prompt_width = int(input_ids.size(1))

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=int(generation.max_new_tokens),
        do_sample=bool(generation.do_sample),
        pad_token_id=pad_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if int(getattr(generation, "min_new_tokens", 0)) > 0:
        gen_kwargs["min_new_tokens"] = int(generation.min_new_tokens)
    if bool(generation.do_sample):
        gen_kwargs["temperature"] = float(generation.temperature)
        gen_kwargs["top_p"] = float(generation.top_p)
    if float(generation.repetition_penalty) != 1.0:
        gen_kwargs["repetition_penalty"] = float(generation.repetition_penalty)
    if int(generation.no_repeat_ngram_size) > 0:
        gen_kwargs["no_repeat_ngram_size"] = int(generation.no_repeat_ngram_size)

    generated = policy.generate(**gen_kwargs)

    response_ids = generated[:, prompt_width:]
    response_lengths = _response_lengths(response_ids, tokenizer)
    full_attention = _build_full_attention(
        attention_mask, generated, prompt_width, response_lengths
    )

    policy_out = policy(generated, full_attention)
    old_logprobs = shifted_token_logprobs(policy_out.logits, generated).detach()
    values = policy_out.values[:, :-1].detach()
    ref_logprobs = reference.token_logprobs(generated, full_attention).detach()
    resp_mask = response_label_mask_from_lengths(
        generated, prompt_width, response_lengths
    )

    expected_shape = resp_mask.shape
    for name, tensor in {
        "old_logprobs": old_logprobs,
        "values": values,
        "ref_logprobs": ref_logprobs,
    }.items():
        if tensor.shape != expected_shape:
            raise RuntimeError(
                f"{name} shape {tuple(tensor.shape)} does not match response_mask {tuple(expected_shape)}"
            )

    raw_scores = reward_model(generated, full_attention).detach().float()
    scores = raw_scores
    if reward_clip_min is not None or reward_clip_max is not None:
        min_v = -float("inf") if reward_clip_min is None else float(reward_clip_min)
        max_v = float("inf") if reward_clip_max is None else float(reward_clip_max)
        scores = scores.clamp(min=min_v, max=max_v)

    eos_ids = _eos_token_ids(tokenizer)
    hit_eos = torch.zeros_like(response_lengths, dtype=torch.bool)
    if eos_ids:
        eos_tensor = torch.tensor(
            sorted(eos_ids), device=response_ids.device, dtype=response_ids.dtype
        )
        hit_eos = (
            (response_ids.unsqueeze(-1) == eos_tensor.view(1, 1, -1))
            .any(dim=-1)
            .any(dim=1)
        )

    terminal_scores = scores - float(length_penalty_coef) * response_lengths.float()
    if float(missing_eos_penalty) != 0.0:
        terminal_scores = (
            terminal_scores - float(missing_eos_penalty) * (~hit_eos).float()
        )
    if int(min_response_tokens) > 0 and float(short_response_penalty) != 0.0:
        shortfall = (int(min_response_tokens) - response_lengths.float()).clamp_min(0.0)
        terminal_scores = terminal_scores - float(
            short_response_penalty
        ) * shortfall / float(max(int(min_response_tokens), 1))

    # Optional group-relative reward baseline.  With one sample per prompt,
    # sequence-level rewards are heavily confounded by prompt/domain difficulty.
    # Repeating each prompt K times and centering scores within each group makes
    # PPO learn which response is better for the same prompt, which is much closer
    # to preference optimization and much less sensitive to prompt mix.
    group_size = max(1, int(group_size))
    if group_size > 1:
        if terminal_scores.numel() % group_size != 0:
            raise ValueError(
                f"Batch size {terminal_scores.numel()} must be divisible by group_size={group_size}"
            )
        grouped = terminal_scores.view(-1, group_size)
        grouped = grouped - grouped.mean(dim=1, keepdim=True)
        if bool(group_normalize) and group_size > 1:
            std = grouped.std(dim=1, unbiased=False, keepdim=True).clamp_min(
                float(group_advantage_eps)
            )
            grouped = grouped / std
        terminal_scores = grouped.reshape_as(terminal_scores)

    kl_per_token = old_logprobs.float() - ref_logprobs.float()
    rewards = (-float(kl_coef) * kl_per_token) * resp_mask.float()

    # Add shaped terminal reward to the final generated token of each sequence.
    for i in range(generated.size(0)):
        positions = torch.nonzero(resp_mask[i], as_tuple=False).flatten()
        if positions.numel() > 0:
            rewards[i, positions[-1]] += terminal_scores[i]

    decoded_responses = []
    for ids, keep in zip(response_ids, response_lengths.tolist()):
        decoded_responses.append(
            tokenizer.decode(ids[: int(keep)], skip_special_tokens=True).strip()
        )

    return LMRolloutBatch(
        input_ids=generated.detach(),
        attention_mask=full_attention.detach(),
        response_mask=resp_mask.detach(),
        old_logprobs=old_logprobs.detach(),
        ref_logprobs=ref_logprobs.detach(),
        values=values.detach(),
        rewards=rewards.detach(),
        scores=raw_scores.detach(),
        prompts=list(prompts),
        responses=decoded_responses,
        metadata=metadata,
    )
