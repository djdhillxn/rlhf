from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class LMRolloutBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    response_mask: torch.Tensor
    old_logprobs: torch.Tensor
    ref_logprobs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    scores: torch.Tensor
    advantages: torch.Tensor | None = None
    returns: torch.Tensor | None = None
    prompts: list[str] | None = None
    responses: list[str] | None = None
    metadata: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.input_ids.ndim != 2:
            raise ValueError(f"input_ids must be rank-2, got {tuple(self.input_ids.shape)}")
        if self.attention_mask.shape != self.input_ids.shape:
            raise ValueError(
                f"attention_mask shape {tuple(self.attention_mask.shape)} must match input_ids {tuple(self.input_ids.shape)}"
            )
        label_shape = (self.input_ids.size(0), max(self.input_ids.size(1) - 1, 0))
        for name in ("response_mask", "old_logprobs", "ref_logprobs", "values", "rewards"):
            value = getattr(self, name)
            if tuple(value.shape) != tuple(label_shape):
                raise ValueError(f"{name} shape {tuple(value.shape)} must be {tuple(label_shape)}")
        if self.scores.ndim != 1 or self.scores.size(0) != self.input_ids.size(0):
            raise ValueError(f"scores shape {tuple(self.scores.shape)} must be ({self.input_ids.size(0)},)")
        if self.advantages is not None and self.advantages.shape != self.response_mask.shape:
            raise ValueError("advantages shape must match response_mask shape")
        if self.returns is not None and self.returns.shape != self.response_mask.shape:
            raise ValueError("returns shape must match response_mask shape")

    def to(self, device: torch.device | str) -> "LMRolloutBatch":
        kwargs = {}
        for field_name, value in self.__dict__.items():
            if torch.is_tensor(value):
                kwargs[field_name] = value.to(device)
            else:
                kwargs[field_name] = value
        return LMRolloutBatch(**kwargs)

    @property
    def batch_size(self) -> int:
        return int(self.input_ids.size(0))

    @property
    def num_response_tokens(self) -> int:
        return int(self.response_mask.sum().item())
