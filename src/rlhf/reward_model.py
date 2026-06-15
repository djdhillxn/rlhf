from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass
class ModelLoadConfig:
    model_name: str
    torch_dtype: str = "auto"
    device_map: str | None = None
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    trust_remote_code: bool = False


def resolve_dtype(name: str | None):
    if name is None or str(name).lower() == "auto":
        return "auto"
    name = str(name).lower()
    if name in {"float16", "fp16", "half"}:
        return torch.float16
    if name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if name in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {name}")


def build_quantization_config(load_in_4bit: bool = False, load_in_8bit: bool = False):
    if not load_in_4bit and not load_in_8bit:
        return None
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:  # pragma: no cover
        raise ImportError("bitsandbytes/transformers quantization support is required for 4/8-bit loading") from exc
    if load_in_4bit and load_in_8bit:
        raise ValueError("Choose only one of load_in_4bit or load_in_8bit.")
    if load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            bnb_4bit_use_double_quant=True,
        )
    return BitsAndBytesConfig(load_in_8bit=True)


def build_lora_config(cfg: dict[str, Any] | None):
    if not cfg or not bool(cfg.get("enabled", True)):
        return None
    try:
        from peft import LoraConfig
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install PEFT first: pip install peft") from exc
    return LoraConfig(
        r=int(cfg.get("r", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        bias=str(cfg.get("bias", "none")),
        task_type=str(cfg.get("task_type", "CAUSAL_LM")),
        target_modules=list(
            cfg.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
        ),
    )


def load_causal_lm(load_cfg: ModelLoadConfig):
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {"trust_remote_code": load_cfg.trust_remote_code}
    dtype = resolve_dtype(load_cfg.torch_dtype)
    if dtype != "auto":
        # Newer Transformers versions prefer `dtype`; older versions accepted
        # `torch_dtype`. Colab currently warns on torch_dtype, so use dtype.
        kwargs["torch_dtype"] = dtype
    quant_cfg = build_quantization_config(load_cfg.load_in_4bit, load_cfg.load_in_8bit)
    if quant_cfg is not None:
        kwargs["quantization_config"] = quant_cfg
    if load_cfg.device_map is not None:
        kwargs["device_map"] = load_cfg.device_map
    return AutoModelForCausalLM.from_pretrained(load_cfg.model_name, **kwargs)


def transformer_core(backbone: nn.Module) -> nn.Module | None:
    """Return the decoder backbone without the LM head when available.

    Reward modeling only needs hidden states, not full vocabulary logits. Calling
    AutoModelForCausalLM directly materializes [batch, seq, vocab] logits, which
    is slow and memory-heavy. For Qwen/Llama-style models, the causal LM exposes
    the decoder as `.model`. For PEFT-wrapped models, the original causal LM sits
    under `base_model.model`. The returned module still contains LoRA layers, so
    reward-model training remains correct.
    """
    module = backbone
    base_model = getattr(module, "base_model", None)
    if base_model is not None and hasattr(base_model, "model"):
        module = base_model.model
    core = getattr(module, "model", None)
    if isinstance(core, nn.Module):
        return core
    return None


def forward_hidden_states(backbone: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    core = transformer_core(backbone)
    if core is not None:
        outputs = core(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            return outputs.hidden_states[-1]

    outputs = backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    return outputs.hidden_states[-1]


def last_attended_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    """Return the absolute index of the last attended token for each row.

    This works for both right- and left-padded batches. The previous implementation
    used attention_mask.sum(dim=1)-1, which is only correct for right padding and
    is wrong for left-padded generation batches.
    """
    if attention_mask.ndim != 2:
        raise ValueError(f"attention_mask must be rank-2, got {tuple(attention_mask.shape)}")
    positions = torch.arange(attention_mask.size(1), device=attention_mask.device).unsqueeze(0)
    masked_positions = positions.masked_fill(~attention_mask.bool(), 0)
    return masked_positions.max(dim=1).values.long()


class RewardModel(nn.Module):
    """Causal LM backbone plus scalar reward head on the last attended token."""

    def __init__(self, backbone: nn.Module, hidden_size: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.reward_head = nn.Linear(hidden_size, 1)

    @classmethod
    def from_model_name(
        cls,
        model_name: str,
        *,
        torch_dtype: str = "auto",
        device_map: str | None = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        lora: dict[str, Any] | None = None,
        gradient_checkpointing: bool = True,
        trust_remote_code: bool = False,
    ) -> "RewardModel":
        backbone = load_causal_lm(
            ModelLoadConfig(
                model_name=model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                trust_remote_code=trust_remote_code,
            )
        )
        if gradient_checkpointing and hasattr(backbone, "gradient_checkpointing_enable"):
            backbone.gradient_checkpointing_enable()
            if hasattr(backbone.config, "use_cache"):
                backbone.config.use_cache = False
        if load_in_4bit or load_in_8bit:
            from peft import prepare_model_for_kbit_training

            backbone = prepare_model_for_kbit_training(backbone)
        lora_cfg = build_lora_config(lora)
        if lora_cfg is not None:
            from peft import get_peft_model

            backbone = get_peft_model(backbone, lora_cfg)
        hidden_size = int(getattr(backbone.config, "hidden_size", getattr(backbone.config, "n_embd", 0)))
        if hidden_size <= 0:
            raise ValueError("Could not infer model hidden size from config.")
        return cls(backbone=backbone, hidden_size=hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = forward_hidden_states(self.backbone, input_ids, attention_mask)
        last_idx = last_attended_indices(attention_mask)
        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        pooled = hidden[batch_idx, last_idx].to(self.reward_head.weight.dtype)
        return self.reward_head(pooled).squeeze(-1)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def save_rlhf_pretrained(self, output_dir: str | Path, tokenizer: Any | None = None) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(self.backbone, "save_pretrained"):
            self.backbone.save_pretrained(output_dir / "adapter_or_model")
        torch.save(self.reward_head.state_dict(), output_dir / "reward_head.pt")
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir / "tokenizer")

    @classmethod
    def load_rlhf_pretrained(
        cls,
        checkpoint_dir: str | Path,
        *,
        base_model_name: str,
        torch_dtype: str = "auto",
        device_map: str | None = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        trust_remote_code: bool = False,
        is_trainable: bool = False,
        strict: bool = True,
    ) -> "RewardModel":
        checkpoint_dir = Path(checkpoint_dir)
        if strict and not checkpoint_dir.exists():
            raise FileNotFoundError(f"Reward checkpoint directory does not exist: {checkpoint_dir}")
        model = cls.from_model_name(
            base_model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            lora=None,
            gradient_checkpointing=False,
            trust_remote_code=trust_remote_code,
        )
        adapter_dir = checkpoint_dir / "adapter_or_model"
        if adapter_dir.exists():
            from peft import PeftModel

            model.backbone = PeftModel.from_pretrained(model.backbone, adapter_dir, is_trainable=is_trainable)
        elif strict:
            raise FileNotFoundError(
                f"Reward adapter/model directory missing: {adapter_dir}. "
                "Check that checkpoint_dir points at a saved reward checkpoint."
            )
        reward_head_path = checkpoint_dir / "reward_head.pt"
        if reward_head_path.exists():
            model.reward_head.load_state_dict(torch.load(reward_head_path, map_location="cpu"))
        else:
            raise FileNotFoundError(f"Missing reward head: {reward_head_path}")
        return model
