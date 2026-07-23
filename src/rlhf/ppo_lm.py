import random

import torch

from .lm_policy import shifted_token_logprobs
from .metrics import explained_variance, masked_mean


class LMRolloutBatch:
    def __init__(
        self,
        input_ids,
        attention_mask,
        response_mask,
        old_logprobs,
        ref_logprobs,
        values,
        rewards,
        scores,
        advantages=None,
        returns=None,
        prompts=None,
        responses=None,
        metadata=None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.response_mask = response_mask
        self.old_logprobs = old_logprobs
        self.ref_logprobs = ref_logprobs
        self.values = values
        self.rewards = rewards
        self.scores = scores
        self.advantages = advantages
        self.returns = returns
        self.prompts = prompts
        self.responses = responses
        self.metadata = metadata
        self.validate()

    def validate(self):
        if self.input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must be rank-2, got {tuple(self.input_ids.shape)}"
            )
        if self.attention_mask.shape != self.input_ids.shape:
            raise ValueError(
                f"attention_mask shape {tuple(self.attention_mask.shape)} must match input_ids {tuple(self.input_ids.shape)}"
            )
        label_shape = (self.input_ids.size(0), max(self.input_ids.size(1) - 1, 0))
        for name in (
            "response_mask",
            "old_logprobs",
            "ref_logprobs",
            "values",
            "rewards",
        ):
            value = getattr(self, name)
            if tuple(value.shape) != tuple(label_shape):
                raise ValueError(
                    f"{name} shape {tuple(value.shape)} must be {tuple(label_shape)}"
                )
        if self.scores.ndim != 1 or self.scores.size(0) != self.input_ids.size(0):
            raise ValueError(
                f"scores shape {tuple(self.scores.shape)} must be ({self.input_ids.size(0)},)"
            )
        if (
            self.advantages is not None
            and self.advantages.shape != self.response_mask.shape
        ):
            raise ValueError("advantages shape must match response_mask shape")
        if self.returns is not None and self.returns.shape != self.response_mask.shape:
            raise ValueError("returns shape must match response_mask shape")

    def to(self, device):
        kwargs = {}
        for field_name, value in self.__dict__.items():
            if torch.is_tensor(value):
                kwargs[field_name] = value.to(device)
            else:
                kwargs[field_name] = value
        return LMRolloutBatch(**kwargs)

    @property
    def batch_size(self):
        return int(self.input_ids.size(0))

    @property
    def num_response_tokens(self):
        return int(self.response_mask.sum().item())


class LMPPOStats:
    def __init__(
        self,
        loss,
        policy_loss,
        value_loss,
        approx_kl,
        clip_fraction,
        entropy,
        reward_model_score,
        non_score_reward,
        objective_kl,
        abs_ref_logratio,
        total_reward,
        value_explained_variance,
        kl_coef,
        num_response_tokens,
    ):
        self.loss = loss
        self.policy_loss = policy_loss
        self.value_loss = value_loss
        self.approx_kl = approx_kl
        self.clip_fraction = clip_fraction
        self.entropy = entropy
        self.reward_model_score = reward_model_score
        self.non_score_reward = non_score_reward
        self.objective_kl = objective_kl
        self.abs_ref_logratio = abs_ref_logratio
        self.total_reward = total_reward
        self.value_explained_variance = value_explained_variance
        self.kl_coef = kl_coef
        self.num_response_tokens = num_response_tokens


def compute_gae(
    rewards,
    values,
    mask,
    *,
    gamma=1.0,
    lam=0.95,
):
    """GAE over generated response tokens."""
    rewards = rewards.float()
    values = values.float()
    mask_f = mask.float()
    bsz, seq_len = rewards.shape
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros(bsz, device=rewards.device)
    for t in reversed(range(seq_len)):
        if t == seq_len - 1:
            next_values = torch.zeros(bsz, device=rewards.device)
            next_nonterminal = torch.zeros(bsz, device=rewards.device)
        else:
            next_values = values[:, t + 1]
            next_nonterminal = mask_f[:, t + 1]
        delta = rewards[:, t] + gamma * next_values * next_nonterminal - values[:, t]
        lastgaelam = delta + gamma * lam * next_nonterminal * lastgaelam
        lastgaelam = lastgaelam * mask_f[:, t]
        advantages[:, t] = lastgaelam
    returns = advantages + values
    return advantages * mask_f, returns * mask_f


def normalize_advantages(advantages, mask, eps=1e-8):
    valid = mask.bool()
    out = advantages.clone()
    if valid.sum() > 1:
        mean = out[valid].mean()
        std = out[valid].std(unbiased=False).clamp_min(eps)
        out[valid] = (out[valid] - mean) / std
    out[~valid] = 0
    return out


class AdaptiveKLController:
    def __init__(
        self,
        init_kl_coef=0.05,
        target_kl=0.05,
        horizon=10000,
        min_kl_coef=0.02,
        max_kl_coef=1.0,
        adaptive=True,
    ):
        self.value = float(init_kl_coef)
        self.target = float(target_kl)
        self.horizon = max(1, int(horizon))
        self.min_value = float(min_kl_coef)
        self.max_value = float(max_kl_coef)
        self.adaptive = bool(adaptive)
        self.value = float(min(max(self.value, self.min_value), self.max_value))

    def update(self, measured_kl, n_steps):
        # Empirical sampled log-ratio estimates can be negative on small batches,
        # even though the true KL is non-negative in expectation.  Negative noisy
        # estimates previously drove the KL coefficient almost to zero, allowing
        # the LM policy to drift into non-language / reward-hacking modes.
        if (not self.adaptive) or self.target <= 0:
            return self.value
        measured = max(float(measured_kl), 0.0)
        proportional_error = max(min(measured / self.target - 1.0, 0.2), -0.2)
        mult = 1.0 + proportional_error * float(n_steps) / float(self.horizon)
        self.value = float(min(max(self.value * mult, self.min_value), self.max_value))
        return self.value


class LMPPOTrainer:
    def __init__(self, policy, cfg):
        self.policy = policy
        self.cfg = cfg
        self.clip_range = float(cfg.get("clip_range", 0.2))
        self.value_clip_range = cfg.get("value_clip_range", 0.2)
        self.value_clip_range = (
            None if self.value_clip_range is None else float(self.value_clip_range)
        )
        self.value_coef = float(cfg.get("value_coef", 0.5))
        self.entropy_coef = float(cfg.get("entropy_coef", 0.0))
        self.learning_rate = float(cfg.get("learning_rate", 1e-5))
        self.weight_decay = float(cfg.get("weight_decay", 0.0))
        self.max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
        self.ppo_epochs = int(cfg.get("ppo_epochs", 4))
        self.minibatch_size = int(cfg.get("minibatch_size", 4))
        self.target_policy_kl = float(cfg.get("target_policy_kl", 0.05))
        self.gamma = float(cfg.get("gamma", 1.0))
        self.lam = float(cfg.get("lam", 0.95))
        self.optimizer = torch.optim.AdamW(
            self.policy.trainable_parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def prepare_batch(self, batch):
        advantages, returns = compute_gae(
            batch.rewards,
            batch.values,
            batch.response_mask,
            gamma=self.gamma,
            lam=self.lam,
        )
        advantages = normalize_advantages(advantages, batch.response_mask)
        batch.advantages = advantages.detach()
        batch.returns = returns.detach()
        return batch

    def update(self, batch, *, kl_coef):
        if batch.advantages is None or batch.returns is None:
            batch = self.prepare_batch(batch)
        self.policy.train()
        n = batch.batch_size
        indices = list(range(n))
        last_metrics = {}
        stop_early = False

        for _epoch in range(self.ppo_epochs):
            random.shuffle(indices)
            for start in range(0, n, self.minibatch_size):
                mb_idx = indices[start : start + self.minibatch_size]
                if not mb_idx:
                    continue
                idx = torch.tensor(
                    mb_idx, device=batch.input_ids.device, dtype=torch.long
                )
                metrics = self._update_minibatch(batch, idx)
                last_metrics = metrics
                if metrics["approx_kl"] > 1.5 * self.target_policy_kl:
                    stop_early = True
                    break
            if stop_early:
                break

        with torch.no_grad():
            current_out = self.policy(batch.input_ids, batch.attention_mask)
            current_logprobs = shifted_token_logprobs(
                current_out.logits, batch.input_ids
            )
            current_ref_logratio = current_logprobs.float() - batch.ref_logprobs.float()
            rollout_ref_logratio = (
                batch.old_logprobs.float() - batch.ref_logprobs.float()
            )
            objective_kl = masked_mean(current_ref_logratio, batch.response_mask).item()
            abs_ref_logratio = masked_mean(
                current_ref_logratio.abs(), batch.response_mask
            ).item()
            non_score_reward = masked_mean(
                -float(kl_coef) * rollout_ref_logratio, batch.response_mask
            ).item()
            total_reward = masked_mean(batch.rewards, batch.response_mask).item()
            value_ev = explained_variance(
                batch.values, batch.returns, batch.response_mask
            )

        return LMPPOStats(
            loss=float(last_metrics.get("loss", float("nan"))),
            policy_loss=float(last_metrics.get("policy_loss", float("nan"))),
            value_loss=float(last_metrics.get("value_loss", float("nan"))),
            approx_kl=float(last_metrics.get("approx_kl", float("nan"))),
            clip_fraction=float(last_metrics.get("clip_fraction", float("nan"))),
            entropy=float(last_metrics.get("entropy", float("nan"))),
            reward_model_score=float(batch.scores.float().mean().item()),
            non_score_reward=float(non_score_reward),
            objective_kl=float(objective_kl),
            abs_ref_logratio=float(abs_ref_logratio),
            total_reward=float(total_reward),
            value_explained_variance=float(value_ev),
            kl_coef=float(kl_coef),
            num_response_tokens=batch.num_response_tokens,
        )

    def _update_minibatch(self, batch, idx):
        input_ids = batch.input_ids.index_select(0, idx)
        attention_mask = batch.attention_mask.index_select(0, idx)
        mask = batch.response_mask.index_select(0, idx)
        old_logprobs = batch.old_logprobs.index_select(0, idx)
        old_values = batch.values.index_select(0, idx)
        advantages = batch.advantages.index_select(0, idx)
        returns = batch.returns.index_select(0, idx)

        out = self.policy(input_ids, attention_mask)
        new_logprobs = shifted_token_logprobs(out.logits, input_ids)
        new_values = out.values[:, :-1]

        logratio = new_logprobs - old_logprobs
        ratio = torch.exp(logratio)
        unclipped = ratio * advantages
        clipped = (
            torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
            * advantages
        )
        policy_loss = -masked_mean(torch.minimum(unclipped, clipped), mask)

        if self.value_clip_range is None:
            value_pred = new_values
        else:
            value_pred = old_values + torch.clamp(
                new_values - old_values, -self.value_clip_range, self.value_clip_range
            )
        value_loss_unclipped = (new_values - returns) ** 2
        value_loss_clipped = (value_pred - returns) ** 2
        value_loss = 0.5 * masked_mean(
            torch.maximum(value_loss_unclipped, value_loss_clipped), mask
        )

        entropy = torch.tensor(0.0, device=input_ids.device)
        if self.entropy_coef > 0:
            # Full-vocab entropy is expensive but available for small models/minibatches.
            probs = torch.softmax(out.logits[:, :-1, :].float(), dim=-1)
            log_probs = torch.log_softmax(out.logits[:, :-1, :].float(), dim=-1)
            token_entropy = -(probs * log_probs).sum(dim=-1)
            entropy = masked_mean(token_entropy, mask)

        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.policy.trainable_parameters(), self.max_grad_norm
            )
        self.optimizer.step()

        with torch.no_grad():
            # Schulman-style sampled approximate KL for old policy -> new policy.
            # This is more stable than mean(old_logp - new_logp), which can be
            # negative on small minibatches and fail to trigger early stopping.
            approx_kl_tensor = (ratio - 1.0) - logratio
            approx_kl = masked_mean(approx_kl_tensor, mask).clamp_min(0.0).item()
            clip_fraction = masked_mean(
                (torch.abs(ratio - 1.0) > self.clip_range).float(), mask
            ).item()
        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "approx_kl": float(approx_kl),
            "clip_fraction": float(clip_fraction),
            "entropy": float(entropy.item()),
        }
