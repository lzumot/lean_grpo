"""Group Relative Policy Optimization (GRPO) implementation.

GRPO normalizes rewards within groups of trajectories for the same problem,
eliminating the need for a separate value function.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from lean_grpo.algorithms.base import (
    AdvantageNormalizer,
    PolicyLoss,
    RLAlgorithm,
    RLConfig,
)


@dataclass
class GRPOConfig(RLConfig):
    """Configuration for GRPO.
    
    GRPO uses group-relative advantages:
    - For each theorem, generate G rollouts
    - Compute mean reward within group
    - Advantage = (reward - mean) / std
    """
    
    # Group size for GRPO
    group_size: int = 8
    
    # Advantage normalization
    normalize_advantages: bool = True
    advantage_normalization_method: str = "mean_std"  # 'mean_std', 'max_min', 'none'
    
    # Whether to use importance sampling correction
    use_importance_sampling: bool = True
    
    # Truncated importance sampling (for stability)
    truncated_is: Optional[float] = None  # Upper bound for IS ratio
    
    # Token-level vs sequence-level importance sampling
    importance_sampling_level: str = "token"  # 'token', 'sequence', 'average'
    
    # Max negative advantage IS weight
    max_negative_advantage_is_weight: Optional[float] = None


class GRPO(RLAlgorithm):
    """Group Relative Policy Optimization.
    
    GRPO is designed for LLM training where:
    1. We generate multiple completions for the same prompt (group)
    2. Compute advantages relative to the group mean
    3. No value function needed (advantages come from reward normalization)
    
    Reference: DeepSeekMath paper
    """
    
    def __init__(self, config: Optional[GRPOConfig] = None, **kwargs):
        """Initialize GRPO.
        
        Args:
            config: GRPO configuration
            **kwargs: Override config fields
        """
        if config is None:
            config = GRPOConfig(**kwargs)
        super().__init__(config)
        self.config: GRPOConfig = config
    
    def compute_advantages(
        self,
        rewards: list[float],
        **kwargs
    ) -> dict[str, float]:
        """Compute group-relative advantages.
        
        For GRPO, advantages are computed as:
        A_i = (R_i - mean(R)) / std(R)
        
        Args:
            rewards: List of rewards for trajectories in the same group
            
        Returns:
            Dict mapping trajectory indices to advantages
        """
        if not rewards:
            return {}
        
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        
        if self.config.normalize_advantages:
            advantages = AdvantageNormalizer.normalize(
                rewards_tensor,
                method=self.config.advantage_normalization_method
            )
        else:
            advantages = rewards_tensor
        
        return {i: adv.item() for i, adv in enumerate(advantages)}
    
    def compute_loss(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        **kwargs
    ) -> PolicyLoss:
        """Compute GRPO policy loss.
        
        The loss is:
        L = -E[min(ratio * A, clip(ratio) * A)] + beta * KL
        
        where ratio = exp(new_logprob - old_logprob)
        
        Args:
            model: Policy model
            batch: Contains:
                - input_ids: [B, S] token IDs
                - attention_mask: [B, S] mask
                - old_logprobs: [B, S] old policy logprobs
                - advantages: [B] or [B, S] advantages
                - reference_logprobs: [B, S] reference logprobs (optional)
                - group_ids: [B] group assignments (optional)
                
        Returns:
            PolicyLoss with all components
        """
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        old_logprobs = batch["old_logprobs"]
        advantages = batch["advantages"]
        ref_logprobs = batch.get("reference_logprobs")
        
        # Get new log probabilities
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits
        
        # Compute log probabilities for tokens
        log_probs = self._compute_log_probs(logits, input_ids)
        
        # Mask for valid tokens
        if attention_mask is not None:
            mask = attention_mask.bool()
        else:
            mask = torch.ones_like(input_ids, dtype=torch.bool)
        
        # Compute importance sampling ratio
        log_ratio = log_probs - old_logprobs
        ratio = torch.exp(log_ratio)
        
        # Apply importance sampling constraints
        if self.config.use_importance_sampling:
            ratio = self._apply_is_constraints(ratio, log_ratio, advantages, mask)
        
        # Expand advantages to token level if needed
        if advantages.dim() == 1:
            # [B] -> [B, 1] for broadcasting
            advantages = advantages.unsqueeze(-1)
        
        # PPO-style clipped loss
        epsilon = self.config.epsilon
        epsilon_high = self.config.epsilon_high
        
        clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon_high)
        
        policy_loss1 = -ratio * advantages
        policy_loss2 = -clipped_ratio * advantages
        policy_loss = torch.max(policy_loss1, policy_loss2)
        
        # Apply mask and reduce
        policy_loss = policy_loss * mask
        
        if self.config.reduce_loss == "mean":
            policy_loss = policy_loss.sum() / mask.sum()
        elif self.config.reduce_loss == "sum":
            policy_loss = policy_loss.sum()
        # else: keep per-token losses
        
        # KL penalty
        kl_div = torch.zeros_like(policy_loss)
        if self.config.beta > 0 and ref_logprobs is not None:
            kl_per_token = self.compute_kl_penalty(log_probs, ref_logprobs)
            kl_div = (kl_per_token * mask).sum() / mask.sum()
        
        # Total loss
        loss = policy_loss + self.config.beta * kl_div
        
        # Compute metrics
        metrics = {
            "policy_loss": policy_loss.item(),
            "kl_div": kl_div.item() if isinstance(kl_div, torch.Tensor) else 0.0,
            "mean_ratio": ratio[mask].mean().item(),
            "clip_fraction": (ratio[mask] > 1 + epsilon).float().mean().item(),
        }
        
        return PolicyLoss(
            loss=loss,
            policy_loss=policy_loss,
            kl_div=kl_div if isinstance(kl_div, torch.Tensor) else torch.tensor(0.0),
            metrics=metrics,
        )
    
    def _compute_log_probs(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probabilities for tokens.
        
        Args:
            logits: [B, S, V] model logits
            input_ids: [B, S] token IDs
            
        Returns:
            [B, S] log probabilities
        """
        # Shift for next token prediction
        logits = logits[:, :-1, :]
        target_ids = input_ids[:, 1:]
        
        # Compute log softmax
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Gather log probs for actual tokens
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=target_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        # Pad to match original shape
        token_log_probs = torch.nn.functional.pad(
            token_log_probs,
            (1, 0),
            value=0.0
        )
        
        return token_log_probs
    
    def _apply_is_constraints(
        self,
        ratio: torch.Tensor,
        log_ratio: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply importance sampling constraints.
        
        Args:
            ratio: Importance sampling ratios
            log_ratio: Log ratios
            advantages: Advantages
            mask: Valid token mask
            
        Returns:
            Constrained ratios
        """
        # Max negative advantage IS weight
        if self.config.max_negative_advantage_is_weight is not None:
            # For negative advantages, limit how large ratio can be
            ratio = torch.where(
                advantages.unsqueeze(-1) < 0,
                torch.clamp(ratio, max=self.config.max_negative_advantage_is_weight),
                ratio
            )
        
        # Truncated importance sampling
        if self.config.truncated_is is not None:
            # Detach and clamp the ratio for stability
            ratio = ratio * torch.clamp(
                ratio.detach(),
                max=self.config.truncated_is
            )
        
        return ratio
    
    def compute_group_advantages(
        self,
        rewards: list[float],
        group_assignments: list[int],
    ) -> dict[int, float]:
        """Compute advantages per group.
        
        Args:
            rewards: All rewards
            group_assignments: Group ID for each reward
            
        Returns:
            Dict mapping indices to advantages
        """
        import numpy as np
        
        rewards_array = np.array(rewards)
        groups_array = np.array(group_assignments)
        
        advantages = {}
        
        for group_id in np.unique(groups_array):
            mask = groups_array == group_id
            group_rewards = rewards_array[mask]
            
            mean_reward = group_rewards.mean()
            std_reward = group_rewards.std()
            
            if std_reward < 1e-8:
                group_advantages = np.zeros_like(group_rewards)
            else:
                group_advantages = (group_rewards - mean_reward) / std_reward
            
            # Assign back
            indices = np.where(mask)[0]
            for idx, adv in zip(indices, group_advantages):
                advantages[int(idx)] = float(adv)
        
        return advantages
