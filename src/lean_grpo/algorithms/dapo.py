"""Decoupled Advantage Policy Optimization (DAPO) implementation.

DAPO decouples the advantage estimation from the policy optimization,
allowing for more flexible and stable training.

Key features:
1. Separate advantage and policy networks
2. Decoupled gradient updates
3. Population-based advantage normalization
4. Support for sparse rewards
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from lean_grpo.algorithms.base import PolicyLoss, RLAlgorithm, RLConfig


@dataclass
class DAPOConfig(RLConfig):
    """Configuration for DAPO.
    
    DAPO uses decoupled optimization:
    - Separate learning rates for advantage and policy
    - Population-based normalization
    - Sparse reward handling
    """
    
    # Group size (like GRPO)
    group_size: int = 8
    
    # Decoupled learning rates
    advantage_lr: float = 1e-4
    policy_lr_multiplier: float = 1.0  # policy_lr = lr * multiplier
    
    # Population-based normalization
    use_population_norm: bool = True
    population_size: int = 100  # Number of groups for normalization stats
    
    # EMA for advantage normalization
    advantage_ema_decay: float = 0.99
    
    # Sparse reward handling
    sparse_reward_threshold: float = 0.9  # Threshold for considering reward as sparse
    sparse_reward_bonus: float = 0.1  # Bonus for non-zero rewards in sparse setting
    
    # Decoupled clipping
    advantage_clip: Optional[float] = 5.0
    ratio_clip_high: float = 1.5  # Higher clip bound for positive advantages
    ratio_clip_low: float = 0.5   # Lower clip bound for negative advantages
    
    # Asymmetric loss
    use_asymmetric_loss: bool = True
    positive_advantage_scale: float = 1.0
    negative_advantage_scale: float = 0.8
    
    # Critic (value function) settings
    use_critic: bool = False  # DAPO can optionally use a critic
    critic_coef: float = 0.5
    
    # Reward shaping
    use_reward_shaping: bool = False
    shaping_coef: float = 0.1


class DAPO(RLAlgorithm):
    """Decoupled Advantage Policy Optimization.
    
    DAPO improves upon GRPO by:
    1. Decoupling advantage estimation from policy updates
    2. Using population statistics for more stable normalization
    3. Asymmetric loss for positive vs negative advantages
    4. Better handling of sparse rewards
    """
    
    def __init__(self, config: Optional[DAPOConfig] = None, **kwargs):
        """Initialize DAPO.
        
        Args:
            config: DAPO configuration
            **kwargs: Override config fields
        """
        if config is None:
            config = DAPOConfig(**kwargs)
        super().__init__(config)
        self.config: DAPOConfig = config
        
        # Running statistics for population-based normalization
        self._reward_mean = 0.0
        self._reward_var = 1.0
        self._num_groups = 0
        
        # For sparse reward detection
        self._reward_sparsity = 0.0
    
    def compute_advantages(
        self,
        rewards: list[float],
        **kwargs
    ) -> dict[str, float]:
        """Compute decoupled advantages.
        
        Uses population-based normalization for more stable estimates.
        
        Args:
            rewards: List of rewards
            **kwargs: May contain 'population_stats' for global normalization
            
        Returns:
            Dict mapping indices to advantages
        """
        if not rewards:
            return {}
        
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        
        # Update population statistics
        self._update_population_stats(rewards_tensor)
        
        # Check for sparse rewards
        is_sparse = self._detect_sparse_rewards(rewards_tensor)
        
        # Compute advantages
        if self.config.use_population_norm and self._num_groups > 10:
            # Use population statistics
            mean = self._reward_mean
            std = (self._reward_var ** 0.5)
        else:
            # Use group statistics (like GRPO)
            mean = rewards_tensor.mean()
            std = rewards_tensor.std()
        
        if std < 1e-8:
            advantages = torch.zeros_like(rewards_tensor)
        else:
            advantages = (rewards_tensor - mean) / std
        
        # Apply advantage clipping
        if self.config.advantage_clip is not None:
            advantages = torch.clamp(
                advantages,
                -self.config.advantage_clip,
                self.config.advantage_clip
            )
        
        # Sparse reward handling
        if is_sparse and self.config.use_reward_shaping:
            # Give bonus for non-zero rewards
            non_zero_mask = rewards_tensor != 0
            advantages = advantages + non_zero_mask.float() * self.config.sparse_reward_bonus
        
        return {i: adv.item() for i, adv in enumerate(advantages)}
    
    def _update_population_stats(self, rewards: torch.Tensor):
        """Update running population statistics.
        
        Args:
            rewards: Tensor of rewards from current group
        """
        group_mean = rewards.mean().item()
        group_var = rewards.var().item()
        
        # EMA update
        decay = self.config.advantage_ema_decay
        self._reward_mean = decay * self._reward_mean + (1 - decay) * group_mean
        self._reward_var = decay * self._reward_var + (1 - decay) * group_var
        self._num_groups += 1
    
    def _detect_sparse_rewards(self, rewards: torch.Tensor) -> bool:
        """Detect if rewards are sparse.
        
        Args:
            rewards: Tensor of rewards
            
        Returns:
            True if rewards appear to be sparse
        """
        # Update sparsity estimate
        non_zero_ratio = (rewards != 0).float().mean().item()
        self._reward_sparsity = 0.9 * self._reward_sparsity + 0.1 * (1 - non_zero_ratio)
        
        return self._reward_sparsity > self.config.sparse_reward_threshold
    
    def compute_loss(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        **kwargs
    ) -> PolicyLoss:
        """Compute DAPO policy loss.
        
        Uses decoupled asymmetric clipping and population-based advantages.
        
        Args:
            model: Policy model
            batch: Batch data
            **kwargs: Additional arguments
            
        Returns:
            PolicyLoss
        """
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        old_logprobs = batch["old_logprobs"]
        advantages = batch["advantages"]
        ref_logprobs = batch.get("reference_logprobs")
        
        # Get new logprobs
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        log_probs = self._compute_log_probs(logits, input_ids)
        
        # Mask
        if attention_mask is not None:
            mask = attention_mask.bool()
        else:
            mask = torch.ones_like(input_ids, dtype=torch.bool)
        
        # Ratio
        log_ratio = log_probs - old_logprobs
        ratio = torch.exp(log_ratio)
        
        # Expand advantages to token level
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(-1)
        
        # Asymmetric clipping
        if self.config.use_asymmetric_loss:
            policy_loss = self._compute_asymmetric_loss(ratio, advantages, mask)
        else:
            policy_loss = self._compute_standard_loss(ratio, advantages, mask)
        
        # KL penalty
        kl_div = torch.tensor(0.0)
        if self.config.beta > 0 and ref_logprobs is not None:
            kl_div = self.compute_kl_penalty(log_probs, ref_logprobs)
            kl_div = (kl_div * mask).sum() / mask.sum()
        
        loss = policy_loss + self.config.beta * kl_div
        
        # Optional critic loss
        value_loss = None
        if self.config.use_critic and "returns" in batch and "values" in batch:
            value_loss = self._compute_value_loss(batch)
            loss = loss + self.config.critic_coef * value_loss
        
        metrics = {
            "policy_loss": policy_loss.item(),
            "kl_div": kl_div.item(),
            "mean_ratio": ratio[mask].mean().item(),
            "mean_advantage": advantages[mask].mean().item(),
            "population_mean": self._reward_mean,
            "population_std": self._reward_var ** 0.5,
        }
        
        if value_loss is not None:
            metrics["value_loss"] = value_loss.item()
        
        return PolicyLoss(
            loss=loss,
            policy_loss=policy_loss,
            kl_div=kl_div,
            value_loss=value_loss,
            metrics=metrics,
        )
    
    def _compute_asymmetric_loss(
        self,
        ratio: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute asymmetric loss for positive vs negative advantages.
        
        Args:
            ratio: Importance sampling ratios
            advantages: Advantages
            mask: Valid token mask
            
        Returns:
            Loss tensor
        """
        # Separate positive and negative advantages
        pos_mask = (advantages > 0) & mask
        neg_mask = (advantages <= 0) & mask
        
        # Asymmetric clipping
        clip_high = 1 + self.config.ratio_clip_high
        clip_low = 1 - self.config.ratio_clip_low
        
        # Positive advantages: use higher clip bound
        pos_clipped = torch.clamp(ratio, max=clip_high)
        pos_loss1 = -ratio * advantages * pos_mask.float()
        pos_loss2 = -pos_clipped * advantages * pos_mask.float()
        pos_loss = torch.max(pos_loss1, pos_loss2)
        
        # Negative advantages: use lower clip bound
        neg_clipped = torch.clamp(ratio, min=clip_low)
        neg_loss1 = -ratio * advantages * neg_mask.float()
        neg_loss2 = -neg_clipped * advantages * neg_mask.float()
        neg_loss = torch.max(neg_loss1, neg_loss2)
        
        # Combine with scaling
        pos_scale = self.config.positive_advantage_scale
        neg_scale = self.config.negative_advantage_scale
        
        policy_loss = pos_scale * pos_loss + neg_scale * neg_loss
        
        # Reduce
        if self.config.reduce_loss == "mean":
            policy_loss = policy_loss.sum() / mask.sum()
        elif self.config.reduce_loss == "sum":
            policy_loss = policy_loss.sum()
        
        return policy_loss
    
    def _compute_standard_loss(
        self,
        ratio: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute standard clipped loss."""
        epsilon = self.config.epsilon
        epsilon_high = self.config.epsilon_high
        
        clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon_high)
        
        policy_loss1 = -ratio * advantages
        policy_loss2 = -clipped_ratio * advantages
        policy_loss = torch.max(policy_loss1, policy_loss2)
        
        policy_loss = (policy_loss * mask).sum() / mask.sum()
        
        return policy_loss
    
    def _compute_value_loss(self, batch: dict) -> torch.Tensor:
        """Compute value function loss if using critic."""
        values = batch["values"]
        returns = batch["returns"]
        
        value_loss = F.mse_loss(values, returns)
        return value_loss
    
    def _compute_log_probs(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probabilities."""
        logits = logits[:, :-1, :]
        target_ids = input_ids[:, 1:]
        
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=target_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        token_log_probs = torch.nn.functional.pad(
            token_log_probs,
            (1, 0),
            value=0.0
        )
        
        return token_log_probs
    
    def get_population_stats(self) -> dict[str, float]:
        """Get current population statistics.
        
        Returns:
            Dict with population statistics
        """
        return {
            "reward_mean": self._reward_mean,
            "reward_std": self._reward_var ** 0.5,
            "num_groups": self._num_groups,
            "reward_sparsity": self._reward_sparsity,
        }
