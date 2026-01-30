"""Direct Group Relative Policy Optimization (DGPO) implementation.

DGPO (formerly DrGRPO) is a variant of GRPO that uses direct preference optimization
concepts while maintaining the group-based relative advantage computation.

Key differences from GRPO:
1. Direct optimization without importance sampling (simpler)
2. Bradley-Terry style preference learning within groups
3. Can work with pairwise preferences
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from lean_grpo.algorithms.base import PolicyLoss, RLAlgorithm, RLConfig
from lean_grpo.algorithms.grpo import GRPOConfig


@dataclass
class DGPOConfig(GRPOConfig):
    """Configuration for DGPO.
    
    DGPO extends GRPO with:
    - Direct preference optimization
    - Pairwise comparison within groups
    - Optional DPO-style loss
    """
    
    # Use DPO-style direct optimization
    use_dpo_loss: bool = False
    
    # DPO beta parameter (temperature for preference model)
    dpo_beta: float = 0.1
    
    # Mixture coefficient between GRPO and DPO losses
    dpo_coef: float = 0.0  # 0 = pure GRPO, 1 = pure DPO
    
    # Pairwise margin for preference learning
    pairwise_margin: float = 0.0
    
    # Use ranked preferences instead of scalar rewards
    use_ranked_preferences: bool = False
    
    # Number of preference pairs to sample per group
    num_preference_pairs: int = 4
    
    # Temperature for softmax in preference computation
    preference_temperature: float = 1.0


class DGPO(RLAlgorithm):
    """Direct GRPO algorithm.
    
    DGPO (Direct GRPO) combines the group-relative advantages of GRPO with
    direct preference optimization principles.
    
    Two modes:
    1. Standard: Uses group-normalized rewards like GRPO
    2. Preference: Uses pairwise preferences within groups
    """
    
    def __init__(self, config: Optional[DGPOConfig] = None, **kwargs):
        """Initialize DGPO.
        
        Args:
            config: DGPO configuration
            **kwargs: Override config fields
        """
        if config is None:
            config = DGPOConfig(**kwargs)
        super().__init__(config)
        self.config: DGPOConfig = config
    
    def compute_advantages(
        self,
        rewards: list[float],
        **kwargs
    ) -> dict[str, float]:
        """Compute advantages using DGPO method.
        
        If use_ranked_preferences is True, uses ranked preferences.
        Otherwise, uses standard GRPO advantage computation.
        
        Args:
            rewards: List of rewards
            **kwargs: May contain 'preferences' for pairwise comparisons
            
        Returns:
            Dict mapping indices to advantages
        """
        if self.config.use_ranked_preferences and "preferences" in kwargs:
            return self._compute_preference_advantages(
                rewards,
                kwargs["preferences"]
            )
        
        # Standard GRPO-style advantages
        return self._compute_grpo_advantages(rewards)
    
    def _compute_grpo_advantages(
        self,
        rewards: list[float]
    ) -> dict[str, float]:
        """Standard GRPO advantage computation."""
        if not rewards:
            return {}
        
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        mean = rewards_tensor.mean()
        std = rewards_tensor.std()
        
        if std < 1e-8:
            advantages = torch.zeros_like(rewards_tensor)
        else:
            advantages = (rewards_tensor - mean) / std
        
        return {i: adv.item() for i, adv in enumerate(advantages)}
    
    def _compute_preference_advantages(
        self,
        rewards: list[float],
        preferences: list[tuple[int, int]]
    ) -> dict[str, float]:
        """Compute advantages from pairwise preferences.
        
        Args:
            rewards: Raw rewards
            preferences: List of (winner_idx, loser_idx) tuples
            
        Returns:
            Advantages based on preference strength
        """
        n = len(rewards)
        scores = torch.zeros(n)
        counts = torch.zeros(n)
        
        # Count wins/losses
        for winner, loser in preferences:
            score_diff = rewards[winner] - rewards[loser]
            scores[winner] += 1 + score_diff
            scores[loser] -= 1
            counts[winner] += 1
            counts[loser] += 1
        
        # Average scores
        counts = torch.clamp(counts, min=1)
        avg_scores = scores / counts
        
        # Normalize
        mean = avg_scores.mean()
        std = avg_scores.std()
        
        if std > 1e-8:
            advantages = (avg_scores - mean) / std
        else:
            advantages = avg_scores - mean
        
        return {i: adv.item() for i, adv in enumerate(advantages)}
    
    def compute_loss(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        **kwargs
    ) -> PolicyLoss:
        """Compute DGPO loss.
        
        Combines GRPO loss with optional DPO-style loss.
        
        Args:
            model: Policy model
            batch: Standard batch data
            **kwargs: May contain 'chosen_rejected' for DPO
            
        Returns:
            PolicyLoss
        """
        # Standard GRPO loss
        grpo_loss = self._compute_grpo_loss(model, batch)
        
        # Optional DPO loss
        if self.config.use_dpo_loss and self.config.dpo_coef > 0:
            dpo_loss = self._compute_dpo_loss(model, batch, kwargs)
            
            # Combine losses
            total_loss = (
                (1 - self.config.dpo_coef) * grpo_loss.loss +
                self.config.dpo_coef * dpo_loss
            )
            
            metrics = {
                **grpo_loss.metrics,
                "dpo_coef": self.config.dpo_coef,
            }
            
            return PolicyLoss(
                loss=total_loss,
                policy_loss=grpo_loss.policy_loss,
                kl_div=grpo_loss.kl_div,
                metrics=metrics,
            )
        
        return grpo_loss
    
    def _compute_grpo_loss(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
    ) -> PolicyLoss:
        """Compute standard GRPO loss component."""
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
        
        # Expand advantages
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(-1)
        
        # Clipped loss
        epsilon = self.config.epsilon
        epsilon_high = self.config.epsilon_high
        
        clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon_high)
        
        policy_loss1 = -ratio * advantages
        policy_loss2 = -clipped_ratio * advantages
        policy_loss = torch.max(policy_loss1, policy_loss2)
        
        # Apply mask
        policy_loss = (policy_loss * mask).sum() / mask.sum()
        
        # KL penalty
        kl_div = torch.tensor(0.0)
        if self.config.beta > 0 and ref_logprobs is not None:
            kl_div = self.compute_kl_penalty(log_probs, ref_logprobs)
            kl_div = (kl_div * mask).sum() / mask.sum()
        
        loss = policy_loss + self.config.beta * kl_div
        
        metrics = {
            "policy_loss": policy_loss.item(),
            "kl_div": kl_div.item(),
            "mean_ratio": ratio[mask].mean().item(),
        }
        
        return PolicyLoss(
            loss=loss,
            policy_loss=policy_loss,
            kl_div=kl_div,
            metrics=metrics,
        )
    
    def _compute_dpo_loss(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        kwargs: dict,
    ) -> torch.Tensor:
        """Compute DPO-style loss.
        
        DPO loss: -log(sigmoid(beta * (log_pi_chosen - log_pi_rejected)))
        """
        if "chosen_rejected" not in kwargs:
            return torch.tensor(0.0)
        
        chosen_ids = kwargs["chosen_ids"]  # Better completions
        rejected_ids = kwargs["rejected_ids"]  # Worse completions
        ref_logprobs_chosen = kwargs.get("ref_logprobs_chosen")
        ref_logprobs_rejected = kwargs.get("ref_logprobs_rejected")
        
        beta = self.config.dpo_beta
        
        # Compute policy logprobs
        outputs_chosen = model(input_ids=chosen_ids)
        logprobs_chosen = self._compute_log_probs(outputs_chosen.logits, chosen_ids)
        pi_logratios = logprobs_chosen.sum(dim=-1)
        
        outputs_rejected = model(input_ids=rejected_ids)
        logprobs_rejected = self._compute_log_probs(outputs_rejected.logits, rejected_ids)
        pi_logratios = pi_logratios - logprobs_rejected.sum(dim=-1)
        
        # Reference logprobs
        if ref_logprobs_chosen is not None and ref_logprobs_rejected is not None:
            ref_logratios = ref_logprobs_chosen.sum(dim=-1) - ref_logprobs_rejected.sum(dim=-1)
        else:
            ref_logratios = 0
        
        # DPO loss
        logits = beta * (pi_logratios - ref_logratios)
        losses = -F.logsigmoid(logits)
        
        return losses.mean()
    
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
    
    def sample_preference_pairs(
        self,
        rewards: list[float],
        num_pairs: Optional[int] = None,
    ) -> list[tuple[int, int]]:
        """Sample preference pairs from a group.
        
        Args:
            rewards: Rewards for each trajectory
            num_pairs: Number of pairs to sample
            
        Returns:
            List of (winner_idx, loser_idx) tuples
        """
        import random
        
        if num_pairs is None:
            num_pairs = self.config.num_preference_pairs
        
        n = len(rewards)
        if n < 2:
            return []
        
        # Sort by reward
        indexed_rewards = list(enumerate(rewards))
        indexed_rewards.sort(key=lambda x: x[1], reverse=True)
        
        pairs = []
        
        # Sample pairs with probability based on reward difference
        for _ in range(num_pairs):
            # Pick winner from top half, loser from bottom half
            winner_idx = random.choice(range(n // 2))
            loser_idx = random.choice(range(n // 2, n))
            
            winner = indexed_rewards[winner_idx][0]
            loser = indexed_rewards[loser_idx][0]
            
            if rewards[winner] > rewards[loser] + self.config.pairwise_margin:
                pairs.append((winner, loser))
        
        return pairs
