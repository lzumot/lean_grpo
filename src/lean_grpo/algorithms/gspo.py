"""Group-Synchronized Policy Optimization (GSPO) implementation.

GSPO synchronizes policy updates across multiple groups to improve
sample efficiency and stability.

Key features:
1. Cross-group synchronization
2. Group-level gradient accumulation
3. Dynamic group composition
4. Consensus-based advantage estimation
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist

from lean_grpo.algorithms.base import PolicyLoss, RLAlgorithm, RLConfig


@dataclass
class GSPOConfig(RLConfig):
    """Configuration for GSPO.
    
    GSPO adds synchronization mechanisms to group-based training:
    - Cross-group gradient synchronization
    - Consensus-based normalization
    - Dynamic group sizing
    """
    
    # Group configuration
    min_group_size: int = 4
    max_group_size: int = 16
    target_group_size: int = 8
    
    # Dynamic group sizing
    use_dynamic_groups: bool = False
    group_size_adjustment_rate: float = 0.1
    target_advantage_std: float = 1.0  # Adjust group size to hit this
    
    # Synchronization
    sync_frequency: int = 1  # Sync every N steps
    use_consensus: bool = True
    consensus_weight: float = 0.3  # Weight for consensus term
    
    # Cross-group gradient aggregation
    gradient_sync_method: str = "mean"  # 'mean', 'weighted', 'adaptive'
    gradient_sync_weight: float = 1.0
    
    # Group composition
    composition_strategy: str = "similar_reward"  # 'random', 'similar_reward', 'diverse'
    diversity_bonus: float = 0.1
    
    # Consensus-based advantage
    consensus_window: int = 5  # Number of groups for consensus
    consensus_temperature: float = 0.5
    
    # Adaptive clipping based on group diversity
    adaptive_clipping: bool = True
    diversity_clip_factor: float = 2.0
    
    # Entropy regularization across groups
    cross_group_entropy: bool = False
    cross_group_entropy_coef: float = 0.01


class GSPO(RLAlgorithm):
    """Group-Synchronized Policy Optimization.
    
    GSPO improves group-based training by:
    1. Synchronizing gradients across groups
    2. Using consensus-based advantage estimation
    3. Dynamic group composition
    4. Cross-group regularization
    
    This is particularly useful in distributed settings or when
    training with large batch sizes across multiple groups.
    """
    
    def __init__(self, config: Optional[GSPOConfig] = None, **kwargs):
        """Initialize GSPO.
        
        Args:
            config: GSPO configuration
            **kwargs: Override config fields
        """
        if config is None:
            config = GSPOConfig(**kwargs)
        super().__init__(config)
        self.config: GSPOConfig = config
        
        # Group history for consensus
        self._group_rewards_history: list[list[float]] = []
        self._group_advantages_history: list[dict[int, float]] = []
        
        # Current group size (dynamic)
        self._current_group_size = config.target_group_size
        
        # Gradient synchronization state
        self._step_count = 0
        self._accumulated_gradients: Optional[list[torch.Tensor]] = None
    
    def compute_advantages(
        self,
        rewards: list[float],
        **kwargs
    ) -> dict[str, float]:
        """Compute advantages with consensus.
        
        Uses both group-local and cross-group consensus for advantage estimation.
        
        Args:
            rewards: List of rewards
            **kwargs: May contain 'group_id' for tracking
            
        Returns:
            Dict mapping indices to advantages
        """
        if not rewards:
            return {}
        
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        
        # Compute group-local advantages (like GRPO)
        local_mean = rewards_tensor.mean()
        local_std = rewards_tensor.std()
        
        if local_std < 1e-8:
            local_advantages = torch.zeros_like(rewards_tensor)
        else:
            local_advantages = (rewards_tensor - local_mean) / local_std
        
        # Compute consensus advantages
        if self.config.use_consensus and len(self._group_rewards_history) > 0:
            consensus_advantages = self._compute_consensus_advantages(rewards)
            
            # Blend local and consensus
            weight = self.config.consensus_weight
            final_advantages = (1 - weight) * local_advantages + weight * torch.tensor(consensus_advantages)
        else:
            final_advantages = local_advantages
        
        # Update history
        self._group_rewards_history.append(rewards)
        if len(self._group_rewards_history) > self.config.consensus_window:
            self._group_rewards_history.pop(0)
        
        # Update dynamic group size
        if self.config.use_dynamic_groups:
            self._update_group_size(local_std.item())
        
        return {i: adv.item() for i, adv in enumerate(final_advantages)}
    
    def _compute_consensus_advantages(
        self,
        rewards: list[float],
    ) -> list[float]:
        """Compute consensus advantages from group history.
        
        Args:
            rewards: Current group rewards
            
        Returns:
            Consensus advantages for each trajectory
        """
        if not self._group_rewards_history:
            return [0.0] * len(rewards)
        
        # Compute global statistics from history
        all_rewards = []
        for group_rewards in self._group_rewards_history:
            all_rewards.extend(group_rewards)
        all_rewards.extend(rewards)
        
        global_mean = sum(all_rewards) / len(all_rewards)
        global_var = sum((r - global_mean) ** 2 for r in all_rewards) / len(all_rewards)
        global_std = global_var ** 0.5
        
        if global_std < 1e-8:
            return [0.0] * len(rewards)
        
        # Compute advantages using global stats
        consensus_advantages = [(r - global_mean) / global_std for r in rewards]
        
        return consensus_advantages
    
    def _update_group_size(self, advantage_std: float):
        """Dynamically adjust group size based on advantage variance.
        
        Args:
            advantage_std: Standard deviation of advantages
        """
        target = self.config.target_advantage_std
        diff = target - advantage_std
        
        # Adjust group size
        adjustment = self.config.group_size_adjustment_rate * diff
        new_size = int(self._current_group_size + adjustment)
        
        # Clamp to valid range
        new_size = max(self.config.min_group_size, 
                      min(self.config.max_group_size, new_size))
        
        self._current_group_size = new_size
    
    def compute_loss(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        **kwargs
    ) -> PolicyLoss:
        """Compute GSPO loss with cross-group synchronization.
        
        Args:
            model: Policy model
            batch: Batch data
            **kwargs: May contain 'other_groups_data' for sync
            
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
        
        # Compute group diversity for adaptive clipping
        if self.config.adaptive_clipping:
            diversity = self._compute_group_diversity(advantages)
            epsilon = self.config.epsilon * (1 + diversity * self.config.diversity_clip_factor)
            epsilon_high = self.config.epsilon_high * (1 + diversity * self.config.diversity_clip_factor)
        else:
            epsilon = self.config.epsilon
            epsilon_high = self.config.epsilon_high
        
        # Ratio
        log_ratio = log_probs - old_logprobs
        ratio = torch.exp(log_ratio)
        
        # Expand advantages
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(-1)
        
        # Clipped loss
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
        
        # Cross-group entropy regularization
        cross_entropy_loss = torch.tensor(0.0)
        if self.config.cross_group_entropy and "other_groups_logprobs" in kwargs:
            other_logprobs = kwargs["other_groups_logprobs"]
            cross_entropy_loss = self._compute_cross_group_entropy(
                log_probs, other_logprobs, mask
            )
        
        # Total loss
        loss = (
            policy_loss +
            self.config.beta * kl_div +
            self.config.cross_group_entropy_coef * cross_entropy_loss
        )
        
        # Gradient synchronization (if in distributed setting)
        if self.config.sync_frequency > 0:
            self._step_count += 1
            if self._step_count % self.config.sync_frequency == 0:
                self._synchronize_gradients(model)
        
        metrics = {
            "policy_loss": policy_loss.item(),
            "kl_div": kl_div.item(),
            "mean_ratio": ratio[mask].mean().item(),
            "group_diversity": diversity if self.config.adaptive_clipping else 0.0,
            "current_group_size": self._current_group_size,
        }
        
        if self.config.cross_group_entropy:
            metrics["cross_entropy_loss"] = cross_entropy_loss.item()
        
        return PolicyLoss(
            loss=loss,
            policy_loss=policy_loss,
            kl_div=kl_div,
            metrics=metrics,
        )
    
    def _compute_group_diversity(self, advantages: torch.Tensor) -> float:
        """Compute diversity measure within group.
        
        Args:
            advantages: Group advantages
            
        Returns:
            Diversity score (0 = uniform, 1 = diverse)
        """
        if advantages.dim() > 1:
            advantages = advantages.mean(dim=-1)
        
        # Use coefficient of variation as diversity measure
        mean = advantages.mean()
        std = advantages.std()
        
        if abs(mean) < 1e-8:
            return 0.0
        
        cv = std / abs(mean)
        return min(cv, 1.0)  # Cap at 1
    
    def _compute_cross_group_entropy(
        self,
        log_probs: torch.Tensor,
        other_groups_logprobs: list[torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-group entropy regularization.
        
        Encourages similar policies across groups.
        
        Args:
            log_probs: Current group logprobs
            other_groups_logprobs: Logprobs from other groups
            mask: Valid token mask
            
        Returns:
            Cross-entropy loss
        """
        if not other_groups_logprobs:
            return torch.tensor(0.0)
        
        # Average logprobs from other groups
        avg_other_logprobs = torch.stack(other_groups_logprobs).mean(dim=0)
        
        # KL between current and average other
        kl = torch.exp(avg_other_logprobs - log_probs) - (avg_other_logprobs - log_probs) - 1
        
        # Apply mask and reduce
        kl = (kl * mask).sum() / mask.sum()
        
        return kl
    
    def _synchronize_gradients(self, model: nn.Module):
        """Synchronize gradients across processes.
        
        Args:
            model: Model with gradients
        """
        if not dist.is_initialized():
            return
        
        world_size = dist.get_world_size()
        if world_size <= 1:
            return
        
        # All-reduce gradients
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
    
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
    
    def compose_groups(
        self,
        trajectories: list,
        num_groups: int,
    ) -> list[list[int]]:
        """Compose groups using configured strategy.
        
        Args:
            trajectories: All trajectories
            num_groups: Number of groups to create
            
        Returns:
            List of groups (each group is list of indices)
        """
        import random
        
        n = len(trajectories)
        indices = list(range(n))
        
        if self.config.composition_strategy == "random":
            random.shuffle(indices)
            group_size = n // num_groups
            groups = [
                indices[i * group_size:(i + 1) * group_size]
                for i in range(num_groups)
            ]
            
        elif self.config.composition_strategy == "similar_reward":
            # Sort by reward and group nearby
            sorted_indices = sorted(
                indices,
                key=lambda i: trajectories[i].reward,
                reverse=True
            )
            group_size = n // num_groups
            groups = [
                sorted_indices[i * group_size:(i + 1) * group_size]
                for i in range(num_groups)
            ]
            
        elif self.config.composition_strategy == "diverse":
            # Interleave high and low rewards
            sorted_indices = sorted(
                indices,
                key=lambda i: trajectories[i].reward,
                reverse=True
            )
            groups = [[] for _ in range(num_groups)]
            for i, idx in enumerate(sorted_indices):
                groups[i % num_groups].append(idx)
        
        else:
            raise ValueError(f"Unknown strategy: {self.config.composition_strategy}")
        
        return groups
    
    def get_current_group_size(self) -> int:
        """Get current dynamic group size."""
        return self._current_group_size
