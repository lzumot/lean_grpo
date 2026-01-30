"""Base class for RL algorithms."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel


@dataclass
class RLConfig:
    """Base configuration for RL algorithms."""
    
    # Common hyperparameters
    learning_rate: float = 5e-6
    gamma: float = 1.0  # Discount factor
    
    # PPO-style clipping
    epsilon: float = 0.2
    epsilon_high: Optional[float] = None  # If None, uses epsilon
    
    # KL penalty
    beta: float = 0.0  # KL coefficient (0 = no KL penalty)
    
    # Value function (if used by algorithm)
    use_value_function: bool = False
    value_loss_coef: float = 0.5
    
    # Entropy bonus
    entropy_coef: float = 0.0
    
    # Gradient clipping
    max_grad_norm: float = 0.1
    
    # Loss aggregation
    reduce_loss: str = "mean"  # 'mean', 'sum', or 'none'
    
    def __post_init__(self):
        if self.epsilon_high is None:
            self.epsilon_high = self.epsilon


@dataclass 
class PolicyLoss:
    """Output of policy loss computation."""
    
    loss: torch.Tensor
    policy_loss: torch.Tensor
    kl_div: torch.Tensor
    entropy: Optional[torch.Tensor] = None
    value_loss: Optional[torch.Tensor] = None
    metrics: dict[str, float] = field(default_factory=dict)


class RLAlgorithm(ABC):
    """Abstract base class for RL algorithms.
    
    This class defines the interface that all RL algorithms must implement.
    Algorithms are responsible for:
    1. Computing advantages from rewards
    2. Computing policy loss
    3. Optionally computing value loss
    """
    
    def __init__(self, config: RLConfig):
        """Initialize the algorithm.
        
        Args:
            config: Algorithm configuration
        """
        self.config = config
    
    @abstractmethod
    def compute_advantages(
        self,
        rewards: list[float],
        **kwargs
    ) -> dict[str, float]:
        """Compute advantages from rewards.
        
        Args:
            rewards: List of rewards for a group of trajectories
            **kwargs: Additional algorithm-specific arguments
            
        Returns:
            Dict mapping trajectory indices to advantages
        """
        pass
    
    @abstractmethod
    def compute_loss(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        **kwargs
    ) -> PolicyLoss:
        """Compute the policy loss.
        
        Args:
            model: The policy model
            batch: Batch of data containing:
                - input_ids: Token IDs
                - attention_mask: Attention mask
                - old_logprobs: Log probs from old policy
                - advantages: Computed advantages
                - reference_logprobs: (Optional) Log probs from reference model
            **kwargs: Additional algorithm-specific arguments
            
        Returns:
            PolicyLoss containing all loss components
        """
        pass
    
    def compute_kl_penalty(
        self,
        new_logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence penalty.
        
        Uses the unbiased estimator from Schulman et al.:
        KL = exp(ref - new) - (ref - new) - 1
        
        Args:
            new_logprobs: Log probabilities from current policy
            ref_logprobs: Log probabilities from reference policy
            
        Returns:
            KL divergence tensor
        """
        log_ratio = ref_logprobs - new_logprobs
        kl_div = torch.exp(log_ratio) - log_ratio - 1.0
        return kl_div
    
    def clip_gradients(self, model: nn.Module) -> float:
        """Clip gradients by norm.
        
        Args:
            model: Model with gradients
            
        Returns:
            Global gradient norm before clipping
        """
        return torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            self.config.max_grad_norm
        )
    
    def get_metrics(self) -> dict[str, float]:
        """Get algorithm-specific metrics.
        
        Returns:
            Dict of metric names to values
        """
        return {}
    
    @property
    def name(self) -> str:
        """Get algorithm name."""
        return self.__class__.__name__


class AdvantageNormalizer:
    """Utility for normalizing advantages."""
    
    @staticmethod
    def normalize(
        advantages: torch.Tensor,
        method: str = "mean_std"
    ) -> torch.Tensor:
        """Normalize advantages.
        
        Args:
            advantages: Raw advantages
            method: Normalization method ('mean_std', 'max_min', 'none')
            
        Returns:
            Normalized advantages
        """
        if method == "none":
            return advantages
        
        if method == "mean_std":
            mean = advantages.mean()
            std = advantages.std()
            if std > 1e-8:
                return (advantages - mean) / std
            return advantages - mean
        
        if method == "max_min":
            min_val = advantages.min()
            max_val = advantages.max()
            range_val = max_val - min_val
            if range_val > 1e-8:
                return 2 * (advantages - min_val) / range_val - 1
            return advantages
        
        raise ValueError(f"Unknown normalization method: {method}")
    
    @staticmethod
    def whiten(
        advantages: torch.Tensor,
        shift_mean: bool = True
    ) -> torch.Tensor:
        """Whiten advantages (GRPO-style).
        
        Args:
            advantages: Raw advantages
            shift_mean: Whether to subtract mean
            
        Returns:
            Whitened advantages
        """
        mean = advantages.mean()
        std = advantages.std()
        
        if std < 1e-8:
            return torch.zeros_like(advantages) if shift_mean else advantages
        
        whitened = (advantages - mean) / std
        if not shift_mean:
            whitened = whitened + mean
        
        return whitened
