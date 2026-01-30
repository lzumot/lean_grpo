"""Base class for reward scorers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from lean_grpo.trajectory import ProofTrajectory


@dataclass
class RewardScorerConfig:
    """Base configuration for reward scorers."""
    
    # Base reward for completion
    completion_reward: float = 1.0
    
    # Partial credit for progress
    partial_reward_enabled: bool = True
    max_partial_reward: float = 0.5
    
    # Penalties
    error_penalty: float = -0.1
    timeout_penalty: float = -0.05
    
    # Scaling
    reward_scale: float = 1.0
    reward_shift: float = 0.0
    
    # Clipping
    min_reward: float = -10.0
    max_reward: float = 10.0


class BaseRewardScorer(ABC):
    """Abstract base class for reward scorers.
    
    Reward scorers are responsible for calculating rewards based on
    proof trajectories. Different scorers can be used for different
    problem types, difficulties, or domains.
    
    To create a custom scorer:
    1. Inherit from BaseRewardScorer
    2. Implement the `score` method
    3. Register with the RewardScorerRegistry
    """
    
    def __init__(self, config: Optional[RewardScorerConfig] = None):
        """Initialize the scorer.
        
        Args:
            config: Scorer configuration
        """
        self.config = config or RewardScorerConfig()
    
    @abstractmethod
    def score(self, trajectory: ProofTrajectory, **kwargs) -> tuple[float, dict[str, Any]]:
        """Calculate the reward for a trajectory.
        
        Args:
            trajectory: The proof trajectory to score
            **kwargs: Additional context (problem metadata, etc.)
            
        Returns:
            Tuple of (reward, metadata_dict)
        """
        pass
    
    def batch_score(
        self,
        trajectories: list[ProofTrajectory],
        **kwargs
    ) -> list[tuple[float, dict[str, Any]]]:
        """Score multiple trajectories.
        
        Args:
            trajectories: List of trajectories to score
            **kwargs: Additional context
            
        Returns:
            List of (reward, metadata) tuples
        """
        return [self.score(traj, **kwargs) for traj in trajectories]
    
    def normalize_reward(self, reward: float) -> float:
        """Normalize and clip reward.
        
        Args:
            reward: Raw reward
            
        Returns:
            Normalized and clipped reward
        """
        # Apply scaling and shift
        reward = reward * self.config.reward_scale + self.config.reward_shift
        
        # Clip
        return max(self.config.min_reward, min(self.config.max_reward, reward))
    
    def get_progress_reward(
        self,
        trajectory: ProofTrajectory,
        max_steps: int = 20
    ) -> float:
        """Calculate partial reward based on progress.
        
        Args:
            trajectory: The trajectory
            max_steps: Maximum expected steps
            
        Returns:
            Partial reward (0 to max_partial_reward)
        """
        if not self.config.partial_reward_enabled:
            return 0.0
        
        if trajectory.is_complete:
            return 0.0  # Full reward handled separately
        
        # Calculate progress
        num_steps = trajectory.num_steps
        valid_steps = sum(1 for step in trajectory.steps if step.is_valid)
        
        # Progress based on valid steps
        progress = valid_steps / max_steps
        
        # Scale to partial reward range
        return progress * self.config.max_partial_reward
    
    @property
    def name(self) -> str:
        """Get scorer name."""
        return self.__class__.__name__
    
    def get_metadata(self) -> dict[str, Any]:
        """Get scorer metadata.
        
        Returns:
            Dict with scorer info
        """
        return {
            "name": self.name,
            "config": self.config.__dict__,
        }
