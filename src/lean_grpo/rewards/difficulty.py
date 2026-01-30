"""Difficulty-based reward scorer.

Provides different reward functions for easy, medium, and hard problems.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from lean_grpo.rewards.base import BaseRewardScorer, RewardScorerConfig
from lean_grpo.trajectory import ProofTrajectory


class ProblemDifficulty(Enum):
    """Problem difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


@dataclass
class DifficultyConfig(RewardScorerConfig):
    """Configuration for difficulty-based scoring."""
    
    # Easy problem settings
    easy_completion_bonus: float = 0.2
    easy_step_reward: float = 0.02
    easy_error_penalty: float = -0.05
    
    # Medium problem settings (default)
    medium_completion_bonus: float = 0.5
    medium_step_reward: float = 0.05
    medium_error_penalty: float = -0.1
    
    # Hard problem settings
    hard_completion_bonus: float = 1.0
    hard_step_reward: float = 0.08
    hard_error_penalty: float = -0.15
    
    # Expert problem settings
    expert_completion_bonus: float = 2.0
    expert_step_reward: float = 0.1
    expert_error_penalty: float = -0.2
    
    # Difficulty detection
    auto_detect_difficulty: bool = True
    difficulty_from_steps: bool = True
    easy_step_threshold: int = 5
    hard_step_threshold: int = 15


class DifficultyBasedScorer(BaseRewardScorer):
    """Reward scorer that adjusts based on problem difficulty.
    
    Harder problems get higher rewards for completion but also
    higher penalties for errors.
    
    Difficulty can be:
    1. Explicitly provided (via metadata)
    2. Auto-detected from proof characteristics
    3. Set per-problem in the dataset
    """
    
    def __init__(self, config: Optional[DifficultyConfig] = None):
        super().__init__(config)
        self.config: DifficultyConfig = config or DifficultyConfig()
    
    def score(
        self,
        trajectory: ProofTrajectory,
        difficulty: Optional[ProblemDifficulty] = None,
        **kwargs
    ) -> tuple[float, dict[str, Any]]:
        """Calculate difficulty-based reward.
        
        Args:
            trajectory: The proof trajectory
            difficulty: Problem difficulty (auto-detected if None)
            **kwargs: Additional context
            
        Returns:
            (reward, metadata)
        """
        # Detect difficulty if not provided
        if difficulty is None:
            difficulty = self._detect_difficulty(trajectory, kwargs)
        
        # Get settings for this difficulty
        settings = self._get_difficulty_settings(difficulty)
        
        # Calculate reward
        reward = 0.0
        metrics = {
            "difficulty": difficulty.value,
            "completion_bonus": 0.0,
            "step_rewards": 0.0,
            "error_penalties": 0.0,
            "progress_reward": 0.0,
        }
        
        # Completion bonus
        if trajectory.is_complete:
            reward += settings["completion_bonus"]
            metrics["completion_bonus"] = settings["completion_bonus"]
        else:
            # Partial progress reward
            progress = self.get_progress_reward(trajectory)
            reward += progress
            metrics["progress_reward"] = progress
        
        # Step rewards/penalties
        for step in trajectory.steps:
            if step.is_valid:
                reward += settings["step_reward"]
                metrics["step_rewards"] += settings["step_reward"]
            else:
                reward += settings["error_penalty"]
                metrics["error_penalties"] += settings["error_penalty"]
        
        # Normalize
        reward = self.normalize_reward(reward)
        
        return reward, metrics
    
    def _detect_difficulty(
        self,
        trajectory: ProofTrajectory,
        kwargs: dict
    ) -> ProblemDifficulty:
        """Detect problem difficulty from context.
        
        Args:
            trajectory: The trajectory
            kwargs: Additional context
            
        Returns:
            Detected difficulty
        """
        # Check if difficulty provided in kwargs
        if "difficulty" in kwargs:
            diff_str = kwargs["difficulty"].lower()
            try:
                return ProblemDifficulty(diff_str)
            except ValueError:
                pass
        
        # Check if difficulty provided in trajectory metadata
        if trajectory.metadata.get("difficulty"):
            diff_str = trajectory.metadata["difficulty"].lower()
            try:
                return ProblemDifficulty(diff_str)
            except ValueError:
                pass
        
        # Auto-detect from proof characteristics
        if self.config.auto_detect_difficulty:
            return self._auto_detect_difficulty(trajectory)
        
        return ProblemDifficulty.MEDIUM
    
    def _auto_detect_difficulty(
        self,
        trajectory: ProofTrajectory
    ) -> ProblemDifficulty:
        """Auto-detect difficulty from proof characteristics.
        
        Heuristics:
        - Short proofs (<= 5 steps) -> Easy
        - Medium proofs (5-15 steps) -> Medium
        - Long proofs (> 15 steps) -> Hard
        
        Args:
            trajectory: The trajectory
            
        Returns:
            Detected difficulty
        """
        if not self.config.difficulty_from_steps:
            return ProblemDifficulty.MEDIUM
        
        num_steps = trajectory.num_steps
        
        if num_steps <= self.config.easy_step_threshold:
            return ProblemDifficulty.EASY
        elif num_steps >= self.config.hard_step_threshold:
            return ProblemDifficulty.HARD
        else:
            return ProblemDifficulty.MEDIUM
    
    def _get_difficulty_settings(self, difficulty: ProblemDifficulty) -> dict:
        """Get reward settings for a difficulty level.
        
        Args:
            difficulty: The difficulty level
            
        Returns:
            Dict with settings
        """
        settings_map = {
            ProblemDifficulty.EASY: {
                "completion_bonus": self.config.easy_completion_bonus,
                "step_reward": self.config.easy_step_reward,
                "error_penalty": self.config.easy_error_penalty,
            },
            ProblemDifficulty.MEDIUM: {
                "completion_bonus": self.config.medium_completion_bonus,
                "step_reward": self.config.medium_step_reward,
                "error_penalty": self.config.medium_error_penalty,
            },
            ProblemDifficulty.HARD: {
                "completion_bonus": self.config.hard_completion_bonus,
                "step_reward": self.config.hard_step_reward,
                "error_penalty": self.config.hard_error_penalty,
            },
            ProblemDifficulty.EXPERT: {
                "completion_bonus": self.config.expert_completion_bonus,
                "step_reward": self.config.expert_step_reward,
                "error_penalty": self.config.expert_error_penalty,
            },
        }
        
        return settings_map.get(difficulty, settings_map[ProblemDifficulty.MEDIUM])


# Convenience functions for creating difficulty-specific scorers

def create_easy_scorer() -> DifficultyBasedScorer:
    """Create a scorer optimized for easy problems."""
    config = DifficultyConfig(
        completion_reward=1.0,
        easy_completion_bonus=0.2,
        partial_reward_enabled=True,
        max_partial_reward=0.3,
    )
    return DifficultyBasedScorer(config)


def create_medium_scorer() -> DifficultyBasedScorer:
    """Create a scorer optimized for medium problems."""
    config = DifficultyConfig(
        completion_reward=1.0,
        medium_completion_bonus=0.5,
        partial_reward_enabled=True,
        max_partial_reward=0.5,
    )
    return DifficultyBasedScorer(config)


def create_hard_scorer() -> DifficultyBasedScorer:
    """Create a scorer optimized for hard problems."""
    config = DifficultyConfig(
        completion_reward=1.0,
        hard_completion_bonus=1.0,
        partial_reward_enabled=True,
        max_partial_reward=0.7,
        error_penalty=-0.15,
    )
    return DifficultyBasedScorer(config)
