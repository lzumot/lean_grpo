"""Efficiency-based reward scorer.

Rewards shorter, more elegant proofs and penalizes inefficient proofs.
"""

from dataclasses import dataclass
from typing import Any, Optional

from lean_grpo.rewards.base import BaseRewardScorer, RewardScorerConfig
from lean_grpo.trajectory import ProofTrajectory


@dataclass
class EfficiencyConfig(RewardScorerConfig):
    """Configuration for efficiency-based scoring."""
    
    # Length penalties/rewards
    optimal_length: int = 10  # Optimal number of steps
    length_tolerance: int = 5  # Tolerance around optimal
    
    # Penalty for excessive length
    length_penalty_factor: float = 0.02  # Per step over optimal
    
    # Reward for being concise
    conciseness_bonus: float = 0.1
    
    # Token efficiency
    token_efficiency_bonus: float = 0.05
    max_tokens_per_step: int = 50
    
    # Proof elegance
    elegance_bonus: float = 0.1
    no_redundancy_bonus: float = 0.05
    
    # Repetition penalty
    repetition_penalty: float = -0.02
    
    # Use of automation
    automation_bonus: float = 0.03  # For using automation tactics


class EfficiencyScorer(BaseRewardScorer):
    """Reward scorer that rewards efficient, concise proofs.
    
    Key metrics:
    - Proof length (number of steps)
    - Token efficiency
    - Lack of redundancy
    - Use of automation
    """
    
    AUTOMATION_TACTICS = [
        "simp", "ring", "field", "linarith", "nlinarith",
        "norm_num", "finish", "solve_by_elim", "tidy",
        "aesop", "auto", "blast", "force",
    ]
    
    def __init__(self, config: Optional[EfficiencyConfig] = None):
        super().__init__(config)
        self.config: EfficiencyConfig = config or EfficiencyConfig()
    
    def score(self, trajectory: ProofTrajectory, **kwargs) -> tuple[float, dict[str, Any]]:
        """Score based on efficiency metrics.
        
        Args:
            trajectory: The proof trajectory
            **kwargs: Additional context
            
        Returns:
            (reward, metadata)
        """
        reward = 0.0
        metrics = {
            "scorer": "efficiency",
            "num_steps": trajectory.num_steps,
            "total_tokens": trajectory.total_tokens,
            "length_score": 0.0,
            "token_efficiency": 0.0,
            "automation_used": 0,
            "redundancy_penalty": 0.0,
            "completion_bonus": 0.0,
        }
        
        tactics_text = trajectory.get_tactics_text().lower()
        tactics_list = [t.strip() for t in trajectory.get_tactics_text().split('\n') if t.strip()]
        
        # 1. Length scoring
        num_steps = trajectory.num_steps
        optimal = self.config.optimal_length
        tolerance = self.config.length_tolerance
        
        if num_steps <= optimal + tolerance:
            # Within optimal range - bonus
            metrics["length_score"] = self.config.conciseness_bonus
            reward += metrics["length_score"]
        else:
            # Too long - penalty
            excess = num_steps - (optimal + tolerance)
            penalty = excess * self.config.length_penalty_factor
            metrics["length_score"] = -penalty
            reward += metrics["length_score"]
        
        # 2. Token efficiency
        if num_steps > 0:
            avg_tokens_per_step = trajectory.total_tokens / num_steps
            metrics["token_efficiency"] = avg_tokens_per_step
            
            if avg_tokens_per_step < self.config.max_tokens_per_step:
                reward += self.config.token_efficiency_bonus
        
        # 3. Automation usage
        for tactic in self.AUTOMATION_TACTICS:
            if tactic in tactics_text:
                metrics["automation_used"] += tactics_text.count(tactic)
                reward += self.config.automation_bonus
        
        # 4. Redundancy detection
        unique_tactics = set(tactics_list)
        if len(tactics_list) > 0:
            redundancy_ratio = 1 - len(unique_tactics) / len(tactics_list)
            if redundancy_ratio > 0.3:  # More than 30% redundant
                metrics["redundancy_penalty"] = self.config.repetition_penalty * redundancy_ratio
                reward += metrics["redundancy_penalty"]
        
        # 5. Elegance heuristics
        if self._is_elegant(tactics_list):
            reward += self.config.elegance_bonus
        
        # 6. Completion bonus
        if trajectory.is_complete:
            reward += self.config.completion_reward
            metrics["completion_bonus"] = self.config.completion_reward
        
        # Normalize
        reward = self.normalize_reward(reward)
        
        return reward, metrics
    
    def _is_elegant(self, tactics: list[str]) -> bool:
        """Check if proof has elegant structure.
        
        Args:
            tactics: List of tactics
            
        Returns:
            True if elegant
        """
        if len(tactics) == 0:
            return False
        
        # Check for elegance patterns
        elegance_patterns = [
            # Starts with intro, ends with exact/apply
            lambda t: t[0].startswith("intro") and any(
                t[-1].startswith(end) for end in ["exact", "apply", "rfl"]
            ),
            # Uses calc block
            lambda t: any("calc" in tactic for tactic in t),
            # Uses automation for finishing
            lambda t: any(
                tactic.startswith(auto) for tactic in t[-3:]
                for auto in ["simp", "ring", "field", "linarith"]
            ),
        ]
        
        # Score elegance
        elegance_score = sum(1 for pattern in elegance_patterns if pattern(tactics))
        
        return elegance_score >= 2


class LengthOnlyScorer(EfficiencyScorer):
    """Simple scorer that only considers proof length."""
    
    def score(self, trajectory: ProofTrajectory, **kwargs) -> tuple[float, dict[str, Any]]:
        """Score based only on length."""
        num_steps = trajectory.num_steps
        optimal = self.config.optimal_length
        
        if num_steps <= optimal:
            reward = self.config.completion_reward
        else:
            excess = num_steps - optimal
            penalty = excess * self.config.length_penalty_factor
            reward = self.config.completion_reward - penalty
        
        if not trajectory.is_complete:
            reward = self.get_progress_reward(trajectory)
        
        metrics = {
            "scorer": "length_only",
            "num_steps": num_steps,
            "optimal": optimal,
        }
        
        return self.normalize_reward(reward), metrics


class TokenEfficiencyScorer(EfficiencyScorer):
    """Scorer that focuses on token efficiency."""
    
    def score(self, trajectory: ProofTrajectory, **kwargs) -> tuple[float, dict[str, Any]]:
        """Score based on token efficiency."""
        total_tokens = trajectory.total_tokens
        num_steps = trajectory.num_steps
        
        # Ideal tokens per step
        ideal_tokens_per_step = 20
        
        if num_steps > 0:
            actual_avg = total_tokens / num_steps
            efficiency = ideal_tokens_per_step / max(actual_avg, 1)
            
            reward = efficiency * self.config.completion_reward
        else:
            reward = 0.0
        
        if trajectory.is_complete:
            reward = max(reward, self.config.completion_reward * 0.5)
        else:
            reward = self.get_progress_reward(trajectory)
        
        metrics = {
            "scorer": "token_efficiency",
            "total_tokens": total_tokens,
            "avg_tokens_per_step": total_tokens / max(num_steps, 1),
            "efficiency": efficiency if num_steps > 0 else 0,
        }
        
        return self.normalize_reward(reward), metrics
