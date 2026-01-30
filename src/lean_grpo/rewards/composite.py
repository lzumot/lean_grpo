"""Composite reward scorer that combines multiple scorers."""

from dataclasses import dataclass, field
from typing import Any, Optional

from lean_grpo.rewards.base import BaseRewardScorer, RewardScorerConfig
from lean_grpo.trajectory import ProofTrajectory


@dataclass
class CompositeConfig(RewardScorerConfig):
    """Configuration for composite scoring."""
    
    # Scorer weights (name -> weight)
    scorer_weights: dict[str, float] = field(default_factory=dict)
    
    # Combination method
    combination_method: str = "weighted_sum"  # 'weighted_sum', 'product', 'min', 'max'
    
    # Normalize individual scores before combining
    normalize_before_combine: bool = True
    
    # If True, requires all scorers to return non-negative for positive reward
    strict_mode: bool = False


class CompositeScorer(BaseRewardScorer):
    """Combines multiple reward scorers into one.
    
    This allows you to:
    1. Combine domain-specific and efficiency-based scoring
    2. Weight different aspects of proof quality
    3. Use different scorers for different problem types
    
    Example:
        ```python
        composite = CompositeScorer(
            config=CompositeConfig(
                scorer_weights={
                    "algebra": 0.5,
                    "efficiency": 0.3,
                    "difficulty": 0.2,
                }
            ),
            scorers={
                "algebra": AlgebraScorer(),
                "efficiency": EfficiencyScorer(),
                "difficulty": DifficultyBasedScorer(),
            }
        )
        ```
    """
    
    def __init__(
        self,
        scorers: dict[str, BaseRewardScorer],
        config: Optional[CompositeConfig] = None,
    ):
        """Initialize composite scorer.
        
        Args:
            scorers: Dict mapping names to scorer instances
            config: Composite configuration
        """
        super().__init__(config)
        self.config: CompositeConfig = config or CompositeConfig()
        self.scorers = scorers
        
        # Validate weights
        if self.config.scorer_weights:
            for name in self.config.scorer_weights:
                if name not in self.scorers:
                    raise ValueError(f"Weight provided for unknown scorer: {name}")
        else:
            # Equal weights
            self.config.scorer_weights = {
                name: 1.0 / len(scorers) for name in scorers
            }
    
    def score(self, trajectory: ProofTrajectory, **kwargs) -> tuple[float, dict[str, Any]]:
        """Score using multiple scorers and combine.
        
        Args:
            trajectory: The proof trajectory
            **kwargs: Additional context
            
        Returns:
            (combined_reward, metadata)
        """
        scores = {}
        all_metrics = {}
        
        # Get scores from each scorer
        for name, scorer in self.scorers.items():
            score, metrics = scorer.score(trajectory, **kwargs)
            scores[name] = score
            all_metrics[name] = metrics
        
        # Combine scores
        combined = self._combine_scores(scores)
        
        # Build metadata
        metadata = {
            "scorer": "composite",
            "combination_method": self.config.combination_method,
            "individual_scores": scores,
            "scorer_metrics": all_metrics,
            "weights": self.config.scorer_weights,
        }
        
        # Normalize final reward
        combined = self.normalize_reward(combined)
        
        return combined, metadata
    
    def _combine_scores(self, scores: dict[str, float]) -> float:
        """Combine scores from multiple scorers.
        
        Args:
            scores: Dict mapping scorer names to scores
            
        Returns:
            Combined score
        """
        method = self.config.combination_method
        
        if method == "weighted_sum":
            return self._weighted_sum(scores)
        elif method == "product":
            return self._product_combine(scores)
        elif method == "min":
            return min(scores.values())
        elif method == "max":
            return max(scores.values())
        else:
            raise ValueError(f"Unknown combination method: {method}")
    
    def _weighted_sum(self, scores: dict[str, float]) -> float:
        """Weighted sum of scores."""
        total = 0.0
        total_weight = 0.0
        
        for name, score in scores.items():
            weight = self.config.scorer_weights.get(name, 1.0)
            total += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return total / total_weight
    
    def _product_combine(self, scores: dict[str, float]) -> float:
        """Product of scores (all must be positive for good result)."""
        result = 1.0
        
        for name, score in scores.items():
            weight = self.config.scorer_weights.get(name, 1.0)
            
            # Normalize to [0, 1] for product
            normalized = (score + 10) / 20  # Assuming [-10, 10] range
            normalized = max(0, min(1, normalized))
            
            # Weighted geometric mean
            result *= normalized ** weight
        
        # Scale back to [-10, 10]
        return result * 20 - 10
    
    def add_scorer(self, name: str, scorer: BaseRewardScorer, weight: float = 1.0):
        """Add a scorer to the composite.
        
        Args:
            name: Scorer name
            scorer: Scorer instance
            weight: Weight for this scorer
        """
        self.scorers[name] = scorer
        self.config.scorer_weights[name] = weight
        
        # Renormalize weights
        total = sum(self.config.scorer_weights.values())
        if total > 0:
            for key in self.config.scorer_weights:
                self.config.scorer_weights[key] /= total
    
    def remove_scorer(self, name: str):
        """Remove a scorer from the composite.
        
        Args:
            name: Scorer name
        """
        if name in self.scorers:
            del self.scorers[name]
        if name in self.config.scorer_weights:
            del self.config.scorer_weights[name]


class AdaptiveCompositeScorer(CompositeScorer):
    """Composite scorer that adapts weights based on problem characteristics.
    
    For example:
    - For hard problems: weight efficiency less, completion more
    - For algebra problems: weight algebra scorer higher
    """
    
    def __init__(
        self,
        scorers: dict[str, BaseRewardScorer],
        config: Optional[CompositeConfig] = None,
    ):
        super().__init__(scorers, config)
        self.base_weights = self.config.scorer_weights.copy()
    
    def score(
        self,
        trajectory: ProofTrajectory,
        problem_type: Optional[str] = None,
        difficulty: Optional[str] = None,
        **kwargs
    ) -> tuple[float, dict[str, Any]]:
        """Score with adaptive weights.
        
        Args:
            trajectory: The proof trajectory
            problem_type: Type of problem (e.g., 'algebra', 'topology')
            difficulty: Difficulty level (e.g., 'easy', 'hard')
            **kwargs: Additional context
            
        Returns:
            (reward, metadata)
        """
        # Adapt weights based on context
        self._adapt_weights(problem_type, difficulty)
        
        # Score with adapted weights
        score, metadata = super().score(trajectory, **kwargs)
        
        # Add adaptation info
        metadata["adapted_weights"] = self.config.scorer_weights.copy()
        metadata["problem_type"] = problem_type
        metadata["difficulty"] = difficulty
        
        return score, metadata
    
    def _adapt_weights(self, problem_type: Optional[str], difficulty: Optional[str]):
        """Adapt weights based on problem characteristics."""
        # Start with base weights
        weights = self.base_weights.copy()
        
        # Adapt based on problem type
        if problem_type:
            type_adaptations = {
                "algebra": {"algebra": 1.5},
                "topology": {"topology": 1.5},
                "analysis": {"analysis": 1.5},
                "number_theory": {"number_theory": 1.5},
                "linear_algebra": {"linear_algebra": 1.5},
            }
            
            if problem_type.lower() in type_adaptations:
                for scorer_name, factor in type_adaptations[problem_type.lower()].items():
                    if scorer_name in weights:
                        weights[scorer_name] *= factor
        
        # Adapt based on difficulty
        if difficulty:
            if difficulty.lower() == "hard":
                # For hard problems, weight completion more, efficiency less
                if "difficulty" in weights:
                    weights["difficulty"] *= 1.5
                if "efficiency" in weights:
                    weights["efficiency"] *= 0.7
            elif difficulty.lower() == "easy":
                # For easy problems, weight efficiency more
                if "efficiency" in weights:
                    weights["efficiency"] *= 1.3
        
        # Renormalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        self.config.scorer_weights = weights


def create_default_composite() -> CompositeScorer:
    """Create a default composite scorer with reasonable defaults.
    
    Returns:
        Composite scorer
    """
    from lean_grpo.rewards.difficulty import DifficultyBasedScorer
    from lean_grpo.rewards.efficiency import EfficiencyScorer
    
    scorers = {
        "difficulty": DifficultyBasedScorer(),
        "efficiency": EfficiencyScorer(),
    }
    
    config = CompositeConfig(
        scorer_weights={
            "difficulty": 0.6,
            "efficiency": 0.4,
        },
        combination_method="weighted_sum",
    )
    
    return CompositeScorer(scorers, config)


def create_domain_aware_composite(domain: str) -> AdaptiveCompositeScorer:
    """Create an adaptive composite for a specific domain.
    
    Args:
        domain: Mathematical domain
        
    Returns:
        Adaptive composite scorer
    """
    from lean_grpo.rewards.difficulty import DifficultyBasedScorer
    from lean_grpo.rewards.domain import create_domain_scorer
    from lean_grpo.rewards.efficiency import EfficiencyScorer
    
    scorers = {
        "domain": create_domain_scorer(domain),
        "difficulty": DifficultyBasedScorer(),
        "efficiency": EfficiencyScorer(),
    }
    
    config = CompositeConfig(
        scorer_weights={
            "domain": 0.5,
            "difficulty": 0.3,
            "efficiency": 0.2,
        },
    )
    
    return AdaptiveCompositeScorer(scorers, config)
