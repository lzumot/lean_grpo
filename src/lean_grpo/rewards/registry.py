"""Registry for reward scorers.

Provides easy access to all available scorers.
"""

from typing import Optional, Type

from lean_grpo.rewards.base import BaseRewardScorer, RewardScorerConfig
from lean_grpo.rewards.difficulty import DifficultyBasedScorer, DifficultyConfig
from lean_grpo.rewards.domain import (
    AlgebraScorer,
    AnalysisScorer,
    LinearAlgebraScorer,
    NumberTheoryScorer,
    TopologyScorer,
)
from lean_grpo.rewards.efficiency import EfficiencyScorer, EfficiencyConfig
from lean_grpo.rewards.composite import CompositeScorer, AdaptiveCompositeScorer, CompositeConfig


class RewardScorerRegistry:
    """Registry for reward scorers.
    
    Provides a centralized way to:
    1. Register new scorers
    2. Retrieve scorers by name
    3. List available scorers
    4. Create scorers with default configs
    
    Example:
        ```python
        from lean_grpo.rewards import get_scorer, list_scorers
        
        # List available scorers
        print(list_scorers())
        
        # Get a scorer
        scorer = get_scorer("algebra")
        
        # Get with custom config
        scorer = get_scorer("difficulty", difficulty="hard")
        ```
    """
    
    _scorers: dict[str, Type[BaseRewardScorer]] = {}
    _configs: dict[str, Type[RewardScorerConfig]] = {}
    
    @classmethod
    def register(
        cls,
        name: str,
        scorer_class: Type[BaseRewardScorer],
        config_class: Optional[Type[RewardScorerConfig]] = None,
    ):
        """Register a scorer.
        
        Args:
            name: Scorer name
            scorer_class: Scorer class
            config_class: Optional config class
        """
        cls._scorers[name] = scorer_class
        if config_class:
            cls._configs[name] = config_class
    
    @classmethod
    def get(
        cls,
        name: str,
        config: Optional[RewardScorerConfig] = None,
        **kwargs
    ) -> BaseRewardScorer:
        """Get a scorer by name.
        
        Args:
            name: Scorer name
            config: Optional config instance
            **kwargs: Config parameters
            
        Returns:
            Scorer instance
            
        Raises:
            ValueError: If scorer not found
        """
        # Check if it's a domain
        if name.lower() in ["algebra", "topology", "analysis", "nt", "number_theory", "la", "linear_algebra"]:
            from lean_grpo.rewards.domain import create_domain_scorer
            return create_domain_scorer(name)
        
        if name not in cls._scorers:
            raise ValueError(
                f"Unknown scorer: {name}. "
                f"Available: {list(cls._scorers.keys())}"
            )
        
        scorer_class = cls._scorers[name]
        
        # Create config if needed
        if config is None and kwargs:
            if name in cls._configs:
                config = cls._configs[name](**kwargs)
            else:
                config = RewardScorerConfig(**kwargs)
        
        return scorer_class(config)
    
    @classmethod
    def list(cls) -> list[str]:
        """List available scorers.
        
        Returns:
            List of scorer names
        """
        # Include domains
        domains = ["algebra", "topology", "analysis", "number_theory", "linear_algebra"]
        return list(cls._scorers.keys()) + domains
    
    @classmethod
    def create_composite(
        cls,
        scorer_names: list[str],
        weights: Optional[dict[str, float]] = None,
    ) -> CompositeScorer:
        """Create a composite scorer from multiple scorers.
        
        Args:
            scorer_names: List of scorer names
            weights: Optional weights for each scorer
            
        Returns:
            Composite scorer
        """
        scorers = {}
        for name in scorer_names:
            scorers[name] = cls.get(name)
        
        config = CompositeConfig(
            scorer_weights=weights or {name: 1.0 / len(scorer_names) for name in scorer_names}
        )
        
        return CompositeScorer(scorers, config)


# Register built-in scorers
RewardScorerRegistry.register("difficulty", DifficultyBasedScorer, DifficultyConfig)
RewardScorerRegistry.register("efficiency", EfficiencyScorer, EfficiencyConfig)
RewardScorerRegistry.register("composite", CompositeScorer, CompositeConfig)
RewardScorerRegistry.register("adaptive", AdaptiveCompositeScorer, CompositeConfig)

# Domain scorers (registered under different names)
RewardScorerRegistry.register("algebra_scorer", AlgebraScorer)
RewardScorerRegistry.register("topology_scorer", TopologyScorer)
RewardScorerRegistry.register("analysis_scorer", AnalysisScorer)
RewardScorerRegistry.register("number_theory_scorer", NumberTheoryScorer)
RewardScorerRegistry.register("linear_algebra_scorer", LinearAlgebraScorer)


# Convenience functions

def get_scorer(name: str, **kwargs) -> BaseRewardScorer:
    """Get a scorer by name.
    
    Args:
        name: Scorer name or domain
        **kwargs: Configuration parameters
        
    Returns:
        Scorer instance
        
    Example:
        ```python
        # Get domain scorer
        scorer = get_scorer("algebra")
        
        # Get difficulty scorer with config
        scorer = get_scorer("difficulty", difficulty="hard")
        
        # Get efficiency scorer
        scorer = get_scorer("efficiency", optimal_length=8)
        ```
    """
    return RewardScorerRegistry.get(name, **kwargs)


def list_scorers() -> list[str]:
    """List available scorers.
    
    Returns:
        List of scorer names
    """
    return RewardScorerRegistry.list()


def create_composite_scorer(
    *scorer_names: str,
    weights: Optional[dict[str, float]] = None
) -> CompositeScorer:
    """Create a composite scorer.
    
    Args:
        *scorer_names: Names of scorers to combine
        weights: Optional weights (equal if not provided)
        
    Returns:
        Composite scorer
        
    Example:
        ```python
        # Equal weights
        scorer = create_composite_scorer("algebra", "efficiency", "difficulty")
        
        # Custom weights
        scorer = create_composite_scorer(
            "algebra", "efficiency",
            weights={"algebra": 0.5, "efficiency": 0.5}
        )
        ```
    """
    return RewardScorerRegistry.create_composite(list(scorer_names), weights)


# Pre-configured scorers for common use cases

def get_easy_algebra_scorer() -> BaseRewardScorer:
    """Get scorer optimized for easy algebra problems."""
    from lean_grpo.rewards.composite import create_domain_aware_composite
    return create_domain_aware_composite("algebra")


def get_hard_topology_scorer() -> BaseRewardScorer:
    """Get scorer optimized for hard topology problems."""
    from lean_grpo.rewards.composite import AdaptiveCompositeScorer
    from lean_grpo.rewards.difficulty import DifficultyBasedScorer, DifficultyConfig
    from lean_grpo.rewards.domain import TopologyScorer
    from lean_grpo.rewards.efficiency import EfficiencyScorer
    
    config = DifficultyConfig(
        hard_completion_bonus=2.0,
        hard_step_reward=0.1,
    )
    
    scorers = {
        "topology": TopologyScorer(),
        "difficulty": DifficultyBasedScorer(config),
        "efficiency": EfficiencyScorer(),
    }
    
    composite_config = CompositeConfig(
        scorer_weights={"topology": 0.4, "difficulty": 0.4, "efficiency": 0.2}
    )
    
    return AdaptiveCompositeScorer(scorers, composite_config)


def get_efficiency_focused_scorer() -> BaseRewardScorer:
    """Get scorer that prioritizes proof efficiency."""
    from lean_grpo.rewards.efficiency import EfficiencyScorer, EfficiencyConfig
    
    config = EfficiencyConfig(
        length_penalty_factor=0.05,
        conciseness_bonus=0.2,
        optimal_length=8,
    )
    
    return EfficiencyScorer(config)


def get_lenient_scorer() -> BaseRewardScorer:
    """Get lenient scorer for exploration phase."""
    from lean_grpo.rewards.difficulty import DifficultyBasedScorer, DifficultyConfig
    
    config = DifficultyConfig(
        easy_error_penalty=-0.02,
        medium_error_penalty=-0.05,
        hard_error_penalty=-0.1,
        partial_reward_enabled=True,
        max_partial_reward=0.7,
    )
    
    return DifficultyBasedScorer(config)
