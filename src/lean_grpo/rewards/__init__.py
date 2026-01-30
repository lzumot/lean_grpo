"""Modular reward scoring system for Lean 4 proofs.

This module provides different reward scorers for:
- Problem difficulty (easy, medium, hard)
- Mathematical domains (algebra, topology, analysis, number theory, etc.)
- Proof characteristics (length, elegance, efficiency)
"""

from lean_grpo.rewards.base import BaseRewardScorer, RewardScorerConfig
from lean_grpo.rewards.difficulty import DifficultyBasedScorer, DifficultyConfig
from lean_grpo.rewards.domain import (
    AlgebraScorer,
    TopologyScorer,
    AnalysisScorer,
    NumberTheoryScorer,
    LinearAlgebraScorer,
)
from lean_grpo.rewards.efficiency import EfficiencyScorer, EfficiencyConfig
from lean_grpo.rewards.composite import CompositeScorer, CompositeConfig
from lean_grpo.rewards.registry import RewardScorerRegistry, get_scorer, list_scorers

__all__ = [
    # Base
    "BaseRewardScorer",
    "RewardScorerConfig",
    # Difficulty-based
    "DifficultyBasedScorer",
    "DifficultyConfig",
    # Domain-specific
    "AlgebraScorer",
    "TopologyScorer",
    "AnalysisScorer",
    "NumberTheoryScorer",
    "LinearAlgebraScorer",
    # Efficiency
    "EfficiencyScorer",
    "EfficiencyConfig",
    # Composite
    "CompositeScorer",
    "CompositeConfig",
    # Registry
    "RewardScorerRegistry",
    "get_scorer",
    "list_scorers",
]
