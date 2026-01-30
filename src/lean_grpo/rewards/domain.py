"""Domain-specific reward scorers for different mathematical areas.

Each domain has its own characteristics:
- Algebra: Emphasizes simplification, equation solving
- Topology: Emphasizes continuity, openness/closedness
- Analysis: Emphasizes limits, convergence, differentiation
- Number Theory: Emphasizes divisibility, primes, modular arithmetic
- Linear Algebra: Emphasizes matrix operations, vector spaces
"""

import re
from dataclasses import dataclass
from typing import Any, Optional

from lean_grpo.rewards.base import BaseRewardScorer, RewardScorerConfig
from lean_grpo.trajectory import ProofTrajectory


@dataclass
class DomainConfig(RewardScorerConfig):
    """Configuration for domain-specific scoring."""
    
    # Domain identification
    auto_detect_domain: bool = True
    
    # Tactic preferences (weights for using preferred tactics)
    preferred_tactic_bonus: float = 0.02
    discouraged_tactic_penalty: float = -0.01
    
    # Structural bonuses
    good_structure_bonus: float = 0.05
    bad_structure_penalty: float = -0.03


class AlgebraScorer(BaseRewardScorer):
    """Reward scorer optimized for algebra problems.
    
    Preferred characteristics:
    - Uses simplification tactics (simp, ring, field)
    - Equation solving (linarith, nlinarith)
    - Good use of substitutions
    """
    
    PREFERRED_TACTICS = [
        "simp", "ring", "field", "linarith", "nlinarith",
        "rw", "rewrite", "subst", "calc", "abel",
    ]
    
    DISCOURAGED_TACTICS = [
        "trivial", "sorry", "admit",
    ]
    
    def __init__(self, config: Optional[DomainConfig] = None):
        super().__init__(config)
        self.config: DomainConfig = config or DomainConfig()
    
    def score(self, trajectory: ProofTrajectory, **kwargs) -> tuple[float, dict[str, Any]]:
        """Score an algebra proof trajectory."""
        reward = 0.0
        metrics = {
            "domain": "algebra",
            "preferred_tactics_used": 0,
            "discouraged_tactics_used": 0,
            "simplification_steps": 0,
            "completion_bonus": 0.0,
        }
        
        tactics_text = trajectory.get_tactics_text().lower()
        
        # Check for preferred tactics
        for tactic in self.PREFERRED_TACTICS:
            if tactic in tactics_text:
                reward += self.config.preferred_tactic_bonus
                metrics["preferred_tactics_used"] += 1
        
        # Check for discouraged tactics
        for tactic in self.DISCOURAGED_TACTICS:
            if tactic in tactics_text:
                reward += self.config.discouraged_tactic_penalty
                metrics["discouraged_tactics_used"] += 1
        
        # Bonus for simplification patterns
        simp_patterns = ["simp", "ring", "field"]
        for pattern in simp_patterns:
            metrics["simplification_steps"] += tactics_text.count(pattern)
        
        # Extra bonus for good simplification
        if metrics["simplification_steps"] >= 2:
            reward += self.config.good_structure_bonus
        
        # Completion bonus
        if trajectory.is_complete:
            reward += self.config.completion_reward
            metrics["completion_bonus"] = self.config.completion_reward
        else:
            reward += self.get_progress_reward(trajectory)
        
        # Normalize
        reward = self.normalize_reward(reward)
        
        return reward, metrics


class TopologyScorer(BaseRewardScorer):
    """Reward scorer optimized for topology problems.
    
    Preferred characteristics:
    - Proper handling of open/closed sets
    - Continuity proofs
    - Good use of topological definitions
    """
    
    PREFERRED_TACTICS = [
        "continuity", "is_open", "is_closed", "compact",
        "apply", "have", "show", "let", "intro",
        "rw", "simp", "unfold",
    ]
    
    TOPOLOGY_KEYWORDS = [
        "continuous", "open", "closed", "compact", "hausdorff",
        "neighborhood", "interior", "closure", "boundary",
        "convergence", "limit", "homeomorphism",
    ]
    
    def __init__(self, config: Optional[DomainConfig] = None):
        super().__init__(config)
        self.config: DomainConfig = config or DomainConfig()
    
    def score(self, trajectory: ProofTrajectory, **kwargs) -> tuple[float, dict[str, Any]]:
        """Score a topology proof trajectory."""
        reward = 0.0
        metrics = {
            "domain": "topology",
            "topological_concepts": 0,
            "definition_unfolds": 0,
            "structured_proof": False,
            "completion_bonus": 0.0,
        }
        
        tactics_text = trajectory.get_tactics_text().lower()
        
        # Count topological concepts
        for keyword in self.TOPOLOGY_KEYWORDS:
            if keyword in tactics_text:
                metrics["topological_concepts"] += 1
                reward += 0.01
        
        # Bonus for unfolding definitions (good topology practice)
        metrics["definition_unfolds"] = tactics_text.count("unfold")
        if metrics["definition_unfolds"] > 0:
            reward += self.config.preferred_tactic_bonus * metrics["definition_unfolds"]
        
        # Check for structured proof pattern
        structured_patterns = ["intro", "apply", "have", "show"]
        structured_count = sum(1 for p in structured_patterns if p in tactics_text)
        if structured_count >= 3:
            metrics["structured_proof"] = True
            reward += self.config.good_structure_bonus
        
        # Completion bonus
        if trajectory.is_complete:
            reward += self.config.completion_reward
            metrics["completion_bonus"] = self.config.completion_reward
        else:
            reward += self.get_progress_reward(trajectory)
        
        # Normalize
        reward = self.normalize_reward(reward)
        
        return reward, metrics


class AnalysisScorer(BaseRewardScorer):
    """Reward scorer optimized for analysis problems.
    
    Preferred characteristics:
    - Proper epsilon-delta proofs
    - Limit handling
    - Convergence arguments
    """
    
    ANALYSIS_KEYWORDS = [
        "limit", "converges", "continuous", "differentiable",
        "integrable", "epsilon", "delta", "neighborhood",
        "bounded", "monotone", "cauchy", "uniform",
    ]
    
    PREFERRED_TACTICS = [
        "apply", "have", "obtain", "use", "let",
        "linarith", "nlinarith", "simp", "field_simp",
        "continuity", "differentiability",
    ]
    
    def __init__(self, config: Optional[DomainConfig] = None):
        super().__init__(config)
        self.config: DomainConfig = config or DomainConfig()
    
    def score(self, trajectory: ProofTrajectory, **kwargs) -> tuple[float, dict[str, Any]]:
        """Score an analysis proof trajectory."""
        reward = 0.0
        metrics = {
            "domain": "analysis",
            "analysis_concepts": 0,
            "epsilon_delta_style": False,
            "limit_handling": False,
            "completion_bonus": 0.0,
        }
        
        tactics_text = trajectory.get_tactics_text().lower()
        
        # Count analysis concepts
        for keyword in self.ANALYSIS_KEYWORDS:
            if keyword in tactics_text:
                metrics["analysis_concepts"] += 1
                reward += 0.01
        
        # Check for epsilon-delta style
        if "epsilon" in tactics_text or "delta" in tactics_text:
            metrics["epsilon_delta_style"] = True
            reward += self.config.good_structure_bonus
        
        # Check for limit handling
        if "limit" in tactics_text or "tendsto" in tactics_text:
            metrics["limit_handling"] = True
            reward += self.config.preferred_tactic_bonus
        
        # Bonus for using analysis tactics
        for tactic in self.PREFERRED_TACTICS:
            if tactic in tactics_text:
                reward += 0.01
        
        # Completion bonus
        if trajectory.is_complete:
            reward += self.config.completion_reward
            metrics["completion_bonus"] = self.config.completion_reward
        else:
            reward += self.get_progress_reward(trajectory)
        
        # Normalize
        reward = self.normalize_reward(reward)
        
        return reward, metrics


class NumberTheoryScorer(BaseRewardScorer):
    """Reward scorer optimized for number theory problems.
    
    Preferred characteristics:
    - Divisibility arguments
    - Prime handling
    - Modular arithmetic
    """
    
    NT_KEYWORDS = [
        "prime", "divisible", "gcd", "lcm", "mod", "modulo",
        "congruent", "coprime", "even", "odd", "divisor",
        "factorial", "power", "multiplicative",
    ]
    
    PREFERRED_TACTICS = [
        "norm_num", "ring", "rw", "simp", "calc",
        "have", "obtain", "use", "exists",
    ]
    
    def __init__(self, config: Optional[DomainConfig] = None):
        super().__init__(config)
        self.config: DomainConfig = config or DomainConfig()
    
    def score(self, trajectory: ProofTrajectory, **kwargs) -> tuple[float, dict[str, Any]]:
        """Score a number theory proof trajectory."""
        reward = 0.0
        metrics = {
            "domain": "number_theory",
            "nt_concepts": 0,
            "computation_steps": 0,
            "explicit_construction": False,
            "completion_bonus": 0.0,
        }
        
        tactics_text = trajectory.get_tactics_text().lower()
        
        # Count number theory concepts
        for keyword in self.NT_KEYWORDS:
            if keyword in tactics_text:
                metrics["nt_concepts"] += 1
                reward += 0.01
        
        # Bonus for computational steps (common in NT)
        metrics["computation_steps"] = tactics_text.count("norm_num")
        if metrics["computation_steps"] > 0:
            reward += self.config.preferred_tactic_bonus * metrics["computation_steps"]
        
        # Check for explicit construction (use/exists)
        if "use " in tactics_text or "exists" in tactics_text:
            metrics["explicit_construction"] = True
            reward += self.config.good_structure_bonus
        
        # Completion bonus
        if trajectory.is_complete:
            reward += self.config.completion_reward
            metrics["completion_bonus"] = self.config.completion_reward
        else:
            reward += self.get_progress_reward(trajectory)
        
        # Normalize
        reward = self.normalize_reward(reward)
        
        return reward, metrics


class LinearAlgebraScorer(BaseRewardScorer):
    """Reward scorer optimized for linear algebra problems.
    
    Preferred characteristics:
    - Matrix operations
    - Vector space reasoning
    - Eigenvalue/eigenvector handling
    """
    
    LA_KEYWORDS = [
        "matrix", "vector", "linear", "independent", "basis",
        "dimension", "rank", "determinant", "trace", "eigenvalue",
        "eigenvector", "transpose", "inverse", "dot", "cross",
        "subspace", "span", "kernel", "image",
    ]
    
    PREFERRED_TACTICS = [
        "simp", "ring", "rw", "ext", "funext", "linarith",
        "have", "apply", "constructor", "simpa",
    ]
    
    def __init__(self, config: Optional[DomainConfig] = None):
        super().__init__(config)
        self.config: DomainConfig = config or DomainConfig()
    
    def score(self, trajectory: ProofTrajectory, **kwargs) -> tuple[float, dict[str, Any]]:
        """Score a linear algebra proof trajectory."""
        reward = 0.0
        metrics = {
            "domain": "linear_algebra",
            "la_concepts": 0,
            "extensionality_used": False,
            "computation_used": False,
            "completion_bonus": 0.0,
        }
        
        tactics_text = trajectory.get_tactics_text().lower()
        
        # Count linear algebra concepts
        for keyword in self.LA_KEYWORDS:
            if keyword in tactics_text:
                metrics["la_concepts"] += 1
                reward += 0.01
        
        # Bonus for extensionality (common in LA)
        if "ext" in tactics_text or "funext" in tactics_text:
            metrics["extensionality_used"] = True
            reward += self.config.preferred_tactic_bonus
        
        # Check for computational tactics
        comp_tactics = ["simp", "ring", "field"]
        for tactic in comp_tactics:
            if tactic in tactics_text:
                metrics["computation_used"] = True
                reward += 0.01
        
        # Bonus for constructor (common for showing properties)
        if "constructor" in tactics_text:
            reward += self.config.preferred_tactic_bonus
        
        # Completion bonus
        if trajectory.is_complete:
            reward += self.config.completion_reward
            metrics["completion_bonus"] = self.config.completion_reward
        else:
            reward += self.get_progress_reward(trajectory)
        
        # Normalize
        reward = self.normalize_reward(reward)
        
        return reward, metrics


# Factory function for domain scorers

def create_domain_scorer(domain: str, config: Optional[DomainConfig] = None):
    """Create a domain-specific scorer.
    
    Args:
        domain: Domain name ('algebra', 'topology', 'analysis', 'nt', 'la')
        config: Optional configuration
        
    Returns:
        Domain-specific scorer
    """
    domain_map = {
        "algebra": AlgebraScorer,
        "topology": TopologyScorer,
        "analysis": AnalysisScorer,
        "nt": NumberTheoryScorer,
        "number_theory": NumberTheoryScorer,
        "la": LinearAlgebraScorer,
        "linear_algebra": LinearAlgebraScorer,
    }
    
    domain = domain.lower()
    if domain not in domain_map:
        raise ValueError(f"Unknown domain: {domain}. Available: {list(domain_map.keys())}")
    
    return domain_map[domain](config)
