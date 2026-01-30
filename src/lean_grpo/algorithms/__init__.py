"""RL algorithms for proof generation training.

This module provides multiple RL algorithm implementations:
- GRPO: Group Relative Policy Optimization (default)
- DGPO: Direct GRPO (preference learning variant)
- DrGRPO: Dr. GRPO - GRPO Done Right (fixes for standard GRPO)
- DAPO: Decoupled Advantage Policy Optimization
- GSPO: Group-Synchronized Policy Optimization
"""

from lean_grpo.algorithms.base import RLAlgorithm, RLConfig
from lean_grpo.algorithms.grpo import GRPO, GRPOConfig
from lean_grpo.algorithms.dgrpo import DGPO, DGPOConfig
from lean_grpo.algorithms.drgrpo import DrGRPO, DrGRPOConfig
from lean_grpo.algorithms.dapo import DAPO, DAPOConfig
from lean_grpo.algorithms.gspo import GSPO, GSPOConfig

__all__ = [
    "RLAlgorithm",
    "RLConfig",
    "GRPO",
    "GRPOConfig",
    "DGPO",
    "DGPOConfig",
    "DrGRPO",
    "DrGRPOConfig",
    "DAPO",
    "DAPOConfig",
    "GSPO",
    "GSPOConfig",
    "create_algorithm",
]


def create_algorithm(
    algorithm_type: str,
    **kwargs
) -> RLAlgorithm:
    """Factory function to create RL algorithm instances.
    
    Args:
        algorithm_type: Type of algorithm ('grpo', 'dgpo', 'drgrpo', 'dapo', 'gspo')
        **kwargs: Algorithm-specific configuration
        
    Returns:
        Configured RL algorithm instance
        
    Raises:
        ValueError: If algorithm_type is not recognized
    """
    algorithms = {
        "grpo": GRPO,
        "dgpo": DGPO,
        "drgrpo": DrGRPO,
        "dapo": DAPO,
        "gspo": GSPO,
    }
    
    algorithm_type = algorithm_type.lower()
    if algorithm_type not in algorithms:
        raise ValueError(
            f"Unknown algorithm: {algorithm_type}. "
            f"Available: {list(algorithms.keys())}"
        )
    
    return algorithms[algorithm_type](**kwargs)
