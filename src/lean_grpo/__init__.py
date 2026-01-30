"""Lean GRPO: GRPO training pipeline for Lean 4 proof generation with multiple RL algorithms.

Supported algorithms:
- GRPO: Group Relative Policy Optimization (default)
- DGPO: Direct GRPO (preference learning variant)
- DrGRPO: Dr. GRPO - GRPO Done Right (fixes for standard GRPO)
- DAPO: Decoupled Advantage Policy Optimization
- GSPO: Group-Synchronized Policy Optimization
"""

from lean_grpo.lean_interface import LeanInterface, LeanProofState, TacticResult
from lean_grpo.trajectory import ProofTrajectory, ProofStep
from lean_grpo.reward import LeanRewardCalculator, RewardConfig
from lean_grpo.trainer import LeanGRPOTrainer, LeanGRPOConfig, LeanGRPOPipeline
from lean_grpo.rollout import ProofRolloutGenerator
from lean_grpo.inference_client import InferenceClient

# Reward scorers
from lean_grpo import rewards

# Export algorithms
from lean_grpo.algorithms import (
    RLAlgorithm,
    RLConfig,
    GRPO,
    GRPOConfig,
    DGPO,
    DGPOConfig,
    DrGRPO,
    DrGRPOConfig,
    DAPO,
    DAPOConfig,
    GSPO,
    GSPOConfig,
    create_algorithm,
)

__version__ = "0.1.0"

__all__ = [
    # Core components
    "LeanInterface",
    "LeanProofState", 
    "TacticResult",
    "ProofTrajectory",
    "ProofStep",
    "LeanRewardCalculator",
    "RewardConfig",
    "LeanGRPOTrainer",
    "LeanGRPOConfig",
    "LeanGRPOPipeline",
    "ProofRolloutGenerator",
    "InferenceClient",
    # Algorithms
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
