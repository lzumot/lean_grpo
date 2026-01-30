"""Advanced training examples with different configurations."""

import asyncio
import json
from pathlib import Path

from lean_grpo.inference_client import MockInferenceClient, VLLMClient
from lean_grpo.lean_interface import MockLeanInterface
from lean_grpo.reward import LeanRewardCalculator, RewardConfig
from lean_grpo.rewards import get_scorer, create_composite_scorer
from lean_grpo.trainer import LeanGRPOConfig, LeanGRPOPipeline


def load_theorems(path: str) -> list[dict]:
    """Load theorems from JSONL."""
    with open(path, 'r') as f:
        return [json.loads(line) for line in f if line.strip()]


async def train_drgrpo_winsorized():
    """Train with DrGRPO using winsorized normalization."""
    print("\n" + "="*60)
    print("DrGRPO with Winsorized Normalization")
    print("="*60 + "\n")
    
    config = LeanGRPOConfig(
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        algorithm="drgrpo",
        algorithm_config={
            "advantage_norm_method": "winsorized",
            "winsorize_quantile": 0.95,
            "use_asymmetric_clip": True,
            "kl_estimator": "schulman",
        },
        num_generations=8,
        output_dir="outputs/drgrpo_winsorized",
    )
    
    theorems_path = Path(__file__).parent / "example_theorems.jsonl"
    theorems = load_theorems(str(theorems_path))[:3]
    
    lean = MockLeanInterface()
    inference = MockInferenceClient()
    
    pipeline = LeanGRPOPipeline(config, inference, lean)
    
    print("Configuration:")
    print(f"  Algorithm: DrGRPO")
    print(f"  Normalization: Winsorized (q=0.95)")
    print(f"  KL Estimator: Schulman (unbiased)")
    print(f"  Asymmetric Clipping: Enabled")
    print(f"\nTheorems: {len(theorems)}")
    print("Use --pipeline flag to actually run training")


async def train_dapo_sparse():
    """Train with DAPO for sparse rewards."""
    print("\n" + "="*60)
    print("DAPO for Sparse Rewards")
    print("="*60 + "\n")
    
    config = LeanGRPOConfig(
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        algorithm="dapo",
        algorithm_config={
            "use_population_norm": True,
            "population_size": 100,
            "use_asymmetric_loss": True,
            "positive_advantage_scale": 1.0,
            "negative_advantage_scale": 0.5,
            "use_reward_shaping": True,
            "sparse_reward_threshold": 0.8,
        },
        num_generations=8,
        output_dir="outputs/dapo_sparse",
    )
    
    # Create reward config for sparse rewards
    reward_config = RewardConfig(
        completion_reward=1.0,
        step_reward=0.05,
        error_penalty=-0.02,
        goal_reduction_reward=0.1,
    )
    config.reward_config = reward_config
    
    theorems_path = Path(__file__).parent / "example_theorems.jsonl"
    theorems = load_theorems(str(theorems_path))[:3]
    
    lean = MockLeanInterface()
    inference = MockInferenceClient()
    
    print("Configuration:")
    print(f"  Algorithm: DAPO")
    print(f"  Population Norm: Enabled (size=100)")
    print(f"  Asymmetric Loss: Enabled")
    print(f"  Sparse Handling: Enabled")
    print(f"\nThis configuration is best when most proofs fail (sparse success)")


async def train_algebra_domain():
    """Train with domain-specific scoring for algebra."""
    print("\n" + "="*60)
    print("Domain-Specific Training: Algebra")
    print("="*60 + "\n")
    
    config = LeanGRPOConfig(
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        algorithm="grpo",
        num_generations=8,
        output_dir="outputs/algebra_domain",
    )
    
    # Create algebra-specific reward calculator
    lean = MockLeanInterface()
    
    # Get algebra scorer
    algebra_scorer = get_scorer("algebra")
    
    print("Domain Scorer Created:")
    print(f"  Name: {algebra_scorer.name}")
    print(f"  Preferred tactics: {algebra_scorer.PREFERRED_TACTICS[:5]}")
    print(f"\nThis scorer rewards:")
    print("  - Use of simplification tactics (simp, ring, field)")
    print("  - Equation solving (linarith, nlinarith)")
    print("  - Proper substitutions")


async def train_difficulty_based():
    """Train with difficulty-based rewards."""
    print("\n" + "="*60)
    print("Difficulty-Based Rewards")
    print("="*60 + "\n")
    
    config = LeanGRPOConfig(
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        algorithm="grpo",
        num_generations=8,
        output_dir="outputs/difficulty_based",
    )
    
    from lean_grpo.rewards.difficulty import DifficultyBasedScorer, DifficultyConfig
    
    # Create config with custom difficulty settings
    difficulty_config = DifficultyConfig(
        easy_completion_bonus=0.3,
        medium_completion_bonus=0.7,
        hard_completion_bonus=1.5,
        expert_completion_bonus=3.0,
        auto_detect_difficulty=True,
    )
    
    scorer = DifficultyBasedScorer(difficulty_config)
    
    print("Difficulty Scorer Created:")
    print(f"  Easy bonus: {scorer.config.easy_completion_bonus}")
    print(f"  Medium bonus: {scorer.config.medium_completion_bonus}")
    print(f"  Hard bonus: {scorer.config.hard_completion_bonus}")
    print(f"  Expert bonus: {scorer.config.expert_completion_bonus}")
    print(f"\nDifficulty can be:")
    print("  - Provided in theorem metadata")
    print("  - Auto-detected from proof length")


async def train_composite_rewards():
    """Train with composite reward scoring."""
    print("\n" + "="*60)
    print("Composite Reward Scoring")
    print("="*60 + "\n")
    
    config = LeanGRPOConfig(
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        algorithm="grpo",
        num_generations=8,
        output_dir="outputs/composite",
    )
    
    # Create composite scorer
    composite = create_composite_scorer(
        "algebra",
        "efficiency",
        "difficulty",
        weights={
            "algebra": 0.4,
            "efficiency": 0.3,
            "difficulty": 0.3,
        }
    )
    
    print("Composite Scorer Created:")
    print(f"  Components: {list(composite.scorers.keys())}")
    print(f"  Weights: {composite.config.scorer_weights}")
    print(f"\nThis scorer combines:")
    print("  - Domain knowledge (algebra)")
    print("  - Proof efficiency")
    print("  - Problem difficulty")


async def train_efficiency_focused():
    """Train focusing on proof efficiency."""
    print("\n" + "="*60)
    print("Efficiency-Focused Training")
    print("="*60 + "\n")
    
    config = LeanGRPOConfig(
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        algorithm="grpo",
        num_generations=8,
        output_dir="outputs/efficiency",
    )
    
    from lean_grpo.rewards.efficiency import EfficiencyScorer, EfficiencyConfig
    
    # Create efficiency-focused config
    efficiency_config = EfficiencyConfig(
        optimal_length=6,
        length_tolerance=2,
        length_penalty_factor=0.05,
        conciseness_bonus=0.15,
        token_efficiency_bonus=0.1,
    )
    
    scorer = EfficiencyScorer(efficiency_config)
    
    print("Efficiency Scorer Created:")
    print(f"  Optimal length: {scorer.config.optimal_length}")
    print(f"  Length penalty: {scorer.config.length_penalty_factor}/step")
    print(f"  Conciseness bonus: {scorer.config.conciseness_bonus}")
    print(f"\nThis scorer rewards:")
    print("  - Short, concise proofs")
    print("  - Use of automation")
    print("  - Token efficiency")
    print("  - Elegant proof structure")


async def train_gspo_distributed():
    """Train with GSPO for distributed settings."""
    print("\n" + "="*60)
    print("GSPO for Distributed Training")
    print("="*60 + "\n")
    
    config = LeanGRPOConfig(
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        algorithm="gspo",
        algorithm_config={
            "sync_frequency": 4,
            "use_consensus": True,
            "consensus_weight": 0.3,
            "use_dynamic_groups": True,
            "composition_strategy": "diverse",
        },
        num_generations=16,  # Larger groups for distributed
        output_dir="outputs/gspo_distributed",
    )
    
    print("Configuration:")
    print(f"  Algorithm: GSPO")
    print(f"  Sync Frequency: Every 4 steps")
    print(f"  Consensus: Enabled (weight=0.3)")
    print(f"  Dynamic Groups: Enabled")
    print(f"  Group Size: 16")
    print(f"\nBest for:")
    print("  - Multi-GPU training")
    print("  - Large batch sizes")
    print("  - Cross-group gradient sync")


async def train_drgrpo_vs_grpo():
    """Compare DrGRPO and GRPO side by side."""
    print("\n" + "="*60)
    print("DrGRPO vs GRPO Comparison")
    print("="*60 + "\n")
    
    # Standard GRPO config
    grpo_config = LeanGRPOConfig(
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        algorithm="grpo",
        num_generations=8,
        output_dir="outputs/compare_grpo",
    )
    
    # DrGRPO config with fixes
    drgrpo_config = LeanGRPOConfig(
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        algorithm="drgrpo",
        algorithm_config={
            "is_level": "token",
            "use_unbiased_kl": True,
            "kl_estimator": "schulman",
            "advantage_norm_method": "winsorized",
            "use_asymmetric_clip": True,
        },
        num_generations=8,
        output_dir="outputs/compare_drgrpo",
    )
    
    print("GRPO Configuration:")
    print(f"  Standard implementation")
    print(f"  May have IS and KL estimation issues")
    
    print("\nDrGRPO (GRPO Done Right) Configuration:")
    print(f"  Fixed IS correction")
    print(f"  Unbiased KL (Schulman estimator)")
    print(f"  Winsorized normalization")
    print(f"  Asymmetric clipping")
    
    print("\nUse DrGRPO when:")
    print("  - GRPO training is unstable")
    print("  - High variance in rewards")
    print("  - Training not converging")


async def main():
    """Run all advanced examples."""
    examples = [
        train_drgrpo_winsorized,
        train_dapo_sparse,
        train_algebra_domain,
        train_difficulty_based,
        train_composite_rewards,
        train_efficiency_focused,
        train_gspo_distributed,
        train_drgrpo_vs_grpo,
    ]
    
    for example in examples:
        try:
            await example()
        except Exception as e:
            print(f"Error in {example.__name__}: {e}")
    
    print("\n" + "="*60)
    print("Advanced Examples Complete!")
    print("="*60)
    print("\nTo run actual training, adapt these configurations in")
    print("your training script with proper infrastructure setup.")


if __name__ == "__main__":
    asyncio.run(main())
