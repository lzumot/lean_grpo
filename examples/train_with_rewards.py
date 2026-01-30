"""Example showing how to use different reward scorers."""

import argparse
import asyncio
import json
from pathlib import Path

from lean_grpo.inference_client import MockInferenceClient
from lean_grpo.lean_interface import MockLeanInterface
from lean_grpo.reward import LeanRewardCalculator
from lean_grpo.rewards import (
    get_scorer,
    create_composite_scorer,
    list_scorers,
)
from lean_grpo.rewards.registry import (
    get_easy_algebra_scorer,
    get_hard_topology_scorer,
    get_efficiency_focused_scorer,
)
from lean_grpo.trainer import LeanGRPOConfig, LeanGRPOTrainer


def load_theorems(path: str) -> list[dict]:
    """Load theorems from JSONL file."""
    theorems = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                theorems.append(json.loads(line))
    return theorems


def example_list_scorers():
    """Show all available scorers."""
    print("\n" + "="*60)
    print("Available Reward Scorers")
    print("="*60 + "\n")
    
    scorers = list_scorers()
    
    print("Built-in scorers:")
    for scorer in sorted(scorers):
        print(f"  - {scorer}")
    
    print("\nPre-configured scorers:")
    print("  - get_easy_algebra_scorer()")
    print("  - get_hard_topology_scorer()")
    print("  - get_efficiency_focused_scorer()")
    print("  - get_lenient_scorer()")


def example_basic_scorer():
    """Example using a basic scorer."""
    print("\n" + "="*60)
    print("Example: Basic Domain Scorer")
    print("="*60 + "\n")
    
    # Get algebra scorer
    scorer = get_scorer("algebra")
    
    print(f"Created scorer: {scorer.name}")
    print(f"Config: {scorer.config.__dict__}")


def example_difficulty_scorer():
    """Example using difficulty-based scorer."""
    print("\n" + "="*60)
    print("Example: Difficulty-Based Scorer")
    print("="*60 + "\n")
    
    # Easy problems
    easy_scorer = get_scorer("difficulty", difficulty="easy")
    print(f"Easy scorer config: {easy_scorer.config.easy_completion_bonus}")
    
    # Hard problems
    hard_scorer = get_scorer("difficulty", difficulty="hard")
    print(f"Hard scorer config: {hard_scorer.config.hard_completion_bonus}")


def example_composite_scorer():
    """Example combining multiple scorers."""
    print("\n" + "="*60)
    print("Example: Composite Scorer")
    print("="*60 + "\n")
    
    # Create composite with equal weights
    composite = create_composite_scorer("algebra", "efficiency", "difficulty")
    
    print(f"Created composite scorer with {len(composite.scorers)} components:")
    for name in composite.scorers:
        print(f"  - {name}")
    
    # Create with custom weights
    weighted = create_composite_scorer(
        "algebra", "efficiency",
        weights={"algebra": 0.7, "efficiency": 0.3}
    )
    
    print(f"\nWeighted composite: {weighted.config.scorer_weights}")


def example_preconfigured():
    """Example using pre-configured scorers."""
    print("\n" + "="*60)
    print("Example: Pre-configured Scorers")
    print("="*60 + "\n")
    
    # Easy algebra
    easy_algebra = get_easy_algebra_scorer()
    print(f"Easy algebra scorer: {easy_algebra.name}")
    
    # Hard topology
    hard_topology = get_hard_topology_scorer()
    print(f"Hard topology scorer: {hard_topology.name}")
    
    # Efficiency focused
    efficiency = get_efficiency_focused_scorer()
    print(f"Efficiency scorer: {efficiency.name}")


def example_custom_config():
    """Example with custom configuration."""
    print("\n" + "="*60)
    print("Example: Custom Configuration")
    print("="*60 + "\n")
    
    from lean_grpo.rewards.efficiency import EfficiencyScorer, EfficiencyConfig
    
    # Create custom efficiency config
    config = EfficiencyConfig(
        optimal_length=5,
        length_penalty_factor=0.05,
        conciseness_bonus=0.2,
    )
    
    scorer = EfficiencyScorer(config)
    print(f"Custom efficiency scorer:")
    print(f"  Optimal length: {scorer.config.optimal_length}")
    print(f"  Penalty factor: {scorer.config.length_penalty_factor}")
    print(f"  Conciseness bonus: {scorer.config.conciseness_bonus}")


def example_training_with_custom_reward():
    """Example training with a custom reward scorer."""
    print("\n" + "="*60)
    print("Example: Training with Custom Reward")
    print("="*60 + "\n")
    
    # Load theorems
    theorems_path = Path(__file__).parent / "example_theorems.jsonl"
    theorems = load_theorems(str(theorems_path))
    
    # Create custom composite scorer
    scorer = create_composite_scorer(
        "algebra", "efficiency",
        weights={"algebra": 0.6, "efficiency": 0.4}
    )
    
    # Create reward calculator with custom scorer
    lean = MockLeanInterface()
    reward_calc = LeanRewardCalculator(
        lean_interface=lean,
        # Note: In real usage, you'd integrate the scorer here
    )
    
    # Create trainer
    config = LeanGRPOConfig(
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        algorithm="grpo",
        num_generations=4,
        output_dir="outputs/custom_reward",
    )
    
    inference = MockInferenceClient()
    trainer = LeanGRPOTrainer(config, lean, inference)
    trainer.setup()
    
    print(f"Trainer created with custom reward configuration")
    print(f"Algorithm: {config.algorithm}")
    print(f"Scorer: Composite (algebra + efficiency)")
    
    # In real usage, you would train here
    print("\nNote: This example doesn't actually train (using mock)")
    print("To train for real, use train_example.py with proper setup")


def example_adaptive_scorer():
    """Example using adaptive scorer."""
    print("\n" + "="*60)
    print("Example: Adaptive Scorer")
    print("="*60 + "\n")
    
    from lean_grpo.rewards.composite import AdaptiveCompositeScorer, CompositeConfig
    from lean_grpo.rewards.difficulty import DifficultyBasedScorer
    from lean_grpo.rewards.domain import AlgebraScorer
    from lean_grpo.rewards.efficiency import EfficiencyScorer
    
    # Create adaptive composite
    scorers = {
        "algebra": AlgebraScorer(),
        "difficulty": DifficultyBasedScorer(),
        "efficiency": EfficiencyScorer(),
    }
    
    config = CompositeConfig(
        scorer_weights={"algebra": 0.4, "difficulty": 0.3, "efficiency": 0.3}
    )
    
    adaptive = AdaptiveCompositeScorer(scorers, config)
    
    print("Created adaptive composite scorer")
    print("Weights adapt based on problem type and difficulty")
    
    # Show adaptation for different problem types
    print("\nExample adaptations:")
    print("  For 'hard' problems: difficulty weight increases")
    print("  For 'algebra' domain: algebra weight increases")


def main():
    """Run examples based on command line arguments."""
    parser = argparse.ArgumentParser(
        description="Examples of using different reward scorers"
    )
    parser.add_argument(
        "example",
        nargs="?",
        choices=[
            "list",
            "basic",
            "difficulty",
            "composite",
            "preconfigured",
            "custom",
            "training",
            "adaptive",
            "all",
        ],
        default="all",
        help="Which example to run",
    )
    args = parser.parse_args()
    
    examples = {
        "list": example_list_scorers,
        "basic": example_basic_scorer,
        "difficulty": example_difficulty_scorer,
        "composite": example_composite_scorer,
        "preconfigured": example_preconfigured,
        "custom": example_custom_config,
        "training": example_training_with_custom_reward,
        "adaptive": example_adaptive_scorer,
    }
    
    if args.example == "all":
        for name, func in examples.items():
            try:
                func()
            except Exception as e:
                print(f"Error in {name}: {e}")
    else:
        examples[args.example]()
    
    print("\n" + "="*60)
    print("Examples complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
