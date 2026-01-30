"""Complete example of domain-specific training with custom rewards."""

import argparse
import asyncio
import json
from pathlib import Path

from lean_grpo.inference_client import VLLMClient
from lean_grpo.lean_interface import LeanInterface, MockLeanInterface
from lean_grpo.rewards import (
    get_scorer,
    create_composite_scorer,
    DifficultyBasedScorer,
    DifficultyConfig,
)
from lean_grpo.rewards.registry import (
    get_easy_algebra_scorer,
    get_hard_topology_scorer,
)
from lean_grpo.trainer import LeanGRPOConfig, LeanGRPOTrainer, LeanGRPOPipeline


def load_theorems(path: str) -> list[dict]:
    """Load theorems from JSONL."""
    with open(path, 'r') as f:
        return [json.loads(line) for line in f if line.strip()]


def create_algebra_theorems() -> list[dict]:
    """Create sample algebra theorems."""
    return [
        {
            "name": "add_comm",
            "statement": "theorem add_comm (n m : Nat) : n + m = m + n",
            "context": "",
            "imports": ["Mathlib"],
            "difficulty": "easy",
            "domain": "algebra",
        },
        {
            "name": "mul_distrib",
            "statement": "theorem mul_add (n m p : Nat) : n * (m + p) = n * m + n * p",
            "context": "",
            "imports": ["Mathlib"],
            "difficulty": "medium",
            "domain": "algebra",
        },
        {
            "name": "pow_mul",
            "statement": "theorem pow_mul (a : Nat) (m n : Nat) : a ^ (m * n) = (a ^ m) ^ n",
            "context": "",
            "imports": ["Mathlib"],
            "difficulty": "hard",
            "domain": "algebra",
        },
    ]


def create_topology_theorems() -> list[dict]:
    """Create sample topology theorems."""
    return [
        {
            "name": "union_open",
            "statement": "theorem union_open {X : Type} [TopologicalSpace X] (S : Set (Set X)) (h : ∀ s ∈ S, IsOpen s) : IsOpen (⋃₀ S)",
            "context": "",
            "imports": ["Mathlib"],
            "difficulty": "easy",
            "domain": "topology",
        },
        {
            "name": "closed_intersection",
            "statement": "theorem closed_intersection {X : Type} [TopologicalSpace X] (S : Set (Set X)) (h : ∀ s ∈ S, IsClosed s) : IsClosed (⋂₀ S)",
            "context": "",
            "imports": ["Mathlib"],
            "difficulty": "medium",
            "domain": "topology",
        },
    ]


async def train_algebra_easy():
    """Train on easy algebra problems."""
    print("\n" + "="*60)
    print("Training: Easy Algebra Problems")
    print("="*60 + "\n")
    
    # Get pre-configured scorer for easy algebra
    scorer = get_easy_algebra_scorer()
    print(f"Using scorer: {scorer.name}")
    
    # Configuration
    config = LeanGRPOConfig(
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        algorithm="grpo",
        num_generations=4,
        learning_rate=1e-5,  # Higher LR for easy problems
        output_dir="outputs/algebra_easy",
    )
    
    theorems = create_algebra_theorems()
    
    lean = MockLeanInterface()
    inference = MockInferenceClient()
    
    print(f"Training on {len(theorems)} algebra theorems")
    print("Configuration optimized for easy problems")
    
    # In real usage, train here
    # trainer = LeanGRPOTrainer(config, lean, inference)
    # trainer.setup()
    # trainer.train(...)


async def train_algebra_hard():
    """Train on hard algebra problems."""
    print("\n" + "="*60)
    print("Training: Hard Algebra Problems")
    print("="*60 + "\n")
    
    # Custom difficulty config for hard problems
    difficulty_config = DifficultyConfig(
        hard_completion_bonus=2.0,
        hard_step_reward=0.1,
        error_penalty=-0.05,  # Lenient errors
    )
    scorer = DifficultyBasedScorer(difficulty_config)
    
    # Use DrGRPO for stability on hard problems
    config = LeanGRPOConfig(
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        algorithm="drgrpo",
        algorithm_config={
            "advantage_norm_method": "winsorized",
            "use_asymmetric_clip": True,
        },
        num_generations=8,
        learning_rate=3e-6,  # Lower LR for stability
        output_dir="outputs/algebra_hard",
    )
    
    theorems = [t for t in create_algebra_theorems() if t.get("difficulty") == "hard"]
    
    print(f"Using scorer: DifficultyBased (hard settings)")
    print(f"Using algorithm: DrGRPO (winsorized normalization)")
    print(f"Training on {len(theorems)} hard theorems")


async def train_topology_composite():
    """Train on topology with composite scoring."""
    print("\n" + "="*60)
    print("Training: Topology with Composite Scoring")
    print("="*60 + "\n")
    
    # Create composite scorer for topology
    composite = create_composite_scorer(
        "topology",
        "difficulty",
        "efficiency",
        weights={
            "topology": 0.5,      # Domain knowledge
            "difficulty": 0.3,    # Difficulty handling
            "efficiency": 0.2,    # Proof efficiency
        }
    )
    
    print(f"Composite scorer components:")
    for name in composite.scorers:
        weight = composite.config.scorer_weights.get(name, 0)
        print(f"  - {name}: {weight:.1%}")
    
    config = LeanGRPOConfig(
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        algorithm="grpo",
        num_generations=6,
        output_dir="outputs/topology_composite",
    )
    
    theorems = create_topology_theorems()
    print(f"\nTraining on {len(theorems)} topology theorems")


async def train_adaptive_scoring():
    """Train with adaptive scoring based on problem metadata."""
    print("\n" + "="*60)
    print("Training: Adaptive Scoring")
    print("="*60 + "\n")
    
    from lean_grpo.rewards.composite import AdaptiveCompositeScorer, CompositeConfig
    from lean_grpo.rewards.domain import AlgebraScorer, TopologyScorer
    from lean_grpo.rewards.difficulty import DifficultyBasedScorer
    
    # Create adaptive composite
    scorers = {
        "algebra": AlgebraScorer(),
        "topology": TopologyScorer(),
        "difficulty": DifficultyBasedScorer(),
    }
    
    config = CompositeConfig(
        scorer_weights={
            "algebra": 0.33,
            "topology": 0.33,
            "difficulty": 0.34,
        }
    )
    
    adaptive = AdaptiveCompositeScorer(scorers, config)
    
    print("Adaptive scorer created")
    print("Weights automatically adjust based on:")
    print("  - Problem domain (algebra vs topology)")
    print("  - Problem difficulty (easy vs hard)")
    
    # Example: Train with mixed problems
    algebra = create_algebra_theorems()
    topology = create_topology_theorems()
    mixed = algebra + topology
    
    print(f"\nMixed dataset: {len(algebra)} algebra, {len(topology)} topology")
    print("Scorer adapts weights per-problem")


async def train_difficulty_progression():
    """Train with progressive difficulty."""
    print("\n" + "="*60)
    print("Training: Progressive Difficulty")
    print("="*60 + "\n")
    
    all_theorems = create_algebra_theorems()
    
    # Sort by difficulty
    difficulty_order = ["easy", "medium", "hard"]
    
    for difficulty in difficulty_order:
        theorems = [t for t in all_theorems if t.get("difficulty") == difficulty]
        if not theorems:
            continue
        
        print(f"\n--- Training on {difficulty} problems ---")
        
        # Adjust scorer for difficulty
        scorer = get_scorer("difficulty", difficulty=difficulty)
        
        # Adjust algorithm based on difficulty
        if difficulty == "easy":
            algo = "grpo"
            lr = 1e-5
        elif difficulty == "medium":
            algo = "grpo"
            lr = 5e-6
        else:  # hard
            algo = "drgrpo"
            lr = 3e-6
        
        config = LeanGRPOConfig(
            base_model="Qwen/Qwen2.5-0.5B-Instruct",
            algorithm=algo,
            learning_rate=lr,
            num_generations=4 if difficulty == "easy" else 8,
            output_dir=f"outputs/progression_{difficulty}",
        )
        
        print(f"  Algorithm: {algo}")
        print(f"  Learning rate: {lr}")
        print(f"  Theorems: {len(theorems)}")


async def main():
    """Run domain-specific examples."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "example",
        nargs="?",
        choices=[
            "algebra_easy",
            "algebra_hard",
            "topology",
            "adaptive",
            "progression",
            "all",
        ],
        default="all",
    )
    args = parser.parse_args()
    
    examples = {
        "algebra_easy": train_algebra_easy,
        "algebra_hard": train_algebra_hard,
        "topology": train_topology_composite,
        "adaptive": train_adaptive_scoring,
        "progression": train_difficulty_progression,
    }
    
    if args.example == "all":
        for name, func in examples.items():
            try:
                await func()
            except Exception as e:
                print(f"Error in {name}: {e}")
    else:
        await examples[args.example]()
    
    print("\n" + "="*60)
    print("Domain-Specific Training Examples Complete!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
