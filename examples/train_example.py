"""Example training script for Lean GRPO with algorithm selection."""

import argparse
import asyncio
import json
import os
from pathlib import Path

from lean_grpo.inference_client import VLLMClient
from lean_grpo.lean_interface import LeanInterface, MockLeanInterface
from lean_grpo.reward import REWARD_CONFIG_SHAPED
from lean_grpo.trainer import LeanGRPOConfig, LeanGRPOTrainer, LeanGRPOPipeline


def load_theorems(path: str) -> list[dict]:
    """Load theorems from JSONL file."""
    theorems = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                theorems.append(json.loads(line))
    return theorems


async def train_with_algorithm(
    algorithm: str,
    theorems: list[dict],
    use_pipeline: bool = False,
):
    """Train with a specific algorithm.
    
    Args:
        algorithm: Algorithm name ('grpo', 'dgpo', 'drgrpo', 'dapo', 'gspo')
        theorems: List of theorems to train on
        use_pipeline: Whether to use iterative pipeline
    """
    algo_display = algorithm.upper()
    if algorithm == "drgrpo":
        algo_display = "Dr. GRPO (GRPO Done Right)"
    
    print(f"\n{'='*60}")
    print(f"Training with {algo_display}")
    print(f"{'='*60}\n")
    
    # Algorithm-specific configurations
    algo_configs = {
        "grpo": {
            "group_size": 4,
        },
        "dgpo": {
            "group_size": 4,
            "use_dpo_loss": False,  # Set to True to enable DPO
            "dpo_coef": 0.0,
        },
        "drgrpo": {
            "group_size": 4,
            "is_level": "token",
            "use_unbiased_kl": True,
            "kl_estimator": "schulman",
            "advantage_norm_method": "winsorized",
            "use_asymmetric_clip": True,
        },
        "dapo": {
            "group_size": 4,
            "use_population_norm": True,
            "population_size": 50,
            "use_asymmetric_loss": True,
        },
        "gspo": {
            "target_group_size": 4,
            "use_consensus": True,
            "consensus_weight": 0.3,
        },
    }
    
    # Create configuration
    config = LeanGRPOConfig(
        base_model="Qwen/Qwen2.5-0.5B-Instruct",  # Small model for testing
        algorithm=algorithm,
        algorithm_config=algo_configs.get(algorithm, {}),
        lora_rank=8,
        num_generations=4,  # Smaller for testing
        learning_rate=5e-6,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        max_proof_steps=10,
        reward_config=REWARD_CONFIG_SHAPED,
        output_dir=f"outputs/example_{algorithm}",
        use_vllm=True,
    )
    
    # Setup interfaces
    lean = MockLeanInterface()
    
    # Setup inference client
    inference = VLLMClient(
        base_url=os.environ.get("VLLM_URL", "http://localhost:8000/v1"),
        api_key=os.environ.get("OPENAI_API_KEY", "dummy-key"),
    )
    
    if use_pipeline:
        print(f"Running iterative pipeline with {algo_display}...")
        pipeline = LeanGRPOPipeline(config, inference, lean)
        await pipeline.run_training_loop(
            theorems=theorems[:5],  # Use subset for testing
            num_iterations=2,
            rollouts_per_iteration=20,
        )
    else:
        print(f"Running single training with {algo_display}...")
        trainer = LeanGRPOTrainer(config, lean, inference)
        trainer.setup()
        
        # Prepare dataset
        dataset = trainer.prepare_dataset(theorems[:5], trainer.tokenizer)
        print(f"Dataset size: {len(dataset)}")
        
        # Train
        if len(dataset) > 0:
            trainer.train(dataset)
            trainer.save_model()
            print(f"Training complete! Model saved to {config.output_dir}")


async def compare_algorithms(theorems: list[dict]):
    """Compare different algorithms on the same theorems."""
    print("\n" + "="*60)
    print("Comparing RL Algorithms")
    print("="*60)
    
    algorithms = ["grpo", "drgrpo", "dgpo", "dapo", "gspo"]
    
    for algo in algorithms:
        try:
            await train_with_algorithm(algo, theorems, use_pipeline=False)
        except Exception as e:
            print(f"Error with {algo}: {e}")
            continue


async def main():
    """Run example training."""
    parser = argparse.ArgumentParser(description="Train with different RL algorithms")
    parser.add_argument(
        "--algorithm",
        choices=["grpo", "dgpo", "drgrpo", "dapo", "gspo", "all"],
        default="grpo",
        help="Algorithm to use (default: grpo)",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Use iterative pipeline",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all algorithms",
    )
    args = parser.parse_args()
    
    # Load theorems
    theorems_path = Path(__file__).parent / "example_theorems.jsonl"
    theorems = load_theorems(str(theorems_path))
    print(f"Loaded {len(theorems)} theorems")
    
    if args.compare:
        await compare_algorithms(theorems)
    elif args.algorithm == "all":
        await compare_algorithms(theorems)
    else:
        await train_with_algorithm(args.algorithm, theorems, args.pipeline)


if __name__ == "__main__":
    asyncio.run(main())
