"""GRPO Trainer for Lean 4 proof generation with multi-algorithm support."""

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer
from trl import GRPOConfig, GRPOTrainer as BaseGRPOTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from vllm import SamplingParams

from lean_grpo.algorithms import (
    DAPO,
    DAPOConfig,
    DGPO,
    DGPOConfig,
    DrGRPO,
    DrGRPOConfig,
    GRPO,
    GRPOConfig as AlgoGRPOConfig,
    GSPO,
    GSPOConfig,
    RLAlgorithm,
    create_algorithm,
)
from lean_grpo.inference_client import InferenceClient, VLLMClient
from lean_grpo.lean_interface import LeanInterface, MockLeanInterface
from lean_grpo.reward import LeanRewardCalculator, RewardConfig
from lean_grpo.rollout import BatchRolloutManager, ProofRolloutGenerator
from lean_grpo.trajectory import ProofTrajectory


@dataclass
class LeanGRPOConfig:
    """Configuration for Lean GRPO training.
    
    Supports multiple RL algorithms: GRPO, DGPO, DrGRPO, DAPO, GSPO
    """
    
    # Model configuration
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    lora_rank: int = 8
    lora_alpha: int = 8
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    
    # Algorithm selection
    algorithm: str = "grpo"  # 'grpo', 'dgpo', 'drgrpo', 'dapo', 'gspo'
    algorithm_config: dict[str, Any] = field(default_factory=dict)
    
    # Training configuration
    learning_rate: float = 5e-6
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.1
    max_grad_norm: float = 0.1
    
    # GRPO/Algorithm configuration
    num_generations: int = 8  # Group size
    max_prompt_length: int = 2048
    max_completion_length: int = 512
    beta: float = 0.0  # KL penalty coefficient
    
    # Proof generation configuration
    max_proof_steps: int = 20
    temperature: float = 1.0
    
    # Reward configuration
    reward_config: Optional[RewardConfig] = None
    
    # vLLM configuration
    use_vllm: bool = True
    gpu_memory_utilization: float = 0.6
    
    # Logging
    output_dir: str = "outputs"
    logging_steps: int = 10
    save_steps: int = 500
    report_to: str = "wandb"


class LeanGRPOTrainer:
    """GRPO Trainer for training LLMs to generate Lean 4 proofs.
    
    Supports multiple RL algorithms:
    - GRPO: Group Relative Policy Optimization (default)
    - DGPO: Direct GRPO (preference learning)
    - DrGRPO: Dr. GRPO - GRPO Done Right (fixes for standard GRPO)
    - DAPO: Decoupled Advantage Policy Optimization
    - GSPO: Group-Synchronized Policy Optimization
    
    Uses TRL's GRPOTrainer as base with algorithm-specific adaptations.
    """
    
    def __init__(
        self,
        config: LeanGRPOConfig,
        lean_interface: Optional[LeanInterface] = None,
        inference_client: Optional[InferenceClient] = None,
    ):
        """Initialize the trainer.
        
        Args:
            config: Training configuration
            lean_interface: Interface to Lean 4
            inference_client: External inference client (optional)
        """
        self.config = config
        self.lean = lean_interface or MockLeanInterface()
        self.inference = inference_client
        
        # These will be initialized during setup
        self.model: Optional[FastLanguageModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.trainer: Optional[BaseGRPOTrainer] = None
        self.reward_calc: Optional[LeanRewardCalculator] = None
        self.algorithm: Optional[RLAlgorithm] = None
        
    def setup(self) -> "LeanGRPOTrainer":
        """Set up the model, tokenizer, and algorithm."""
        print(f"Loading base model: {self.config.base_model}")
        
        # Load model with Unsloth for efficiency
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.base_model,
            max_seq_length=self.config.max_prompt_length + self.config.max_completion_length,
            load_in_4bit=False,
            fast_inference=self.config.use_vllm,
            max_lora_rank=self.config.lora_rank,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
        )
        
        # Add LoRA adapters
        print(f"Adding LoRA adapters (rank={self.config.lora_rank})")
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora_rank,
            target_modules=list(self.config.target_modules),
            lora_alpha=self.config.lora_alpha,
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        
        # Initialize reward calculator
        self.reward_calc = LeanRewardCalculator(
            lean_interface=self.lean,
            config=self.config.reward_config or RewardConfig(),
        )
        
        # Initialize RL algorithm
        self._setup_algorithm()
        
        return self
    
    def _setup_algorithm(self):
        """Initialize the RL algorithm based on config."""
        algo_type = self.config.algorithm.lower()
        print(f"Using algorithm: {algo_type.upper()}")
        
        # Build algorithm config
        algo_config_kwargs = {
            "learning_rate": self.config.learning_rate,
            "epsilon": 0.2,
            "beta": self.config.beta,
            "max_grad_norm": self.config.max_grad_norm,
            **self.config.algorithm_config,
        }
        
        if algo_type == "grpo":
            algo_config = AlgoGRPOConfig(
                group_size=self.config.num_generations,
                **algo_config_kwargs,
            )
            self.algorithm = GRPO(algo_config)
            
        elif algo_type == "dgpo":
            algo_config = DGPOConfig(
                group_size=self.config.num_generations,
                **algo_config_kwargs,
            )
            self.algorithm = DGPO(algo_config)
            
        elif algo_type == "drgrpo":
            # Dr. GRPO - GRPO Done Right
            algo_config = DrGRPOConfig(
                group_size=self.config.num_generations,
                **algo_config_kwargs,
            )
            self.algorithm = DrGRPO(algo_config)
            print("  (GRPO Done Right - with fixes for IS, KL, and normalization)")
            
        elif algo_type == "dapo":
            algo_config = DAPOConfig(
                group_size=self.config.num_generations,
                **algo_config_kwargs,
            )
            self.algorithm = DAPO(algo_config)
            
        elif algo_type == "gspo":
            algo_config = GSPOConfig(
                target_group_size=self.config.num_generations,
                **algo_config_kwargs,
            )
            self.algorithm = GSPO(algo_config)
            
        else:
            raise ValueError(f"Unknown algorithm: {algo_type}")
    
    def prepare_dataset(
        self,
        theorems: list[dict],
        tokenizer: PreTrainedTokenizer,
    ) -> Dataset:
        """Prepare dataset from theorems.
        
        Args:
            theorems: List of theorem dictionaries
            tokenizer: Tokenizer for length filtering
            
        Returns:
            HuggingFace Dataset
        """
        # Convert theorems to training examples
        examples = []
        for theorem in theorems:
            prompt = self._build_prompt(
                theorem["statement"],
                theorem.get("context", ""),
            )
            
            examples.append({
                "prompt": prompt,
                "theorem_name": theorem["name"],
                "theorem_statement": theorem["statement"],
                "context": theorem.get("context", ""),
                "imports": theorem.get("imports", ["Mathlib"]),
            })
        
        dataset = Dataset.from_list(examples)
        
        # Filter by length
        def check_length(example):
            tokens = tokenizer.apply_chat_template(
                example["prompt"],
                tokenize=True,
                add_generation_prompt=True,
            )
            return len(tokens) <= self.config.max_prompt_length
        
        dataset = dataset.filter(check_length)
        print(f"Dataset size after filtering: {len(dataset)}")
        
        return dataset
    
    def _build_prompt(
        self,
        theorem_statement: str,
        context: str = "",
    ) -> list[dict[str, str]]:
        """Build the prompt for a theorem."""
        system_msg = (
            "You are a Lean 4 proof assistant. Generate tactics to prove the theorem. "
            "Respond with valid Lean 4 tactics. Be concise and effective."
        )
        
        user_msg = "Prove the following theorem:\n\n"
        if context:
            user_msg += f"{context}\n\n"
        user_msg += theorem_statement
        
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
    
    def create_reward_function(
        self,
    ) -> Callable[[list, list, Any], list[float]]:
        """Create the reward function for training.
        
        The reward function uses the configured algorithm's advantage computation.
        
        Returns:
            Function that takes prompts, completions, and kwargs,
            returns list of rewards.
        """
        async def calculate_rewards_async(
            prompts: list[list[dict]],
            completions: list[list[dict]],
            **kwargs,
        ) -> list[tuple[float, dict]]:
            """Async reward calculation."""
            results = []
            
            for i, (prompt, completion) in enumerate(zip(prompts, completions)):
                # Extract theorem info from kwargs
                theorem_name = kwargs.get("theorem_name", [f"theorem_{i}"] * len(prompts))[i]
                theorem_statement = kwargs.get("theorem_statement", [""] * len(prompts))[i]
                context = kwargs.get("context", [""] * len(prompts))[i]
                imports = kwargs.get("imports", [["Mathlib"]] * len(prompts))[i]
                
                # Build trajectory from completion
                trajectory = self._build_trajectory_from_completion(
                    theorem_name=theorem_name,
                    theorem_statement=theorem_statement,
                    context=context,
                    imports=imports,
                    completion=completion,
                )
                
                # Calculate reward
                reward, metrics = await self.reward_calc.validate_and_score(trajectory)
                results.append((reward, metrics))
            
            return results
        
        def reward_fn(
            prompts: list[list[dict]],
            completions: list[list[dict]],
            **kwargs,
        ) -> list[float]:
            """Synchronous wrapper for reward function."""
            results = asyncio.run(calculate_rewards_async(prompts, completions, **kwargs))
            return [r[0] for r in results]
        
        return reward_fn
    
    def _build_trajectory_from_completion(
        self,
        theorem_name: str,
        theorem_statement: str,
        context: str,
        imports: list[str],
        completion: list[dict],
    ) -> ProofTrajectory:
        """Build a ProofTrajectory from a completion."""
        trajectory = ProofTrajectory(
            theorem_name=theorem_name,
            theorem_statement=theorem_statement,
            context=context,
            imports=imports,
        )
        
        # Extract tactics from completion
        if completion:
            content = completion[0].get("content", "")
            # Split into tactics (simple heuristic)
            tactics = [t.strip() for t in content.split('\n') if t.strip()]
            
            # Build state step by step
            for tactic in tactics:
                # Create a step (simplified - in practice would execute in Lean)
                from lean_grpo.lean_interface import TacticResult, TacticStatus
                
                step = ProofStep(
                    tactic=tactic,
                    result=TacticResult(
                        status=TacticStatus.SUCCESS,
                        tactic=tactic,
                        goals=[],  # Would be populated by actual Lean execution
                    ),
                )
                trajectory.steps.append(step)
        
        return trajectory
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        callbacks: Optional[list] = None,
    ) -> None:
        """Train the model using the configured algorithm.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            callbacks: Optional training callbacks
        """
        if self.model is None:
            raise RuntimeError("Trainer not set up. Call setup() first.")
        
        # Create reward function
        reward_fn = self.create_reward_function()
        
        # Configure GRPO with algorithm-specific settings
        training_args = self._create_training_args()
        
        # Create trainer
        self.trainer = BaseGRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            reward_funcs=reward_fn,
        )
        
        # Add callbacks
        if callbacks:
            for callback in callbacks:
                self.trainer.add_callback(callback)
        
        # Monkey-patch compute_loss if using custom algorithm
        if self.algorithm is not None and self.config.algorithm != "grpo":
            self._patch_trainer_loss()
        
        # Train
        print(f"Starting {self.config.algorithm.upper()} training...")
        self.trainer.train()
        print("Training complete!")
    
    def _create_training_args(self) -> GRPOConfig:
        """Create TRL GRPOConfig with algorithm-specific settings."""
        return GRPOConfig(
            use_vllm=self.config.use_vllm,
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            max_grad_norm=self.config.max_grad_norm,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=2,
            report_to=self.config.report_to,
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            beta=self.config.beta,
            num_generations=self.config.num_generations,
            max_prompt_length=self.config.max_prompt_length,
            max_completion_length=self.config.max_completion_length,
            remove_unused_columns=False,
        )
    
    def _patch_trainer_loss(self):
        """Patch the trainer's compute_loss to use our algorithm."""
        original_compute_loss = self.trainer.compute_loss
        algorithm = self.algorithm
        
        def patched_compute_loss(model, inputs, return_outputs=False, num_items_in_batch=None):
            # Use our algorithm's compute_loss
            loss_obj = algorithm.compute_loss(model, inputs)
            
            # Log metrics
            if hasattr(self.trainer, '_metrics'):
                for key, value in loss_obj.metrics.items():
                    if isinstance(value, (int, float)):
                        self.trainer._metrics.setdefault("train", {}).setdefault(key, []).append(value)
            
            return loss_obj.loss
        
        self.trainer.compute_loss = patched_compute_loss
    
    def save_model(self, path: Optional[str] = None) -> None:
        """Save the trained model.
        
        Args:
            path: Path to save to (uses output_dir if None)
        """
        save_path = path or os.path.join(self.config.output_dir, "final")
        os.makedirs(save_path, exist_ok=True)
        
        print(f"Saving model to {save_path}")
        self.model.save_pretrained(save_path)  # type: ignore
        self.tokenizer.save_pretrained(save_path)  # type: ignore
    
    def generate_proof(
        self,
        theorem_statement: str,
        context: str = "",
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        """Generate a proof for a theorem.
        
        Args:
            theorem_statement: The theorem to prove
            context: Additional context
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            
        Returns:
            Generated proof text
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        prompt = self._build_prompt(theorem_statement, context)
        
        # Apply chat template
        prompt_text = self.tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Generate
        outputs = self.model.fast_generate(  # type: ignore
            [prompt_text],
            sampling_params=SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
            ),
        )
        
        return outputs[0].outputs[0].text


class LeanGRPOPipeline:
    """End-to-end pipeline for Lean GRPO training with algorithm support.
    
    This combines rollout generation, reward calculation, and training
    into a single pipeline inspired by ART's architecture.
    """
    
    def __init__(
        self,
        config: LeanGRPOConfig,
        inference_client: InferenceClient,
        lean_interface: Optional[LeanInterface] = None,
    ):
        """Initialize the pipeline.
        
        Args:
            config: Training configuration
            inference_client: Client for inference
            lean_interface: Interface to Lean 4
        """
        self.config = config
        self.inference = inference_client
        self.lean = lean_interface or MockLeanInterface()
        
        self.trainer = LeanGRPOTrainer(config, self.lean)
        self.rollout_manager: Optional[BatchRolloutManager] = None
        
    async def run_training_loop(
        self,
        theorems: list[dict],
        num_iterations: int = 10,
        rollouts_per_iteration: int = 100,
    ) -> None:
        """Run the full training loop.
        
        This implements a training loop similar to ART:
        1. Generate rollouts with current model
        2. Calculate rewards using Lean
        3. Train using configured algorithm
        4. Repeat
        
        Args:
            theorems: List of theorems to train on
            num_iterations: Number of training iterations
            rollouts_per_iteration: Rollouts per iteration
        """
        # Setup trainer
        self.trainer.setup()
        
        # Create rollout manager
        self.rollout_manager = BatchRolloutManager(
            inference_client=self.inference,
            lean_interface=self.lean,
            reward_calculator=self.trainer.reward_calc,
            rollouts_per_theorem=self.config.num_generations,
        )
        
        print(f"\nAlgorithm: {self.config.algorithm.upper()}")
        if self.config.algorithm.lower() == "drgrpo":
            print("  (GRPO Done Right - with fixes for IS, KL, and normalization)")
        print(f"Group size: {self.config.num_generations}")
        
        for iteration in range(num_iterations):
            print(f"\n{'='*50}")
            print(f"Iteration {iteration + 1}/{num_iterations}")
            print(f"{'='*50}\n")
            
            # Sample theorems for this iteration
            import random
            sampled_theorems = random.sample(
                theorems,
                min(rollouts_per_iteration // self.config.num_generations, len(theorems)),
            )
            
            # Generate rollouts
            print("Generating rollouts...")
            trajectory_groups = await self.rollout_manager.generate_batch_rollouts(
                sampled_theorems,
                progress_callback=lambda c, t: print(f"  Progress: {c}/{t}"),
            )
            
            # Flatten trajectories
            all_trajectories = [
                traj for group in trajectory_groups for traj in group
            ]
            
            print(f"Generated {len(all_trajectories)} trajectories")
            
            # Calculate statistics
            rewards = [t.reward for t in all_trajectories]
            mean_reward = sum(rewards) / len(rewards) if rewards else 0
            completion_rate = sum(1 for t in all_trajectories if t.is_complete) / len(all_trajectories)
            
            print(f"Mean reward: {mean_reward:.3f}")
            print(f"Completion rate: {completion_rate:.3%}")
            
            # Use algorithm-specific advantage computation if available
            if self.trainer.algorithm:
                advantages = self.trainer.algorithm.compute_advantages(rewards)
                mean_advantage = sum(advantages.values()) / len(advantages) if advantages else 0
                print(f"Mean advantage: {mean_advantage:.3f}")
            
            # Prepare training data from successful trajectories
            training_data = self._trajectories_to_dataset(all_trajectories)
            
            # Train
            if len(training_data) > 0:
                print(f"Training on {len(training_data)} examples...")
                self.trainer.train(training_data)
            
            # Save checkpoint
            checkpoint_dir = os.path.join(
                self.config.output_dir,
                f"checkpoint-{iteration + 1}"
            )
            self.trainer.save_model(checkpoint_dir)
        
        print("\nTraining complete!")
    
    def _trajectories_to_dataset(self, trajectories: list[ProofTrajectory]) -> Dataset:
        """Convert trajectories to a HuggingFace Dataset."""
        examples = []
        
        for traj in trajectories:
            example = traj.to_training_example()
            examples.append(example)
        
        return Dataset.from_list(examples)


# Fix circular import
from lean_grpo.trajectory import ProofStep
