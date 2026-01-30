"""Rollout generation for proof generation."""

import asyncio
from typing import Optional

from lean_grpo.inference_client import InferenceClient
from lean_grpo.lean_interface import LeanInterface, LeanProofState, MockLeanInterface
from lean_grpo.reward import LeanRewardCalculator
from lean_grpo.trajectory import ProofStep, ProofTrajectory


class ProofRolloutGenerator:
    """Generator for proof rollouts.
    
    This class handles the interaction between the LLM and Lean 4
    to generate proof attempts, which are then used for GRPO training.
    """
    
    def __init__(
        self,
        inference_client: InferenceClient,
        lean_interface: Optional[LeanInterface] = None,
        reward_calculator: Optional[LeanRewardCalculator] = None,
        max_steps: int = 20,
        temperature: float = 1.0,
        max_tokens_per_tactic: int = 256,
        stop_sequences: Optional[list[str]] = None,
    ):
        """Initialize the rollout generator.
        
        Args:
            inference_client: Client for LLM inference
            lean_interface: Interface to Lean 4
            reward_calculator: Calculator for proof rewards
            max_steps: Maximum proof steps per rollout
            temperature: Sampling temperature
            max_tokens_per_tactic: Max tokens per tactic generation
            stop_sequences: Sequences to stop generation
        """
        self.inference = inference_client
        self.lean = lean_interface or MockLeanInterface()
        self.reward_calc = reward_calculator or LeanRewardCalculator(self.lean)
        self.max_steps = max_steps
        self.temperature = temperature
        self.max_tokens = max_tokens_per_tactic
        self.stop_sequences = stop_sequences or ["\n\n", "```", "</proof>"]
    
    async def generate_rollout(
        self,
        theorem_name: str,
        theorem_statement: str,
        context: str = "",
        imports: Optional[list[str]] = None,
    ) -> ProofTrajectory:
        """Generate a single proof rollout.
        
        Args:
            theorem_name: Name of the theorem
            theorem_statement: The theorem statement to prove
            context: Additional context (definitions, etc.)
            imports: List of imports
            
        Returns:
            ProofTrajectory with the complete proof attempt
        """
        trajectory = ProofTrajectory(
            theorem_name=theorem_name,
            theorem_statement=theorem_statement,
            context=context,
            imports=imports or ["Mathlib"],
        )
        
        # Initial state
        state = LeanProofState(
            theorem_name=theorem_name,
            theorem_statement=theorem_statement,
            context=context,
            imports=imports or ["Mathlib"],
        )
        
        # Build messages for the conversation
        messages = self._build_initial_messages(theorem_statement, context)
        
        for step_num in range(self.max_steps):
            # Generate next tactic
            try:
                result = await self.inference.generate_tactic(
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stop=self.stop_sequences,
                )
            except Exception as e:
                trajectory.error = f"Generation failed at step {step_num}: {e}"
                break
            
            tactic = result["content"].strip()
            
            # Skip empty tactics
            if not tactic:
                continue
            
            # Execute tactic in Lean
            tactic_result = await self.lean.execute_tactic(state, tactic)
            
            # Create proof step
            step = ProofStep(
                tactic=tactic,
                result=tactic_result,
                state_before=state,
                prompt_tokens=result.get("prompt_tokens", 0),
                completion_tokens=result.get("completion_tokens", 0),
                logprobs=result.get("logprobs"),
            )
            
            # Update state
            if tactic_result.is_valid:
                new_state = LeanProofState(
                    theorem_name=theorem_name,
                    theorem_statement=theorem_statement,
                    context=context,
                    imports=imports or ["Mathlib"],
                    proof_steps=state.proof_steps + [tactic_result],
                )
                step.state_after = new_state
                state = new_state
            
            trajectory.steps.append(step)
            
            # Check if proof is complete
            if tactic_result.proof_complete:
                break
            
            # Check for errors (stop on error)
            if tactic_result.status.value == "error":
                break
            
            # Update messages for next iteration
            messages = self._update_messages(messages, tactic, state)
        
        # Calculate final reward
        reward, metrics = await self.reward_calc.calculate_reward(trajectory)
        trajectory.reward = reward
        trajectory.metrics = metrics
        
        return trajectory
    
    async def generate_rollouts(
        self,
        theorem_name: str,
        theorem_statement: str,
        num_rollouts: int,
        context: str = "",
        imports: Optional[list[str]] = None,
        max_concurrent: int = 5,
    ) -> list[ProofTrajectory]:
        """Generate multiple rollouts for the same theorem.
        
        Args:
            theorem_name: Name of the theorem
            theorem_statement: The theorem statement to prove
            num_rollouts: Number of rollouts to generate
            context: Additional context
            imports: List of imports
            max_concurrent: Maximum concurrent rollouts
            
        Returns:
            List of ProofTrajectories
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def rollout_with_limit():
            async with semaphore:
                return await self.generate_rollout(
                    theorem_name=theorem_name,
                    theorem_statement=theorem_statement,
                    context=context,
                    imports=imports,
                )
        
        tasks = [rollout_with_limit() for _ in range(num_rollouts)]
        return await asyncio.gather(*tasks)
    
    def _build_initial_messages(
        self,
        theorem_statement: str,
        context: str = "",
    ) -> list[dict[str, str]]:
        """Build the initial conversation messages."""
        system_prompt = (
            "You are a Lean 4 proof assistant. Your task is to generate tactics "
            "to prove the given theorem. Generate one tactic at a time. "
            "Use standard Lean 4 tactics like 'intro', 'apply', 'exact', 'simp', "
            "'rw', 'cases', 'induction', etc. Respond with only the tactic, "
            "no explanation."
        )
        
        user_prompt = "Prove the following theorem:\n\n"
        if context:
            user_prompt += f"{context}\n\n"
        user_prompt += theorem_statement
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    
    def _update_messages(
        self,
        messages: list[dict[str, str]],
        tactic: str,
        state: LeanProofState,
    ) -> list[dict[str, str]]:
        """Update messages after a tactic is applied.
        
        This creates a new conversation that includes:
        1. The tactic that was just applied
        2. The current goals (as feedback)
        """
        new_messages = messages.copy()
        
        # Add the tactic as assistant response
        new_messages.append({"role": "assistant", "content": tactic})
        
        # Add current goals as user message (for context)
        if state.num_goals_remaining > 0:
            goals_text = f"Remaining goals: {state.num_goals_remaining}\n"
            if state.current_goals:
                goals_text += "Current goal:\n"
                for goal in state.current_goals[:1]:  # Show first goal
                    goals_text += f"  {goal.get('text', str(goal))}\n"
            goals_text += "\nNext tactic:"
            
            new_messages.append({"role": "user", "content": goals_text})
        
        return new_messages


class BatchRolloutManager:
    """Manager for batch rollout generation across multiple theorems."""
    
    def __init__(
        self,
        inference_client: InferenceClient,
        lean_interface: Optional[LeanInterface] = None,
        reward_calculator: Optional[LeanRewardCalculator] = None,
        rollouts_per_theorem: int = 8,
        max_concurrent_rollouts: int = 10,
    ):
        """Initialize the batch rollout manager.
        
        Args:
            inference_client: Client for LLM inference
            lean_interface: Interface to Lean 4
            reward_calculator: Calculator for proof rewards
            rollouts_per_theorem: Number of rollouts per theorem
            max_concurrent_rollouts: Maximum concurrent rollouts
        """
        self.generator = ProofRolloutGenerator(
            inference_client=inference_client,
            lean_interface=lean_interface,
            reward_calculator=reward_calculator,
        )
        self.rollouts_per_theorem = rollouts_per_theorem
        self.max_concurrent = max_concurrent_rollouts
    
    async def generate_batch_rollouts(
        self,
        theorems: list[dict],
        progress_callback: Optional[callable] = None,
    ) -> list[list[ProofTrajectory]]:
        """Generate rollouts for multiple theorems.
        
        Args:
            theorems: List of theorem dicts with 'name', 'statement', etc.
            progress_callback: Optional callback(current, total)
            
        Returns:
            List of trajectory groups (one group per theorem)
        """
        results = []
        total = len(theorems)
        
        for i, theorem in enumerate(theorems):
            trajectories = await self.generator.generate_rollouts(
                theorem_name=theorem["name"],
                theorem_statement=theorem["statement"],
                context=theorem.get("context", ""),
                imports=theorem.get("imports", ["Mathlib"]),
                num_rollouts=self.rollouts_per_theorem,
                max_concurrent=self.max_concurrent,
            )
            
            results.append(trajectories)
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        return results
