"""Reward calculation for Lean 4 proofs."""

from dataclasses import dataclass, field
from typing import Callable, Optional

from lean_grpo.lean_interface import (
    LeanInterface,
    LeanProofState,
    MockLeanInterface,
    TacticStatus,
)
from lean_grpo.trajectory import ProofTrajectory


@dataclass
class RewardConfig:
    """Configuration for reward calculation."""
    
    # Base reward for completing a proof
    completion_reward: float = 1.0
    
    # Reward for each valid step (partial credit)
    step_reward: float = 0.05
    
    # Penalty for errors
    error_penalty: float = -0.1
    
    # Penalty for timeout
    timeout_penalty: float = -0.05
    
    # Maximum steps to consider for partial rewards
    max_steps_for_partial: int = 20
    
    # Penalty for excessive length
    length_penalty_factor: float = 0.01
    
    # Reward for reducing goals
    goal_reduction_reward: float = 0.1
    
    # Whether to use Lean interface for validation
    use_lean_validation: bool = True
    
    # Custom reward functions
    custom_rewards: list[Callable[[ProofTrajectory], float]] = field(
        default_factory=list
    )
    
    # Reward shaping function
    shaping_fn: Optional[Callable[[ProofTrajectory, float], float]] = None


class LeanRewardCalculator:
    """Calculates rewards for proof trajectories.
    
    This class implements various reward strategies for training
    LLMs to generate Lean 4 proofs, including:
    - Binary rewards (success/failure)
    - Partial rewards based on proof progress
    - Shaped rewards for intermediate progress
    """
    
    def __init__(
        self,
        lean_interface: Optional[LeanInterface] = None,
        config: Optional[RewardConfig] = None,
    ):
        """Initialize the reward calculator.
        
        Args:
            lean_interface: Interface to Lean 4 (uses mock if None)
            config: Reward configuration
        """
        self.lean = lean_interface or MockLeanInterface()
        self.config = config or RewardConfig()
    
    async def calculate_reward(
        self,
        trajectory: ProofTrajectory,
        use_partial: bool = True,
    ) -> tuple[float, dict]:
        """Calculate the reward for a trajectory.
        
        Args:
            trajectory: The proof trajectory to evaluate
            use_partial: Whether to use partial reward shaping
            
        Returns:
            Tuple of (reward, metrics_dict)
        """
        metrics = {
            "num_steps": trajectory.num_steps,
            "has_errors": trajectory.has_errors,
            "is_complete": trajectory.is_complete,
            "total_tokens": trajectory.total_tokens,
        }
        
        # Start with base reward
        reward = 0.0
        
        # If proof is complete, give completion reward
        if trajectory.is_complete:
            reward = self.config.completion_reward
            metrics["completion_bonus"] = self.config.completion_reward
        elif use_partial:
            # Partial reward based on progress
            partial_reward = self._calculate_partial_reward(trajectory)
            reward += partial_reward
            metrics["partial_reward"] = partial_reward
        
        # Add/subtract per-step rewards/penalties
        step_contribution = self._calculate_step_contribution(trajectory)
        reward += step_contribution
        metrics["step_contribution"] = step_contribution
        
        # Apply length penalty
        if trajectory.num_steps > self.config.max_steps_for_partial:
            excess = trajectory.num_steps - self.config.max_steps_for_partial
            length_penalty = excess * self.config.length_penalty_factor
            reward -= length_penalty
            metrics["length_penalty"] = length_penalty
        
        # Apply custom rewards
        for custom_fn in self.config.custom_rewards:
            try:
                custom_reward = custom_fn(trajectory)
                reward += custom_reward
                metrics.setdefault("custom_rewards", []).append(custom_reward)
            except Exception as e:
                metrics.setdefault("custom_reward_errors", []).append(str(e))
        
        # Apply reward shaping if configured
        if self.config.shaping_fn:
            try:
                shaped_reward = self.config.shaping_fn(trajectory, reward)
                metrics["shaped_reward_delta"] = shaped_reward - reward
                reward = shaped_reward
            except Exception as e:
                metrics["shaping_error"] = str(e)
        
        # Ensure reward is in reasonable range
        reward = max(-10.0, min(10.0, reward))
        
        metrics["final_reward"] = reward
        return reward, metrics
    
    def _calculate_partial_reward(self, trajectory: ProofTrajectory) -> float:
        """Calculate partial reward for incomplete proofs."""
        if not trajectory.steps:
            return 0.0
        
        # Reward based on progress
        progress = min(
            trajectory.num_steps / self.config.max_steps_for_partial,
            1.0
        )
        
        # Base progress reward
        reward = progress * 0.3  # Max 0.3 for progress
        
        # Reward for goal reduction
        if trajectory.steps:
            final_state = trajectory.get_final_state()
            if final_state:
                # Reward for each goal solved
                initial_goals = 1  # Assume 1 initial goal
                remaining_goals = final_state.num_goals_remaining
                goals_solved = max(0, initial_goals - remaining_goals)
                reward += goals_solved * self.config.goal_reduction_reward
        
        return reward
    
    def _calculate_step_contribution(self, trajectory: ProofTrajectory) -> float:
        """Calculate reward contribution from individual steps."""
        contribution = 0.0
        
        for step in trajectory.steps:
            if step.result.status == TacticStatus.SUCCESS:
                contribution += self.config.step_reward
            elif step.result.status == TacticStatus.ERROR:
                contribution += self.config.error_penalty
            elif step.result.status == TacticStatus.TIMEOUT:
                contribution += self.config.timeout_penalty
            # INCOMPLETE gets no reward or penalty
        
        return contribution
    
    async def validate_and_score(
        self,
        trajectory: ProofTrajectory,
    ) -> tuple[float, dict]:
        """Validate a proof with Lean and calculate score.
        
        This method actually checks the proof with Lean 4 to
        ensure it's valid, then calculates the reward.
        
        Args:
            trajectory: The trajectory to validate
            
        Returns:
            Tuple of (reward, metrics)
        """
        metrics = {"validated": False}
        
        if not self.config.use_lean_validation:
            # Skip validation, just calculate reward
            reward, calc_metrics = await self.calculate_reward(trajectory)
            metrics.update(calc_metrics)
            return reward, metrics
        
        # Build the final state
        final_state = LeanProofState(
            theorem_name=trajectory.theorem_name,
            theorem_statement=trajectory.theorem_statement,
            context=trajectory.context,
            imports=trajectory.imports,
            proof_steps=[step.result for step in trajectory.steps],
        )
        
        # Validate with Lean
        is_valid, error = await self.lean.check_proof(final_state)
        metrics["lean_valid"] = is_valid
        metrics["lean_error"] = error
        
        if is_valid:
            # Proof is valid - give full reward
            reward = self.config.completion_reward
            metrics["validation_bonus"] = 0.5  # Bonus for validated proof
            reward += metrics["validation_bonus"]
        else:
            # Proof is invalid - calculate partial reward
            reward, calc_metrics = await self.calculate_reward(trajectory)
            metrics.update(calc_metrics)
            # Penalty for invalid proof
            reward -= 0.2
            metrics["invalid_penalty"] = -0.2
        
        metrics["final_reward"] = reward
        metrics["validated"] = True
        
        return reward, metrics
    
    def compute_grpo_advantages(
        self,
        trajectories: list[ProofTrajectory],
    ) -> dict[str, float]:
        """Compute advantages for GRPO training.
        
        GRPO (Group Relative Policy Optimization) normalizes rewards
        within a group of trajectories for the same problem.
        
        Args:
            trajectories: List of trajectories for the same theorem
            
        Returns:
            Dict mapping trajectory IDs to advantages
        """
        if not trajectories:
            return {}
        
        rewards = [t.reward for t in trajectories]
        mean_reward = sum(rewards) / len(rewards)
        
        # Compute standard deviation
        if len(rewards) > 1:
            variance = sum((r - mean_reward) ** 2 for r in rewards) / (len(rewards) - 1)
            std_reward = variance ** 0.5
        else:
            std_reward = 1.0
        
        # Compute advantages
        advantages = {}
        for traj in trajectories:
            if std_reward > 1e-6:
                advantages[traj.id] = (traj.reward - mean_reward) / std_reward
            else:
                advantages[traj.id] = 0.0
        
        return advantages


# Predefined reward configurations

REWARD_CONFIG_BINARY = RewardConfig(
    completion_reward=1.0,
    step_reward=0.0,
    error_penalty=0.0,
    use_lean_validation=True,
)
"""Binary reward: 1 for complete proof, 0 otherwise."""

REWARD_CONFIG_SHAPED = RewardConfig(
    completion_reward=1.0,
    step_reward=0.02,
    error_penalty=-0.05,
    timeout_penalty=-0.02,
    goal_reduction_reward=0.1,
    length_penalty_factor=0.005,
    use_lean_validation=True,
)
"""Shaped reward with partial credit for progress."""

REWARD_CONFIG_LENIENT = RewardConfig(
    completion_reward=1.0,
    step_reward=0.05,
    error_penalty=-0.02,
    timeout_penalty=0.0,
    goal_reduction_reward=0.15,
    length_penalty_factor=0.002,
    use_lean_validation=True,
)
"""Lenient reward configuration that encourages exploration."""

REWARD_CONFIG_STRICT = RewardConfig(
    completion_reward=1.0,
    step_reward=0.0,
    error_penalty=-0.2,
    timeout_penalty=-0.1,
    goal_reduction_reward=0.0,
    length_penalty_factor=0.02,
    use_lean_validation=True,
)
"""Strict reward configuration that only rewards valid proofs."""
