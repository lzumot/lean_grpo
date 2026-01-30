"""Trajectory management for proof generation."""

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from lean_grpo.lean_interface import LeanProofState, TacticResult


@dataclass
class ProofStep:
    """A single step in a proof trajectory."""
    
    # The tactic that was generated
    tactic: str
    
    # The result of executing the tactic
    result: TacticResult
    
    # The state before this step
    state_before: Optional[LeanProofState] = None
    
    # The state after this step
    state_after: Optional[LeanProofState] = None
    
    # Token usage for this step
    prompt_tokens: int = 0
    completion_tokens: int = 0
    
    # Generation metadata
    logprobs: Optional[list[float]] = None
    
    @property
    def is_valid(self) -> bool:
        """Check if this step was valid."""
        return self.result.is_valid


@dataclass
class ProofTrajectory:
    """A complete trajectory for proof generation.
    
    This represents one attempt at proving a theorem, including
    all tactics tried and the final outcome.
    """
    
    # Unique identifier for this trajectory
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # Initial theorem to prove
    theorem_name: str = ""
    theorem_statement: str = ""
    
    # Context (imports, definitions, etc.)
    context: str = ""
    imports: list[str] = field(default_factory=lambda: ["Mathlib"])
    
    # The proof steps taken
    steps: list[ProofStep] = field(default_factory=list)
    
    # Final reward for this trajectory
    reward: float = 0.0
    
    # Additional metrics
    metrics: dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Error information if trajectory failed
    error: Optional[str] = None
    
    @property
    def is_complete(self) -> bool:
        """Check if the proof is complete."""
        if not self.steps:
            return False
        return self.steps[-1].result.proof_complete
    
    @property
    def has_errors(self) -> bool:
        """Check if any step had errors."""
        return any(not step.is_valid for step in self.steps)
    
    @property
    def num_steps(self) -> int:
        """Number of proof steps."""
        return len(self.steps)
    
    @property
    def total_tokens(self) -> int:
        """Total tokens used in this trajectory."""
        return sum(
            step.prompt_tokens + step.completion_tokens
            for step in self.steps
        )
    
    def get_final_state(self) -> Optional[LeanProofState]:
        """Get the final proof state."""
        if not self.steps:
            return None
        return self.steps[-1].state_after
    
    def to_training_example(self) -> dict[str, Any]:
        """Convert this trajectory to a training example.
        
        Returns a dictionary suitable for training with TRL/GRPO.
        """
        # Build the conversation
        messages = []
        
        # System prompt
        messages.append({
            "role": "system",
            "content": (
                "You are a Lean 4 proof assistant. "
                "Generate tactics to prove the given theorem. "
                "Respond with valid Lean 4 tactics only."
            )
        })
        
        # User prompt with theorem
        theorem_text = self.theorem_statement
        if self.context:
            theorem_text = f"{self.context}\n\n{theorem_text}"
        
        messages.append({
            "role": "user",
            "content": f"Prove the following theorem:\n\n{theorem_text}"
        })
        
        # Assistant responses (tactics)
        assistant_content = ""
        for i, step in enumerate(self.steps):
            if i > 0:
                assistant_content += "\n"
            assistant_content += step.tactic
        
        if assistant_content:
            messages.append({
                "role": "assistant",
                "content": assistant_content
            })
        
        return {
            "messages": messages,
            "reward": self.reward,
            "trajectory_id": self.id,
            "theorem_name": self.theorem_name,
            "num_steps": self.num_steps,
            "is_complete": self.is_complete,
            **self.metrics,
        }
    
    def to_lean_code(self) -> str:
        """Generate the complete Lean 4 code for this proof."""
        state = LeanProofState(
            theorem_name=self.theorem_name,
            theorem_statement=self.theorem_statement,
            context=self.context,
            imports=self.imports,
            proof_steps=[step.result for step in self.steps],
        )
        return state.to_lean_code()
    
    def get_tactics_text(self) -> str:
        """Get all tactics as a single text string."""
        return '\n'.join(step.tactic for step in self.steps)
    
    def copy(self) -> "ProofTrajectory":
        """Create a copy of this trajectory."""
        return ProofTrajectory(
            id=str(uuid.uuid4())[:8],
            theorem_name=self.theorem_name,
            theorem_statement=self.theorem_statement,
            context=self.context,
            imports=self.imports.copy(),
            steps=[],  # Start fresh
            reward=0.0,
            metrics=self.metrics.copy(),
            metadata=self.metadata.copy(),
        )


@dataclass
class TrajectoryGroup:
    """A group of trajectories for the same theorem.
    
    This is used for GRPO training where multiple attempts
    at the same theorem are grouped together.
    """
    
    theorem_name: str
    theorem_statement: str
    trajectories: list[ProofTrajectory] = field(default_factory=list)
    
    def __len__(self) -> int:
        return len(self.trajectories)
    
    def __iter__(self):
        return iter(self.trajectories)
    
    def add_trajectory(self, trajectory: ProofTrajectory) -> None:
        """Add a trajectory to this group."""
        self.trajectories.append(trajectory)
    
    def get_rewards(self) -> list[float]:
        """Get all rewards in this group."""
        return [t.reward for t in self.trajectories]
    
    def get_mean_reward(self) -> float:
        """Get the mean reward of this group."""
        if not self.trajectories:
            return 0.0
        return sum(t.reward for t in self.trajectories) / len(self.trajectories)
    
    def get_best_trajectory(self) -> Optional[ProofTrajectory]:
        """Get the trajectory with the highest reward."""
        if not self.trajectories:
            return None
        return max(self.trajectories, key=lambda t: t.reward)
    
    def compute_advantages(self) -> dict[str, float]:
        """Compute advantages for each trajectory using GRPO.
        
        Returns a dict mapping trajectory IDs to advantages.
        """
        if not self.trajectories:
            return {}
        
        rewards = [t.reward for t in self.trajectories]
        mean_reward = sum(rewards) / len(rewards)
        
        # Compute standard deviation
        if len(rewards) > 1:
            variance = sum((r - mean_reward) ** 2 for r in rewards) / (len(rewards) - 1)
            std_reward = variance ** 0.5
        else:
            std_reward = 1.0
        
        # Compute advantages (normalized rewards)
        advantages = {}
        for traj in self.trajectories:
            if std_reward > 0:
                advantages[traj.id] = (traj.reward - mean_reward) / std_reward
            else:
                advantages[traj.id] = 0.0
        
        return advantages
