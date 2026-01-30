"""Tests for trajectory management."""

import pytest

from lean_grpo.lean_interface import LeanProofState, TacticResult, TacticStatus
from lean_grpo.trajectory import ProofStep, ProofTrajectory, TrajectoryGroup


def test_proof_trajectory_creation():
    trajectory = ProofTrajectory(
        theorem_name="test_theorem",
        theorem_statement="theorem test : True",
    )
    
    assert trajectory.theorem_name == "test_theorem"
    assert not trajectory.is_complete
    assert trajectory.num_steps == 0


def test_proof_trajectory_with_steps():
    trajectory = ProofTrajectory(
        theorem_name="test",
        theorem_statement="theorem test : True",
    )
    
    step1 = ProofStep(
        tactic="intro",
        result=TacticResult(
            status=TacticStatus.SUCCESS,
            tactic="intro",
            goals=[{}],
        ),
    )
    
    step2 = ProofStep(
        tactic="trivial",
        result=TacticResult(
            status=TacticStatus.SUCCESS,
            tactic="trivial",
            goals=[],
            proof_complete=True,
        ),
    )
    
    trajectory.steps = [step1, step2]
    
    assert trajectory.num_steps == 2
    assert trajectory.is_complete
    assert not trajectory.has_errors


def test_proof_trajectory_to_training_example():
    trajectory = ProofTrajectory(
        theorem_name="add_zero",
        theorem_statement="(n : Nat) : n + 0 = n",
    )
    
    trajectory.steps = [
        ProofStep(
            tactic="intro n",
            result=TacticResult(
                status=TacticStatus.SUCCESS,
                tactic="intro n",
                goals=[],
            ),
        ),
        ProofStep(
            tactic="rfl",
            result=TacticResult(
                status=TacticStatus.SUCCESS,
                tactic="rfl",
                goals=[],
                proof_complete=True,
            ),
        ),
    ]
    trajectory.reward = 1.0
    
    example = trajectory.to_training_example()
    
    assert "messages" in example
    assert "reward" in example
    assert example["reward"] == 1.0
    assert example["is_complete"] is True


def test_trajectory_group():
    group = TrajectoryGroup(
        theorem_name="test",
        theorem_statement="theorem test : True",
    )
    
    # Add trajectories
    for i in range(3):
        traj = ProofTrajectory(
            theorem_name="test",
            theorem_statement="theorem test : True",
        )
        traj.reward = 0.5 * (i + 1)
        group.add_trajectory(traj)
    
    assert len(group) == 3
    assert group.get_mean_reward() == 1.0
    
    best = group.get_best_trajectory()
    assert best is not None
    assert best.reward == 1.5


def test_trajectory_group_advantages():
    group = TrajectoryGroup(
        theorem_name="test",
        theorem_statement="theorem test : True",
    )
    
    # Add trajectories with different rewards
    for reward in [0.0, 0.5, 1.0]:
        traj = ProofTrajectory(
            theorem_name="test",
            theorem_statement="theorem test : True",
        )
        traj.reward = reward
        group.add_trajectory(traj)
    
    advantages = group.compute_advantages()
    
    assert len(advantages) == 3
    # Mean is 0.5, so 0.0 -> -1.22, 0.5 -> 0, 1.0 -> 1.22 (approximately)
    assert advantages[group.trajectories[0].id] < 0
    assert abs(advantages[group.trajectories[1].id]) < 0.01
    assert advantages[group.trajectories[2].id] > 0
