"""Tests for Lean interface."""

import pytest

from lean_grpo.lean_interface import (
    LeanProofState,
    MockLeanInterface,
    TacticResult,
    TacticStatus,
)


@pytest.fixture
def mock_lean():
    return MockLeanInterface()


@pytest.fixture
def sample_state():
    return LeanProofState(
        theorem_name="test",
        theorem_statement="theorem test : True",
        imports=["Mathlib"],
    )


@pytest.mark.asyncio
async def test_mock_execute_valid_tactic(mock_lean, sample_state):
    result = await mock_lean.execute_tactic(sample_state, "intro")
    
    assert result.status == TacticStatus.SUCCESS
    assert result.tactic == "intro"
    assert result.is_valid


@pytest.mark.asyncio
async def test_mock_execute_invalid_tactic(mock_lean, sample_state):
    result = await mock_lean.execute_tactic(sample_state, "invalid_tactic_xyz")
    
    assert result.status == TacticStatus.ERROR
    assert not result.is_valid


@pytest.mark.asyncio
async def test_check_proof_valid(mock_lean):
    state = LeanProofState(
        theorem_name="test",
        theorem_statement="theorem test : True",
        proof_steps=[
            TacticResult(
                status=TacticStatus.SUCCESS,
                tactic="trivial",
                proof_complete=True,
            )
        ],
    )
    
    is_valid, error = await mock_lean.check_proof(state)
    
    assert is_valid
    assert error is None


@pytest.mark.asyncio
async def test_check_proof_invalid(mock_lean):
    state = LeanProofState(
        theorem_name="test",
        theorem_statement="theorem test : True",
        proof_steps=[
            TacticResult(
                status=TacticStatus.ERROR,
                tactic="bad",
                error_message="Unknown tactic",
            )
        ],
    )
    
    is_valid, error = await mock_lean.check_proof(state)
    
    assert not is_valid


def test_proof_state_to_lean_code():
    state = LeanProofState(
        theorem_name="add_zero",
        theorem_statement="(n : Nat) : n + 0 = n",
        imports=["Mathlib"],
        proof_steps=[
            TacticResult(
                status=TacticStatus.SUCCESS,
                tactic="intro n",
                goals=[],
            ),
            TacticResult(
                status=TacticStatus.SUCCESS,
                tactic="rfl",
                proof_complete=True,
            ),
        ],
    )
    
    code = state.to_lean_code()
    
    assert "import Mathlib" in code
    assert "theorem add_zero" in code
    assert "intro n" in code
    assert "rfl" in code


def test_tactic_result_properties():
    result = TacticResult(
        status=TacticStatus.SUCCESS,
        tactic="intro",
        goals=[{}, {}],  # 2 goals
        proof_complete=False,
    )
    
    assert result.num_goals == 2
    assert result.is_valid
    assert not result.proof_complete


def test_tactic_result_error():
    result = TacticResult(
        status=TacticStatus.ERROR,
        tactic="bad",
        error_message="Unknown tactic",
    )
    
    assert not result.is_valid
    assert result.num_goals == 0
