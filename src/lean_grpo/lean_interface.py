"""Lean 4 REPL interface for proof checking and state management."""

import json
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import AsyncIterator, Optional

import aiofiles


class TacticStatus(Enum):
    """Status of a tactic execution."""
    
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    INCOMPLETE = "incomplete"


@dataclass
class TacticResult:
    """Result of executing a tactic in Lean 4."""
    
    status: TacticStatus
    tactic: str
    goals: list[dict] = field(default_factory=list)
    error_message: Optional[str] = None
    proof_complete: bool = False
    line_number: Optional[int] = None
    
    @property
    def num_goals(self) -> int:
        """Return the number of remaining goals."""
        return len(self.goals)
    
    @property
    def is_valid(self) -> bool:
        """Check if the tactic was valid (no errors)."""
        return self.status in (TacticStatus.SUCCESS, TacticStatus.INCOMPLETE)


@dataclass 
class LeanProofState:
    """Represents the current state of a proof attempt."""
    
    theorem_name: str
    theorem_statement: str
    proof_steps: list[TacticResult] = field(default_factory=list)
    context: str = ""
    imports: list[str] = field(default_factory=lambda: ["Mathlib"])
    
    @property
    def current_goals(self) -> list[dict]:
        """Get the current goals from the last proof step."""
        if not self.proof_steps:
            return []
        return self.proof_steps[-1].goals
    
    @property
    def num_goals_remaining(self) -> int:
        """Get the number of goals remaining."""
        if not self.proof_steps:
            return 1  # Initial goal
        return self.proof_steps[-1].num_goals
    
    @property
    def is_complete(self) -> bool:
        """Check if the proof is complete (no goals remaining)."""
        if not self.proof_steps:
            return False
        return self.proof_steps[-1].proof_complete
    
    @property
    def has_errors(self) -> bool:
        """Check if any proof step has errors."""
        return any(
            step.status == TacticStatus.ERROR for step in self.proof_steps
        )
    
    def to_lean_code(self) -> str:
        """Generate the complete Lean 4 code for this proof."""
        lines = []
        
        # Add imports
        for imp in self.imports:
            lines.append(f"import {imp}")
        lines.append("")
        
        # Add context if provided
        if self.context:
            lines.append(self.context)
            lines.append("")
        
        # Add theorem statement
        lines.append(f"theorem {self.theorem_name} {self.theorem_statement}")
        lines.append("  := by")
        
        # Add proof steps
        for step in self.proof_steps:
            tactic_lines = step.tactic.strip().split('\n')
            for i, tac_line in enumerate(tactic_lines):
                if i == 0:
                    lines.append(f"    {tac_line}")
                else:
                    lines.append(f"      {tac_line}")
        
        return '\n'.join(lines)
    
    def get_proof_so_far(self) -> str:
        """Get just the proof tactics applied so far."""
        return '\n'.join(step.tactic for step in self.proof_steps)


class LeanInterface:
    """Interface to Lean 4 for proof checking and tactic execution.
    
    This class provides methods to interact with Lean 4 through the REPL
    or by checking files. It supports both sync and async operations.
    """
    
    def __init__(
        self,
        lean_cmd: str = "lake",
        timeout: float = 30.0,
        max_memory_mb: int = 4096,
        temp_dir: Optional[Path] = None,
    ):
        """Initialize the Lean interface.
        
        Args:
            lean_cmd: Command to run Lean (usually 'lake' or 'lean')
            timeout: Default timeout for Lean operations in seconds
            max_memory_mb: Maximum memory for Lean process
            temp_dir: Directory for temporary Lean files
        """
        self.lean_cmd = lean_cmd
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.temp_dir = temp_dir or Path(tempfile.gettempdir()) / "lean_grpo"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for theorem states
        self._state_cache: dict[str, LeanProofState] = {}
    
    async def execute_tactic(
        self,
        state: LeanProofState,
        tactic: str,
        timeout: Optional[float] = None,
    ) -> TacticResult:
        """Execute a tactic on the current proof state.
        
        Args:
            state: Current proof state
            tactic: Tactic to apply
            timeout: Optional timeout override
            
        Returns:
            TacticResult with the outcome
        """
        # Create a temporary Lean file with the proof so far
        temp_file = self.temp_dir / f"proof_{id(state)}_{time.time_ns()}.lean"
        
        # Add the new tactic to the state
        test_state = LeanProofState(
            theorem_name=state.theorem_name,
            theorem_statement=state.theorem_statement,
            proof_steps=state.proof_steps + [
                TacticResult(
                    status=TacticStatus.SUCCESS,
                    tactic=tactic,
                    goals=[],
                )
            ],
            context=state.context,
            imports=state.imports,
        )
        
        lean_code = test_state.to_lean_code()
        
        try:
            async with aiofiles.open(temp_file, 'w') as f:
                await f.write(lean_code)
            
            # Run Lean on the file
            result = await self._run_lean_check(temp_file, timeout)
            
            # Parse the result
            return self._parse_tactic_result(result, tactic)
            
        except asyncio.TimeoutError:
            return TacticResult(
                status=TacticStatus.TIMEOUT,
                tactic=tactic,
                error_message="Tactic execution timed out",
            )
        except Exception as e:
            return TacticResult(
                status=TacticStatus.ERROR,
                tactic=tactic,
                error_message=str(e),
            )
        finally:
            # Cleanup
            if temp_file.exists():
                temp_file.unlink()
    
    async def check_proof(
        self,
        state: LeanProofState,
        timeout: Optional[float] = None,
    ) -> tuple[bool, Optional[str]]:
        """Check if a complete proof is valid.
        
        Args:
            state: The proof state to check
            timeout: Optional timeout override
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        temp_file = self.temp_dir / f"check_{id(state)}_{time.time_ns()}.lean"
        lean_code = state.to_lean_code()
        
        try:
            async with aiofiles.open(temp_file, 'w') as f:
                await f.write(lean_code)
            
            result = await self._run_lean_check(temp_file, timeout)
            
            # Check for errors
            if result.returncode == 0:
                return True, None
            else:
                return False, result.stderr or result.stdout
                
        except asyncio.TimeoutError:
            return False, "Proof checking timed out"
        except Exception as e:
            return False, str(e)
        finally:
            if temp_file.exists():
                temp_file.unlink()
    
    async def _run_lean_check(
        self,
        file_path: Path,
        timeout: Optional[float] = None,
    ) -> subprocess.CompletedProcess:
        """Run Lean check on a file."""
        import asyncio
        
        cmd = [self.lean_cmd, "build", str(file_path)]
        
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.temp_dir,
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout or self.timeout,
            )
            return subprocess.CompletedProcess(
                cmd=cmd,
                returncode=proc.returncode or 0,
                stdout=stdout.decode('utf-8', errors='replace'),
                stderr=stderr.decode('utf-8', errors='replace'),
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise
    
    def _parse_tactic_result(
        self,
        result: subprocess.CompletedProcess,
        tactic: str,
    ) -> TacticResult:
        """Parse the result of running Lean on a tactic."""
        if result.returncode == 0:
            # No errors - proof is complete or has remaining goals
            # Check if proof is complete by looking for "goals accomplished"
            output = result.stdout + result.stderr
            
            proof_complete = "goals accomplished" in output.lower()
            
            # Try to extract goals from output
            goals = self._extract_goals(output)
            
            return TacticResult(
                status=TacticStatus.SUCCESS,
                tactic=tactic,
                goals=goals,
                proof_complete=proof_complete,
            )
        else:
            # Error occurred
            error_msg = result.stderr or result.stdout
            
            # Categorize errors
            if "unknown tactic" in error_msg.lower():
                status = TacticStatus.ERROR
            elif "timeout" in error_msg.lower():
                status = TacticStatus.TIMEOUT
            elif "unexpected end of input" in error_msg.lower():
                status = TacticStatus.INCOMPLETE
            else:
                status = TacticStatus.ERROR
            
            return TacticResult(
                status=status,
                tactic=tactic,
                error_message=error_msg[:1000],  # Truncate long errors
            )
    
    def _extract_goals(self, output: str) -> list[dict]:
        """Extract goals from Lean output.
        
        This is a simplified parser - in practice, you'd want to use
        Lean's JSON output format or LSP for more reliable parsing.
        """
        goals = []
        
        # Look for goal patterns like "⊢ Type" or "case name"
        goal_pattern = r'[⊢|case]\s*(.+?)(?=\n\s*[⊢|case]|\Z)'
        matches = re.finditer(goal_pattern, output, re.DOTALL)
        
        for match in matches:
            goal_text = match.group(1).strip()
            goals.append({
                "type": "standard",
                "text": goal_text,
            })
        
        return goals
    
    async def batch_check_proofs(
        self,
        states: list[LeanProofState],
        timeout: Optional[float] = None,
    ) -> list[tuple[bool, Optional[str]]]:
        """Check multiple proofs in parallel.
        
        Args:
            states: List of proof states to check
            timeout: Optional timeout per proof
            
        Returns:
            List of (is_valid, error_message) tuples
        """
        import asyncio
        
        tasks = [self.check_proof(state, timeout) for state in states]
        return await asyncio.gather(*tasks)
    
    def get_partial_proof_score(
        self,
        state: LeanProofState,
        max_steps: int = 20,
    ) -> float:
        """Calculate a partial score for an incomplete proof.
        
        This provides a reward signal for proofs that make progress
        even if they don't complete.
        
        Args:
            state: The proof state
            max_steps: Maximum expected proof steps
            
        Returns:
            Score between 0 and 1
        """
        if state.is_complete:
            return 1.0
        
        if state.has_errors:
            # Penalize errors, but still give some credit for progress
            valid_steps = sum(
                1 for step in state.proof_steps
                if step.status != TacticStatus.ERROR
            )
            return 0.1 * (valid_steps / max_steps)
        
        # Reward based on progress (steps taken without errors)
        # and remaining goals
        step_progress = len(state.proof_steps) / max_steps
        goal_penalty = state.num_goals_remaining * 0.1
        
        score = 0.3 * step_progress - goal_penalty
        return max(0.0, min(0.9, score))  # Cap at 0.9 for incomplete proofs


class MockLeanInterface(LeanInterface):
    """Mock Lean interface for testing without Lean installed."""
    
    async def execute_tactic(
        self,
        state: LeanProofState,
        tactic: str,
        timeout: Optional[float] = None,
    ) -> TacticResult:
        """Mock tactic execution."""
        # Simple heuristic: certain tactics are "valid"
        valid_tactics = ['rfl', 'simp', 'exact', 'apply', 'intro', 'cases']
        
        is_valid = any(tac in tactic.lower() for tac in valid_tactics)
        
        if is_valid:
            # Simulate reducing goals
            remaining_goals = max(0, state.num_goals_remaining - 1)
            return TacticResult(
                status=TacticStatus.SUCCESS,
                tactic=tactic,
                goals=[{}] * remaining_goals if remaining_goals > 0 else [],
                proof_complete=remaining_goals == 0,
            )
        else:
            return TacticResult(
                status=TacticStatus.ERROR,
                tactic=tactic,
                error_message=f"Unknown tactic: {tactic}",
            )
    
    async def check_proof(
        self,
        state: LeanProofState,
        timeout: Optional[float] = None,
    ) -> tuple[bool, Optional[str]]:
        """Mock proof checking."""
        is_valid = not state.has_errors and state.is_complete
        return is_valid, None if is_valid else "Proof incomplete or has errors"
