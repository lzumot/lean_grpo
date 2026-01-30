# Lean GRPO Repository Guide

A comprehensive guide to understanding the Lean GRPO codebase for first-time contributors and users.

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Core Components](#core-components)
4. [Algorithms](#algorithms)
5. [Reward System](#reward-system)
6. [Configuration System](#configuration-system)
7. [Examples](#examples)
8. [How to Navigate](#how-to-navigate)
9. [Getting Started](#getting-started)

---

## Overview

Lean GRPO is a training pipeline for teaching Large Language Models (LLMs) to generate valid Lean 4 proofs using Group Relative Policy Optimization (GRPO) and related RL algorithms.

### Key Philosophy

- **Modularity**: Swap algorithms, rewards, and configurations easily
- **Multiple Algorithms**: Support for GRPO, DrGRPO, DGPO, DAPO, GSPO
- **Domain Awareness**: Reward scorers for different mathematical domains
- **Configurable**: YAML/JSON configs for reproducible experiments

---

## Repository Structure

```
lean_grpo/
├── configs/              # Pre-built configuration files
├── docs/                 # Documentation
├── examples/             # Example scripts and tutorials
├── src/lean_grpo/        # Main source code
│   ├── algorithms/       # RL algorithm implementations
│   ├── rewards/          # Modular reward scorers
│   ├── __init__.py       # Package exports
│   ├── cli.py            # Command-line interface
│   ├── inference_client.py   # API client for LLMs
│   ├── lean_interface.py     # Lean 4 integration
│   ├── reward.py             # Legacy reward calculator
│   ├── rollout.py            # Proof generation rollout
│   ├── trainer.py            # Main training loop
│   └── trajectory.py         # Proof trajectory handling
├── tests/                # Test suite
├── CONTRIBUTING.md       # Contribution guidelines
├── Dockerfile            # Container setup
├── Makefile              # Development commands
├── README.md             # Main documentation
└── pyproject.toml        # Package dependencies
```

---

## Core Components

### 1. Trainer (`src/lean_grpo/trainer.py`)

**What it does**: Orchestrates the entire training process.

**Key Classes**:
- `LeanGRPOConfig`: Configuration dataclass for training
- `LeanGRPOTrainer`: Main trainer class
- `LeanGRPOPipeline`: End-to-end training pipeline

**How to use**:
```python
from lean_grpo import LeanGRPOConfig, LeanGRPOTrainer

config = LeanGRPOConfig(algorithm="drgrpo", num_generations=8)
trainer = LeanGRPOTrainer(config, lean_interface, inference_client)
trainer.setup()
trainer.train(dataset)
```

### 2. Lean Interface (`src/lean_grpo/lean_interface.py`)

**What it does**: Communicates with Lean 4 to check proofs.

**Key Classes**:
- `LeanInterface`: Real Lean 4 integration (requires Lean installed)
- `MockLeanInterface`: Mock version for testing
- `LeanProofState`: Represents proof state
- `TacticResult`: Result of tactic execution

**Key Point**: Use `MockLeanInterface` for development/testing without Lean installed.

### 3. Inference Client (`src/lean_grpo/inference_client.py`)

**What it does**: Connects to LLM inference endpoints.

**Key Classes**:
- `InferenceClient`: Base client for OpenAI-compatible APIs
- `VLLMClient`: Specialized for vLLM with GRPO support
- `MockInferenceClient`: Mock for testing

**Usage**:
```python
client = VLLMClient(
    base_url="http://localhost:8000/v1",
    api_key="your-key"
)
```

### 4. Rollout Generator (`src/lean_grpo/rollout.py`)

**What it does**: Generates proof attempts using the LLM.

**Key Classes**:
- `ProofRolloutGenerator`: Generates individual rollouts
- `BatchRolloutManager`: Manages parallel rollout generation

**Flow**: 
1. Theorem → Prompt → LLM → Tactic
2. Tactic → Lean → Result
3. Repeat until complete or max steps

### 5. Trajectory (`src/lean_grpo/trajectory.py`)

**What it does**: Tracks proof attempts and results.

**Key Classes**:
- `ProofStep`: Single tactic execution
- `ProofTrajectory`: Complete proof attempt
- `TrajectoryGroup`: Group of trajectories for same theorem

**Usage**: Trajectories store tactics, states, rewards, and metadata.

---

## Algorithms

Located in `src/lean_grpo/algorithms/`

### Architecture

All algorithms inherit from `RLAlgorithm` base class:

```
RLAlgorithm (base.py)
    ├── GRPO (grpo.py)          # Default
    ├── DGPO (dgrpo.py)         # Direct GRPO (preference learning)
    ├── DrGRPO (drgrpo.py)      # GRPO Done Right (fixes)
    ├── DAPO (dapo.py)          # Decoupled Advantage
    └── GSPO (gspo.py)          # Group-Synchronized
```

### How to Add a New Algorithm

1. Create file in `algorithms/my_algo.py`
2. Inherit from `RLAlgorithm`
3. Implement `compute_advantages()` and `compute_loss()`
4. Register in `algorithms/__init__.py`

### Algorithm Selection Guide

| Algorithm | File | Use When |
|-----------|------|----------|
| GRPO | `grpo.py` | Starting point |
| DrGRPO | `drgrpo.py` | GRPO unstable |
| DGPO | `dgrpo.py` | Have preferences |
| DAPO | `dapo.py` | Sparse rewards |
| GSPO | `gspo.py` | Distributed training |

---

## Reward System

Located in `src/lean_grpo/rewards/`

### Design Philosophy

The reward system is **modular and swappable**. You can:
- Use pre-built scorers
- Combine multiple scorers
- Create custom scorers
- Swap at runtime

### Structure

```
rewards/
├── base.py          # BaseRewardScorer class
├── difficulty.py    # DifficultyBasedScorer
├── domain.py        # Domain-specific scorers
├── efficiency.py    # EfficiencyScorer
├── composite.py     # Combining scorers
└── registry.py      # Easy access functions
```

### Quick Usage

```python
from lean_grpo.rewards import get_scorer, create_composite_scorer

# Single scorer
scorer = get_scorer("algebra")

# Composite
scorer = create_composite_scorer(
    "algebra", "efficiency",
    weights={"algebra": 0.6, "efficiency": 0.4}
)
```

### Creating Custom Scorers

```python
from lean_grpo.rewards.base import BaseRewardScorer

class MyScorer(BaseRewardScorer):
    def score(self, trajectory, **kwargs):
        reward = 0.0
        # Your logic
        return self.normalize_reward(reward), {}
```

---

## Configuration System

### Config Files Location

All configs are in `configs/` directory.

### Config Hierarchy

1. **Default values** in `LeanGRPOConfig` dataclass
2. **YAML config file** overrides defaults
3. **Command line args** override YAML
4. **Algorithm-specific config** in `algorithm_config` dict

### Config File Structure

```yaml
# configs/example.yaml

# Model
base_model: "Qwen/Qwen2.5-7B-Instruct"
lora_rank: 8

# Algorithm selection
algorithm: "drgrpo"
algorithm_config:
  advantage_norm_method: "winsorized"
  kl_estimator: "schulman"

# Training
learning_rate: 5e-6
num_train_epochs: 3

# Generation
num_generations: 8
max_proof_steps: 20

# Reward
reward_type: "shaped"
```

### Using Configs

```bash
# Use a config file
lean-grpo train theorems.jsonl --config configs/drgrpo.yaml

# Override specific values
lean-grpo train theorems.jsonl --config configs/drgrpo.yaml --lr 1e-5
```

---

## Examples

Located in `examples/`

| File | Purpose | Run It |
|------|---------|--------|
| `train_example.py` | Basic training with all algorithms | `python examples/train_example.py --algorithm grpo` |
| `train_advanced.py` | Advanced configurations | `python examples/train_advanced.py` |
| `train_with_rewards.py` | Custom reward scorers | `python examples/train_with_rewards.py` |
| `train_domain_specific.py` | Domain-specific training | `python examples/train_domain_specific.py` |
| `example_theorems.jsonl` | Sample theorems | - |

### Example Patterns

Each example follows this pattern:
1. Load theorems
2. Create config
3. Setup interfaces (Lean, Inference)
4. Create trainer
5. Train

---

## How to Navigate

### Finding What You Need

| I want to... | Look in... |
|--------------|------------|
| Change training hyperparameters | `configs/*.yaml` or `trainer.py` |
| Use a different algorithm | `algorithms/` or `--algorithm` flag |
| Customize rewards | `rewards/` or `examples/train_with_rewards.py` |
| Add a new domain scorer | `rewards/domain.py` |
| Connect to different LLM | `inference_client.py` |
| Debug Lean integration | `lean_interface.py` |
| See usage examples | `examples/` |
| Run tests | `tests/` |

### Key Files for Common Tasks

**Adding a new algorithm**:
1. `src/lean_grpo/algorithms/my_algo.py` - Implement
2. `src/lean_grpo/algorithms/__init__.py` - Register
3. `src/lean_grpo/trainer.py` - Add to setup
4. `configs/` - Create example config

**Adding a new reward scorer**:
1. `src/lean_grpo/rewards/my_scorer.py` - Implement
2. `src/lean_grpo/rewards/registry.py` - Register
3. `docs/REWARDS.md` - Document

**Adding a new example**:
1. `examples/my_example.py` - Create
2. Follow pattern from existing examples
3. Update this guide

---

## Getting Started

### For Users

1. **Install**: `pip install -e "."`
2. **Try an example**: `python examples/train_example.py`
3. **Use CLI**: `lean-grpo train examples/example_theorems.jsonl`
4. **Read docs**: Start with `README.md`, then `docs/QUICK_REFERENCE.md`

### For Developers

1. **Install with dev dependencies**: `pip install -e ".[dev]"`
2. **Run tests**: `make test`
3. **Explore code**: Start with `trainer.py`, then `algorithms/`
4. **Make changes**: Follow existing patterns
5. **Add tests**: In `tests/` directory

### For Researchers

1. **Understand algorithms**: Read `algorithms/*.py` files
2. **Experiment with rewards**: Use `examples/train_with_rewards.py`
3. **Try different configs**: Use files in `configs/`
4. **Compare approaches**: Run `examples/train_advanced.py`

---

## Code Patterns

### Async Pattern

Most operations are async for efficiency:

```python
async def train():
    trajectory = await generator.generate_rollout(...)
    result = await lean.execute_tactic(...)
```

### Factory Pattern

Algorithms and scorers use factory functions:

```python
algo = create_algorithm("drgrpo", **config)
scorer = get_scorer("algebra")
```

### Config Pattern

Dataclasses with defaults, overridden by YAML/CLI:

```python
@dataclass
class Config:
    param: str = "default"

config = Config()
# Override from YAML/CLI
```

---

## Testing

### Test Structure

```
tests/
├── test_algorithms.py      # Algorithm tests
├── test_lean_interface.py  # Lean interface tests
└── test_trajectory.py      # Trajectory tests
```

### Running Tests

```bash
# All tests
make test

# Specific test
pytest tests/test_algorithms.py::TestGRPO -v

# With coverage
make test-cov
```

---

## Common Pitfalls

1. **Forgetting to call `setup()`**: Trainer needs `trainer.setup()` before use
2. **Not using mock interfaces**: Use `MockLeanInterface` and `MockInferenceClient` for testing
3. **Wrong config path**: Configs are relative to working directory
4. **Missing Lean**: Real Lean 4 must be installed for production use

---

## Next Steps

- **Read**: `README.md` for overview, `docs/QUICK_REFERENCE.md` for commands
- **Run**: `examples/train_example.py` to see it in action
- **Experiment**: Try different configs in `configs/`
- **Customize**: Create your own reward scorer or algorithm
- **Contribute**: See `CONTRIBUTING.md` for guidelines

---

## Questions?

- Check `docs/` directory for more guides
- Look at `examples/` for working code
- Read docstrings in source files
- Run with `--help` for CLI options
