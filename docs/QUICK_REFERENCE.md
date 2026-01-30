# Lean GRPO Quick Reference

## Installation

```bash
pip install -e ".[dev]"
```

## Basic Usage

### Train with GRPO (Default)

```bash
lean-grpo train theorems.jsonl --algorithm grpo
```

### Train with Different Algorithms

```bash
# GRPO Done Right (fixed IS, KL, normalization)
lean-grpo train theorems.jsonl --algorithm drgrpo

# Direct GRPO (preference learning)
lean-grpo train theorems.jsonl --algorithm dgpo

# DAPO (sparse rewards)
lean-grpo train theorems.jsonl --algorithm dapo

# GSPO (distributed training)
lean-grpo train theorems.jsonl --algorithm gspo
```

### Use Different Reward Scorers

```python
from lean_grpo.rewards import get_scorer

# Domain-specific
scorer = get_scorer("algebra")
scorer = get_scorer("topology")

# Difficulty-based
scorer = get_scorer("difficulty", difficulty="hard")

# Composite
from lean_grpo.rewards import create_composite_scorer
scorer = create_composite_scorer("algebra", "efficiency")
```

## Configuration Files

| Config | Use Case |
|--------|----------|
| `grpo_basic.yaml` | General purpose, good starting point |
| `grpo_aggressive.yaml` | Quick experiments, fast iteration |
| `grpo_conservative.yaml` | Hard problems, stable training |
| `drgrpo.yaml` | When GRPO has stability issues |
| `drgrpo_rank.yaml` | DrGRPO with rank normalization |
| `dgpo_preferences.yaml` | With pairwise preferences |
| `dapo.yaml` | Sparse rewards |
| `dapo_expert.yaml` | Very hard problems |
| `gspo.yaml` | Distributed training |
| `gspo_multi_gpu.yaml` | Multi-GPU setup |
| `hard_problems.yaml` | Expert-level problems |
| `algebra_domain.yaml` | Algebra-specific training |

## Example Scripts

| Script | Description |
|--------|-------------|
| `train_example.py` | Basic training with all algorithms |
| `train_advanced.py` | Advanced configurations |
| `train_with_rewards.py` | Custom reward scorers |

## Algorithm Selection Guide

| Algorithm | Best For | When to Use |
|-----------|----------|-------------|
| **GRPO** | General use | Starting point, dense rewards |
| **DrGRPO** | Stability issues | Unstable training, high variance |
| **DGPO** | Preferences | Pairwise proof comparisons |
| **DAPO** | Sparse rewards | Most proofs fail |
| **GSPO** | Distributed | Multi-GPU, large batches |

## Command Line Options

```bash
lean-grpo train THEOREMS.jsonl \
    --algorithm grpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --num-generations 8 \
    --lr 5e-6 \
    --epochs 3 \
    --output outputs/my_experiment
```

## Python API

```python
from lean_grpo import *

# Configure
config = LeanGRPOConfig(
    algorithm="drgrpo",
    algorithm_config={
        "advantage_norm_method": "winsorized",
    },
    num_generations=8,
)

# Setup
trainer = LeanGRPOTrainer(config, lean, inference)
trainer.setup()

# Train
trainer.train(dataset)
```

## Common Issues

### Training unstable
→ Use `--algorithm drgrpo` with `winsorized` normalization

### Rewards too sparse
→ Use `--algorithm dapo` with `use_reward_shaping: true`

### Proofs too long
→ Use efficiency scorer: `get_scorer("efficiency")`

### Multi-GPU training
→ Use `--algorithm gspo` with larger group size

## Quick Examples

```bash
# List available scorers
lean-grpo algorithms

# Train with custom config
lean-grpo train theorems.jsonl --config configs/drgrpo.yaml

# Train hard problems
lean-grpo train theorems.jsonl --config configs/hard_problems.yaml

# Compare algorithms
python examples/train_example.py --compare
```
