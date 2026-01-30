# Lean GRPO: RL Training for Lean 4 Proof Generation

A flexible GRPO training pipeline for training Large Language Models to generate valid Lean 4 proofs with good tactics. Uses Lean 4 for partial reward calculation and supports multiple RL algorithms.

## Features

- 🎯 **Multiple RL Algorithms**: GRPO, DGPO, DrGRPO, DAPO, GSPO - easily swappable
- ✅ **Lean 4 Integration**: Partial rewards based on actual proof progress
- ⚡ **Fast Inference**: vLLM support for efficient generation during training
- 🔄 **API Endpoint Support**: Use any OpenAI-compatible inference endpoint
- 📊 **Flexible Rewards**: Multiple reward configurations
- 🚀 **Efficient Training**: LoRA/QLoRA via Unsloth
- 🧪 **Mock Testing**: Mock interfaces for testing without Lean 4 or GPU

## Supported Algorithms

| Algorithm | Full Name | Key Features | Best For |
|-----------|-----------|--------------|----------|
| **GRPO** | Group Relative Policy Optimization | Group-normalized advantages, no value function | General use, stable training |
| **DGPO** | Direct GRPO | Preference learning, DPO-style loss | Pairwise preferences |
| **DrGRPO** | Dr. GRPO (GRPO Done Right) | Fixed IS, unbiased KL, robust normalization | When GRPO has stability issues |
| **DAPO** | Decoupled Advantage Policy Optimization | Population stats, asymmetric loss, sparse reward handling | Sparse/diverse rewards |
| **GSPO** | Group-Synchronized Policy Optimization | Cross-group sync, consensus, dynamic groups | Large-scale distributed |

## Installation

```bash
# Clone and install
pip install -e ".[dev]"

# Or install from source
pip install git+https://github.com/lzumot/lean-grpo.git
```

### Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)
- Lean 4 (optional, mock mode available)

## Quick Start

### 1. Prepare Your Theorems

```jsonl
{"name": "add_zero", "statement": "theorem add_zero (n : Nat) : n + 0 = n", "context": "", "imports": ["Mathlib"]}
{"name": "add_comm", "statement": "theorem add_comm (n m : Nat) : n + m = m + n", "context": "", "imports": ["Mathlib"]}
```

### 2. Start vLLM Server (Optional)

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000
```

### 3. Train with Different Algorithms

```bash
# Default GRPO
lean-grpo train theorems.jsonl --algorithm grpo --num-generations 8

# DAPO for sparse rewards
lean-grpo train theorems.jsonl --algorithm dapo --num-generations 8

# GSPO for distributed training
lean-grpo train theorems.jsonl --algorithm gspo --num-generations 8

# DGPO with preference learning
lean-grpo train theorems.jsonl --algorithm dgpo --algo-config '{"use_dpo_loss": true, "dpo_coef": 0.5}'

# DrGRPO (GRPO Done Right) for stability
lean-grpo train theorems.jsonl --algorithm drgrpo --algo-config '{"kl_estimator": "schulman"}'
```

## Algorithm Selection Guide

### GRPO (Default)
```bash
lean-grpo train theorems.jsonl --algorithm grpo
```
- **Best for**: General use cases, stable training
- **Key feature**: Group-relative advantage normalization
- **When to use**: Good starting point, works well with dense rewards

### DGPO (Direct GRPO)
```bash
lean-grpo train theorems.jsonl --algorithm dgpo
```
- **Best for**: When you have pairwise preferences
- **Key feature**: Combines GRPO with DPO-style direct optimization
- **When to use**: You can compare proofs and say "this one is better"

### DrGRPO (Dr. GRPO - GRPO Done Right)
```bash
lean-grpo train theorems.jsonl --algorithm drgrpo
```
- **Best for**: When standard GRPO has stability issues
- **Key fixes**: 
  - Proper importance sampling with correction
  - Unbiased KL estimation (Schulman estimator)
  - Winsorized advantage normalization
  - Asymmetric clipping for positive/negative advantages
- **When to use**: GRPO training is unstable, high variance, or not converging

### DAPO (Decoupled Advantage Policy Optimization)
```bash
lean-grpo train theorems.jsonl --algorithm dapo
```
- **Best for**: Sparse rewards, diverse problem difficulties
- **Key features**: 
  - Population-based normalization (more stable than per-group)
  - Asymmetric loss (different treatment for positive/negative advantages)
  - Sparse reward detection and handling
- **When to use**: Most proofs fail (sparse success), problems vary greatly in difficulty

### GSPO (Group-Synchronized Policy Optimization)
```bash
lean-grpo train theorems.jsonl --algorithm gspo
```
- **Best for**: Large-scale distributed training
- **Key features**:
  - Cross-group gradient synchronization
  - Consensus-based advantages
  - Dynamic group sizing
- **When to use**: Training on many GPUs with large batch sizes

## Usage Examples

### Python API

```python
from lean_grpo import *

# Choose your algorithm
config = LeanGRPOConfig(
    base_model="Qwen/Qwen2.5-7B-Instruct",
    algorithm="dapo",  # or "grpo", "DGPO", "gspo"
    algorithm_config={
        "population_size": 100,
        "use_asymmetric_loss": True,
    },
    num_generations=8,
)

# Train
trainer = LeanGRPOTrainer(config, lean, inference)
trainer.setup()
trainer.train(dataset)
```

### Custom Algorithm Configuration

```python
# DAPO with custom settings
config = LeanGRPOConfig(
    algorithm="dapo",
    algorithm_config={
        "population_size": 200,
        "advantage_ema_decay": 0.99,
        "use_asymmetric_loss": True,
        "positive_advantage_scale": 1.0,
        "negative_advantage_scale": 0.8,
        "sparse_reward_threshold": 0.8,
    }
)

# GSPO for distributed training
config = LeanGRPOConfig(
    algorithm="gspo",
    algorithm_config={
        "sync_frequency": 4,
        "use_consensus": True,
        "consensus_weight": 0.3,
        "use_dynamic_groups": True,
    }
)
```

### Using the Algorithm Directly

```python
from lean_grpo.algorithms import GRPO, DAPO, GSPOConfig

# Create algorithm
algo = GRPO(group_size=8, normalize_advantages=True)

# Compute advantages
rewards = [0.0, 0.5, 1.0, 0.3, 0.8]
advantages = algo.compute_advantages(rewards)
print(advantages)  # {0: -1.2, 1: -0.2, 2: 1.3, ...}
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Lean GRPO Pipeline                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌─────────────┐ │
│  │   Theorems   │────▶│   Rollout    │────▶│    Lean 4   │ │
│  │   Dataset    │     │   Generator  │     │   Checker   │ │
│  └──────────────┘     └──────────────┘     └─────────────┘ │
│                              │                      │       │
│                              ▼                      ▼       │
│                       ┌──────────────┐     ┌─────────────┐ │
│                       │   LLM API    │     │   Reward    │ │
│                       │   (vLLM)     │◀────│ Calculator  │ │
│                       └──────────────┘     └─────────────┘ │
│                              │                               │
│                              ▼                               │
│              ┌───────────────────────────────┐              │
│              │      RL Algorithm            │              │
│              │  ┌─────┐┌─────┐┌─────┐┌────┐┌────┐│              │
│              │  │GRPO ││DGPO ││DrGRPO││DAPO││GSPO││              │
│              │  └─────┘└─────┘└─────┘└────┘└────┘│              │
│              └───────────────────────────────┘              │
│                              │                               │
│                              ▼                               │
│                       ┌──────────────┐                      │
│                       │   Trainer    │                      │
│                       │   (Unsloth)  │                      │
│                       └──────────────┘                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Algorithm Details

### GRPO (Group Relative Policy Optimization)

The default algorithm. For each theorem:
1. Generate G completions (group)
2. Compute mean reward within group
3. Advantage = (reward - mean) / std

**Key hyperparameters:**
- `group_size`: Number of rollouts per theorem
- `normalize_advantages`: Whether to normalize
- `epsilon`: Clipping parameter

### DGPO (Direct GRPO)

Adds preference learning to GRPO:
1. Standard GRPO advantages
2. Optional DPO-style loss on pairwise preferences
3. Can work with ranked preferences

**Key hyperparameters:**
- `use_dpo_loss`: Enable DPO component
- `dpo_beta`: Temperature for preference model
- `dpo_coef`: Weight of DPO loss (0 = pure GRPO)

### DrGRPO (Dr. GRPO - GRPO Done Right)

Fixes common issues with standard GRPO:
1. **Fixed Importance Sampling**: Proper token-level IS with correction
2. **Unbiased KL Estimation**: Uses proper Schulman estimator
3. **Robust Normalization**: Winsorized normalization handles outliers
4. **Asymmetric Clipping**: Different clip bounds for +/- advantages
5. **Numerical Stability**: Careful log-space computations

**Key hyperparameters:**
- `is_level`: Importance sampling level ('token', 'sequence', 'geometric_mean')
- `kl_estimator`: KL estimator ('schulman', 'abs', 'mse')
- `advantage_norm_method`: Normalization ('standard', 'winsorized', 'rank')
- `use_asymmetric_clip`: Enable asymmetric clipping
- `use_unbiased_kl`: Use unbiased KL estimator

### DAPO (Decoupled Advantage Policy Optimization)

Improves stability with:
1. **Population-based normalization**: Uses running statistics across groups
2. **Asymmetric loss**: Different clip bounds for positive/negative advantages
3. **Sparse reward handling**: Detects and adapts to sparse rewards

**Key hyperparameters:**
- `population_size`: Size of reward history
- `use_asymmetric_loss`: Enable asymmetric clipping
- `positive_advantage_scale`: Scale for positive advantages
- `negative_advantage_scale`: Scale for negative advantages

### GSPO (Group-Synchronized Policy Optimization)

For distributed training:
1. **Cross-group synchronization**: Syncs gradients across groups
2. **Consensus-based advantages**: Uses multiple groups for advantage estimation
3. **Dynamic groups**: Adjusts group size based on diversity

**Key hyperparameters:**
- `sync_frequency`: How often to sync gradients
- `use_consensus`: Enable consensus advantages
- `consensus_weight`: Weight of consensus term
- `use_dynamic_groups`: Enable dynamic group sizing

## CLI Reference

### Train

```bash
lean-grpo train THEOREMS.jsonl [OPTIONS]

Algorithm Options:
  --algorithm, -a TEXT          Algorithm: grpo, DGPO, dapo, gspo
  --algo-config TEXT            JSON config for algorithm-specific settings
  --num-generations, -n INT     Group size (default: 8)

Examples:
  # GRPO (default)
  lean-grpo train theorems.jsonl --algorithm grpo -n 8
  
  # DAPO with custom population
  lean-grpo train theorems.jsonl --algorithm dapo --algo-config '{"population_size": 200}'
  
  # GSPO with consensus
  lean-grpo train theorems.jsonl --algorithm gspo --algo-config '{"consensus_weight": 0.5}'
```

### Show Available Algorithms

```bash
lean-grpo algorithms
```

Output:
```
┌──────────┬───────────────────────────────────────────────┬──────────────────────────┐
│Algorithm │Description                                    │Best For                  │
├──────────┼───────────────────────────────────────────────┼──────────────────────────┤
│GRPO      │Group Relative Policy Optimization             │General use               │
│DGRPO     │Direct GRPO with preference learning           │Pairwise preferences      │
│DAPO      │Decoupled Advantage Policy Optimization        │Sparse rewards            │
│GSPO      │Group-Synchronized Policy Optimization         │Distributed training      │
└──────────┴───────────────────────────────────────────────┴──────────────────────────┘
```

## Configuration Files

### YAML Config with Algorithm

```yaml
# config_dapo.yaml
base_model: "Qwen/Qwen2.5-7B-Instruct"
algorithm: "dapo"
algorithm_config:
  population_size: 100
  use_asymmetric_loss: true
  positive_advantage_scale: 1.0
  negative_advantage_scale: 0.8
  use_reward_shaping: true

num_generations: 8
learning_rate: 5e-6
```

Usage:
```bash
lean-grpo train theorems.jsonl --config config_dapo.yaml
```

## Docker Compose with Algorithm Selection

```yaml
services:
  training:
    build: .
    command: >
      lean-grpo train theorems.jsonl
      --algorithm dapo
      --algo-config '{"population_size": 100}'
      --num-generations 8
```

## Testing Algorithms

```bash
# Run tests for all algorithms
pytest tests/test_algorithms.py -v

# Test specific algorithm
pytest tests/test_algorithms.py::TestDAPO -v
```

## Citation

If you use Lean GRPO in your research, please cite:

```bibtex
@software{lean_grpo,
  title={Lean GRPO: Multi-Algorithm RL Training for Lean 4 Proof Generation},
  year={2025},
  url={https://github.com/lzumot/lean-grpo}
}
```

## Acknowledgments

- [OpenPipe ART](https://github.com/OpenPipe/ART) - Architecture inspiration
- [TRL](https://github.com/huggingface/trl) - GRPO implementation
- [Unsloth](https://github.com/unslothai/unsloth) - Efficient fine-tuning
- [Lean 4](https://github.com/leanprover/lean4) - Theorem prover

## License

MIT License
