# Reward Scoring System

This document describes the modular reward scoring system in Lean GRPO.

## Overview

The reward system is designed to be modular and swappable. You can:
- Use pre-built scorers for different domains and difficulties
- Combine multiple scorers
- Create custom scorers
- Swap scorers easily for different training runs

## Quick Start

```python
from lean_grpo.rewards import get_scorer, list_scorers

# See available scorers
print(list_scorers())

# Get a domain scorer
scorer = get_scorer("algebra")

# Get a difficulty-based scorer
scorer = get_scorer("difficulty", difficulty="hard")

# Combine scorers
from lean_grpo.rewards import create_composite_scorer
scorer = create_composite_scorer("algebra", "efficiency")
```

## Available Scorers

### 1. Difficulty-Based Scorer

Adjusts rewards based on problem difficulty (easy, medium, hard, expert).

```python
scorer = get_scorer("difficulty", difficulty="hard")
```

### 2. Domain-Specific Scorers

- `algebra` - Optimized for algebra problems
- `topology` - Optimized for topology problems  
- `analysis` - Optimized for analysis problems
- `number_theory` - Optimized for number theory
- `linear_algebra` - Optimized for linear algebra

### 3. Efficiency Scorer

Rewards shorter, more elegant proofs.

```python
scorer = get_scorer("efficiency")
```

### 4. Composite Scorer

Combine multiple scorers with custom weights.

```python
scorer = create_composite_scorer(
    "algebra", "efficiency", "difficulty",
    weights={"algebra": 0.4, "efficiency": 0.3, "difficulty": 0.3}
)
```

## Swapping Reward Scorers

### Method 1: Python API

```python
from lean_grpo.rewards import get_scorer
from lean_grpo.reward import LeanRewardCalculator

scorer = get_scorer("algebra")
reward_calc = LeanRewardCalculator(lean, scorer.config)
```

### Method 2: Command Line

```bash
# Use domain-specific scorer
lean-grpo train theorems.jsonl --reward-scorer algebra

# Use difficulty-based with custom settings
lean-grpo train theorems.jsonl --reward-scorer difficulty --difficulty hard
```

## Creating Custom Scorers

```python
from lean_grpo.rewards.base import BaseRewardScorer, RewardScorerConfig

class MyScorer(BaseRewardScorer):
    def score(self, trajectory, **kwargs):
        reward = 0.0
        if trajectory.is_complete:
            reward += 1.0
        return self.normalize_reward(reward), {}
```

## Configuration Files

Example config with custom reward:

```yaml
# config.yaml
algorithm: "grpo"
reward_scorer: "composite"
scorers:
  - name: "algebra"
    weight: 0.5
  - name: "efficiency"
    weight: 0.5
```
