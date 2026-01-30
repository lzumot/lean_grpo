"""Tests for RL algorithms."""

import pytest
import torch

from lean_grpo.algorithms import (
    DAPO,
    DAPOConfig,
    DGPO,
    DGPOConfig,
    DrGRPO,
    DrGRPOConfig,
    GRPO,
    GRPOConfig,
    GSPO,
    GSPOConfig,
    create_algorithm,
)


class TestGRPO:
    """Tests for GRPO algorithm."""
    
    def test_creation(self):
        config = GRPOConfig(group_size=8)
        algo = GRPO(config)
        assert algo.name == "GRPO"
        assert algo.config.group_size == 8
    
    def test_compute_advantages(self):
        algo = GRPO(GRPOConfig())
        rewards = [0.0, 0.5, 1.0]
        
        advantages = algo.compute_advantages(rewards)
        
        assert len(advantages) == 3
        # Mean reward is 0.5, so middle should be ~0
        assert abs(advantages[1]) < 0.1
        # First should be negative, last positive
        assert advantages[0] < advantages[1] < advantages[2]
    
    def test_compute_advantages_empty(self):
        algo = GRPO(GRPOConfig())
        advantages = algo.compute_advantages([])
        assert advantages == {}
    
    def test_compute_advantages_single(self):
        algo = GRPO(GRPOConfig())
        advantages = algo.compute_advantages([1.0])
        # Single reward should have advantage of 0 (or very small)
        assert len(advantages) == 1
    
    def test_compute_kl_penalty(self):
        algo = GRPO(GRPOConfig())
        
        new_logprobs = torch.log(torch.tensor([0.5, 0.5]))
        ref_logprobs = torch.log(torch.tensor([0.4, 0.6]))
        
        kl = algo.compute_kl_penalty(new_logprobs, ref_logprobs)
        
        assert kl.shape == (2,)
        assert torch.all(kl >= 0)  # KL is always non-negative
    
    def test_normalize_advantages_false(self):
        config = GRPOConfig(normalize_advantages=False)
        algo = GRPO(config)
        rewards = [0.0, 1.0, 2.0]
        
        advantages = algo.compute_advantages(rewards)
        
        # Without normalization, advantages = rewards
        assert advantages[0] == 0.0
        assert advantages[2] == 2.0


class TestDGPO:
    """Tests for DGPO (Direct GRPO) algorithm."""
    
    def test_creation(self):
        config = DGPOConfig(group_size=8)
        algo = DGPO(config)
        assert algo.name == "DGPO"
    
    def test_compute_grpo_advantages(self):
        algo = DGPO(DGPOConfig())
        rewards = [0.0, 0.5, 1.0]
        
        advantages = algo._compute_grpo_advantages(rewards)
        
        assert len(advantages) == 3
        assert abs(advantages[1]) < 0.1  # Middle should be ~0
    
    def test_compute_preference_advantages(self):
        algo = DGPO(DGPOConfig())
        rewards = [0.0, 0.5, 1.0]
        preferences = [(2, 0), (2, 1), (1, 0)]  # 2 > 1 > 0
        
        advantages = algo._compute_preference_advantages(rewards, preferences)
        
        assert len(advantages) == 3
        # Winner (index 2) should have highest advantage
        assert advantages[2] > advantages[1] > advantages[0]
    
    def test_sample_preference_pairs(self):
        algo = DGPO(DGPOConfig(num_preference_pairs=2))
        rewards = [0.0, 0.2, 0.5, 0.8, 1.0]
        
        pairs = algo.sample_preference_pairs(rewards)
        
        assert len(pairs) <= 2
        for winner, loser in pairs:
            assert rewards[winner] > rewards[loser]
    
    def test_use_dpo_loss_flag(self):
        config = DGPOConfig(use_dpo_loss=True, dpo_coef=0.5)
        algo = DGPO(config)
        assert algo.config.use_dpo_loss is True
        assert algo.config.dpo_coef == 0.5


class TestDrGRPO:
    """Tests for DrGRPO (GRPO Done Right) algorithm."""
    
    def test_creation(self):
        config = DrGRPOConfig(group_size=8)
        algo = DrGRPO(config)
        assert algo.name == "DrGRPO"
    
    def test_standard_normalization(self):
        algo = DrGRPO(DrGRPOConfig(advantage_norm_method="standard"))
        rewards = [0.0, 0.5, 1.0]
        
        advantages = algo.compute_advantages(rewards)
        
        assert len(advantages) == 3
        # Standard normalization: (x - mean) / std
        mean = sum(rewards) / len(rewards)
        assert abs(advantages[1]) < 0.1  # Middle (0.5) should be ~0
    
    def test_winsorized_normalization(self):
        algo = DrGRPO(DrGRPOConfig(
            advantage_norm_method="winsorized",
            winsorize_quantile=0.95
        ))
        # Add some outliers
        rewards = [0.0, 0.5, 1.0, 10.0, -10.0]
        
        advantages = algo.compute_advantages(rewards)
        
        assert len(advantages) == 5
        # Outliers should be clipped
        max_adv = max(advantages.values())
        assert max_adv < 5.0  # Should be winsorized
    
    def test_rank_normalization(self):
        algo = DrGRPO(DrGRPOConfig(advantage_norm_method="rank"))
        rewards = [0.0, 0.5, 1.0, 0.3]
        
        advantages = algo.compute_advantages(rewards)
        
        assert len(advantages) == 4
        # Rank normalization gives equal spacing
        sorted_advs = sorted(advantages.values())
        # Should be roughly equally spaced
        assert sorted_advs[0] < sorted_advs[-1]
    
    def test_asymmetric_clip_config(self):
        config = DrGRPOConfig(
            use_asymmetric_clip=True,
            clip_high=0.3,
            clip_low=0.1
        )
        algo = DrGRPO(config)
        assert algo.config.use_asymmetric_clip is True
        assert algo.config.clip_high == 0.3
        assert algo.config.clip_low == 0.1
    
    def test_kl_estimators(self):
        for estimator in ["schulman", "abs", "mse"]:
            algo = DrGRPO(DrGRPOConfig(kl_estimator=estimator))
            
            new_logprobs = torch.log(torch.tensor([[0.5, 0.5]]))
            ref_logprobs = torch.log(torch.tensor([[0.4, 0.6]]))
            mask = torch.ones_like(new_logprobs, dtype=torch.bool)
            
            kl = algo._compute_unbiased_kl(new_logprobs, ref_logprobs, mask)
            
            assert kl.item() >= 0  # KL should be non-negative
    
    def test_is_levels(self):
        for level in ["token", "sequence", "geometric_mean"]:
            algo = DrGRPO(DrGRPOConfig(is_level=level))
            assert algo.config.is_level == level
    
    def test_remove_outliers(self):
        algo = DrGRPO(DrGRPOConfig(remove_outliers=True, outlier_threshold=2.0))
        rewards = torch.tensor([0.0, 0.1, 0.2, 10.0])  # 10.0 is outlier
        
        cleaned = algo._remove_outliers(rewards)
        
        # Outlier should be clipped, not removed
        assert cleaned.max() < 10.0
    
    def test_diagnostics(self):
        algo = DrGRPO(DrGRPOConfig())
        diagnostics = algo.get_diagnostics()
        
        assert "is_level" in diagnostics
        assert "kl_estimator" in diagnostics
        assert "advantage_norm" in diagnostics


class TestDAPO:
    """Tests for DAPO algorithm."""
    
    def test_creation(self):
        config = DAPOConfig(group_size=8)
        algo = DAPO(config)
        assert algo.name == "DAPO"
    
    def test_population_stats_update(self):
        algo = DAPO(DAPOConfig())
        
        # Simulate multiple groups
        for _ in range(10):
            rewards = torch.tensor([0.0, 0.5, 1.0])
            algo._update_population_stats(rewards)
        
        stats = algo.get_population_stats()
        assert stats["num_groups"] == 10
        assert stats["reward_mean"] > 0  # Should converge towards ~0.5
    
    def test_sparse_reward_detection(self):
        algo = DAPO(DAPOConfig(sparse_reward_threshold=0.5))
        
        # Mostly zeros - should be sparse
        sparse_rewards = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])
        is_sparse = algo._detect_sparse_rewards(sparse_rewards)
        assert is_sparse is True
        
        # Dense rewards
        dense_rewards = torch.tensor([0.2, 0.3, 0.4, 0.5, 0.6])
        is_sparse = algo._detect_sparse_rewards(dense_rewards)
        # May or may not be sparse depending on history
    
    def test_advantage_clipping(self):
        config = DAPOConfig(advantage_clip=2.0)
        algo = DAPO(config)
        
        rewards = [0.0, 10.0, -10.0]  # Very spread out
        advantages = algo.compute_advantages(rewards)
        
        # All advantages should be within clip bounds
        for adv in advantages.values():
            assert -2.0 <= adv <= 2.0
    
    def test_asymmetric_loss_config(self):
        config = DAPOConfig(
            use_asymmetric_loss=True,
            positive_advantage_scale=1.0,
            negative_advantage_scale=0.5,
        )
        algo = DAPO(config)
        assert algo.config.use_asymmetric_loss is True
        assert algo.config.positive_advantage_scale == 1.0
        assert algo.config.negative_advantage_scale == 0.5


class TestGSPO:
    """Tests for GSPO algorithm."""
    
    def test_creation(self):
        config = GSPOConfig(target_group_size=8)
        algo = GSPO(config)
        assert algo.name == "GSPO"
        assert algo.get_current_group_size() == 8
    
    def test_compute_advantages_consensus(self):
        config = GSPOConfig(use_consensus=True, consensus_weight=0.3)
        algo = GSPO(config)
        
        # First group - no consensus yet
        rewards1 = [0.0, 0.5, 1.0]
        adv1 = algo.compute_advantages(rewards1)
        
        # Second group - should use consensus
        rewards2 = [0.2, 0.6, 0.9]
        adv2 = algo.compute_advantages(rewards2)
        
        assert len(adv1) == 3
        assert len(adv2) == 3
    
    def test_group_diversity(self):
        algo = GSPO(GSPOConfig())
        
        # Uniform advantages - low diversity
        uniform = torch.tensor([1.0, 1.0, 1.0, 1.0])
        diversity_uniform = algo._compute_group_diversity(uniform)
        assert diversity_uniform == 0.0
        
        # Diverse advantages - high diversity
        diverse = torch.tensor([-2.0, -1.0, 1.0, 2.0])
        diversity_diverse = algo._compute_group_diversity(diverse)
        assert diversity_diverse > diversity_uniform
    
    def test_dynamic_group_size(self):
        config = GSPOConfig(
            use_dynamic_groups=True,
            target_group_size=8,
            min_group_size=4,
            max_group_size=16,
        )
        algo = GSPO(config)
        
        initial_size = algo.get_current_group_size()
        
        # Low variance should decrease group size
        for _ in range(5):
            rewards = torch.tensor([0.48, 0.49, 0.50, 0.51, 0.52])
            algo.compute_advantages(rewards.tolist())
        
        # High variance should increase group size
        for _ in range(5):
            rewards = torch.tensor([0.0, 0.2, 0.5, 0.8, 1.0])
            algo.compute_advantages(rewards.tolist())
        
        final_size = algo.get_current_group_size()
        assert 4 <= final_size <= 16
    
    def test_compose_groups_random(self):
        algo = GSPO(GSPOConfig(composition_strategy="random"))
        
        class MockTrajectory:
            def __init__(self, reward):
                self.reward = reward
        
        trajectories = [MockTrajectory(i * 0.1) for i in range(10)]
        groups = algo.compose_groups(trajectories, num_groups=2)
        
        assert len(groups) == 2
        assert sum(len(g) for g in groups) == 10
    
    def test_compose_groups_diverse(self):
        algo = GSPO(GSPOConfig(composition_strategy="diverse"))
        
        class MockTrajectory:
            def __init__(self, reward):
                self.reward = reward
        
        trajectories = [MockTrajectory(i * 0.1) for i in range(10)]
        groups = algo.compose_groups(trajectories, num_groups=2)
        
        assert len(groups) == 2
        # Each group should have mixed high/low rewards


class TestAlgorithmFactory:
    """Tests for algorithm factory."""
    
    def test_create_grpo(self):
        algo = create_algorithm("grpo", group_size=8)
        assert isinstance(algo, GRPO)
    
    def test_create_dgpo(self):
        algo = create_algorithm("dgpo", group_size=8)
        assert isinstance(algo, DGPO)
    
    def test_create_drgrpo(self):
        algo = create_algorithm("drgrpo", group_size=8)
        assert isinstance(algo, DrGRPO)
    
    def test_create_dapo(self):
        algo = create_algorithm("dapo", group_size=8)
        assert isinstance(algo, DAPO)
    
    def test_create_gspo(self):
        algo = create_algorithm("gspo", target_group_size=8)
        assert isinstance(algo, GSPO)
    
    def test_create_unknown(self):
        with pytest.raises(ValueError, match="Unknown algorithm"):
            create_algorithm("unknown")
    
    def test_create_case_insensitive(self):
        algo = create_algorithm("GRPO")
        assert isinstance(algo, GRPO)
        
        algo = create_algorithm("DgPo")
        assert isinstance(algo, DGPO)
        
        algo = create_algorithm("DrGrPo")
        assert isinstance(algo, DrGRPO)


class TestAlgorithmConfig:
    """Tests for algorithm configurations."""
    
    def test_rl_config_defaults(self):
        from lean_grpo.algorithms.base import RLConfig
        config = RLConfig()
        assert config.learning_rate == 5e-6
        assert config.epsilon == 0.2
        assert config.beta == 0.0
    
    def test_grpo_config_group_size(self):
        config = GRPOConfig(group_size=16)
        assert config.group_size == 16
    
    def test_dgpo_config(self):
        config = DGPOConfig(use_dpo_loss=True, dpo_coef=0.5)
        assert config.use_dpo_loss is True
        assert config.dpo_coef == 0.5
    
    def test_drgrpo_config(self):
        config = DrGRPOConfig(
            is_level="sequence",
            kl_estimator="schulman",
            use_unbiased_kl=True
        )
        assert config.is_level == "sequence"
        assert config.kl_estimator == "schulman"
        assert config.use_unbiased_kl is True
    
    def test_dapo_config_population(self):
        config = DAPOConfig(population_size=200)
        assert config.population_size == 200
    
    def test_gspo_config_sync(self):
        config = GSPOConfig(sync_frequency=5)
        assert config.sync_frequency == 5
