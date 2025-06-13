# tests/unit_tests/test_reward_system.py
import unittest
import logging
from typing import Dict, Any

# Ensure the logger is configured to avoid NoHandlerFoundError
# You might want to set up a more sophisticated logging configuration for tests
logging.basicConfig(level=logging.DEBUG)

# Attempt to import from the correct location
# This assumes your project structure allows this import path
# If 'src' is not in PYTHONPATH for tests, this might fail.
# Consider using conftest.py or adjusting PYTHONPATH.
try:
    from src.environment.progressive_reward_system import SimpleReward, IntermediateReward, ComplexReward, ProgressiveLearningSystem, BaseRewardStrategy
except ImportError:
    # Fallback for environments where src is not directly in path, common in some test setups
    # This is a simplified approach; a robust solution involves proper path management (e.g., conftest.py)
    import sys
    import os
    # Adjust path to go up one level from 'tests/unit_tests' to project root, then into 'src'
    # This specific path adjustment depends on where your tests are run from.
    # If tests are run from project root, 'src.xxx' should work if 'src' is a package.
    # If run from 'tests/unit_tests', you need to go up to 'Oanda_Trading_Bot/' then 'src/'
    # Assuming tests are run from the project root (e.g., 'pytest' from Oanda_Trading_Bot/)
    # and conftest.py handles adding 'src' to sys.path, this fallback might not be strictly necessary
    # but is included for robustness in different execution contexts.
    # For now, we rely on conftest.py to have added 'src' to the path.
    # If that's not the case, the initial import will fail, and this message will be less relevant.
    print("Failed to import reward system components directly. Ensure 'src' is in PYTHONPATH or use conftest.py.")
    # Re-raise the error if the import truly fails, as the tests cannot run.
    raise

class TestRewardSystemBase(unittest.TestCase):
    def test_module_importable(self):
        """Test that the progressive_reward_system module and its components can be imported."""
        # The try/except block for imports at the top of the file already serves as a basic import test.
        # If we reach this point, the necessary classes were imported.
        self.assertTrue(True, "Module and components imported successfully.")

    def test_base_reward_strategy_instantiation(self):
        """Test that BaseRewardStrategy can be (conceptually) instantiated by its children."""
        # BaseRewardStrategy is abstract, so we test instantiation through a concrete child.
        class DummyReward(BaseRewardStrategy):
            def calculate_reward(self, market_data: Any, portfolio_context: Any, trade_action: Any) -> float:
                return 1.0
        
        strategy = DummyReward(config={})
        self.assertIsInstance(strategy, BaseRewardStrategy)
        self.assertIsInstance(strategy, DummyReward)

    def test_simple_reward_instantiation(self):
        """Test that SimpleReward can be instantiated."""
        strategy = SimpleReward(config={})
        self.assertIsInstance(strategy, SimpleReward)

    def test_intermediate_reward_instantiation(self):
        """Test that IntermediateReward can be instantiated."""
        strategy = IntermediateReward(config={})
        self.assertIsInstance(strategy, IntermediateReward)

    def test_complex_reward_instantiation(self):
        """Test that ComplexReward can be instantiated."""
        strategy = ComplexReward(config={})
        self.assertIsInstance(strategy, ComplexReward)

    def test_progressive_learning_system_instantiation_minimal(self):
        """Test ProgressiveLearningSystem instantiation with minimal valid config."""
        # Minimal config for ProgressiveLearningSystem to initialize
        # It needs at least one stage defined, typically stage 1.
        minimal_stage_configs = {
            1: {
                'reward_strategy_class': SimpleReward,
                'reward_config': {},
                'stage_criteria': lambda metrics: True # Simplest criteria
            }
        }
        system = ProgressiveLearningSystem(stage_configs=minimal_stage_configs)
        self.assertIsInstance(system, ProgressiveLearningSystem)
        self.assertEqual(system.current_stage_number, 1) # Changed current_stage to current_stage_number
        self.assertIsInstance(system.get_current_reward_function(), SimpleReward)

    def test_progressive_learning_system_instantiation_empty_config(self):
        """Test ProgressiveLearningSystem instantiation with an empty stages_config (should raise ValueError)."""
        # The system is designed to raise ValueError if stage_configs is empty.
        with self.assertRaises(ValueError) as context:
            ProgressiveLearningSystem(stage_configs={}, initial_stage=1)
        self.assertIn("Stage configurations cannot be empty", str(context.exception))


class TestSimpleReward(unittest.TestCase):
    def test_instantiation_default_weights(self):
        """Test SimpleReward instantiation with default weights."""
        reward_strategy = SimpleReward(config={})
        self.assertEqual(reward_strategy.profit_weight, 0.8) # Changed profit_loss_weight to profit_weight
        self.assertEqual(reward_strategy.risk_penalty_weight, 0.2)
        self.assertIsInstance(reward_strategy, BaseRewardStrategy)

    def test_instantiation_custom_weights(self):
        """Test SimpleReward instantiation with custom weights."""
        custom_config = {'profit_weight': 0.7, 'risk_penalty_weight': 0.3} # Changed profit_loss_weight to profit_weight
        reward_strategy = SimpleReward(config=custom_config)
        self.assertEqual(reward_strategy.profit_weight, 0.7) # Changed profit_loss_weight to profit_weight
        self.assertEqual(reward_strategy.risk_penalty_weight, 0.3)

    def test_calculate_reward_positive_pnl_zero_risk(self):
        """Test reward calculation with positive PnL and zero risk."""
        reward_strategy = SimpleReward(config={})
        # Map old portfolio_context keys to new trade_info keys
        trade_info = {'realized_pnl': 100, 'drawdown': 0} 
        expected_reward = (100 * 0.8) - (0 * 0.2)
        # Corrected arguments for calculate_reward
        self.assertEqual(reward_strategy.calculate_reward(trade_info), expected_reward)

    def test_calculate_reward_positive_pnl_with_risk(self):
        """Test reward calculation with positive PnL and some risk."""
        reward_strategy = SimpleReward(config={})
        trade_info = {'realized_pnl': 100, 'drawdown': 50}
        expected_reward = (100 * 0.8) - (50 * 0.2)
        self.assertEqual(reward_strategy.calculate_reward(trade_info), expected_reward)

    def test_calculate_reward_negative_pnl_zero_risk(self):
        """Test reward calculation with negative PnL and zero risk."""
        reward_strategy = SimpleReward(config={})
        trade_info = {'realized_pnl': -100, 'drawdown': 0}
        expected_reward = (-100 * 0.8) - (0 * 0.2)
        self.assertEqual(reward_strategy.calculate_reward(trade_info), expected_reward)

    def test_calculate_reward_negative_pnl_with_risk(self):
        """Test reward calculation with negative PnL and some risk."""
        reward_strategy = SimpleReward(config={})
        trade_info = {'realized_pnl': -100, 'drawdown': 50}
        expected_reward = (-100 * 0.8) - (50 * 0.2)
        self.assertEqual(reward_strategy.calculate_reward(trade_info), expected_reward)

    def test_calculate_reward_zero_pnl_with_risk(self):
        """Test reward calculation with zero PnL and some risk."""
        reward_strategy = SimpleReward(config={})
        trade_info = {'realized_pnl': 0, 'drawdown': 50}
        expected_reward = (0 * 0.8) - (50 * 0.2)
        self.assertEqual(reward_strategy.calculate_reward(trade_info), expected_reward)

    def test_calculate_reward_custom_weights(self):
        """Test reward calculation with custom weights."""
        custom_config = {'profit_weight': 0.6, 'risk_penalty_weight': 0.4} # Changed profit_loss_weight
        reward_strategy = SimpleReward(config=custom_config)
        trade_info = {'realized_pnl': 100, 'drawdown': 50}
        expected_reward = (100 * 0.6) - (50 * 0.4)
        self.assertEqual(reward_strategy.calculate_reward(trade_info), expected_reward)

    def test_calculate_reward_missing_keys_in_context(self):
        """Test reward calculation when PnL or risk keys are missing from context (should use defaults)."""
        reward_strategy = SimpleReward(config={})
        
        # Missing 'realized_pnl' and 'drawdown'
        trade_info_missing_all = {} 
        expected_reward_missing_all = (0 * 0.8) - (0 * 0.2) # Defaults to 0 for both
        self.assertEqual(reward_strategy.calculate_reward(trade_info_missing_all), expected_reward_missing_all)

        # Missing 'drawdown'
        trade_info_missing_risk = {'realized_pnl': 100}
        expected_reward_missing_risk = (100 * 0.8) - (0 * 0.2)
        self.assertEqual(reward_strategy.calculate_reward(trade_info_missing_risk), expected_reward_missing_risk)

        # Missing 'realized_pnl'
        trade_info_missing_pnl = {'drawdown': 50}
        expected_reward_missing_pnl = (0 * 0.8) - (50 * 0.2)
        self.assertEqual(reward_strategy.calculate_reward(trade_info_missing_pnl), expected_reward_missing_pnl)

    def test_calculate_reward_custom_risk_metric(self):
        """Test reward calculation with a custom risk metric like 'max_drawdown'."""
        custom_config = {'risk_metric': 'max_drawdown', 'profit_weight': 0.8, 'risk_penalty_weight': 0.2}
        reward_strategy = SimpleReward(config=custom_config)
        trade_info = {'realized_pnl': 100, 'max_drawdown': 70, 'drawdown': 30} # Provide both, ensure max_drawdown is used
        
        # Reward should be based on max_drawdown (70), not drawdown (30)
        expected_reward = (100 * 0.8) - (abs(70) * 0.2) 
        self.assertEqual(reward_strategy.calculate_reward(trade_info), expected_reward)

class TestIntermediateReward(unittest.TestCase):
    def test_instantiation_default_weights(self):
        """Test IntermediateReward instantiation with default weights."""
        reward_strategy = IntermediateReward(config={})
        self.assertEqual(reward_strategy.sharpe_weight, 0.5)
        self.assertEqual(reward_strategy.pnl_weight, 0.3)
        self.assertEqual(reward_strategy.drawdown_penalty_weight, 0.15)
        self.assertEqual(reward_strategy.cost_penalty_weight, 0.05)
        self.assertIsInstance(reward_strategy, BaseRewardStrategy)

    def test_instantiation_custom_weights(self):
        """Test IntermediateReward instantiation with custom weights."""
        custom_config = {
            'sharpe_weight': 0.4,
            'pnl_weight': 0.25,
            'drawdown_penalty_weight': 0.2,
            'cost_penalty_weight': 0.15
        }
        reward_strategy = IntermediateReward(config=custom_config)
        self.assertEqual(reward_strategy.sharpe_weight, 0.4)
        self.assertEqual(reward_strategy.pnl_weight, 0.25)
        self.assertEqual(reward_strategy.drawdown_penalty_weight, 0.2)
        self.assertEqual(reward_strategy.cost_penalty_weight, 0.15)

    def test_calculate_reward_all_positive_inputs(self):
        """Test reward calculation with all positive inputs (or typical values)."""
        reward_strategy = IntermediateReward(config={})
        trade_info = {
            'sharpe_ratio': 1.5,
            'realized_pnl': 200,
            'drawdown': 50,  # Drawdown is a positive value representing magnitude
            'trade_cost': 10
        }
        expected_reward = (1.5 * 0.5) + (200 * 0.3) - (abs(50) * 0.15) - (abs(10) * 0.05)
        # 0.75 + 60 - 7.5 - 0.5 = 52.75
        self.assertAlmostEqual(reward_strategy.calculate_reward(trade_info), expected_reward, places=5)

    def test_calculate_reward_mixed_inputs(self):
        """Test reward calculation with mixed (positive/negative) inputs."""
        reward_strategy = IntermediateReward(config={})
        trade_info = {
            'sharpe_ratio': -0.5,  # Negative Sharpe
            'realized_pnl': -100, # Negative PnL
            'drawdown': 70,
            'trade_cost': 5
        }
        expected_reward = (-0.5 * 0.5) + (-100 * 0.3) - (abs(70) * 0.15) - (abs(5) * 0.05)
        self.assertAlmostEqual(reward_strategy.calculate_reward(trade_info), expected_reward, places=5)

    def test_calculate_reward_zero_values(self):
        """Test reward calculation when all relevant inputs are zero."""
        reward_strategy = IntermediateReward(config={})
        trade_info = {
            'sharpe_ratio': 0,
            'realized_pnl': 0,
            'drawdown': 0,
            'trade_cost': 0
        }
        expected_reward = (0 * 0.5) + (0 * 0.3) - (0 * 0.15) - (0 * 0.05)
        # 0 + 0 - 0 - 0 = 0
        self.assertAlmostEqual(reward_strategy.calculate_reward(trade_info), expected_reward, places=5)

    def test_calculate_reward_custom_weights(self):
        """Test reward calculation with custom weights."""
        custom_config = {
            'sharpe_weight': 0.6,
            'pnl_weight': 0.2,
            'drawdown_penalty_weight': 0.1,
            'cost_penalty_weight': 0.1
        }
        reward_strategy = IntermediateReward(config=custom_config)
        trade_info = {
            'sharpe_ratio': 1.2,
            'realized_pnl': 250,
            'drawdown': 80,
            'trade_cost': 15
        }
        expected_reward = (1.2 * 0.6) + (250 * 0.2) - (abs(80) * 0.1) - (abs(15) * 0.1)
        # 0.72 + 50 - 8 - 1.5 = 41.22
        self.assertAlmostEqual(reward_strategy.calculate_reward(trade_info), expected_reward, places=5)

    def test_calculate_reward_missing_keys_in_trade_info(self):
        """Test reward calculation when keys are missing from trade_info (should use defaults of 0.0)."""
        reward_strategy = IntermediateReward(config={})
        trade_info_empty = {}
        # All get('key', 0.0) will result in 0 for all components
        expected_reward_empty = (0 * 0.5) + (0 * 0.3) - (0 * 0.15) - (0 * 0.05) # = 0
        self.assertAlmostEqual(reward_strategy.calculate_reward(trade_info_empty), expected_reward_empty, places=5)

        trade_info_partial = {'realized_pnl': 100, 'trade_cost': 5}
        # sharpe_ratio defaults to 0, drawdown defaults to 0
        expected_reward_partial = (0 * 0.5) + (100 * 0.3) - (0 * 0.15) - (abs(5) * 0.05)
        # 0 + 30 - 0 - 0.25 = 29.75
        self.assertAlmostEqual(reward_strategy.calculate_reward(trade_info_partial), expected_reward_partial, places=5)

class TestComplexReward(unittest.TestCase):
    def test_instantiation_default_weights(self):
        """Test ComplexReward instantiation with default weights."""
        reward_strategy = ComplexReward(config={})
        self.assertEqual(reward_strategy.sortino_weight, 0.3)
        self.assertEqual(reward_strategy.profit_factor_weight, 0.2)
        self.assertEqual(reward_strategy.win_rate_weight, 0.1)
        self.assertEqual(reward_strategy.market_adaptability_weight, 0.2)
        self.assertEqual(reward_strategy.consistency_weight, 0.1)
        self.assertEqual(reward_strategy.max_drawdown_penalty_weight, 0.1)
        self.assertIsInstance(reward_strategy, BaseRewardStrategy)

    def test_instantiation_custom_weights(self):
        """Test ComplexReward instantiation with custom weights."""
        custom_config = {
            'sortino_weight': 0.25,
            'profit_factor_weight': 0.15,
            'win_rate_weight': 0.15,
            'market_adaptability_weight': 0.25,
            'consistency_weight': 0.1,
            'max_drawdown_penalty_weight': 0.1
        }
        reward_strategy = ComplexReward(config=custom_config)
        self.assertEqual(reward_strategy.sortino_weight, 0.25)
        self.assertEqual(reward_strategy.profit_factor_weight, 0.15)
        self.assertEqual(reward_strategy.win_rate_weight, 0.15)
        self.assertEqual(reward_strategy.market_adaptability_weight, 0.25)
        self.assertEqual(reward_strategy.consistency_weight, 0.1)
        self.assertEqual(reward_strategy.max_drawdown_penalty_weight, 0.1)

    def test_calculate_reward_positive_scenario(self):
        """Test ComplexReward calculation with a generally positive scenario."""
        reward_strategy = ComplexReward(config={})
        trade_info = {
            'sortino_ratio': 2.0,
            'profit_factor': 3.0, # Total P / Total L
            'win_rate': 0.7,    # 70% win rate
            'max_drawdown': 0.1, # 10% max drawdown
            'market_adaptability_score': 0.8, # External score
            'behavioral_consistency_score': 0.75 # External score
        }
        expected_reward = (
            2.0 * 0.3 +              # sortino
            (3.0 - 1) * 0.2 +        # profit_factor (value is profit_factor - 1)
            (0.7 - 0.5) * 0.1 +      # win_rate (value is win_rate - 0.5)
            0.8 * 0.2 +              # market_adaptability
            0.75 * 0.1 -             # consistency
            abs(0.1) * 0.1           # max_drawdown penalty
        )
        self.assertAlmostEqual(reward_strategy.calculate_reward(trade_info), expected_reward)

    def test_calculate_reward_mixed_scenario(self):
        """Test ComplexReward calculation with mixed performance metrics."""
        reward_strategy = ComplexReward(config={})
        trade_info = {
            'sortino_ratio': -0.5, # Poor Sortino
            'profit_factor': 0.8,  # Losing more than winning
            'win_rate': 0.4,     # Low win rate
            'max_drawdown': 0.25, # High max drawdown
            'market_adaptability_score': 0.3,
            'behavioral_consistency_score': 0.4
        }
        expected_reward = (
            -0.5 * 0.3 +             # sortino
            (0.8 - 1) * 0.2 +        # profit_factor
            (0.4 - 0.5) * 0.1 +      # win_rate
            0.3 * 0.2 +              # market_adaptability
            0.4 * 0.1 -              # consistency
            abs(0.25) * 0.1          # max_drawdown penalty
        )
        self.assertAlmostEqual(reward_strategy.calculate_reward(trade_info), expected_reward)

    def test_calculate_reward_missing_keys(self):
        """Test ComplexReward calculation with missing keys (should use defaults)."""
        reward_strategy = ComplexReward(config={})
        trade_info_missing = {
            'sortino_ratio': 1.0
            # Other keys are missing
        }
        # Defaults: profit_factor=1.0, win_rate=0.5, max_drawdown=0.0
        # market_adaptability_score=0.0, behavioral_consistency_score=0.0
        expected_reward = (
            1.0 * 0.3 +              # sortino
            (1.0 - 1) * 0.2 +        # profit_factor (default 1.0)
            (0.5 - 0.5) * 0.1 +      # win_rate (default 0.5)
            0.0 * 0.2 +              # market_adaptability (default 0.0)
            0.0 * 0.1 -              # consistency (default 0.0)
            abs(0.0) * 0.1           # max_drawdown penalty (default 0.0)
        )
        self.assertAlmostEqual(reward_strategy.calculate_reward(trade_info_missing), expected_reward)

    def test_calculate_reward_custom_weights_scenario(self):
        """Test ComplexReward calculation with custom weights."""
        custom_config = {
            'sortino_weight': 0.4,
            'profit_factor_weight': 0.1,
            'win_rate_weight': 0.2,
            'market_adaptability_weight': 0.1,
            'consistency_weight': 0.1,
            'max_drawdown_penalty_weight': 0.1
        }
        reward_strategy = ComplexReward(config=custom_config)
        trade_info = {
            'sortino_ratio': 1.5,
            'profit_factor': 2.5,
            'win_rate': 0.6,
            'max_drawdown': 0.15,
            'market_adaptability_score': 0.9,
            'behavioral_consistency_score': 0.8
        }
        expected_reward = (
            1.5 * 0.4 +              # sortino
            (2.5 - 1) * 0.1 +        # profit_factor
            (0.6 - 0.5) * 0.2 +      # win_rate
            0.9 * 0.1 +              # market_adaptability
            0.8 * 0.1 -              # consistency
            abs(0.15) * 0.1          # max_drawdown penalty
        )
        self.assertAlmostEqual(reward_strategy.calculate_reward(trade_info), expected_reward)


class TestProgressiveLearningSystem(unittest.TestCase):
    def setUp(self):
        """Set up common configurations for testing ProgressiveLearningSystem."""
        self.mock_trade_info_simple = {
            'realized_pnl': 100,
            'drawdown': 10
        }
        self.mock_metrics_stage1_pass = {'cumulative_reward': 600, 'episodes': 50} # Ensure cumulative_reward > 500
        self.mock_metrics_stage1_fail = {'cumulative_reward': 100, 'episodes': 50}

        self.mock_trade_info_intermediate = {
            'sharpe_ratio': 1.5,
            'realized_pnl': 200,
            'drawdown': 50,
            'trade_cost': 10
        }
        self.mock_metrics_stage2_pass = {'sharpe_ratio': 0.9, 'episodes': 100} # Ensure sharpe_ratio > 0.8
        self.mock_metrics_stage2_fail = {'sharpe_ratio': 0.5, 'episodes': 100}

        self.mock_trade_info_complex = {
            'sortino_ratio': 2.0,
            'profit_factor': 3.0,
            'win_rate': 0.7,
            'max_drawdown': 0.1,
            'market_adaptability_score': 0.8,
            'behavioral_consistency_score': 0.75
        }
        # Metrics for advancing to a hypothetical Stage 4 (not used for advancement from Stage 3 in current config)
        self.mock_metrics_stage3_pass = {'alpha_vs_benchmark': 0.05, 'episodes': 150}

        self.stage_configs = {
            1: {
                'reward_strategy_class': SimpleReward,
                'reward_config': {'profit_weight': 0.8, 'risk_penalty_weight': 0.2},
                'criteria_to_advance': lambda metrics: metrics.get('cumulative_reward', 0) > 500 and metrics.get('episodes', 0) >= 50,
                'max_episodes_or_steps': 100
            },
            2: {
                'reward_strategy_class': IntermediateReward,
                'reward_config': {'sharpe_weight': 0.5, 'pnl_weight': 0.3, 'drawdown_penalty_weight': 0.15, 'cost_penalty_weight': 0.05},
                'criteria_to_advance': lambda metrics: metrics.get('sharpe_ratio', 0) > 0.8 and metrics.get('episodes', 0) >= 100,
                'max_episodes_or_steps': 200
            },
            3: { 
                'reward_strategy_class': ComplexReward, 
                'reward_config': { # More complete config for ComplexReward for testing
                    'sortino_weight': 0.3, 'profit_factor_weight': 0.2, 'win_rate_weight': 0.1,
                    'market_adaptability_weight': 0.15, 'consistency_weight': 0.1, 'max_drawdown_penalty_weight': 0.15
                },
                'criteria_to_advance': None, # No auto advancement from stage 3 in this test setup
                'max_episodes_or_steps': 50
            }
        }

    def test_initialization_stage1(self):
        """Test system initializes to stage 1 correctly."""
        system = ProgressiveLearningSystem(stage_configs=self.stage_configs, initial_stage=1)
        self.assertEqual(system.current_stage_number, 1)
        self.assertIsInstance(system.get_current_reward_function(), SimpleReward)
        current_reward_func = system.get_current_reward_function()
        self.assertEqual(current_reward_func.profit_weight, 0.8)

    def test_get_reward_stage1(self):
        """Test reward calculation at stage 1."""
        system = ProgressiveLearningSystem(stage_configs=self.stage_configs, initial_stage=1)
        expected_reward_stage1 = (100 * 0.8) - (10 * 0.2) # From self.mock_trade_info_simple and stage 1 config
        self.assertAlmostEqual(system.calculate_reward(self.mock_trade_info_simple), expected_reward_stage1) # Changed get_reward to calculate_reward

    def test_advance_stage_manually(self):
        """Test manual advancement to stage 2."""
        system = ProgressiveLearningSystem(stage_configs=self.stage_configs, initial_stage=1)
        system.advance_stage_manually() # Changed advance_stage to advance_stage_manually
        self.assertEqual(system.current_stage_number, 2)
        self.assertIsInstance(system.get_current_reward_function(), IntermediateReward)
        current_reward_func = system.get_current_reward_function()
        self.assertEqual(current_reward_func.sharpe_weight, 0.5)

    def test_get_reward_stage2_after_manual_advance(self):
        """Test reward calculation at stage 2 after manual advancement."""
        system = ProgressiveLearningSystem(stage_configs=self.stage_configs, initial_stage=1)
        system.advance_stage_manually() # Changed advance_stage to advance_stage_manually
        self.assertEqual(system.current_stage_number, 2) # Ensure we are in stage 2
        expected_reward_stage2 = (1.5 * 0.5) + (200 * 0.3) - (abs(50) * 0.15) - (abs(10) * 0.05) # From mock_trade_info_intermediate
        self.assertAlmostEqual(system.calculate_reward(self.mock_trade_info_intermediate), expected_reward_stage2, places=5) # Changed get_reward to calculate_reward

    def test_check_and_advance_stage1_pass(self):
        """Test automatic advancement from stage 1 when criteria are met."""
        system = ProgressiveLearningSystem(stage_configs=self.stage_configs, initial_stage=1)
        system.check_and_advance_stage(self.mock_metrics_stage1_pass) # Changed check_and_advance to check_and_advance_stage
        self.assertEqual(system.current_stage_number, 2)
        self.assertIsInstance(system.get_current_reward_function(), IntermediateReward)

    def test_check_and_advance_stage1_fail(self):
        """Test no advancement from stage 1 when criteria are not met."""
        system = ProgressiveLearningSystem(stage_configs=self.stage_configs, initial_stage=1)
        system.check_and_advance_stage(self.mock_metrics_stage1_fail) # Changed check_and_advance to check_and_advance_stage
        self.assertEqual(system.current_stage_number, 1)
        self.assertIsInstance(system.get_current_reward_function(), SimpleReward)

    def test_get_reward_stage2_after_auto_advance(self):
        """Test reward calculation at stage 2 after automatic advancement."""
        system = ProgressiveLearningSystem(stage_configs=self.stage_configs, initial_stage=1)
        system.check_and_advance_stage(self.mock_metrics_stage1_pass) # Advance to Stage 2
        self.assertEqual(system.current_stage_number, 2)
        expected_reward_stage2 = (1.5 * 0.5) + (200 * 0.3) - (abs(50) * 0.15) - (abs(10) * 0.05)
        self.assertAlmostEqual(system.calculate_reward(self.mock_trade_info_intermediate), expected_reward_stage2, places=5) # Changed get_reward to calculate_reward

    def test_check_and_advance_stage2_pass_to_stage3(self):
        """Test automatic advancement from stage 2 to stage 3 when criteria are met."""
        system = ProgressiveLearningSystem(stage_configs=self.stage_configs, initial_stage=2) # Start at stage 2
        self.assertIsInstance(system.get_current_reward_function(), IntermediateReward)
        system.check_and_advance_stage(self.mock_metrics_stage2_pass) # Use metrics defined to pass stage 2
        self.assertEqual(system.current_stage_number, 3)
        self.assertIsInstance(system.get_current_reward_function(), ComplexReward)

    def test_check_and_advance_stage2_fail(self):
        """Test no advancement from stage 2 when criteria for stage 3 are not met."""
        system = ProgressiveLearningSystem(stage_configs=self.stage_configs, initial_stage=2) # Start at stage 2
        system.check_and_advance_stage(self.mock_metrics_stage2_fail) # Use metrics defined to fail stage 2 advancement
        self.assertEqual(system.current_stage_number, 2)
        self.assertIsInstance(system.get_current_reward_function(), IntermediateReward)

    def test_get_reward_stage3_after_advancing_from_stage2(self):
        """Test reward calculation at stage 3 after advancing from stage 2."""
        system = ProgressiveLearningSystem(stage_configs=self.stage_configs, initial_stage=2)
        system.check_and_advance_stage(self.mock_metrics_stage2_pass) # Advance to Stage 3
        self.assertEqual(system.current_stage_number, 3)
        self.assertIsInstance(system.get_current_reward_function(), ComplexReward)
        
        complex_reward_strategy = system.get_current_reward_function()
        cfg = complex_reward_strategy.config # This will be self.stage_configs[3][\'reward_config\']
        
        expected_reward_stage3 = (
            self.mock_trade_info_complex['sortino_ratio'] * cfg['sortino_weight'] +
            (self.mock_trade_info_complex['profit_factor'] - 1) * cfg['profit_factor_weight'] +
            (self.mock_trade_info_complex['win_rate'] - 0.5) * cfg['win_rate_weight'] +
            self.mock_trade_info_complex['market_adaptability_score'] * cfg['market_adaptability_weight'] +
            self.mock_trade_info_complex['behavioral_consistency_score'] * cfg['consistency_weight'] -
            abs(self.mock_trade_info_complex['max_drawdown']) * cfg['max_drawdown_penalty_weight']
        )
        self.assertAlmostEqual(system.calculate_reward(self.mock_trade_info_complex), expected_reward_stage3, places=5)

    def test_advance_stage_to_max_stops(self):
        """Test that advancing beyond the highest configured stage does not change stage."""
        system = ProgressiveLearningSystem(stage_configs=self.stage_configs, initial_stage=3) # Start at stage 3
        self.assertEqual(system.current_stage_number, 3)
        self.assertIsInstance(system.get_current_reward_function(), ComplexReward)
        system.advance_stage_manually() # Changed advance_stage to advance_stage_manually
        self.assertEqual(system.current_stage_number, 3) # Should remain at stage 3
        self.assertIsInstance(system.get_current_reward_function(), ComplexReward)

    def test_check_and_advance_at_max_stage(self):
        """Test check_and_advance at the highest stage with criteria (should not advance)."""
        # Modify stage 3 to have criteria for this test, though it won't advance further
        self.stage_configs[3]['criteria_to_advance'] = lambda metrics: True
        system = ProgressiveLearningSystem(stage_configs=self.stage_configs, initial_stage=3)
        system.check_and_advance_stage({'some_metric': 100}) # Changed check_and_advance to check_and_advance_stage
        self.assertEqual(system.current_stage_number, 3) # Still stage 3
        self.assertIsInstance(system.get_current_reward_function(), ComplexReward)
        # Reset criteria for other tests if setUp is not run per method by test runner (unittest default is per method)
        self.stage_configs[3]['criteria_to_advance'] = None 

    def test_invalid_initial_stage(self):
        """Test system behavior with an invalid initial stage (falls back to min stage)."""
        # ProgressiveLearningSystem is designed to fallback to the minimum stage if initial_stage is invalid
        system = ProgressiveLearningSystem(stage_configs=self.stage_configs, initial_stage=99) 
        self.assertEqual(system.current_stage_number, 1) # Falls back to stage 1 (min defined stage)
        self.assertIsInstance(system.get_current_reward_function(), SimpleReward)

    def test_stage_config_missing_reward_class(self):
        """Test error handling if a stage config is missing reward_strategy_class."""
        faulty_configs = {
            1: {
                # Missing 'reward_strategy_class'
                'reward_config': {},
                'criteria_to_advance': lambda m: True
            }
        }
        with self.assertRaises(ValueError) as context: # Expecting ValueError from _setup_current_stage
            ProgressiveLearningSystem(stage_configs=faulty_configs, initial_stage=1)
        self.assertIn("Invalid reward_strategy_class for stage 1: None", str(context.exception)) # Corrected assertion message

    def test_update_episode_step_counters(self):
        """Test that episode and step counters are updated and reset on stage advance."""
        system = ProgressiveLearningSystem(stage_configs=self.stage_configs, initial_stage=1)
        system.record_episode_end()
        system.record_step()
        system.record_step()
        self.assertEqual(system.episode_in_current_stage, 1) # Changed episodes_in_current_stage to episode_in_current_stage
        self.assertEqual(system.steps_in_current_stage, 2)

        # Advance stage and check if counters reset
        system.advance_stage_manually() # Changed advance_stage to advance_stage_manually
        self.assertEqual(system.current_stage_number, 2)
        self.assertEqual(system.episode_in_current_stage, 0) # Changed episodes_in_current_stage to episode_in_current_stage
        self.assertEqual(system.steps_in_current_stage, 0)

        system.record_episode_end()
        system.record_step()
        self.assertEqual(system.episode_in_current_stage, 1) # Changed episodes_in_current_stage to episode_in_current_stage
        self.assertEqual(system.steps_in_current_stage, 1)

if __name__ == '__main__':
    unittest.main()
