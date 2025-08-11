import pytest
import asyncio
import shutil
import time
import json
import os # Added import
import pandas as pd # Added import
import numpy as np # Added import
from pathlib import Path # Added import
from decimal import Decimal # Added import
from unittest.mock import patch, MagicMock, AsyncMock, PropertyMock
from datetime import datetime, timedelta, timezone # Added timezone

# --- Project-specific imports ---
# Trainer and core components
from src.trainer.universal_trainer import UniversalTrainer
from src.agent.sac_agent_wrapper import QuantumEnhancedSAC
from src.environment.trading_env import UniversalTradingEnvV4
from src.data_manager.mmap_dataset import UniversalMemoryMappedDataset
from src.data_manager.instrument_info_manager import InstrumentInfoManager
from src.data_manager.currency_manager import CurrencyDependencyManager, ensure_currency_data_for_trading
from src.common.shared_data_manager import get_shared_data_manager
from src.trainer.callbacks import UniversalCheckpointCallback

# Agent and model components
from src.agent.transformer_feature_extractor import TransformerFeatureExtractor
from src.models.transformer_model import UniversalTradingTransformer # Corrected import path
from src.agent.enhanced_quantum_strategy_layer import EnhancedStrategySuperposition

# Market Analysis
from src.market_analysis.market_regime_identifier import MarketRegimeIdentifier, MacroRegime

# Configuration constants
from src.common.config import (
    BASE_DIR, WEIGHTS_DIR, LOGS_DIR, MMAP_DATA_DIR as MMAP_UNIVERSAL_DIR,
    DEFAULT_SYMBOLS, GRANULARITY, MAX_SYMBOLS_ALLOWED, PRICE_COLUMNS, PRICE_TYPES,
    TIMESTEPS, TRANSFORMER_MODEL_DIM, TRANSFORMER_NUM_LAYERS, TRANSFORMER_NUM_HEADS,
    TRANSFORMER_FFN_DIM, TRANSFORMER_DROPOUT_RATE, TRANSFORMER_LAYER_NORM_EPS,
    TRANSFORMER_MAX_SEQ_LEN_POS_ENCODING, TRANSFORMER_OUTPUT_DIM_PER_SYMBOL,
    TRAINER_DEFAULT_TOTAL_TIMESTEPS, TRAINER_MODEL_NAME_PREFIX,
    TRAINER_SAVE_FREQ_STEPS, TRAINER_EVAL_FREQ_STEPS, TRAINER_N_EVAL_EPISODES,
    TRAINER_DETERMINISTIC_EVAL,
    DEFAULT_TRAIN_START_ISO, DEFAULT_TRAIN_END_ISO,
    DEFAULT_EVAL_START_ISO, DEFAULT_EVAL_END_ISO,
    SAC_GAMMA, SAC_LEARNING_RATE, SAC_BATCH_SIZE, SAC_BUFFER_SIZE_PER_SYMBOL_FACTOR,
    SAC_LEARNING_STARTS_FACTOR, SAC_TRAIN_FREQ_STEPS, SAC_GRADIENT_STEPS,
    SAC_ENT_COEF, SAC_TARGET_UPDATE_INTERVAL, SAC_TAU,
    INITIAL_CAPITAL, MAX_EPISODE_STEPS_DEFAULT,
    DEVICE, USE_AMP, GRADIENT_CLIP_NORM, ENABLE_GRADIENT_CLIPPING
    # DEFAULT_TIMEFRAME, # Removed, GRANULARITY is used
    # Config paths are mocked locally, not imported directly:
    # TRANSFORMER_CONFIG_PATH, STRATEGY_CONFIG_PATH,
    # MARKET_ANALYSIS_CONFIG_PATH, MAIN_CONFIG_PATH
)

# Reward system components (Import if directly instantiated or type-hinted in test, otherwise Env handles it)
# from src.environment.progressive_reward_calculator import ProgressiveRewardCalculator
# from src.environment.enhanced_reward_calculator import EnhancedRewardCalculator

# Utilities (Import if used for mock configs or directly in test logic)
# from src.utils.config_synchronizer import sync_configs, validate_configs
# from src.utils.technical_indicators import calculate_atr, calculate_adx

# Define a root directory for test artifacts if needed
TEST_ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "test_artifacts_pipeline")
# Override WEIGHTS_DIR and LOGS_DIR for testing to use the artifacts directory
TEST_WEIGHTS_DIR = os.path.join(TEST_ARTIFACTS_DIR, "weights")
TEST_LOGS_DIR = os.path.join(TEST_ARTIFACTS_DIR, "logs")

os.makedirs(TEST_ARTIFACTS_DIR, exist_ok=True)
os.makedirs(TEST_WEIGHTS_DIR, exist_ok=True)
os.makedirs(TEST_LOGS_DIR, exist_ok=True)

# Define mock config paths here, as they are not in the main config.py
MOCK_MAIN_CONFIG_PATH = os.path.join(TEST_ARTIFACTS_DIR, "mock_main_config.json")
MOCK_TRANSFORMER_CONFIG_PATH = os.path.join(TEST_ARTIFACTS_DIR, "mock_transformer_config.json")
MOCK_STRATEGY_CONFIG_PATH = os.path.join(TEST_ARTIFACTS_DIR, "mock_strategy_config.json")
MOCK_MARKET_ANALYSIS_CONFIG_PATH = os.path.join(TEST_ARTIFACTS_DIR, "mock_market_analysis_config.json")


def create_minimal_mock_configs_if_needed():
    # Create simplified mock JSON configuration files for components that might still try to load them.
    # These are minimal and might need expansion if specific components fail due to missing keys.
    if not os.path.exists(MOCK_TRANSFORMER_CONFIG_PATH):
        transformer_config_data = {
            "model": {
                "model_dim": 64,
                "num_heads": 4,
                "num_layers": 1,
                "ffn_dim": 128,
                "dropout_rate": 0.1,
                "timesteps": TIMESTEPS, # from src.common.config
                "output_dim_per_symbol": 32,
                "use_multi_scale": False,
                "use_cross_time_fusion": False
            },
            "training": { # Add a minimal training block if EnhancedTransformerModel expects it
                "batch_size": 32,
                "learning_rate": 1e-4
            }
        }
        with open(MOCK_TRANSFORMER_CONFIG_PATH, 'w') as f:
            json.dump(transformer_config_data, f)
            print(f"Created mock transformer config at {MOCK_TRANSFORMER_CONFIG_PATH}")

    if not os.path.exists(MOCK_STRATEGY_CONFIG_PATH):
        strategy_config_data = {
            "strategies": [
                {"name": "MomentumStrategy", "params": {"momentum_window": 20}, "input_dim": 32}
            ],
            "global_strategy_input_dim": 32 # Example, ensure matches transformer output_dim_per_symbol
        }
        with open(MOCK_STRATEGY_CONFIG_PATH, 'w') as f:
            json.dump(strategy_config_data, f)
            print(f"Created mock strategy config at {MOCK_STRATEGY_CONFIG_PATH}")

    if not os.path.exists(MOCK_MAIN_CONFIG_PATH):
        main_config_data = {
            "symbols": [
                {"name": "EUR_USD", "granularity": "S5", "max_lookback_days": 30, "tradable": True},
                {"name": "USD_JPY", "granularity": "S5", "max_lookback_days": 30, "tradable": True},
                # Add a non-tradable symbol for completeness if needed by any logic
                # {"name": "GBP_AUD", "granularity": "S5", "max_lookback_days": 30, "tradable": False}
            ],
            "data_source": "oanda",
            "account_currency": "AUD",
            "initial_capital": 100000.0,
            "max_concurrent_trades": 2,
            "global_stop_loss_percentage": 0.10,
            "global_take_profit_percentage": 0.20,
            "feature_config": {
                "use_time_features": True,
                "use_fourier_features": False,
                "use_wavelet_features": False
            },
            "reward_config":{
                "balance_change_weight": 1.0,
                "risk_adjusted_return_weight": 0.5,
                "drawdown_penalty_weight": 0.2,
                "trade_frequency_penalty_weight": 0.01,
                "holding_period_penalty_weight": 0.01
            }
        }
        with open(MOCK_MAIN_CONFIG_PATH, 'w') as f:
            json.dump(main_config_data, f, indent=4)
            print(f"Created mock main config at {MOCK_MAIN_CONFIG_PATH}")

@pytest.fixture(scope="module", autouse=True)
def setup_module_env():
    create_minimal_mock_configs_if_needed()
    # Patch global config variables to use test directories
    with patch('src.common.config.WEIGHTS_DIR', Path(TEST_WEIGHTS_DIR)), \
         patch('src.common.config.LOGS_DIR', Path(TEST_LOGS_DIR)):
        # Potentially mock other environment variables or global settings here
        # Ensure OANDA_API_KEY is set to avoid issues with data downloaders, even if mocked
        with patch.dict(os.environ, {"OANDA_API_KEY": "test_api_key", "OANDA_ACCOUNT_ID": "test_account_id"}):
            yield
    # Teardown: remove mock config files and artifact directory can be done manually or with a flag
    # import shutil
    # shutil.rmtree(TEST_ARTIFACTS_DIR) # Careful with this in active development

class TestFullEndToEndPipeline:

    @pytest.fixture(autouse=True)
    def setup_method_env(self, monkeypatch):
        # Load the mock main config data first, as it's created by the module fixture
        # and needed for other mocks in this method-scoped fixture.
        if os.path.exists(MOCK_MAIN_CONFIG_PATH):
            with open(MOCK_MAIN_CONFIG_PATH, 'r') as f:
                self.mock_config_data = json.load(f)
        else:
            # Fallback or raise error if main mock config is unexpectedly missing
            self.mock_config_data = {"symbols": [{"name": "EUR_USD"}, {"name": "USD_JPY"}]} # Minimal fallback
            print(f"Warning: {MOCK_MAIN_CONFIG_PATH} not found, using minimal fallback for mock_config_data.")

        # Mock the shared data manager to prevent actual multiprocessing issues during tests
        self.mock_shared_data_manager = MagicMock()
        self.mock_shared_data_manager.get_current_status.return_value = {'status': 'idle', 'progress': 0, 'message': ''}
        self.mock_shared_data_manager.is_stop_requested.return_value = False
        # Mock methods that would be called by the trainer
        self.mock_shared_data_manager.update_training_status = MagicMock()
        self.mock_shared_data_manager.set_actual_initial_capital = MagicMock()
        self.mock_shared_data_manager.clear_data = MagicMock()
        self.mock_shared_data_manager.request_stop = MagicMock()

        # Patch get_shared_data_manager to return our mock
        monkeypatch.setattr("src.trainer.universal_trainer.get_shared_data_manager", lambda: self.mock_shared_data_manager)
        # monkeypatch.setattr("src.trainer.callbacks.get_shared_data_manager", lambda: self.mock_shared_data_manager) # If callback also c
        # The callbacks module itself does not have get_shared_data_manager.
        # The UniversalCheckpointCallback instance receives it in its constructor.
        # We will mock the callback instance later if direct interaction with shared_data_manager is needed from callback.
        
        # Mock InstrumentInfoManager to prevent API calls
        mock_instrument_manager = MagicMock(spec=InstrumentInfoManager)
        # The method is get_details, not get_instrument_details
        mock_instrument_manager.get_details.return_value = MagicMock(
            pip_location=-4, margin_rate=Decimal("0.05"),
            symbol="EUR_USD", display_name="EUR/USD", type="CURRENCY",
            minimum_trade_size=Decimal("1"), trade_units_precision=0,
            quote_currency="USD", base_currency="EUR",
            pip_value_in_quote_currency_per_unit=Decimal("0.0001"),
            contract_size=Decimal("1")
        )
        mock_instrument_manager.get_all_available_symbols.return_value = [s["name"] for s in self.mock_config_data["symbols"]]
        # monkeypatch.setattr("src.data_manager.mmap_dataset.InstrumentInfoManager", lambda: mock_instrument_manager) # Incorrect: mmap_dataset doesn't directly have/import InstrumentInfoManager
        monkeypatch.setattr("src.environment.trading_env.InstrumentInfoManager", lambda: mock_instrument_manager)
        monkeypatch.setattr("src.trainer.universal_trainer.InstrumentInfoManager", lambda: mock_instrument_manager) # Added for trainer
        # If currency_manager uses InstrumentInfoManager, and mmap_dataset uses currency_manager, that's the path.
        monkeypatch.setattr("src.data_manager.currency_manager.InstrumentInfoManager", lambda: mock_instrument_manager)

        # Mock CurrencyDependencyManager
        mock_currency_manager = MagicMock(spec=CurrencyDependencyManager)
        # Simulate behavior of ensure_currency_data_for_trading
        mock_currency_manager.ensure_currency_data_for_trading.return_value = (True, {"EUR_USD", "USD_JPY", "AUD_USD", "EUR_AUD", "AUD_JPY"})
        monkeypatch.setattr("src.data_manager.currency_manager.CurrencyDependencyManager", lambda: mock_currency_manager)

        # Mock UniversalMemoryMappedDataset to avoid file system operations and use in-memory pandas DFs
        # The mock needs to simulate the behavior of the dataset, especially __getitem__ and __len__.
        class MockMmapDataset:
            def __init__(self, symbols, start_time_iso, end_time_iso, granularity, timesteps_history):
                self.symbols = symbols
                self.timesteps_history = timesteps_history
                self.granularity = granularity
                self.price_columns = ['bid_open', 'bid_high', 'bid_low', 'bid_close', 'ask_open', 'ask_high', 'ask_low', 'ask_close', 'volume']
                self.num_features_per_symbol = len(self.price_columns)
                self.data = {}
                # Populate with some mock data based on symbols
                if "EUR_USD" in self.symbols: self.data["EUR_USD"] = self.mock_eur_usd_data
                if "USD_JPY" in self.symbols: self.data["USD_JPY"] = self.mock_usd_jpy_data
                # Add more mock data for other symbols if needed by ensure_currency_data_for_trading
                if "AUD_USD" in self.symbols: self.data["AUD_USD"] = self.mock_eur_usd_data.copy() * 0.7
                if "EUR_AUD" in self.symbols: self.data["EUR_AUD"] = self.mock_eur_usd_data.copy() * 1.5
                if "AUD_JPY" in self.symbols: self.data["AUD_JPY"] = self.mock_eur_usd_data.copy() * 70
                
                self.min_len = min(len(df) for df in self.data.values()) if self.data else 0
                print(f"MockMmapDataset initialized for {self.symbols}. Min length: {self.min_len}")

            def __len__(self):
                return max(0, self.min_len - self.timesteps_history)

            def __getitem__(self, index):
                if index >= len(self):
                    raise IndexError("Index out of bounds in MockMmapDataset")
                # Return a dictionary of numpy arrays: {symbol: data_slice}
                # Data slice shape: (timesteps_history, num_features)
                item_data = {}
                for symbol in self.symbols:
                    if symbol in self.data:
                        df_slice = self.data[symbol].iloc[index : index + self.timesteps_history]
                        item_data[symbol] = df_slice[self.price_columns].values.astype(np.float32)
                    else: # Create zero array if symbol data is missing (should ideally not happen with good mocking)
                        item_data[symbol] = np.zeros((self.timesteps_history, self.num_features_per_symbol), dtype=np.float32)
                
                # Also need to return a list of active symbols for this step, and padding mask
                # For simplicity, assume all symbols in self.symbols are active
                active_symbols_mask = np.ones(len(self.symbols), dtype=bool)
                return item_data, list(self.symbols), active_symbols_mask

            def get_all_symbol_names(self):
                return self.symbols

            def get_num_features_per_symbol(self):
                return self.num_features_per_symbol

        monkeypatch.setattr("src.trainer.universal_trainer.UniversalMemoryMappedDataset", MockMmapDataset)
        monkeypatch.setattr("src.environment.trading_env.UniversalMemoryMappedDataset", MockMmapDataset) # Env uses it too

        # Mock the EnhancedTransformerModel's config loading if it tries to load from a file
        # This is a deeper mock, might be better to ensure mock JSON exists or mock the model itself.
        # For now, let's assume create_minimal_mock_configs_if_needed handles it.
        # Or, if UniversalTradingTransformer takes config dict directly:
        # @patch("src.models.enhanced_transformer.UniversalTradingTransformer.__init__")
        # def mock_transformer_init(slf, config_dict_or_path, *args, **kwargs):
        #     # Call original init with a predefined dict to avoid file access
        #     actual_config = {
        #         "model_dim": 64, "num_heads": 4, "num_layers": 1, "ffn_dim": 128, "dropout_rate": 0.1,
        #         "timesteps": TIMESTEPS, "output_dim_per_symbol": 32,
        #         "use_multi_scale": False, "use_cross_time_fusion": False
        #     }
        #     # This is tricky, as __init__ is complex. Better to ensure mock file exists.
        #     # Or mock the entire QuantumEnhancedSAC if it becomes too complex.
        #     pass 

    # This method will contain the core test logic, called after mocks are set up.
    def _run_pipeline_test_after_currency_mock(self, mock_ensure_currency_data):
        print("Stage 1: Trainer Instantiation...")
        # Trainer parameters
        test_symbols = ["EUR_USD", "USD_JPY"] # Fewer symbols for faster test
        # Ensure start_time and end_time are timezone-aware (UTC)
        end_time = datetime.now(timezone.utc) - timedelta(days=1)
        start_time = end_time - timedelta(days=7) # 7 days of data for test
        test_total_timesteps = 10 # Very few steps for quick training run
        test_save_freq = 5
        test_eval_freq = 5

        trainer = None # Initialize trainer to None
        try:
            trainer = UniversalTrainer(
                trading_symbols=test_symbols,
                start_time=start_time,
                end_time=end_time,
                granularity="S5", # Keep as S5 to match mock data assumptions
                timesteps_history=TIMESTEPS, # from src.common.config
                account_currency="AUD", # Test with a non-USD currency
                initial_capital=10000.0,
                total_timesteps=test_total_timesteps,
                save_freq=test_save_freq,
                eval_freq=test_eval_freq,
                model_name_prefix="test_pipeline_model",
                # Pass mock UI components as None (already default, but explicit)
                streamlit_progress_bar=None,
                streamlit_status_text=None,
                streamlit_session_state=None,
                # Risk parameters
                risk_percentage=2.0,
                atr_stop_loss_multiplier=1.5,
                max_position_percentage=5.0,
                custom_atr_period=10
            )
            assert trainer is not None, "Trainer instantiation failed"
            print("Trainer instantiated successfully.")
            # Verify shared data manager was set and used
            self.mock_shared_data_manager.set_actual_initial_capital.assert_called_with(10000.0)

        except Exception as e:
            pytest.fail(f"Trainer instantiation failed: {e}")

        # --- Run full pipeline --- 
        # We will call trainer.run_full_training_pipeline() which internally calls prepare_data, setup_env, etc.
        print("\nStage 2-7: Running Full Training Pipeline (shortened)...")
        pipeline_success = False
        try:
            # Mock the actual model training within QuantumEnhancedSAC.train to speed up the test
            # and avoid GPU/CPU intensive work if not needed for this specific integration test focus.
            # The goal is to test the pipeline flow, not the algorithm's convergence.
            with patch.object(QuantumEnhancedSAC, 'train', return_value=None) as mock_sac_train,\
                 patch.object(UniversalCheckpointCallback, '_on_step', return_value=True) as mock_callback_on_step, \
                 patch.object(UniversalCheckpointCallback, '_on_training_end', return_value=None) as mock_callback_on_end, \
                 patch.object(QuantumEnhancedSAC, 'save') as mock_sac_save: # Mock save to prevent file writes if needed
                
                pipeline_success = trainer.run_full_training_pipeline()
                
                # Assert that the mocked train method was called
                # The number of calls to _on_step depends on total_timesteps and log_interval/callback structure
                # For total_timesteps=10, it should be called multiple times by the callback.
                # mock_sac_train.assert_called_once() # SAC.train is called by trainer.train()
                                                    # which is called by run_full_training_pipeline
                assert mock_callback_on_step.call_count > 0, "Callback _on_step was not called"
                # mock_sac_save.assert_called() # Check if model saving was attempted

            assert pipeline_success, "trainer.run_full_training_pipeline() reported failure"
            print("Full training pipeline (mocked training) completed a short run.")
            self.mock_shared_data_manager.update_training_status.assert_any_call(status='completed', progress=100)

        except Exception as e:
            pytest.fail(f"Full training pipeline execution failed: {e}")
        finally:
            if trainer: # Ensure cleanup is called even if pipeline fails mid-way
                print("\nStage 8: Cleanup...")
                trainer.cleanup()
                print("Trainer cleanup called.")

    # The main test method that pytest will discover
    # It now calls the internal method that is wrapped by mocks
    @patch("src.trainer.universal_trainer.manage_data_download_for_symbols")
    @patch("src.data_manager.currency_manager.manage_data_download_for_symbols")
    @patch("src.trainer.universal_trainer.ensure_currency_data_for_trading")
    def test_pipeline_execution(self, mock_ensure_currency, mock_dl_cm, mock_dl_trainer):
        # Configure mocks that are arguments to this method
        mock_dl_trainer.return_value = None
        mock_dl_cm.return_value = None
        # ensure_currency_data_for_trading returns: success (bool), all_symbols_for_dataset (set)
        mock_ensure_currency.return_value = (True, {"EUR_USD", "USD_JPY", "AUD_USD", "EUR_AUD", "AUD_JPY"})

        # Call the internal method that contains the actual test logic
        self._run_pipeline_test_after_currency_mock(mock_ensure_currency)
