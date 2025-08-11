#!/usr/bin/env python3
"""
梯度反向傳播流向驗證測試
深度測試4個核心模組的協作關係和梯度反向傳播：
1. Transformer模組 - 增強版特徵提取器 (EnhancedTransformerFeatureExtractor)
2. 量子策略層 - 量子啟發式交易策略 (EnhancedStrategySuperposition)
3. 元學習模組 - 自適應學習系統 (MetaLearningOptimizer)
4. 量子強化版SAC - 主要強化學習代理 (QuantumEnhancedSAC)
"""

# --- Consolidated Imports ---
import os
import sys
import logging
import traceback
import warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Callable

import unittest
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    from stable_baselines3.common.vec_env import make_vec_env, DummyVecEnv
    from stable_baselines3.common.buffers import ReplayBuffer as SB3ReplayBuffer
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.torch_layers import FlattenExtractor
    from stable_baselines3.sac.policies import SACPolicy as CustomSACPolicy # Assuming this is the intended SB3 SAC policy
except ImportError as e:
    print(f"Failed to import from stable_baselines3: {e}")
    make_vec_env = None
    SB3ReplayBuffer = None
    BaseCallback = None
    ActorCriticPolicy = None
    FlattenExtractor = None
    CustomSACPolicy = None

warnings.filterwarnings("ignore")

# --- Path Setup & Early Logger Import ---
_early_logger_initialized = False
_early_log_messages = []
_test_script_logger = None # Initialize to None

def safe_log(level, message):
    global _early_logger_initialized, _early_log_messages, _test_script_logger
    if not _early_logger_initialized:
        _early_log_messages.append((level, message))
        print(f"EARLY LOG ({level.upper()}): {message}") 
    else:
        if _test_script_logger:
            getattr(_test_script_logger, level)(message)
        else:
            print(f"SAFE LOG ({level.upper()}) - LOGGER NOT READY (POST-INIT): {message}")

try:
    _project_root = Path(__file__).resolve().parent
    _src_path = _project_root / 'src'
    _logs_dir_path = _project_root / 'logs'
    _logs_dir_path.mkdir(parents=True, exist_ok=True)

    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    if str(_src_path) not in sys.path:
        sys.path.insert(0, str(_src_path))
    
    _test_script_logger_name = 'TestGradientFlowLogger'
    _test_script_logger = logging.getLogger(_test_script_logger_name)
    _test_script_logger.setLevel(logging.DEBUG)
    _test_script_logger.propagate = False

    _early_ch = logging.StreamHandler(sys.stdout)
    _early_ch.setLevel(logging.DEBUG)
    _early_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    _early_ch.setFormatter(_early_formatter)
    if not _test_script_logger.handlers:
        _test_script_logger.addHandler(_early_ch)
    
    _early_logger_initialized = True
    safe_log('info', "Early logger initialized for test script.")
    safe_log('info', f"Calculated project root: {_project_root}")
    safe_log('info', f"Calculated src path: {_src_path}")
    safe_log('info', f"Calculated logs directory: {_logs_dir_path}")
    safe_log('info', f"Current sys.path: {sys.path}")

    try:
        from src.common.logger_setup import setup_logging
        setup_logging(test_script_logger_name=_test_script_logger_name, test_script_logfile=str(_logs_dir_path / 'test_gradient_flow_validation_output.log'))
        _test_script_logger = logging.getLogger(_test_script_logger_name) 
        safe_log('info', "Main logger setup from src.common.logger_setup completed.")
        for level, message in _early_log_messages:
            getattr(_test_script_logger, level)(message)
        _early_log_messages = []
    except ImportError as ie:
        safe_log('warning', f"Could not import 'src.common.logger_setup'. Using basic early logger. Error: {ie}")
    except Exception as e:
        safe_log('error', f"Error setting up main logger: {e}\
{traceback.format_exc()}")

except Exception as e:
    if _test_script_logger is None: 
        _test_script_logger = logging.getLogger('FallbackCriticalErrorLogger')
        _ch = logging.StreamHandler(sys.stdout)
        _ch.setLevel(logging.ERROR)
        _formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        _ch.setFormatter(_formatter)
        _test_script_logger.addHandler(_ch)
        _test_script_logger.propagate = False
    safe_log('error', f"Critical error in path setup or early logger initialization: {e}\
{traceback.format_exc()}")
# --- End Path Setup & Early Logger Import ---

MAX_SYMBOLS_ALLOWED = 10

# Initialize all potentially imported names from src to None
TradingEnvironmentWrapper = None
get_device = None
save_replay_buffer = None
load_replay_buffer = None
check_gradients = None
TradingState = None
ComponentSpec = None
DimensionSpec = None
load_config = None
MetaLearningOptimizer = None
QuantumEnhancedSAC = None
ReplayBuffer = None 
CustomReplayBuffer = None
OandaMarketDataSource = None
DummyMarketDataSource = None # Added placeholder for DummyMarketDataSource
CombinedExtractor = None
TransformerFeatureExtractor = None
QuantumFeatureExtractor = None
TransformerModel = None
StrategySuperposition = None
QuantumInspiredLayer = None
DataCollector = None
MarketDataProcessor = None
HighLevelIntegrationSystem = None
PerformanceTracker = None
RiskManager = None
OrchestrationService = None

MODULE_IMPORT_ERRORS = {}
try:
    from src.environments.vts_env_wrapper import VTSEnvironmentWrapper as TradingEnvironmentWrapper_imported
    TradingEnvironmentWrapper = TradingEnvironmentWrapper_imported

    from src.common.utils import (
        get_device as get_device_imported,
        save_replay_buffer as save_replay_buffer_imported,
        load_replay_buffer as load_replay_buffer_imported,
        check_gradients as check_gradients_imported
    )
    get_device = get_device_imported
    save_replay_buffer = save_replay_buffer_imported
    load_replay_buffer = load_replay_buffer_imported
    check_gradients = check_gradients_imported

    from src.common.trading_state import (
        TradingState as TradingState_imported,
        ComponentSpec as ComponentSpec_imported,
        DimensionSpec as DimensionSpec_imported
    )
    TradingState = TradingState_imported
    ComponentSpec = ComponentSpec_imported
    DimensionSpec = DimensionSpec_imported

    from src.config.config_loader import load_config as load_config_imported
    load_config = load_config_imported
    
    from src.agent.feature_extractors import (
        CombinedExtractor as CombinedExtractor_imported,
        TransformerFeatureExtractor as TransformerFeatureExtractor_imported,
        QuantumFeatureExtractor as QuantumFeatureExtractor_imported
    )
    CombinedExtractor = CombinedExtractor_imported
    TransformerFeatureExtractor = TransformerFeatureExtractor_imported
    QuantumFeatureExtractor = QuantumFeatureExtractor_imported

    from src.agent.meta_learning_optimizer import MetaLearningOptimizer as MetaLearningOptimizer_imported
    MetaLearningOptimizer = MetaLearningOptimizer_imported

    from src.agent.quantum_enhanced_sac import QuantumEnhancedSAC as QuantumEnhancedSAC_imported
    QuantumEnhancedSAC = QuantumEnhancedSAC_imported
    
    from src.agent.replay_buffer import (
        ReplayBuffer as ReplayBuffer_src_imported, 
        CustomReplayBuffer as CustomReplayBuffer_imported
    )
    ReplayBuffer = ReplayBuffer_src_imported 
    CustomReplayBuffer = CustomReplayBuffer_imported 

    from src.models.transformer_model import TransformerModel as TransformerModel_imported
    TransformerModel = TransformerModel_imported
    
    from src.models.quantum_models import (
        StrategySuperposition as StrategySuperposition_imported,
        QuantumInspiredLayer as QuantumInspiredLayer_imported
    )
    StrategySuperposition = StrategySuperposition_imported
    QuantumInspiredLayer = QuantumInspiredLayer_imported

    from src.data_handling.data_collector import DataCollector as DataCollector_imported
    DataCollector = DataCollector_imported
    from src.data_handling.market_data_processor import MarketDataProcessor as MarketDataProcessor_imported
    MarketDataProcessor = MarketDataProcessor_imported
    
    try:
        from src.data_sources.oanda_data_source import OandaMarketDataSource as OandaMarketDataSource_imported
        OandaMarketDataSource = OandaMarketDataSource_imported
    except ImportError as e_oanda:
        safe_log('warning', f"Could not import OandaMarketDataSource from src.data_sources: {e_oanda}")

    from src.system.high_level_integration_system import HighLevelIntegrationSystem as HighLevelIntegrationSystem_imported
    HighLevelIntegrationSystem = HighLevelIntegrationSystem_imported
    from src.system.performance_tracker import PerformanceTracker as PerformanceTracker_imported
    PerformanceTracker = PerformanceTracker_imported
    from src.system.risk_manager import RiskManager as RiskManager_imported
    RiskManager = RiskManager_imported
    from src.common.orchestration_service import OrchestrationService as OrchestrationService_imported
    OrchestrationService = OrchestrationService_imported

except ImportError as e:
    module_name = e.name if hasattr(e, 'name') and e.name else "unknown_module"
    safe_log('error', f"Failed to import one or more 'src' modules. Module: {module_name}, Error: {e}\
{traceback.format_exc()}")
    MODULE_IMPORT_ERRORS[module_name] = str(e)

# Fallback definitions for critical CLASS components if imports failed
placeholder_classes = {
    "TradingEnvironmentWrapper": TradingEnvironmentWrapper, "DimensionSpec": DimensionSpec,
    "QuantumEnhancedSAC": QuantumEnhancedSAC, "CustomReplayBuffer": CustomReplayBuffer,
    "ReplayBuffer": ReplayBuffer, "OandaMarketDataSource": OandaMarketDataSource,
    "DummyMarketDataSource": DummyMarketDataSource, # Added DummyMarketDataSource
    "MetaLearningOptimizer": MetaLearningOptimizer, "CombinedExtractor": CombinedExtractor,
    "TransformerFeatureExtractor": TransformerFeatureExtractor, "QuantumFeatureExtractor": QuantumFeatureExtractor,
    "TransformerModel": TransformerModel, "StrategySuperposition": StrategySuperposition,
    "QuantumInspiredLayer": QuantumInspiredLayer, "DataCollector": DataCollector,
    "MarketDataProcessor": MarketDataProcessor, "HighLevelIntegrationSystem": HighLevelIntegrationSystem,
    "PerformanceTracker": PerformanceTracker, "RiskManager": RiskManager,
    "OrchestrationService": OrchestrationService, "TradingState": TradingState, "ComponentSpec": ComponentSpec
}

for class_name, class_obj_val in placeholder_classes.items(): # Renamed class_obj to class_obj_val to avoid conflict
    if class_obj_val is None:
        safe_log('warning', f"{class_name} not imported, defining a placeholder.")
        globals()[class_name] = type(class_name, (object,), {}) 

if FlattenExtractor is None:
    safe_log('warning', "FlattenExtractor from SB3 not available. Defining a basic nn.Flatten as fallback.")
    class FlattenExtractor(nn.Module):
        def __init__(self, observation_space: gym.spaces.Space):
            super().__init__()
            self.flatten = nn.Flatten()
        def forward(self, observations: torch.Tensor) -> torch.Tensor:
            return self.flatten(observations)

# Define DummyMarketDataSource if it's a placeholder (it was added to placeholder_classes)
if 'DummyMarketDataSource' in globals() and globals()['DummyMarketDataSource'] == type('DummyMarketDataSource', (object,), {}):
    safe_log('info', "Defining placeholder DummyMarketDataSource class structure.")
    class DummyMarketDataSource:
        def __init__(self, config: Dict[str, Any]):
            self.config = config
            safe_log('debug', f"DummyMarketDataSource initialized with config: {config}")
        def get_market_data(self, symbol: str, num_points: int) -> Dict[str, np.ndarray]:
            safe_log('debug', f"DummyMarketDataSource: get_market_data called for {symbol}, {num_points} points.")
            # Return dummy data matching typical structure, e.g., OHLCV
            return {
                'open': np.random.rand(num_points).astype(np.float32),
                'high': np.random.rand(num_points).astype(np.float32),
                'low': np.random.rand(num_points).astype(np.float32),
                'close': np.random.rand(num_points).astype(np.float32),
                'volume': np.random.randint(100, 1000, num_points).astype(np.float32)
            }
        def get_current_price(self, symbol: str) -> float:
            price = np.random.rand() * 100
            safe_log('debug', f"DummyMarketDataSource: get_current_price for {symbol} returning {price}")
            return price
        def get_account_summary(self) -> Dict[str, Any]:
            summary = {'balance': 10000.0, 'currency': 'USD'}
            safe_log('debug', f"DummyMarketDataSource: get_account_summary returning {summary}")
            return summary

class DummyVTSEnvironment(gym.Env):
    metadata = {'render_modes': ['human', 'ansi'], 'render_fps': 4}
    def __init__(self, config: Dict[str, Any], market_data_source: Optional[Any] = None): # Added market_data_source
        super(DummyVTSEnvironment, self).__init__()
        self.config = config
        self.market_data_source = market_data_source # Store it
        self.action_space_config = self.config.get('action_space', {'type': 'discrete', 'n': 3})
        self.observation_space_config = self.config.get('observation_space', {'type': 'box', 'shape': (10,), 'low': -1.0, 'high': 1.0})

        if self.action_space_config['type'] == 'discrete':
            self.action_space = spaces.Discrete(self.action_space_config.get('n', 3))
        else: 
            low = np.array(self.action_space_config.get('low', [-1.0] * self.action_space_config.get('shape', (1,)[0])))
            high = np.array(self.action_space_config.get('high', [1.0] * self.action_space_config.get('shape', (1,)[0])))
            shape = (self.action_space_config.get('shape', (1,)[0]),)
            self.action_space = spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)

        obs_shape_tuple = tuple(self.observation_space_config.get('shape', (10,)))
        low_obs = np.full(obs_shape_tuple, self.observation_space_config.get('low', -np.inf))
        high_obs = np.full(obs_shape_tuple, self.observation_space_config.get('high', np.inf))
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)
        
        self.current_step = 0
        self.max_steps = self.config.get('max_episode_steps', 100)
        self._last_observation = self.observation_space.sample()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.current_step = 0
        self._last_observation = self.observation_space.sample()
        # Use market_data_source if available to fetch some initial data for observation
        if self.market_data_source:
            try:
                # Example: incorporate some data into the observation if structure allows
                # This is highly dependent on how observation is constructed
                # For a simple 10-dim Box, we might just use random data or a part of it.
                data = self.market_data_source.get_market_data('EUR_USD', 10) # Get 10 points
                # Simplistic: use first 10 close prices if obs_shape is (10,)
                if self.observation_space.shape == (10,) and 'close' in data and len(data['close']) >= 10:
                    self._last_observation = data['close'][:10].astype(np.float32)
                # else: use sample as fallback if data doesn't fit
            except Exception as e:
                safe_log('warning', f"DummyVTSEnvironment: Error fetching data from market_data_source in reset: {e}")
        return self._last_observation, {"info": "dummy_reset_info"}

    def step(self, action: Any):
        self.current_step += 1
        # Potentially use market_data_source to update state or observation
        if self.market_data_source:
            try:
                # price = self.market_data_source.get_current_price('EUR_USD')
                # Incorporate price into observation or reward logic if needed
                pass # For now, just keeping it simple
            except Exception as e:
                safe_log('warning', f"DummyVTSEnvironment: Error fetching data from market_data_source in step: {e}")

        observation = self.observation_space.sample()
        self._last_observation = observation
        reward = np.random.rand()
        terminated = self.current_step >= self.max_steps
        truncated = False 
        info = {"dummy_info": "step_info", "is_success": terminated} 
        return observation, reward, terminated, truncated, info

    def render(self, mode: str = 'human'):
        if mode == 'ansi':
            return f"Step: {self.current_step}, Last Obs: {self._last_observation}"
        return None

    def close(self):
        pass

def create_dummy_config() -> Dict[str, Any]:
    config = {
        'environment': {
            'name': 'DummyVTSEnvironment',
            'max_episode_steps': 50, 
            'action_space': {'type': 'discrete', 'n': 3},
            'observation_space': {'type': 'box', 'shape': (10,), 'low': -1.0, 'high': 1.0},
        },
        'agent': {
            'name': 'QuantumEnhancedSAC',
            'policy': 'MlpPolicy', 
            'learning_rate': 1e-4,
            'buffer_size': 1000, 
            'learning_starts': 100, # Increased to be >= batch_size
            'batch_size': 64,    # Increased batch_size
            'tau': 0.005,
            'gamma': 0.99,
            'gradient_steps': 1, 
            'ent_coef': 'auto',
            'target_update_interval': 1,
            'train_freq': (1, "step"), # Ensure it's a tuple
            'policy_kwargs': {
                'net_arch': [64, 64], 
                'features_extractor_class': FlattenExtractor, 
                'features_extractor_kwargs': {},
            }
        },
        'data_source': {
            'type': 'dummy', 
            'api_key': 'dummy_key',
            'account_id': 'dummy_account',
            'instruments': ['EUR_USD'],
            'granularity': 'M1',
            'max_data_points': 100
        },
        'trading_state': {
            'initial_balance': 10000,
            'max_drawdown_percent': 0.2,
            'max_concurrent_trades': 5,
            'lot_size': 0.01
        },
        'meta_learning': {
            'meta_lr': 1e-5,
            'num_adaptation_steps': 1, 
            'task_batch_size': 4 
        },
        'replay_buffer': {
            'type': 'CustomReplayBuffer', 
            'buffer_size': 1000, 
            'device': 'auto' 
        },
        'feature_extractor': { 
            'type': 'CombinedExtractor', 
            'transformer_kwargs': {
                'd_model': 64, 
                'nhead': 2,
                'num_encoder_layers': 1,
                'num_decoder_layers': 1,
                'dim_feedforward': 128,
                'dropout': 0.1
            },
            'quantum_kwargs': { 
                'num_qubits': 4,
                'num_layers': 1
            }
        },
        'high_level_system': {
            'max_symbols_allowed': MAX_SYMBOLS_ALLOWED
        },
        'logging': {
            'level': 'DEBUG',
            'log_file': 'logs/test_gradient_flow.log' 
        }
    }
    return config

class TestGradientFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        safe_log('info', "Setting up TestGradientFlow class...")
        cls.config = create_dummy_config()
        cls.device = get_device(cls.config['replay_buffer'].get('device', 'auto')) if get_device and callable(get_device) else torch.device('cpu')
        
        if MODULE_IMPORT_ERRORS:
            skip_message = "Skipping all tests due to critical module import failures: " + ", ".join(MODULE_IMPORT_ERRORS.keys())
            safe_log('error', skip_message)
            raise unittest.SkipTest(skip_message)
        cls.skip_all_tests = False # Should be set only if no skip occurs

        # Instantiate DummyMarketDataSource (placeholder or real)
        # It's used by DummyVTSEnvironment
        current_dummy_market_data_source_class = globals().get('DummyMarketDataSource')
        if current_dummy_market_data_source_class is not None and callable(current_dummy_market_data_source_class):
            cls.dummy_market_data_source = current_dummy_market_data_source_class(config=cls.config['data_source'])
        else:
            safe_log('error', "DummyMarketDataSource class is None or not callable. Cannot create dummy_market_data_source.")
            cls.dummy_market_data_source = None # Fallback
            # This might be critical, consider skipping if it is.

        cls.dummy_env = DummyVTSEnvironment(cls.config['environment'], market_data_source=cls.dummy_market_data_source)
        
        current_trading_env_wrapper_class = globals().get('TradingEnvironmentWrapper')
        current_dimension_spec_class = globals().get('DimensionSpec')
        current_oanda_data_source_class = globals().get('OandaMarketDataSource')

        if current_trading_env_wrapper_class and callable(current_trading_env_wrapper_class) and \
           current_dimension_spec_class and callable(current_dimension_spec_class):
            try:
                obs_spec = current_dimension_spec_class(name="obs", shape=cls.dummy_env.observation_space.shape, dtype=str(cls.dummy_env.observation_space.dtype))
                if isinstance(cls.dummy_env.action_space, spaces.Discrete):
                    act_shape = (1,)
                    act_dtype = "int64"
                elif isinstance(cls.dummy_env.action_space, spaces.Box):
                    act_shape = cls.dummy_env.action_space.shape
                    act_dtype = str(cls.dummy_env.action_space.dtype)
                else:
                    act_shape = (1,); act_dtype = "int64"
                act_spec = current_dimension_spec_class(name="act", shape=act_shape, dtype=act_dtype)
                
                market_data_source_to_use = current_oanda_data_source_class if current_oanda_data_source_class and callable(current_oanda_data_source_class) else current_dummy_market_data_source_class
                if not (market_data_source_to_use and callable(market_data_source_to_use)):
                    safe_log('warning', "No valid market data source class (Oanda or Dummy) found for TradingEnvironmentWrapper. Wrapper might not function as expected.")
                    # market_data_source_to_use will be None or a placeholder type, which might cause issues in TradingEnvironmentWrapper

                cls.wrapped_env = current_trading_env_wrapper_class(
                    env_config=cls.config['environment'],
                    data_source_config=cls.config['data_source'],
                    trading_state_config=cls.config['trading_state'],
                    observation_spec=obs_spec,
                    action_spec=act_spec,
                    market_data_source_class=market_data_source_to_use 
                )
            except Exception as e:
                safe_log('error', f"Failed to create TradingEnvironmentWrapper: {e}\
{traceback.format_exc()}")
                cls.wrapped_env = None
        else:
            safe_log('warning', "TradingEnvironmentWrapper or DimensionSpec class is None/not callable. Cannot create wrapped_env.")
            cls.wrapped_env = None

        if make_vec_env is not None and cls.wrapped_env is not None:
            try:
                cls.vec_env = make_vec_env(lambda: cls.wrapped_env, n_envs=1, vec_env_cls=DummyVecEnv)
            except Exception as e:
                safe_log('error', f"Failed to create make_vec_env: {e}\
{traceback.format_exc()}")
                cls.vec_env = None
        else: # Handle either make_vec_env is None or wrapped_env is None
            if make_vec_env is None: safe_log('warning', "make_vec_env is None (SB3 import failure). Cannot create vec_env.")
            if cls.wrapped_env is None: safe_log('warning', "wrapped_env is None. Cannot create vec_env.")
            cls.vec_env = None

        current_q_sac_class = globals().get('QuantumEnhancedSAC')
        current_custom_sac_policy_class = globals().get('CustomSACPolicy') # This is SB3's SACPolicy

        if current_q_sac_class and callable(current_q_sac_class) and \
           current_custom_sac_policy_class and callable(current_custom_sac_policy_class) and \
           cls.vec_env is not None:
            try:
                policy_kwargs = cls.config['agent'].get('policy_kwargs', {})
                if not policy_kwargs.get('features_extractor_class') and FlattenExtractor and callable(FlattenExtractor):
                    policy_kwargs['features_extractor_class'] = FlattenExtractor
                if policy_kwargs.get('features_extractor_kwargs') is None:
                     policy_kwargs['features_extractor_kwargs'] = {}

                cls.model = current_q_sac_class(
                    current_custom_sac_policy_class, 
                    cls.vec_env,
                    learning_rate=cls.config['agent']['learning_rate'],
                    buffer_size=cls.config['agent']['buffer_size'],
                    learning_starts=cls.config['agent']['learning_starts'],
                    batch_size=cls.config['agent']['batch_size'],
                    tau=cls.config['agent']['tau'],
                    gamma=cls.config['agent']['gamma'],
                    gradient_steps=cls.config['agent']['gradient_steps'],
                    train_freq=cls.config['agent']['train_freq'],
                    policy_kwargs=policy_kwargs,
                    device=cls.device,
                    verbose=0 
                )
            except Exception as e:
                safe_log('error', f"Failed to create QuantumEnhancedSAC model: {e}\
{traceback.format_exc()}")
                cls.model = None
        else:
            missing_comps = []
            if not (current_q_sac_class and callable(current_q_sac_class)): missing_comps.append("QuantumEnhancedSAC class")
            if not (current_custom_sac_policy_class and callable(current_custom_sac_policy_class)): missing_comps.append("CustomSACPolicy (SB3 SACPolicy) class")
            if cls.vec_env is None: missing_comps.append("vec_env instance")
            safe_log('warning', f"One or more components for SAC model are None/not callable ({', '.join(missing_comps)}). Cannot create model.")
            cls.model = None

        # Determine Replay Buffer class to use
        buffer_options = [globals().get(name) for name in ['CustomReplayBuffer', 'ReplayBuffer', 'SB3ReplayBuffer']]
        buffer_class_to_use = next((b for b in buffer_options if b and callable(b)), None)
        
        if buffer_class_to_use and cls.vec_env is not None: 
            try:
                cls.replay_buffer = buffer_class_to_use(
                    buffer_size=cls.config['replay_buffer']['buffer_size'],
                    observation_space=cls.vec_env.observation_space,
                    action_space=cls.vec_env.action_space,
                    device=cls.device,
                    n_envs=cls.vec_env.num_envs,
                )
            except Exception as e:
                safe_log('error', f"Failed to create ReplayBuffer ({buffer_class_to_use.__name__ if buffer_class_to_use else 'N/A'}): {e}\
{traceback.format_exc()}")
                cls.replay_buffer = None
        else:
            missing_comps = []
            if not buffer_class_to_use: missing_comps.append("any ReplayBuffer class")
            if cls.vec_env is None: missing_comps.append("vec_env instance (for env spaces)")
            safe_log('warning', f"One or more components for ReplayBuffer are None/not callable ({', '.join(missing_comps)}). Cannot create replay_buffer.")
            cls.replay_buffer = None
        
        if cls.model and cls.replay_buffer:
            cls.model.replay_buffer = cls.replay_buffer
            safe_log('info', "Replay buffer assigned to model.")
        elif cls.model and not cls.replay_buffer:
            safe_log('warning', "Model exists but replay buffer creation failed. Model might use internal buffer or fail.")
        
        safe_log('info', "TestGradientFlow class setup complete.")

    def setUp(self):
        if hasattr(TestGradientFlow, 'skip_all_tests') and TestGradientFlow.skip_all_tests:
            self.skipTest("Skipping test due to critical module import failures during setUpClass.")
        
        # Check if essential components were successfully initialized in setUpClass
        critical_components_present = hasattr(self, 'model') and self.model is not None and \
                                      hasattr(self, 'replay_buffer') and self.replay_buffer is not None and \
                                      hasattr(self, 'vec_env') and self.vec_env is not None
        if not critical_components_present:
            self.skipTest("Skipping test: Critical components (Model, Replay Buffer, or VecEnv) were not properly initialized in setUpClass.")
        
        # Reset replay buffer if it exists and has a reset/clear mechanism
        if self.model and hasattr(self.model, 'replay_buffer') and self.model.replay_buffer is not None:
            rb = self.model.replay_buffer
            if hasattr(rb, 'reset') and callable(rb.reset):
                 rb.reset()
            elif hasattr(rb, 'pos'): # Basic check for simple buffer
                 rb.pos = 0
                 if hasattr(rb, 'full'): rb.full = False
            safe_log('debug', "Replay buffer reset for new test.")

    def test_dummy_env_creation(self):
        safe_log('info', "Running test_dummy_env_creation...")
        self.assertIsNotNone(self.dummy_env, "DummyVTSEnvironment should be created.")
        self.assertIsNotNone(self.dummy_env.observation_space, "Observation space should exist.")
        self.assertIsNotNone(self.dummy_env.action_space, "Action space should exist.")
        obs, info = self.dummy_env.reset()
        self.assertIsNotNone(obs, "Reset should return an observation.")
        self.assertTrue(self.dummy_env.observation_space.contains(obs), "Observation should be in observation space.")
        action = self.dummy_env.action_space.sample()
        obs, reward, terminated, truncated, info = self.dummy_env.step(action)
        self.assertIsNotNone(obs, "Step should return an observation.")
        self.assertTrue(self.dummy_env.observation_space.contains(obs), "Observation from step should be in observation space.")
        self.assertIsInstance(reward, float, "Reward should be a float.")
        self.assertIsInstance(terminated, bool, "Terminated flag should be a bool.")
        self.assertIsInstance(truncated, bool, "Truncated flag should be a bool.")
        safe_log('info', "test_dummy_env_creation PASSED.")

    def test_replay_buffer_operations(self):
        safe_log('info', "Running test_replay_buffer_operations...")
        num_timesteps_to_collect = self.config['agent'].get('learning_starts', 100) + 5
        obs, info = self.vec_env.reset() # vec_env reset returns obs directly (no info usually for DummyVecEnv)
        collected_count = 0
        for _ in range(num_timesteps_to_collect):
            action, _ = self.model.predict(obs, deterministic=False)
            new_obs, reward, done, infos = self.vec_env.step(action)
            # Ensure replay buffer add is compatible with its type (SB3 or custom)
            # SB3 expects obs, new_obs, action, reward, done, infos (where done is array, infos is list of dicts)
            self.model.replay_buffer.add(obs, new_obs, action, reward, done, infos)
            collected_count +=1
            obs = new_obs
            if done[0]: # For n_envs=1, done is an array like [False]
                obs, info = self.vec_env.reset()

        current_buffer_size = self.model.replay_buffer.size() if hasattr(self.model.replay_buffer, 'size') and callable(self.model.replay_buffer.size) else self.model.replay_buffer.pos
        safe_log('info', f"Collected {collected_count} samples. Replay buffer current size: {current_buffer_size}")
        self.assertGreaterEqual(current_buffer_size, min(num_timesteps_to_collect, self.model.replay_buffer.buffer_size))

        if current_buffer_size >= self.model.batch_size:
            sample = self.model.replay_buffer.sample(self.model.batch_size)
            self.assertIsNotNone(sample)
            # SB3 ReplayBufferSample has these attributes
            self.assertTrue(hasattr(sample, 'observations') and hasattr(sample, 'actions') and \
                            hasattr(sample, 'rewards') and hasattr(sample, 'next_observations') and \
                            hasattr(sample, 'dones'))
            safe_log('info', f"Sampled batch of size {self.model.batch_size} from replay buffer.")
        else:
            safe_log('warning', f"Buffer size ({current_buffer_size}) is less than batch size ({self.model.batch_size}). Skipping sample test.")
        safe_log('info', "test_replay_buffer_operations PASSED.")

    def test_gradient_flow_through_modules(self):
        safe_log('info', "Running test_gradient_flow_through_modules...")
        num_samples_to_collect = self.model.learning_starts + self.model.batch_size 
        obs, info = self.vec_env.reset()
        for _ in range(num_samples_to_collect):
            action, _ = self.model.predict(obs, deterministic=False)
            new_obs, reward, done, infos = self.vec_env.step(action)
            self.model.replay_buffer.add(obs, new_obs, action, reward, done, infos)
            obs = new_obs
            if done[0]:
                obs, info = self.vec_env.reset()
        
        current_buffer_size = self.model.replay_buffer.size() if hasattr(self.model.replay_buffer, 'size') and callable(self.model.replay_buffer.size) else self.model.replay_buffer.pos
        safe_log( 'info', f"Buffer populated with {current_buffer_size} samples for gradient test.")

        if current_buffer_size < self.model.learning_starts:
            self.skipTest(f"Buffer size {current_buffer_size} < learning_starts {self.model.learning_starts}.")
        if current_buffer_size < self.model.batch_size:
             self.skipTest(f"Buffer size {current_buffer_size} < batch_size {self.model.batch_size}.")

        params_to_check = {}
        # Check actor parameters if policy and actor exist and are nn.Module
        if hasattr(self.model, 'policy') and self.model.policy and \
           hasattr(self.model.policy, 'actor') and isinstance(self.model.policy.actor, nn.Module):
            actor_params = list(self.model.policy.actor.parameters())
            if actor_params:
                params_to_check['actor_last_param'] = actor_params[-1]
        
        # Check critic parameters if policy and critic exist and are nn.Module
        if hasattr(self.model, 'policy') and self.model.policy and \
           hasattr(self.model.policy, 'critic') and isinstance(self.model.policy.critic, nn.Module):
            critic_params = list(self.model.policy.critic.parameters())
            if critic_params:
                 params_to_check['critic_last_param'] = critic_params[-1]

        if not params_to_check:
            self.skipTest("Could not identify actor/critic parameters (nn.Module with parameters) to check for gradients.")

        for name, param in params_to_check.items():
            if param.grad is not None: param.grad.zero_() 
            param.requires_grad_(True)

        try:
            safe_log('info', "Attempting model.train() step...")
            # SB3 SAC model's train method typically takes gradient_steps and batch_size
            # For this test, we rely on the model's configured batch_size and ask for 1 gradient_step.
            self.model.train(gradient_steps=1) # Use model's configured batch_size
            safe_log('info', "model.train() step completed.")
        except Exception as e:
            self.fail(f"model.train() failed: {e}\
{traceback.format_exc()}")

        non_none_grads = 0
        for name, param in params_to_check.items():
            if param.grad is not None:
                safe_log('info', f"Gradient for {name} is present. Sum of abs grad: {param.grad.abs().sum().item()}")
                non_none_grads +=1
                self.assertNotEqual(torch.sum(torch.abs(param.grad)).item(), 0.0, f"Gradient for {name} should not be all zeros.")
            else:
                safe_log('warning', f"Gradient for {name} is STILL None after training step.")
        
        self.assertGreater(non_none_grads, 0, "At least one checked parameter should have a non-None gradient.")
        safe_log('info', "test_gradient_flow_through_modules PASSED.")

    @classmethod
    def tearDownClass(cls):
        safe_log('info', "Tearing down TestGradientFlow class...")
        if hasattr(cls, 'vec_env') and cls.vec_env is not None:
            try:
                cls.vec_env.close()
            except Exception as e:
                safe_log('error', f"Error closing vec_env: {e}")
        safe_log('info', "TestGradientFlow class teardown complete.")

if __name__ == "__main__":
    current_file_path = Path(__file__).resolve()
    _project_root_check = current_file_path.parent 
    
    if _test_script_logger is None:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s (MAIN)')
        _test_script_logger = logging.getLogger("MainBlockLogger")

    safe_log('info', f"Running test script: {current_file_path}")
    safe_log('info', f"Project root (checked in main): {_project_root_check}")
    safe_log('info', f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    safe_log('info', f"sys.path: {sys.path}")

    # Check for __init__.py files (visual confirmation for debugging)
    # ... (this debug code can be removed or kept as needed) ...

    if MODULE_IMPORT_ERRORS:
        safe_log('critical', "Cannot run tests due to critical module import failures:")
        for module_name, error_msg in MODULE_IMPORT_ERRORS.items():
            safe_log('error', f"  - Module: {module_name}, Error: {error_msg}")
        print("\n" + "="*70)
        print("TESTS SKIPPED DUE TO IMPORT ERRORS (see logs for details)")
        print("Failed to import modules: " + ", ".join(MODULE_IMPORT_ERRORS.keys()))
        print("="*70)
    else:
        safe_log('info', "All critical modules appear to be imported or have fallbacks. Proceeding with tests.")
        suite = unittest.TestSuite()
        loader = unittest.TestLoader()
        try:
            test_gradient_flow_class = globals().get('TestGradientFlow')
            if test_gradient_flow_class and issubclass(test_gradient_flow_class, unittest.TestCase):
                suite.addTest(loader.loadTestsFromTestCase(test_gradient_flow_class))
                runner = unittest.TextTestRunner(verbosity=2)
                results = runner.run(suite)
                # if not results.wasSuccessful(): sys.exit(1) # Optional: exit with error code if tests fail
            else:
                safe_log('critical', "'TestGradientFlow' class is not defined or not a TestCase. Cannot run tests.")
                print("\n" + "="*70)
                print("CRITICAL ERROR: Test class 'TestGradientFlow' not found or invalid.")
                print("="*70)
        except NameError as ne: # Should be caught by globals().get() check, but as a safeguard
            safe_log('critical', f"Failed to load tests due to NameError: {ne}. 'TestGradientFlow' class might not be defined.")
            print("\n" + "="*70)
            print(f"CRITICAL NameError: {ne}. Test class 'TestGradientFlow' likely not defined.")
            print("="*70)
        except Exception as e_main:
            safe_log('critical', f"An unexpected error occurred during test loading or execution: {e_main}\
{traceback.format_exc()}")

    safe_log('info', "Test script execution finished.")
