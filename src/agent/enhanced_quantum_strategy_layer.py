# src/agent/enhanced_quantum_strategy_layer.py
"""
增強版量子策略層實現
擴展原有3種策略至15+種策略，實現階段一核心架構增強

主要增強：
1. 15種專業交易策略實現
2. 動態策略生成器
3. 量子策略組合優化
4. 自適應權重極習機制
5. 策略創新引擎（基礎版）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from abc import ABC, abstractmethod
import math

try:
    from src.common.logger_setup import logger
    from src.common.config import DEVICE, MAX_SYMBOLS_ALLOWED
    from .strategies import BaseStrategy, StrategyConfig
    from .strategies import (
        MomentumStrategy,
        BreakoutStrategy,
        TrendFollowingStrategy,
        ReversalStrategy,
        StatisticalArbitrageStrategy,
        VolatilityStrategy,
        PairsTradeStrategy,
        MeanReversionStrategy,
        CointegrationStrategy,
        VolatilityArbitrageStrategy,
        ReinforcementLearningStrategy,
        DeepLearningPredictionStrategy,
        EnsembleLearningStrategy,
        TransferLearningStrategy,
        DynamicHedgingStrategy,
        RiskParityStrategy,
        VaRControlStrategy,
        MaxDrawdownControlStrategy,
        OptionFlowStrategy,
        MicrostructureStrategy,
        CarryTradeStrategy,
        MacroEconomicStrategy,
        EventDrivenStrategy,
        SentimentStrategy,
        QuantitativeStrategy,
        MarketMakingStrategy,
        HighFrequencyStrategy,
        AlgorithmicStrategy
    )

except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"Import error in enhanced_quantum_strategy_layer: {e}")
    DEVICE = "cpu"
    MAX_SYMBOLS_ALLOWED = 5
    class BaseStrategy(nn.Module, ABC):
        def __init__(self, params: dict = None, config: Optional[Any] = None):
            super().__init__()
            self.params = params if params is not None else {}
            self.config = config
            self.strategy_id = self.__class__.__name__ if config is None else config.name

        @abstractmethod
        def forward(self, market_data: pd.DataFrame, portfolio_context: dict = None) -> pd.DataFrame:
            pass

        @abstractmethod
        def generate_signals(self, processed_data: pd.DataFrame, portfolio_context: dict = None) -> pd.DataFrame:
            pass

        def get_strategy_name(self) -> str:
            return self.strategy_id
    
    from dataclasses import dataclass
    @dataclass
    class StrategyConfig:
        name: str
        description: str
        risk_level: float
        market_regime: str
        complexity: int
        base_performance: float = 0.5

# 自定義激活函數實現
class Swish(nn.Module):
    """Swish激活函數實現"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class Mish(nn.Module):
    """Mish激活函數實現"""
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


# ===============================
# 15種專業交易策略實現
# ===============================

# All original strategy class definitions (MomentumStrategy, BreakoutStrategy, ..., AlgorithmicStrategy)
# have been moved to their respective files under src/agent/strategies/
# and are now imported above.

# Placeholder classes like CointegrationStrategy, VolatilityArbitrageStrategy, 
# ReinforcementLearningStrategy, etc., are also imported from their new locations.
# Ensure these placeholder classes are properly defined in their respective files 
# (e.g., statistical_arbitrage_strategies.py, ml_strategies.py)
# and inherit from the new BaseStrategy.

# ===============================
# 動態策略生成器
# ===============================

class DynamicStrategyGenerator:
    """
    Generates and manages trading strategies dynamically based on market conditions
    and other factors.
    """
    def __init__(self, config: Optional[Dict] = None, strategies: Optional[List[BaseStrategy]] = None):
        self.config = config if config is not None else {}
        # self.strategies = strategies if strategies is not None else [] # This will be populated by available strategies
        self.available_strategies_classes = { 
            strategy_cls.get_strategy_name(strategy_cls) if hasattr(strategy_cls, 'get_strategy_name') and callable(strategy_cls.get_strategy_name) else strategy_cls.__name__: strategy_cls
            for strategy_cls in [
                MomentumStrategy,
                BreakoutStrategy,
                TrendFollowingStrategy,
                ReversalStrategy,
                StatisticalArbitrageStrategy,
                VolatilityStrategy,
                PairsTradeStrategy,
                MeanReversionStrategy,
                CointegrationStrategy, 
                VolatilityArbitrageStrategy, 
                ReinforcementLearningStrategy, 
                DeepLearningPredictionStrategy, 
                EnsembleLearningStrategy, 
                TransferLearningStrategy, 
                DynamicHedgingStrategy,
                RiskParityStrategy,
                VaRControlStrategy,
                MaxDrawdownControlStrategy,
                OptionFlowStrategy,
                MicrostructureStrategy,
                CarryTradeStrategy,
                MacroEconomicStrategy,
                EventDrivenStrategy,
                SentimentStrategy,
                QuantitativeStrategy,
                MarketMakingStrategy,
                HighFrequencyStrategy,
                AlgorithmicStrategy
            ]
        }
        self.active_strategies_instances: List[BaseStrategy] = [] # Store instances of strategies
        if strategies:
            for s_instance in strategies:
                if isinstance(s_instance, BaseStrategy):
                    self.active_strategies_instances.append(s_instance)
                elif isinstance(s_instance, type) and issubclass(s_instance, BaseStrategy): # if a class is passed
                    try:
                        self.active_strategies_instances.append(s_instance()) # Basic instantiation
                    except Exception as e:
                        logger.error(f"Could not instantiate {s_instance.__name__}: {e}")

        self.genetic_optimizer = None  # Placeholder for GeneticOptimizer
        self.nas_optimizer = None      # Placeholder for NeuralArchitectureSearch
        logger.info("DynamicStrategyGenerator initialized.")

    def add_strategy(self, strategy_instance: BaseStrategy):
        """Adds a strategy instance to the generator."""
        if strategy_instance not in self.active_strategies_instances:
            self.active_strategies_instances.append(strategy_instance)
            logger.info(f"Strategy instance {strategy_instance.get_strategy_name()} added to DynamicStrategyGenerator.")
        else:
            logger.warning(f"Strategy instance {strategy_instance.get_strategy_name()} already exists.")

    def generate_new_strategy(self, market_data: pd.DataFrame, current_context: Dict) -> Optional[BaseStrategy]:
        """
        Generates or selects a new strategy based on the provided market data and context.
        """
        logger.info(f"Attempting to generate new strategy based on market_data and context.")
        if not self.available_strategies_classes:
            logger.warning("No strategy classes available to select from.")
            return None
        
        market_regime = current_context.get("market_regime", "all")
        suitable_strategy_names = []
        
        # This part needs access to StrategyConfig for each strategy type
        # Assuming strategy_configs are passed in self.config or accessible otherwise
        strategy_configs_list = self.config.get("strategy_configs_list", [])
        strategy_config_map = {cfg.name: cfg for cfg in strategy_configs_list if isinstance(cfg, StrategyConfig)}

        for name, s_class in self.available_strategies_classes.items():
            s_config = strategy_config_map.get(name)
            if s_config and (s_config.market_regime == market_regime or s_config.market_regime == "all"):
                suitable_strategy_names.append(name)
            elif not s_config: # If no specific config, consider it suitable for now
                suitable_strategy_names.append(name)

        if suitable_strategy_names:
            import random
            selected_strategy_name = random.choice(suitable_strategy_names)
            StrategyClassToInstantiate = self.available_strategies_classes[selected_strategy_name]
            try:
                # Here, we need parameters for the strategy. This is a simplification.
                # In a real scenario, params might come from GA, NAS, or predefined sets.
                strategy_params = current_context.get("default_params", {}).get(selected_strategy_name, {})
                strategy_config_obj = strategy_config_map.get(selected_strategy_name)
                
                new_strategy_instance = StrategyClassToInstantiate(params=strategy_params, config=strategy_config_obj)
                logger.info(f"Selected and instantiated strategy: {new_strategy_instance.get_strategy_name()}")
                # self.add_strategy(new_strategy_instance) # Optionally add to active list
                return new_strategy_instance
            except Exception as e:
                logger.error(f"Error instantiating strategy {selected_strategy_name}: {e}")
                return None
        else:
            logger.warning(f"No suitable strategy class found for market regime: {market_regime}")
            return None

    def integrate_genetic_optimizer(self, optimizer_config: Dict):
        """
        Integrates and configures the GeneticOptimizer.
        """
        # self.genetic_optimizer = GeneticOptimizer(**optimizer_config) # To be implemented
        logger.info("GeneticOptimizer integration placeholder.")

    def integrate_nas(self, nas_config: Dict):
        """
        Integrates and configures Neural Architecture Search.
        """
        # self.nas_optimizer = NeuralArchitectureSearch(**nas_config) # To be implemented
        logger.info("NeuralArchitectureSearch integration placeholder.")


# ===============================
# 增強版策略疊加系統
# ===============================

class EnhancedStrategySuperposition(nn.Module):
    """
    增強版策略疊加系統
    管理15+種策略的量子疊加，包含動態策略生成
    支持完全動態自適應維度配置
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 enable_dynamic_generation: bool = True,
                 initial_strategy_params: Optional[Dict[str, Dict]] = None,
                 strategy_configs_list: Optional[List[StrategyConfig]] = None):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim # Note: action_dim for PyTorch layers, but strategies now use pandas
        self.enable_dynamic_generation = enable_dynamic_generation
        self.initial_strategy_params = initial_strategy_params if initial_strategy_params else {}
        self.strategy_configs_list = strategy_configs_list if strategy_configs_list else []
        self.strategy_config_map = {cfg.name: cfg for cfg in self.strategy_configs_list}

        # Initialize all strategies based on the new structure
        self.base_strategies_instances = nn.ModuleList()
        
        strategy_classes_to_instantiate = [
            MomentumStrategy, BreakoutStrategy, TrendFollowingStrategy, ReversalStrategy,
            StatisticalArbitrageStrategy, VolatilityStrategy, PairsTradeStrategy, MeanReversionStrategy,
            CointegrationStrategy, VolatilityArbitrageStrategy, 
            ReinforcementLearningStrategy, DeepLearningPredictionStrategy, EnsembleLearningStrategy, TransferLearningStrategy,
            DynamicHedgingStrategy, RiskParityStrategy, VaRControlStrategy, MaxDrawdownControlStrategy,
            OptionFlowStrategy, MicrostructureStrategy, CarryTradeStrategy, MacroEconomicStrategy,
            EventDrivenStrategy, SentimentStrategy, QuantitativeStrategy, MarketMakingStrategy,
            HighFrequencyStrategy, AlgorithmicStrategy
        ]

        for SClass in strategy_classes_to_instantiate:
            strategy_name = SClass.get_strategy_name(SClass) if hasattr(SClass, 'get_strategy_name') else SClass.__name__
            params = self.initial_strategy_params.get(strategy_name, {})
            config = self.strategy_config_map.get(strategy_name)
            if not config:
                 # Create a default config if not provided - this might need more robust handling
                logger.warning(f"StrategyConfig not found for {strategy_name}, using default placeholder.")
                config = StrategyConfig(name=strategy_name, description="Default", risk_level=0.5, market_regime="all", complexity=3)

            try:
                self.base_strategies_instances.append(SClass(params=params, config=config))
            except Exception as e:
                logger.error(f"Error instantiating {strategy_name} in Superposition: {e}")
        
        self.num_base_strategies = len(self.base_strategies_instances)
        
        # 動態策略生成器
        if enable_dynamic_generation:
            # Pass available strategy classes and their configs to the generator
            generator_config = {
                'strategy_configs_list': self.strategy_configs_list,
                # Potentially other configs for GA/NAS if they are part of the generator directly
            }
            self.dynamic_generator = DynamicStrategyGenerator(
                config=generator_config, 
                strategies=None # Generator will use its available_strategies_classes
            )
            # Max generated strategies needs to be managed by the generator itself or via config
            self.max_generated_strategies = self.config.get('max_generated_strategies', 5) 
            self.total_strategies = self.num_base_strategies + self.max_generated_strategies
        else:
            self.dynamic_generator = None
            self.max_generated_strategies = 0
            self.total_strategies = self.num_base_strategies
        
        # 動態計算權重網絡的隱藏層維度 - 自適應縮放
        # These nn.Linear layers expect PyTorch tensors, but strategies now output pandas DataFrames.
        # The forward pass will need significant changes to reconcile this.
        # For now, keeping the layer definitions, but their inputs will be problematic.
        self.vol_hidden_dim = max(16, min(64, self.total_strategies))
        self.regime_hidden_dim = max(32, min(128, state_dim // 4))
        self.corr_hidden_dim = max(32, min(128, (state_dim + 1) // 4))
        
        # 量子振幅參數（可學習）
        self.quantum_amplitudes = nn.Parameter(
            torch.ones(self.total_strategies) / math.sqrt(self.total_strategies)
        )
        
        # 多層次權重調整網絡 - 動態維度配置
        self.weight_networks = nn.ModuleDict({
            'volatility_net': nn.Sequential(
                nn.Linear(1, self.vol_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.vol_hidden_dim, self.total_strategies),
                nn.Sigmoid()
            ),
            'regime_net': nn.Sequential(
                nn.Linear(state_dim, self.regime_hidden_dim),
                nn.GELU(),
                nn.Linear(self.regime_hidden_dim, self.total_strategies),
                nn.Softmax(dim=-1)
            ),
            'correlation_net': nn.Sequential(
                nn.Linear(state_dim + 1, self.corr_hidden_dim),
                nn.GELU(),
                nn.Linear(self.corr_hidden_dim, self.total_strategies),
                nn.Softmax(dim=-1)
            )
        })
        
        # 策略相互作用矩陣
        self.interaction_matrix = nn.Parameter(
            torch.eye(self.total_strategies) + torch.randn(self.total_strategies, self.total_strategies) * 0.05
        )
        
        # 策略性能追蹤
        self.register_buffer('strategy_performance_history', 
                           torch.zeros(self.total_strategies, 100))  # 記錄最近100次性能
        self.register_buffer('performance_index', torch.tensor(0))
        
        # 量子糾纏效應模擬
        self.entanglement_strength = nn.Parameter(torch.tensor(0.1))
        
        logger.info(f"🌟 初始化增強版策略疊加系統: {self.total_strategies}種策略")
        logger.info(f"📐 動態維度配置 - State: {state_dim}, Action: {action_dim}")
        logger.info(f"🔧 權重網絡隱藏層 - Vol: {self.vol_hidden_dim}, Regime: {self.regime_hidden_dim}, Corr: {self.corr_hidden_dim}")
    
    def get_dynamic_dimensions(self) -> Dict[str, int]:
        """獲取當前動態維度配置信息"""
        return {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'total_strategies': self.total_strategies,
            'vol_hidden_dim': self.vol_hidden_dim,
            'regime_hidden_dim': self.regime_hidden_dim,
            'corr_hidden_dim': self.corr_hidden_dim,
            'num_base_strategies': self.num_base_strategies,
            'dynamic_generation_enabled': self.enable_dynamic_generation
        }
    
    def _adaptive_dimension_handler(self, tensor: torch.Tensor, 
                                   expected_dim: int, 
                                   operation_name: str = "unknown") -> torch.Tensor:
        """
        動態維度適配處理器
        自動調整輸入張量以匹配期望維度
        
        Args:
            tensor: 輸入張量
            expected_dim: 期望的最後一個維度
            operation_name: 操作名稱，用於日誌
            
        Returns:
            適配後的張量
        """
        current_shape = tensor.shape
        current_last_dim = current_shape[-1]
        
        if current_last_dim == expected_dim:
            return tensor
        
        batch_dims = current_shape[:-1]
        
        if current_last_dim > expected_dim:
            # 維度過大：使用線性投影降維
            if not hasattr(self, f'_adaptive_projector_{operation_name}_{current_last_dim}_{expected_dim}'):
                projector = nn.Linear(current_last_dim, expected_dim).to(tensor.dev极)
                setattr(self, f'_adaptive_projector_{operation_name}_{current_last_dim}_{expected_dim}', projector)
                logger.info(f"🔧 創建動態投影器: {operation_name} {current_last_dim}→{expected_dim}")
            
            projector = getattr(self, f'_adaptive_projector_{operation_name}_{current_last_dim}_{expected_dim}')
            adapted_tensor = projector(tensor)
            
        elif current_last_dim < expected_dim:
            # 維度過小：使用零填充擴展
            pad_size = expected_dim - current_last_dim
            padding = torch.zeros(*batch_dims, pad_size, device=tensor.device, dtype=tensor.dtype)
            adapted_tensor = torch.cat([tensor, padding], dim=-1)
            logger.info(f"🔧 動態零填充: {operation_name} {current_last_dim}→{expected_dim}")
        
        return adapted_tensor
    
    def _validate_and_adapt_inputs(self, state: torch.Tensor, 
                                  volatility: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        驗證並適配輸入維度
        
        Args:
            state: 市場狀態張量
            volatility: 波動率張量
            
        Returns:
            適配後的狀態和波動率張量
        """
        # 適配狀態張量
        adapted_state = self._adaptive_dimension_handler(
            state, self.state_dim, "state_input"
        )
        
        # 確保波動率是一維的
        if volatility.dim() > 1 and volatility.shape[-1] != 1:
            volatility = volatility.mean(dim=-1, keepdim=True)
        
        if volatility.dim() == 1:
            volatility = volatility.unsqueeze(-1)
        
        return adapted_state, volatility.squeeze(-1)
    
    def forward(self, market_data_dict: Dict[str, pd.DataFrame], 
                portfolio_context: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        核心前向傳播邏輯，整合所有策略並計算最終行動建議
        
        Args:
            market_data_dict: 包含各個交易對市場數據的字典，鍵為交易對名稱，值為Pandas DataFrame。
                              DataFrame應包含 'close', 'volume', 'high', 'low' 等列。
            portfolio_context: 包含投資組合當前狀態和上下文信息的字典。
                               例如: {'cash': 10000, 'positions': {'EUR_USD': 100}, 
                                     'market_regime': 'trending', 'overall_volatility': 0.02,
                                     'risk_appetite': 'high'}

        Returns:
            Tuple containing:
            - final_action_distribution (torch.Tensor): 最終的行動分佈 (e.g., [batch_size, num_assets, num_actions])
                                                        or a more abstract representation.
            - strategy_weights (torch.Tensor): 各策略的計算權重 (e.g., [batch_size, total_strategies])
            - diagnostic_info (Dict[str, Any]): 包含診斷信息的字典
        """
        batch_size = 1 # Assuming batch_size of 1 for now, as market_data_dict is not batched.
                       # This needs to be generalized if batch processing is intended.

        # 0. Prepare data for strategies
        # Assuming all strategies operate on a common primary market data (e.g., a specific symbol or aggregated data)
        # This is a simplification. In reality, each strategy might need different parts of market_data_dict
        # or specific pre-processing.
        # For now, let's assume strategies can handle the raw dict or we pick one primary symbol.
        # Let's assume 'EUR_USD' is a primary symbol if available, otherwise the first one.
        primary_symbol = next(iter(market_data_dict)) if market_data_dict else None
        if not primary_symbol:
            logger.error("Market data dictionary is empty. Cannot proceed.")
            # Return dummy tensors of appropriate (though possibly incorrect) shape
            # This part needs careful consideration for robust error handling
            dummy_action = torch.zeros(batch_size, self.action_dim, device=DEVICE) 
            dummy_weights = torch.zeros(batch_size, self.total_strategies, device=DEVICE)
            return dummy_action, dummy_weights, {}

        primary_market_data_df = market_data_dict[primary_symbol]

        # 1. 執行所有基礎策略和動態生成策略
        strategy_outputs_dfs: List[pd.DataFrame] = []
        active_strategies_for_forward = list(self.base_strategies_instances)

        # Handle dynamically generated strategies if enabled
        if self.enable_dynamic_generation and self.dynamic_generator:
            # This part is conceptual. Dynamic generation might happen less frequently
            # or be triggered by specific conditions, not necessarily every forward pass.
            # For now, let's assume we might have some pre-generated dynamic strategies.
            # Or, if we want to generate one on the fly (less likely for every forward pass):
            # new_dynamic_strategy = self.dynamic_generator.generate_new_strategy(primary_market_data_df, portfolio_context)
            # if new_dynamic_strategy:
            #     active_strategies_for_forward.append(new_dynamic_strategy)
            # For simplicity, let's assume dynamic_generator.active_strategies_instances holds them
            active_strategies_for_forward.extend(self.dynamic_generator.active_strategies_instances)
        
        current_num_strategies = len(active_strategies_for_forward)
        if current_num_strategies == 0:
            logger.error("No active strategies available. Cannot proceed.")
            dummy_action = torch.zeros(batch_size, self.action_dim, device=DEVICE)
            dummy_weights = torch.zeros(batch_size, self.total_strategies, device=DEVICE) # total_strategies might be > 0
            return dummy_action, dummy_weights, {}

        for i, strategy_instance in enumerate(active_strategies_for_forward):
            try:
                # Each strategy's forward method processes market_data and returns processed_data (pd.DataFrame)
                # Then generate_signals uses this processed_data to output signals (pd.DataFrame)
                # The signals DataFrame should ideally have a consistent format, e.g., a 'signal' column.
                # For multi-asset strategies, it might have signals per asset.
                
                # Pass the full market_data_dict and portfolio_context to each strategy
                processed_data = strategy_instance.forward(market_data_dict, portfolio_context)
                signals_df = strategy_instance.generate_signals(processed_data, portfolio_context)
                
                # Ensure signals_df is not empty and has expected structure
                if signals_df is None or signals_df.empty:
                    logger.warning(f"Strategy {strategy_instance.get_strategy_name()} produced empty signals. Using zeros.")
                    # Assuming signals are for the primary_symbol and a single 'signal' column
                    # This needs to be robust: define expected output structure for strategies
                    num_timesteps = len(primary_market_data_df) # Or a fixed lookback window
                    signals_df = pd.DataFrame({'signal': np.zeros(num_timesteps)}, index=primary_market_data_df.index)

                strategy_outputs_dfs.append(signals_df)

            except Exception as e:
                logger.error(f"Error executing strategy {strategy_instance.get_strategy_name()}: {e}", exc_info=True)
                # Append a DataFrame of zeros or handle error appropriately
                num_timesteps = len(primary_market_data_df)
                error_signals_df = pd.DataFrame({'signal': np.zeros(num_timesteps)}, index=primary_market_data_df.index)
                strategy_outputs_dfs.append(error_signals_df)
        
        # Pad with zero signals if current_num_strategies < self.total_strategies (due to dynamic part)
        num_padding_strategies = self.total_strategies - current_num_strategies
        if num_padding_strategies < 0: # Should not happen if total_strategies is correctly managed
            logger.error(f"Current number of strategies ({current_num_strategies}) exceeds total_strategies ({self.total_strategies}). Truncating.")
            strategy_outputs_dfs = strategy_outputs_dfs[:self.total_strategies]
            current_num_strategies = self.total_strategies
            num_padding_strategies = 0
            
        for _ in range(num_padding_strategies):
            num_timesteps = len(primary_market_data_df)
            padding_signals_df = pd.DataFrame({'signal': np.zeros(num_timesteps)}, index=primary_market_data_df.index)
            strategy_outputs_dfs.append(padding_signals_df)

        # 2. Convert strategy outputs (List[pd.DataFrame]) to a PyTorch tensor `strategy_tensor`
        # Assumption: Each DataFrame in strategy_outputs_dfs contains a 'signal' column 
        # for the primary_symbol, representing the strategy's output (e.g., -1, 0, 1).
        # We'll take the latest signal from each strategy.
        # Shape: [batch_size, total_strategies, num_features_per_strategy]
        # For now, num_features_per_strategy = 1 (the signal itself)
        
        latest_signals = []
        for df in strategy_outputs_dfs:
            if not df.empty and 'signal' in df.columns:
                latest_signals.append(df['signal'].iloc[-1] if not df['signal'].empty else 0.0)
            else:
                logger.warning(f"Signal DataFrame is empty or missing 'signal' column. Appending 0.0.")
                latest_signals.append(0.0)
        
        # Ensure latest_signals has self.total_strategies elements
        if len(latest_signals) != self.total_strategies:
            logger.error(f"Mismatch in number of signals ({len(latest_signals)}) and total_strategies ({self.total_strategies}). This indicates an issue in padding or strategy execution.")
            # Fallback: pad or truncate latest_signals to match self.total_strategies
            if len(latest_signals) < self.total_strategies:
                latest_signals.extend([0.0] * (self.total_strategies - len(latest_signals)))
            else:
                latest_signals = latest_signals[:self.total_strategies]

        strategy_tensor = torch.tensor(latest_signals, dtype=torch.float32, device=DEVICE).unsqueeze(0) # [1, total_strategies]
        strategy_tensor = strategy_tensor.unsqueeze(-1) # [1, total_strategies, 1] (feature_dim=1)

        # 3. Derive `state_features_for_weighting` from market_data_dict and portfolio_context
        # Shape: [batch_size, self.state_dim]
        # This is a placeholder. Actual feature engineering will be more complex.
        # Example: use 'close' price of primary_symbol, and some portfolio context.
        if self.state_dim > 0:
            # Example features: latest price, moving average, portfolio cash
            # These features must match what the weighting networks were trained on or expect.
            # This part is highly dependent on the definition of `state_dim`
            
            # Placeholder: Use last N closing prices of the primary symbol, padded/truncated to fit state_dim
            # This is a very naive approach and needs proper design.
            lookback_for_state = self.state_dim 
            if primary_market_data_df is not None and 'close' in primary_market_data_df.columns:
                close_prices = primary_market_data_df['close'].values
                if len(close_prices) >= lookback_for_state:
                    state_features_raw = close_prices[-lookback_for_state:]
                else: # Pad if not enough data
                    state_features_raw = np.pad(close_prices, (lookback_for_state - len(close_prices), 0), 'edge')
            else: # Fallback if no market data
                state_features_raw = np.zeros(lookback_for_state)

            state_features_for_weighting = torch.tensor(state_features_raw, dtype=torch.float32, device=DEVICE).unsqueeze(0) # [1, state_dim]
            
            # Ensure the dimension matches self.state_dim exactly
            if state_features_for_weighting.shape[1] != self.state_dim:
                logger.warning(f"Constructed state_features_for_weighting shape {state_features_for_weighting.shape} does not match self.state_dim {self.state_dim}. Adjusting/Padding.")
                if state_features_for_weighting.shape[1] < self.state_dim:
                    padding = torch.zeros(batch_size, self.state_dim - state_features_for_weighting.shape[1], device=DEVICE)
                    state_features_for_weighting = torch.cat((state_features_for_weighting, padding), dim=1)
                else:
                    state_features_for_weighting = state_features_for_weighting[:, :self.state_dim]
        else: # No state features if state_dim is 0
             state_features_for_weighting = torch.empty(batch_size, 0, device=DEVICE)


        # 4. Derive `volatility_for_weighting` (e.g., from portfolio_context or market_data)
        # Shape: [batch_size, 1]
        # Placeholder: Use overall_volatility from portfolio_context if available, else calculate from primary_market_data
        if 'overall_volatility' in portfolio_context:
            volatility_value = float(portfolio_context['overall_volatility'])
        elif primary_market_data_df is not None and 'close' in primary_market_data_df.columns and len(primary_market_data_df['close']) > 1:
            # Calculate simple volatility (e.g., std of log returns over a short window)
            log_returns = np.log(primary_market_data_df['close'] / primary_market_data_df['close'].shift(1)).dropna()
            if len(log_returns) > 1:
                volatility_value = float(log_returns.std())
            else:
                volatility_value = 0.01 # Default small volatility
        else:
            volatility_value = 0.01 # Default small volatility
            
        volatility_for_weighting = torch.tensor([[volatility_value]], dtype=torch.float32, device=DEVICE) # [1, 1]

        # 5. 計算策略權重 (Unchanged from original logic, but inputs are now derived)
        # Ensure inputs to networks have correct batch_size if it's > 1 in the future.
        # For now, batch_size is 1.
        
        weights_vol = self.weight_networks['volatility_net'](volatility_for_weighting) # Input: [1,1], Output: [1, total_strategies]
        
        weights_regime = torch.zeros(batch_size, self.total_strategies, device=DEVICE)
        if self.state_dim > 0 :
            weights_regime = self.weight_networks['regime_net'](state_features_for_weighting) # Input: [1, state_dim], Output: [1, total_strategies]
        else: # If state_dim is 0, regime_net might not be meaningful or should be handled differently
            weights_regime = torch.ones(batch_size, self.total_strategies, device=DEVICE) / self.total_strategies # Equal weights

        # For correlation_net, input is state_features + volatility
        if self.state_dim > 0:
            corr_net_input = torch.cat([state_features_for_weighting, volatility_for_weighting], dim=1) # [1, state_dim + 1]
        else: # If state_dim is 0, only use volatility
            corr_net_input = volatility_for_weighting # [1,1] - this might require correlation_net to adapt input size or a different handling
            # Adjusting correlation_net input layer if state_dim is 0
            # This is a runtime check; ideally, network structure is fixed at init.
            # For now, let's assume correlation_net's first layer input dim was set considering this.
            # If self.weight_networks['correlation_net'][0].in_features != corr_net_input.shape[1]:
            #    logger.warning("Correlation net input dim mismatch due to state_dim=0. This might lead to errors.")
            #    # A more robust solution would be to have a separate net or logic for state_dim=0
            #    # Or ensure the Linear layer can handle it, which it can't directly if fixed.
            #    # Fallback: use regime weights or equal weights if corr_net cannot run.
            #    weights_corr = weights_regime 
            pass # The nn.Linear will throw an error if input size is wrong.
                 # This needs to be handled by ensuring state_dim + 1 (or 1 if state_dim=0) matches the Linear layer's in_features.
                 # The __init__ already sets this up based on state_dim.
                 # If state_dim is 0, then correlation_net's input is nn.Linear(1, self.corr_hidden_dim)

        weights_corr = self.weight_networks['correlation_net'](corr_net_input) # Output: [1, total_strategies]
        
        # 基礎權重組合
        base_weights = (weights_vol + weights_regime + weights_corr) / 3.0
        
        # 應用量子振幅 (歸一化)
        normalized_amplitudes = F.softmax(self.quantum_amplitudes, dim=0) # Ensure they sum to 1
        weighted_amplitudes = base_weights * normalized_amplitudes.unsqueeze(0) # [1, total_strategies]
        
        # 策略相互作用調整
        # Ensure interaction_matrix is on the correct device
        interaction_matrix_device = self.interaction_matrix.to(DEVICE)
        interacted_weights = torch.matmul(weighted_amplitudes, interaction_matrix_device) # [1, total_strategies]
        
        # 考慮策略歷史表現 (簡化版 - 可以擴展為更複雜的動態調整)
        # Placeholder: current_performance should be calculated based on recent PnL or other metrics
        # For now, let's assume a simple mechanism or skip if not fully designed.
        # Example: if strategy_performance_history has recent returns, use them.
        # This part needs more design for how performance is measured and incorporated.
        # performance_adjustment = torch.tanh(self.strategy_performance_history.mean(dim=1)).unsqueeze(0) # [1, total_strategies]
        # strategy_weights = F.softmax(interacted_weights + performance_adjustment * 0.1, dim=1) # Small adjustment
        
        strategy_weights = F.softmax(interacted_weights, dim=1) # [1, total_strategies]

        # 量子糾纏效應 (簡化版)
        # This is a conceptual step. How entanglement affects weights or actions needs clear definition.
        # Example: slightly bias weights towards a 'consensus' or average.
        # mean_weight = strategy_weights.mean(dim=1, keepdim=True)
        # entanglement_effect = (mean_weight - strategy_weights) * self.entanglement_strength
        # strategy_weights = F.softmax(strategy_weights + entanglement_effect, dim=1)

        # 6. 組合策略輸出得到最終行動
        # strategy_tensor is [batch_size, total_strategies, num_features_per_strategy]
        # strategy_weights is [batch_size, total_strategies]
        
        # Reshape weights for broadcasting: [batch_size, total_strategies, 1]
        weights_reshaped = strategy_weights.unsqueeze(-1)
        
        # Weighted sum of strategy outputs (signals)
        # final_output_features is [batch_size, num_features_per_strategy]
        final_output_features = (strategy_tensor * weights_reshaped).sum(dim=1)
        
        # Convert final_output_features to an action distribution.
        # This is highly dependent on the nature of `action_dim` and what the agent is supposed to output.
        # If action_dim represents [buy, hold, sell] probabilities for ONE asset:
        #   And final_output_features is a single value (e.g. signal strength from -1 to 1)
        #   We need a mapping layer or logic.
        # If action_dim is for multiple assets, strategy_tensor should also be multi-asset.
        
        # Placeholder: Assuming final_output_features (shape [batch_size, 1]) is a raw signal strength.
        # We need to map this to the defined self.action_dim.
        # If action_dim is, for example, 3 (buy, hold, sell probabilities for one asset):
        if self.action_dim == 3: # Example: Buy, Hold, Sell probabilities
            # This is a very simplistic mapping and needs proper design
            signal_strength = final_output_features[:, 0] # Assuming feature_dim was 1
            
            # Map signal_strength (-1 to 1) to probabilities
            buy_prob = torch.sigmoid(signal_strength * 2.0) # Scaled to make it more responsive
            sell_prob = torch.sigmoid(-signal_strength * 2.0)
            hold_prob = 1.0 - (buy_prob + sell_prob) # This can be problematic if sum > 1
            
            # A better way for probabilities that sum to 1:
            # Use softmax on logits derived from signal_strength
            # e.g., logits for buy, hold, sell
            buy_logit = signal_strength 
            sell_logit = -signal_strength
            hold_logit = torch.zeros_like(signal_strength) # Neutral for hold
            
            action_logits = torch.stack([buy_logit, hold_logit, sell_logit], dim=1) # [batch_size, 3]
            final_action_distribution = F.softmax(action_logits, dim=1) # [batch_size, 3]
        
        elif self.action_dim == 1: # Example: Direct output of the signal as action
            final_action_distribution = final_output_features # [batch_size, 1]
        else:
            # Default: if action_dim is different, create a zero tensor.
            # This part MUST be implemented according to the specific meaning of action_dim.
            logger.warning(f"Action_dim {self.action_dim} handling is not fully implemented. Returning zeros.")
            final_action_distribution = torch.zeros(batch_size, self.action_dim, device=DEVICE)

        # 7. 更新策略性能歷史 (簡化)
        # This requires a measure of actual performance post-action, which is not available here.
        # This should ideally be done in a separate update/learning step.
        # For now, we can log the signals or a proxy.
        # self.performance_index = (self.performance_index + 1) % self.strategy_performance_history.size(1)
        # self.strategy_performance_history[:, self.performance_index] = strategy_tensor.squeeze().detach() # Log raw signals

        diagnostic_info = {
            "raw_strategy_signals": strategy_tensor.squeeze(0).detach().cpu().numpy().tolist() if strategy_tensor is not None else [],
            "weights_volatility": weights_vol.squeeze(0).detach().cpu().numpy().tolist(),
            "weights_regime": weights_regime.squeeze(0).detach().cpu().numpy().tolist(),
            "weights_correlation": weights_corr.squeeze(0).detach().cpu().numpy().tolist(),
            "base_combined_weights": base_weights.squeeze(0).detach().cpu().numpy().tolist(),
            "quantum_amplitudes_normalized": normalized_amplitudes.detach().cpu().numpy().tolist(),
            "interacted_weights": interacted_weights.squeeze(0).detach().cpu().numpy().tolist(),
            "final_strategy_weights": strategy_weights.squeeze(0).detach().cpu().numpy().tolist(),
            "state_features_input": state_features_for_weighting.squeeze(0).detach().cpu().numpy().tolist() if self.state_dim > 0 else "N/A",
            "volatility_input": volatility_for_weighting.item(),
            "final_output_features_before_action_mapping": final_output_features.squeeze(0).detach().cpu().numpy().tolist()
        }
        
        return final_action_distribution, strategy_weights, diagnostic_info

    def update_strategy_performance(self, performance_scores: torch.Tensor):
        """更新策略性能歷史"""
        current_idx = self.performance_index.item() % 100
        self.strategy_performance_history[:, current_idx] = performance_scores.mean(dim=0)
        self.performance_index += 1
        
        # 更新動態策略生成器
        if self.enable_dynamic_generation:
            dynamic_performance = performance_scores[:, self.num_base_strategies:]
            if dynamic_performance.numel() > 0:
                self.dynamic_generator.evolve_strategies(dynamic_performance.mean(dim=0))
    
    def get_strategy_analysis(self) -> Dict[str, Any]:
        """獲取策略分析信息"""
        recent_performance = self.strategy_performance_history[:, :min(100, self.performance_index.item())]
        
        return {
            'num_strategies': len(self.base_strategies_instances) + (5 if self.enable_dynamic_generation else 0),
            'avg_performance': recent_performance.mean(dim=-1),
            'performance_std': recent_performance.std(dim=-1),
            'best_strategy_idx': recent_performance.mean(dim=-1).argmax().item(),
            'worst_strategy_idx': recent_performance.mean(dim=-1).argmin().item(),
            'strategy_names': [s.get_strategy_name() for s in self.base_strategies_instances] + 
                            ([f"Dynamic_{i}" for i in range(5)] if self.enable_dynamic_generation else []),
            'quantum_amplitude_distribution': F.softmax(self.quantum_amplitudes, dim=0),
            'entanglement_strength': self.entanglement_strength.item(),
        }


if __name__ == "__main__":
    # 測試增強版量子策略層
    logger.info("開始測試增強版量子策略層...")
    
    # 測試參數
    batch_size = 8
    state_dim = 64
    action_dim = 10
    
    # 創建測試數據
    test_state = torch.randn(batch_size, state_dim)
    test_volatility = torch.rand(batch_size) * 0.5
    
    # 初始化增強版策略疊加系統
    enhanced_strategy_layer = EnhancedStrategySuperposition(
        state_dim=state_dim,
        action_dim=action_dim,
        enable_dynamic_generation=True
    )
    
    try:
        # 前向傳播測試
        with torch.no_grad():
            output, info = enhanced_strategy_layer(test_state, test_volatility)
            
        logger.info(f"測試成功！")
        logger.info(f"輸入狀態形狀: {test_state.shape}")
        logger.info(f"輸出動作形狀: {output.shape}")
        logger.info(f"策略權重形狀: {info['strategy_weights'].shape}")
        logger.info(f"活躍策略數量: {info['num_active_strategies'].mean():.2f}")
        
        # 獲取策略分析
        analysis = enhanced_strategy_layer.get_strategy_analysis()
        logger.info(f"策略總數: {analysis['num_strategies']}")
        logger.info(f"策略名稱: {analysis['strategy_names']}")
        
        # 測試梯度計算
        enhanced_strategy_layer.train()
        output, info = enhanced_strategy_layer(test_state, test_volatility)
        loss = output.abs().mean()
        loss.backward()
        
        logger.info("梯度計算測試通過")
        logger.info(f"總參數量: {sum(p.numel() for p in enhanced_strategy_layer.parameters()):,}")
        
    except Exception as e:
        logger.error(f"測試失敗: {e}")
        raise e

    # ==============================================
    # 測試動態策略生成器的遺傳算法功能
    # ==============================================
    logger.info("\n開始測試動態策略生成器的遺傳算法功能...")
    
    # 加載真實數據
    try:
        import pandas as pd
        # 加載EUR/USD 5秒數據
        data_path = "data/EUR_USD_5S_20250601.csv"
        df = pd.read_csv(data_path)
        logger.info(f"成功加載數據: {data_path}, 形狀: {df.shape}")
        
        # 數據預處理
        # 使用收盤價作為狀態特徵
        prices = df['close'].values
        # 計算波動率
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns)
        
        # 轉換為PyTorch張量
        state_tensor = torch.tensor(prices[-state_dim:], dtype=torch.float32).unsqueeze(0)
        volatility_tensor = torch.tensor([volatility], dtype=torch.float32)
        
        # 初始化動態策略生成器
        generator = DynamicStrategyGenerator(
            state_dim=state_dim,
            action_dim=action_dim,
            base_strategies=enhanced_strategy_layer.base_strategies,
            max_generated_strategies=5
        )
        
        # 測試策略生成
        strategies, fitness_scores = generator.generate_strategies(state_tensor)
        logger.info(f"生成策略數量: {len(strategies)}")
        logger.info(f"策略輸出形狀: {strategies[0].shape}")
        logger.info(f"適應度分數形狀: {fitness_scores.shape}")
        
        # 測試遺傳算法操作
        logger.info("\n測試遺傳算法操作:")
        logger.info("1. 測試輪盤賭選擇...")
        selected_idx = generator.roulette_wheel_selection(fitness_scores)
        logger.info(f"選擇的索引: {selected_idx.item()}")
        
        logger.info("2. 測試單點交叉...")
        parent1 = torch.randn(1, generator.gene_latent_dim)
        parent2 = torch.randn(1, generator.gene_latent_dim)
        child1, child2 = generator.single_point_crossover(parent1, parent2)
        logger.info(f"父代1形狀: {parent1.shape}, 父代2形狀: {parent2.shape}")
        logger.info(f"子代1形狀: {child1.shape}, 子代2形狀: {child2.shape}")
        
        logger.info("3. 測試高斯變異...")
        genes = torch.randn(1, generator.gene_latent_dim)
        mutated_genes = generator.gaussian_mutation(genes)
        logger.info(f"變異前: {genes.mean().item():.4f}, 變異後: {mutated_genes.mean().item():.4f}")
        
        logger.info("✅ 動態策略生成器測試通過")
        
    except Exception as e:
        logger.error(f"動態策略生成器測試失敗: {e}")
        raise e

# Placeholder for GeneticOptimizer and NeuralArchitectureSearch if they are to be defined in this file
# class GeneticOptimizer:
#     def __init__(self, ...):
#         pass
#     def evolve_strategy(self, ...):
#         pass

# class NeuralArchitectureSearch:
#     def __init__(self, ...):
#         pass
#     def search_architecture(self, ...):
#         pass
