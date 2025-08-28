import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium.spaces import Dict as GymDict
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs

from oanda_trading_bot.training_system.agent.enhanced_feature_extractor import EnhancedTransformerFeatureExtractor
from oanda_trading_bot.training_system.strategies.strategy_pool_manager import StrategyPoolManager
from oanda_trading_bot.training_system.common.config import MAX_SYMBOLS_ALLOWED, TIMESTEPS # 假設需要

class CombinedFeatureExtractor(BaseFeaturesExtractor):
    """
    組合特徵提取器。
    它結合了 EnhancedTransformerFeatureExtractor 的輸出和 StrategyPoolManager 的輸出。
    """
    def __init__(self, 
                 observation_space: GymDict, # 來自環境的原始 Dict 觀察空間
                 enhanced_transformer_config: Dict[str, Any],
                 num_features_per_symbol: int,
                 strategy_pool_config: Dict[str, Any],
                 max_symbols_allowed: int = MAX_SYMBOLS_ALLOWED,
                 timesteps_history: int = TIMESTEPS,
                 device: str = "cpu"):
        
        # 實例化 EnhancedTransformerFeatureExtractor
        self.transformer_feature_extractor = EnhancedTransformerFeatureExtractor(
            observation_space=observation_space,
            model_config=enhanced_transformer_config
        )
        
        # 實例化 StrategyPoolManager
        # StrategyPoolManager 的 input_dim 應該是 EnhancedTransformerFeatureExtractor 的 features_dim
        # 並且它需要知道原始特徵的維度
        self.strategy_pool_manager = StrategyPoolManager(
            # input_dim 這裡需要是市場狀態特徵的維度，即 Transformer 的輸出維度
            input_dim=self.transformer_feature_extractor.features_dim,
            num_strategies=strategy_pool_config.get('num_strategies', 15),
            strategy_configs=strategy_pool_config.get('strategy_configs'),
            strategy_config_file_path=strategy_pool_config.get('strategy_config_file_path'),
            use_gumbel_softmax=strategy_pool_config.get('use_gumbel_softmax', True),
            dropout_rate=strategy_pool_config.get('dropout_rate', 0.1),
            # 傳遞原始特徵的維度給 StrategyPoolManager，以便它可以將正確的特徵傳遞給底層策略
            raw_feature_dim=num_features_per_symbol, # 新增參數
            timesteps_history=timesteps_history # 新增參數
        ).to(device) # 確保移到正確的設備

        # 計算最終的 features_dim
        # 來自 TransformerFeatureExtractor 的輸出
        transformer_output_dim = self.transformer_feature_extractor.features_dim
        
        # 來自 StrategyPoolManager 的輸出 (每個 symbol 一個信號，扁平化後)
        # StrategyPoolManager 將輸出 (batch_size, num_symbols * num_actual_strategies)
        # 或者 (batch_size, num_symbols, num_actual_strategies)
        # 如果是 (batch_size, num_symbols, num_actual_strategies)，則扁平化為 (batch_size, num_symbols * num_actual_strategies)
        # 這裡假設 StrategyPoolManager 的輸出是 num_symbols * num_actual_strategies
        # 我們需要從 StrategyPoolManager 中獲取實際的策略數量
        # 但在 __init__ 階段，self.strategy_pool_manager 尚未完全初始化，num_actual_strategies 還未知。
        # 因此，我們需要一個估計值或者在 forward 中動態計算。
        # 暫時使用 max_symbols_allowed * strategy_pool_config.get('num_strategies', 15) 作為估計。
        estimated_strategy_pool_output_dim = max_symbols_allowed * self.strategy_pool_manager.target_num_strategies
        
        final_features_dim = transformer_output_dim + estimated_strategy_pool_output_dim
        
        super().__init__(observation_space, features_dim=final_features_dim)
        
        self.max_symbols_allowed = max_symbols_allowed
        self.timesteps_history = timesteps_history
        self.device = device

    def forward(self, observations: PyTorchObs) -> torch.Tensor:
        # 1. 獲取 EnhancedTransformerFeatureExtractor 的輸出 (高維特徵)
        # EnhancedTransformerFeatureExtractor 期望 Dict 觀察空間
        transformed_features = self.transformer_feature_extractor(observations) # 形狀: (batch_size, transformer_output_dim)

        # 2. 準備原始特徵
        # observations['features_from_dataset'] 的形狀是 (batch_size, num_symbols, timesteps_history, num_features_per_symbol)
        raw_features_batch = observations['features_from_dataset']
        
        # 3. 獲取 StrategyPoolManager 的輸出 (策略信號)
        # StrategyPoolManager 期望 asset_features_batch (原始特徵)
        # 和 market_state_features (高維特徵)
        strategy_signals = self.strategy_pool_manager(
            raw_features_batch=raw_features_batch, # 傳遞原始特徵
            market_state_features=transformed_features, # 傳遞高維特徵 (來自 Transformer)
            current_positions_batch=observations.get('current_positions_nominal_ratio_ac'), # 從環境觀察中獲取持倉
            timestamps=observations.get('time_utc_iso') # 從環境觀察中獲取 timestamps
        ) # 形狀: (batch_size, num_symbols * num_actual_strategies) - 假設 StrategyPoolManager 內部已扁平化

        # 4. 拼接兩種特徵
        # transformed_features 形狀: (batch_size, transformer_output_dim)
        # strategy_signals 形狀: (batch_size, num_symbols * num_actual_strategies)
        
        # 由於 StrategyPoolManager 的輸出現在是扁平化的觀察狀態一部分，
        # 我們直接將其與 transformed_features 拼接。
        combined_output = torch.cat([transformed_features, strategy_signals], dim=1) # 形狀: (batch_size, final_features_dim)
        
        return combined_output
