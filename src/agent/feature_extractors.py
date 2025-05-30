# src/agent/feature_extractors.py
"""
為 Stable Baselines3 定義自定義特徵提取器。
這裡我們將包裝 UniversalTradingTransformer 模型。
"""
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict
import numpy as np

try:
    from src.models.transformer_model import UniversalTradingTransformer
    from src.common.config import (
        MAX_SYMBOLS_ALLOWED,
        TRANSFORMER_OUTPUT_DIM_PER_SYMBOL, 
        DEVICE
    )
    from src.common.logger_setup import logger
except ImportError as e:
    import logging
    logger = logging.getLogger("feature_extractor_fallback")
    logger.error(f"導入錯誤: {e}")

class AdvancedTransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    基於UniversalTradingTransformer的特徵提取器
    完全符合MAX_SYMBOLS_ALLOWED的通用模型設計
    """
    
    def __init__(self, observation_space: spaces.Dict, 
                 transformer_output_dim_per_symbol: int = TRANSFORMER_OUTPUT_DIM_PER_SYMBOL):
        """
        初始化特徵提取器
        
        參數:
            observation_space: 觀察空間 (必須是gymnasium.spaces.Dict)
            transformer_output_dim_per_symbol: 每個交易對的Transformer輸出維度
        """
        # 計算總特徵維度
        total_features = (MAX_SYMBOLS_ALLOWED * transformer_output_dim_per_symbol) + \
                         (MAX_SYMBOLS_ALLOWED * 3) + 1  # 位置+盈虧+保證金+遮罩
        
        super().__init__(observation_space, total_features)
        
        # 處理設備設置
        if isinstance(DEVICE, str) and DEVICE == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(DEVICE)
            
        # 初始化Transformer模型
        self.transformer = UniversalTradingTransformer(
            num_input_features=observation_space.spaces["features_from_dataset"].shape[2],
            output_dim_per_symbol=transformer_output_dim_per_symbol
        ).to(device)
        
        logger.info(f"特徵提取器初始化完成: 輸入維度={observation_space.spaces['features_from_dataset'].shape} 輸出維度={total_features}")

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向傳播
        
        參數:
            observations: 包含以下鍵的字典:
                - "features_from_dataset": 形狀 [batch_size, MAX_SYMBOLS_ALLOWED, timesteps, features]
                - "current_positions": 形狀 [batch_size, MAX_SYMBOLS_ALLOWED]
                - "unrealized_pnl": 形狀 [batch_size, MAX_SYMBOLS_ALLOWED]
                - "margin_level": 形狀 [batch_size, 1]
                - "padding_mask": 形狀 [batch_size, MAX_SYMBOLS_ALLOWED]
                
        返回:
            提取的特徵張量 [batch_size, total_features]
        """
        # 處理時間序列特徵
        features = observations["features_from_dataset"]
        padding_mask = observations["padding_mask"]
        transformer_output = self.transformer(features, padding_mask)
        
        # 展平Transformer輸出
        batch_size = features.shape[0]
        flat_transformer = transformer_output.reshape(batch_size, -1)
        
        # 拼接其他特徵
        other_features = torch.cat([
            observations["current_positions"],
            observations["unrealized_pnl"],
            observations["margin_level"],
            padding_mask.float()
        ], dim=1)
        
        # 合併所有特徵
        combined = torch.cat([flat_transformer, other_features], dim=1)
        return combined