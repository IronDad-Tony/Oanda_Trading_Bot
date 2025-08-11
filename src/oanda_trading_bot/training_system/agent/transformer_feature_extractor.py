# filepath: src/agent/transformer_feature_extractor.py
"""
自定義的FeaturesExtractor，使用UniversalTradingTransformer從觀測字典提取時間序列特徵。
"""
import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from oanda_trading_bot.training_system.models.transformer_model import UniversalTradingTransformer
from oanda_trading_bot.training_system.common.config import TRANSFORMER_OUTPUT_DIM_PER_SYMBOL

class TransformerFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self,
                 observation_space: gym.spaces.Dict,
                 features_key: str = "features_from_dataset",
                 mask_key: str = "padding_mask",
                 transformer_kwargs: dict = None):
        # 讀取原始觀測空間形狀
        feature_space = observation_space.spaces[features_key]
        num_slots, timesteps, feat_dim = feature_space.shape
        # 初始化BaseFeaturesExtractor，暫時features_dim占位1
        super().__init__(observation_space, features_dim=1)
        # 構造Transformer特徵提取器
        tf_kwargs = transformer_kwargs or {}
        # 強制指定輸入維度和槽位數
        tf_kwargs.update({
            'num_input_features': feat_dim,
            'num_symbols_possible': num_slots,
        })
        self.transformer = UniversalTradingTransformer(**tf_kwargs)        # 計算總輸出維度：每個symbol輸出維度 * 符號數
        self._features_dim = num_slots * TRANSFORMER_OUTPUT_DIM_PER_SYMBOL
        self.features_key = features_key
        self.mask_key = mask_key

    def forward(self, observations: dict) -> torch.Tensor:
        # observations 是一個 dict，鍵對應Env.observation_space.keys()
        x = observations[self.features_key]  # Tensor shape (B,S,T,F)
        mask = observations[self.mask_key]   # Tensor shape (B,S)
        # padding_mask傳入cross-asset注意力: True表示要mask
        # 確保mask是布林類型，然後進行邏輯反轉
        if mask.dtype != torch.bool:
            mask = mask.bool()
        # 我們用邏輯反轉 (~mask) 表示dummy slots需要被mask
        out = self.transformer(x, padding_mask_symbols=~mask)
        # out shape (B,S,D)
        # 展平成 (B, S*D)
        batch_size = out.shape[0]
        return out.view(batch_size, -1)
