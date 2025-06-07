# src/agent/enhanced_feature_extractor.py
"""
增強版特徵提取器 - Phase 3
使用增強版UniversalTradingTransformer進行特徵提取
支持多尺度特徵提取、自適應注意力機制和跨時間尺度融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict
import numpy as np

try:
    from src.models.enhanced_transformer import EnhancedUniversalTradingTransformer
    from src.common.config import (
        MAX_SYMBOLS_ALLOWED,
        TRANSFORMER_OUTPUT_DIM_PER_SYMBOL, 
        TRANSFORMER_MODEL_DIM,
        TRANSFORMER_NUM_LAYERS,
        TRANSFORMER_NUM_HEADS,
        TRANSFORMER_FFN_DIM,
        TRANSFORMER_DROPOUT_RATE,
        ENHANCED_TRANSFORMER_USE_MULTI_SCALE,
        ENHANCED_TRANSFORMER_USE_CROSS_TIME_FUSION,
        DEVICE
    )
    from src.common.logger_setup import logger
except ImportError as e:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    try:
        from src.models.enhanced_transformer import EnhancedUniversalTradingTransformer
        from src.common.config import (
            MAX_SYMBOLS_ALLOWED,
            TRANSFORMER_OUTPUT_DIM_PER_SYMBOL, 
            TRANSFORMER_MODEL_DIM,
            TRANSFORMER_NUM_LAYERS,
            TRANSFORMER_NUM_HEADS,
            TRANSFORMER_FFN_DIM,
            TRANSFORMER_DROPOUT_RATE,
            ENHANCED_TRANSFORMER_USE_MULTI_SCALE,
            ENHANCED_TRANSFORMER_USE_CROSS_TIME_FUSION,
            DEVICE
        )
        from src.common.logger_setup import logger
    except ImportError as e2:
        import logging
        logger = logging.getLogger("enhanced_feature_extractor_fallback")
        logger.error(f"導入錯誤: {e2}")
        # 設置默認值
        MAX_SYMBOLS_ALLOWED = 20
        TRANSFORMER_OUTPUT_DIM_PER_SYMBOL = 128
        TRANSFORMER_MODEL_DIM = 512
        TRANSFORMER_NUM_LAYERS = 12
        TRANSFORMER_NUM_HEADS = 16
        TRANSFORMER_FFN_DIM = 2048
        TRANSFORMER_DROPOUT_RATE = 0.1
        ENHANCED_TRANSFORMER_USE_MULTI_SCALE = True
        ENHANCED_TRANSFORMER_USE_CROSS_TIME_FUSION = True
        DEVICE = "cpu"
        
        # 創建一個空的增強版Transformer類以避免錯誤
        class EnhancedUniversalTradingTransformer:
            def __init__(self, *args, **kwargs):
                raise ImportError("EnhancedUniversalTradingTransformer not available")


class EnhancedTransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    增強版Transformer特徵提取器
    基於EnhancedUniversalTradingTransformer的高級特徵提取
    """
    
    def __init__(self, observation_space: spaces.Dict, 
                 enhanced_transformer_output_dim_per_symbol: int = TRANSFORMER_OUTPUT_DIM_PER_SYMBOL):
        """
        初始化增強版特徵提取器
        
        參數:
            observation_space: 觀察空間 (必須是gymnasium.spaces.Dict)
            enhanced_transformer_output_dim_per_symbol: 每個交易對的增強版Transformer輸出維度
        """
        # 計算總特徵維度
        total_features = (MAX_SYMBOLS_ALLOWED * enhanced_transformer_output_dim_per_symbol) + \
                         (MAX_SYMBOLS_ALLOWED * 3) + 1  # 位置+盈虧+保證金+遮罩
        
        super().__init__(observation_space, total_features)
          # 處理設備設置 - 統一使用GPU
        device = torch.device(DEVICE)  # 直接使用配置中的DEVICE
        
        logger.info(f"增強版特徵提取器設備設置: {device}")
        if device.type == 'cuda':
            logger.info(f"- GPU名稱: {torch.cuda.get_device_name(device)}")
            logger.info(f"- GPU記憶體: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f}GB")
            
        # 初始化增強版Transformer模型
        self.enhanced_transformer = EnhancedUniversalTradingTransformer(
            num_input_features=observation_space.spaces["features_from_dataset"].shape[2],
            num_symbols_possible=MAX_SYMBOLS_ALLOWED,
            model_dim=TRANSFORMER_MODEL_DIM,
            num_layers=TRANSFORMER_NUM_LAYERS,
            num_heads=TRANSFORMER_NUM_HEADS,
            ffn_dim=TRANSFORMER_FFN_DIM,
            dropout_rate=TRANSFORMER_DROPOUT_RATE,
            max_seq_len=observation_space.spaces["features_from_dataset"].shape[1],
            output_dim_per_symbol=enhanced_transformer_output_dim_per_symbol,
            use_multi_scale=ENHANCED_TRANSFORMER_USE_MULTI_SCALE,
            use_cross_time_fusion=ENHANCED_TRANSFORMER_USE_CROSS_TIME_FUSION
        ).to(device)
        
        logger.info(f"增強版特徵提取器初始化完成:")
        logger.info(f"- 輸入維度: {observation_space.spaces['features_from_dataset'].shape}")
        logger.info(f"- 輸出維度: {total_features}")
        logger.info(f"- Transformer模型維度: {TRANSFORMER_MODEL_DIM}")
        logger.info(f"- Transformer層數: {TRANSFORMER_NUM_LAYERS}")
        logger.info(f"- 注意力頭數: {TRANSFORMER_NUM_HEADS}")
        logger.info(f"- 多尺度特徵提取: {ENHANCED_TRANSFORMER_USE_MULTI_SCALE}")
        logger.info(f"- 跨時間尺度融合: {ENHANCED_TRANSFORMER_USE_CROSS_TIME_FUSION}")
        
        # 顯示模型參數量
        total_params = sum(p.numel() for p in self.enhanced_transformer.parameters())
        logger.info(f"- 總參數量: {total_params:,}")
        
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
        
        # 確保padding_mask是布爾類型
        if padding_mask.dtype != torch.bool:
            padding_mask = padding_mask.bool()
        
        # 使用增強版Transformer提取特徵
        # padding_mask傳入時需要邏輯反轉，因為True表示要mask的位置
        enhanced_transformer_output = self.enhanced_transformer(features, ~padding_mask)
        
        # 展平增強版Transformer輸出
        batch_size = features.shape[0]
        flat_enhanced_transformer = enhanced_transformer_output.reshape(batch_size, -1)
        
        # 拼接其他特徵
        other_features = torch.cat([
            observations["current_positions"],
            observations["unrealized_pnl"],
            observations["margin_level"],
            padding_mask.float()
        ], dim=1)
        
        # 合併所有特徵
        combined = torch.cat([flat_enhanced_transformer, other_features], dim=1)
        
        # 添加特徵標準化以提高訓練穩定性
        combined = F.layer_norm(combined, combined.shape[1:])
        
        return combined


class EnhancedTransformerFeatureExtractorWithMemory(EnhancedTransformerFeatureExtractor):
    """
    帶記憶機制的增強版特徵提取器
    支持跨episode的長期記憶和短期記憶融合
    """
    
    def __init__(self, observation_space: spaces.Dict, 
                 enhanced_transformer_output_dim_per_symbol: int = TRANSFORMER_OUTPUT_DIM_PER_SYMBOL,
                 memory_size: int = 32):
        super().__init__(observation_space, enhanced_transformer_output_dim_per_symbol)
        
        self.memory_size = memory_size
        self.feature_dim = MAX_SYMBOLS_ALLOWED * enhanced_transformer_output_dim_per_symbol
        
        # 記憶存儲
        self.register_buffer('long_term_memory', 
                           torch.zeros(memory_size, self.feature_dim))
        self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long))
        
        # 記憶融合網絡
        self.memory_fusion = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.GELU(),
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(0.1),
            nn.Linear(self.feature_dim, self.feature_dim)
        )
        
        # 記憶注意力
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=8,
            batch_first=True
        )
        
        logger.info(f"初始化帶記憶機制的增強版特徵提取器，記憶大小: {memory_size}")
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """增強版前向傳播，包含記憶機制"""
        # 基礎特徵提取
        base_features = super().forward(observations)
        
        # 提取Transformer特徵部分
        batch_size = base_features.shape[0]
        transformer_features = base_features[:, :self.feature_dim]
        other_features = base_features[:, self.feature_dim:]
        
        # 記憶增強
        if self.training:
            # 訓練時更新記憶
            self._update_memory(transformer_features.detach())
        
        # 記憶檢索和融合
        memory_enhanced_features = self._retrieve_and_fuse_memory(transformer_features)
        
        # 重新組合特徵
        enhanced_combined = torch.cat([memory_enhanced_features, other_features], dim=1)
        
        return enhanced_combined
    
    def _update_memory(self, features: torch.Tensor):
        """更新長期記憶"""
        batch_size = features.shape[0]
        
        for i in range(batch_size):
            ptr = int(self.memory_ptr.item())
            self.long_term_memory[ptr] = features[i]
            self.memory_ptr[0] = (ptr + 1) % self.memory_size
    
    def _retrieve_and_fuse_memory(self, current_features: torch.Tensor) -> torch.Tensor:
        """檢索和融合記憶"""
        batch_size = current_features.shape[0]
        
        # 與記憶進行注意力交互
        memory_attended, _ = self.memory_attention(
            current_features.unsqueeze(1),  # [batch, 1, feature_dim]
            self.long_term_memory.unsqueeze(0).expand(batch_size, -1, -1),  # [batch, memory_size, feature_dim]
            self.long_term_memory.unsqueeze(0).expand(batch_size, -1, -1)
        )
        
        memory_attended = memory_attended.squeeze(1)  # [batch, feature_dim]
        
        # 記憶融合
        concatenated = torch.cat([current_features, memory_attended], dim=-1)
        fused_features = self.memory_fusion(concatenated)
        
        return fused_features + current_features  # 殘差連接


if __name__ == "__main__":
    # 測試增強版特徵提取器
    logger.info("開始測試增強版特徵提取器...")
    
    # 創建模擬觀察空間
    obs_space = spaces.Dict({
        'features_from_dataset': spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(MAX_SYMBOLS_ALLOWED, 128, 9), dtype=np.float32
        ),
        'current_positions': spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(MAX_SYMBOLS_ALLOWED,), dtype=np.float32
        ),
        'unrealized_pnl': spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(MAX_SYMBOLS_ALLOWED,), dtype=np.float32
        ),
        'margin_level': spaces.Box(
            low=0, high=np.inf, 
            shape=(1,), dtype=np.float32
        ),
        'padding_mask': spaces.Box(
            low=0, high=1, 
            shape=(MAX_SYMBOLS_ALLOWED,), dtype=np.bool_
        )
    })
    
    # 創建測試數據
    test_batch_size = 4
    test_obs = {
        'features_from_dataset': torch.randn(test_batch_size, MAX_SYMBOLS_ALLOWED, 128, 9),
        'current_positions': torch.randn(test_batch_size, MAX_SYMBOLS_ALLOWED),
        'unrealized_pnl': torch.randn(test_batch_size, MAX_SYMBOLS_ALLOWED),
        'margin_level': torch.randn(test_batch_size, 1),
        'padding_mask': torch.zeros(test_batch_size, MAX_SYMBOLS_ALLOWED, dtype=torch.bool)
    }
    
    # 測試基礎增強版特徵提取器
    try:
        extractor = EnhancedTransformerFeatureExtractor(obs_space)
        
        with torch.no_grad():
            features = extractor(test_obs)
            
        logger.info(f"基礎增強版特徵提取器測試成功！")
        logger.info(f"輸入形狀: {test_obs['features_from_dataset'].shape}")
        logger.info(f"輸出形狀: {features.shape}")
        logger.info(f"特徵維度: {extractor.features_dim}")
        
    except Exception as e:
        logger.error(f"基礎增強版特徵提取器測試失敗: {e}")
    
    # 測試帶記憶機制的增強版特徵提取器
    try:
        memory_extractor = EnhancedTransformerFeatureExtractorWithMemory(obs_space)
        
        with torch.no_grad():
            memory_features = memory_extractor(test_obs)
            
        logger.info(f"帶記憶機制的增強版特徵提取器測試成功！")
        logger.info(f"記憶增強特徵形狀: {memory_features.shape}")
        
    except Exception as e:
        logger.error(f"帶記憶機制的增強版特徵提取器測試失敗: {e}")
    
    logger.info("增強版特徵提取器測試完成！")
