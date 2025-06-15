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
from typing import Dict, Any, Optional
import numpy as np

try:
    from src.models.enhanced_transformer import EnhancedTransformer as EnhancedUniversalTradingTransformer
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
        from src.models.enhanced_transformer import EnhancedTransformer as EnhancedUniversalTradingTransformer
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
                 model_config: Optional[Dict[str, Any]] = None, # ADDED: model_config argument
                 enhanced_transformer_output_dim_per_symbol: int = TRANSFORMER_OUTPUT_DIM_PER_SYMBOL):
        """
        初始化增強版特徵提取器
        
        參數:
            observation_space: 觀察空間 (必須是gymnasium.spaces.Dict)
            model_config: 可選的模型配置字典，用於覆蓋默認的Transformer參數
            enhanced_transformer_output_dim_per_symbol: 每個交易對的增強版Transformer輸出維度
        """
        # 計算總特徵維度
        # This uses enhanced_transformer_output_dim_per_symbol,
        # which is managed separately from the internal model_config's hidden_dim, num_layers etc.
        # The EnhancedUniversalTradingTransformer will use the passed output_dim_per_symbol.
        total_features = (MAX_SYMBOLS_ALLOWED * enhanced_transformer_output_dim_per_symbol) + \
                         (MAX_SYMBOLS_ALLOWED * 3) + 1  # 位置+盈虧+保證金+遮罩
        
        super().__init__(observation_space, total_features)
          # 處理設備設置 - 統一使用GPU
        device = torch.device(DEVICE)  # 直接使用配置中的DEVICE
        
        logger.info(f"增強版特徵提取器設備設置: {device}")
        if device.type == 'cuda':
            logger.info(f"- GPU名稱: {torch.cuda.get_device_name(device)}")
            logger.info(f"- GPU記憶體: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f}GB")        # 從model_config中提取配置值，如果沒有則使用默認值
        d_model = model_config.get('hidden_dim', TRANSFORMER_MODEL_DIM) if model_config else TRANSFORMER_MODEL_DIM
        num_encoder_layers = model_config.get('num_layers', TRANSFORMER_NUM_LAYERS) if model_config else TRANSFORMER_NUM_LAYERS
        transformer_nhead = model_config.get('num_heads', TRANSFORMER_NUM_HEADS) if model_config else TRANSFORMER_NUM_HEADS
        dim_feedforward = model_config.get('intermediate_dim', TRANSFORMER_FFN_DIM) if model_config else TRANSFORMER_FFN_DIM
        dropout = model_config.get('dropout_rate', TRANSFORMER_DROPOUT_RATE) if model_config else TRANSFORMER_DROPOUT_RATE
        max_seq_len = model_config.get('max_sequence_length', observation_space.spaces["features_from_dataset"].shape[1]) if model_config else observation_space.spaces["features_from_dataset"].shape[1]
        
        # 初始化增強版Transformer模型
        # 添加調試日誌
        logger.info(f"正在初始化 EnhancedTransformer，參數:")
        logger.info(f"  - use_cts_fusion: {ENHANCED_TRANSFORMER_USE_CROSS_TIME_FUSION}")
        logger.info(f"  - use_msfe: {ENHANCED_TRANSFORMER_USE_MULTI_SCALE}")
        logger.info(f"  - d_model: {d_model}")
        logger.info(f"  - num_encoder_layers: {num_encoder_layers}")
        
        self.enhanced_transformer = EnhancedUniversalTradingTransformer(
            input_dim=observation_space.spaces["features_from_dataset"].shape[2],
            d_model=d_model,
            transformer_nhead=transformer_nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=max_seq_len,
            num_symbols=MAX_SYMBOLS_ALLOWED,
            output_dim=enhanced_transformer_output_dim_per_symbol,
            use_msfe=ENHANCED_TRANSFORMER_USE_MULTI_SCALE,
            use_cts_fusion=ENHANCED_TRANSFORMER_USE_CROSS_TIME_FUSION
        ).to(device)
          # Get the actual config used by the transformer for logging
        try:
            transformer_actual_config = self.enhanced_transformer.get_dynamic_config()
            if transformer_actual_config is None:
                transformer_actual_config = {}
        except Exception as e:
            logger.warning(f"無法獲取Transformer配置: {e}")
            transformer_actual_config = {}

        logger.info(f"增強版特徵提取器初始化完成:")
        logger.info(f"- 輸入維度 (features_from_dataset): {observation_space.spaces['features_from_dataset'].shape}")
        logger.info(f"- 計算出的總輸出特徵維度 (total_features for SB3): {self._features_dim}") # Use self._features_dim for the value passed to super
        logger.info(f"--- Transformer 內部實際配置 ---")
        logger.info(f"- Transformer實際模型維度 (d_model): {transformer_actual_config.get('d_model', 'Unknown')}")
        logger.info(f"- Transformer實際層數 (num_encoder_layers): {transformer_actual_config.get('num_encoder_layers', 'Unknown')}")
        logger.info(f"- Transformer實際注意力頭數 (transformer_nhead): {transformer_actual_config.get('transformer_nhead', 'Unknown')}")
        logger.info(f"- Transformer實際FFN維度 (dim_feedforward): {transformer_actual_config.get('dim_feedforward', 'Unknown')}")
        logger.info(f"- Transformer實際輸出維度 (output_dim): {transformer_actual_config.get('output_dim', 'Unknown')}")
        logger.info(f"- 多尺度特徵提取 (use_msfe): {transformer_actual_config.get('use_msfe', 'Unknown')}")
        logger.info(f"- 跨時間尺度融合 (use_cts_fusion): {transformer_actual_config.get('use_cts_fusion', 'Unknown')}")
        logger.info(f"- Symbol嵌入 (use_symbol_embedding): {transformer_actual_config.get('use_symbol_embedding', 'Unknown')}")
        
        # 如果配置中有額外信息，也顯示出來
        if transformer_actual_config:
            additional_info = {k: v for k, v in transformer_actual_config.items() 
                             if k not in ['d_model', 'num_encoder_layers', 'transformer_nhead', 
                                        'dim_feedforward', 'output_dim', 'use_msfe', 'use_cts_fusion', 'use_symbol_embedding']}
            if additional_info:
                logger.info(f"- 其他配置: {additional_info}")
        
        # 直接從Transformer對象檢查屬性
        logger.info(f"--- 直接檢查Transformer屬性 ---")
        if hasattr(self.enhanced_transformer, 'd_model'):
            logger.info(f"- d_model屬性: {self.enhanced_transformer.d_model}")
        if hasattr(self.enhanced_transformer, 'transformer_nhead'):
            logger.info(f"- transformer_nhead屬性: {self.enhanced_transformer.transformer_nhead}")
        if hasattr(self.enhanced_transformer, 'output_dim'):
            logger.info(f"- output_dim屬性: {self.enhanced_transformer.output_dim}")# 顯示模型參數量
        total_params = sum(p.numel() for p in self.enhanced_transformer.parameters())
        logger.info(f"- 總參數量: {total_params:,}")
        
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向傳播 - 正確處理通用模型的 padding 和 masking
        
        參數:
            observations: 包含以下鍵的字典:
                - "features_from_dataset": 形狀 [batch_size, MAX_SYMBOLS_ALLOWED, timesteps, features]
                - "current_positions_nominal_ratio_ac": 形狀 [batch_size, MAX_SYMBOLS_ALLOWED]
                - "unrealized_pnl_ratio_ac": 形狀 [batch_size, MAX_SYMBOLS_ALLOWED]
                - "margin_level": 形狀 [batch_size, 1]
                - "padding_mask": 形狀 [batch_size, MAX_SYMBOLS_ALLOWED] (標記dummy symbols)
                
        返回:
            提取的特徵張量 [batch_size, total_features]
        """
        try:
            # 處理時間序列特徵
            features = observations["features_from_dataset"]  # [batch, symbols, timesteps, features]
            symbol_padding_mask = observations["padding_mask"]  # [batch, symbols] - True表示dummy symbol
            
            batch_size, num_symbols, seq_len, feature_dim = features.shape
            device = features.device
            
            # 確保symbol_padding_mask是布爾類型
            if symbol_padding_mask.dtype != torch.bool:
                symbol_padding_mask = symbol_padding_mask.bool()
              # 使用增強版Transformer提取特徵
            # EnhancedTransformer期望字典輸入
            transformer_input = {
                "src": features,  # [batch, symbols, seq_len, features]
                "src_key_padding_mask": symbol_padding_mask,  # [batch, symbols] - True表示dummy symbol
                "symbol_ids": None,  # 可選的symbol標識符
                "raw_ohlcv_data_batch": None  # 原始OHLCV數據（如果需要GMM）
            }
            
            enhanced_transformer_output = self.enhanced_transformer(transformer_input)
            # 輸出形狀: [batch_size, num_symbols, output_dim]
            
            # 對dummy symbols的輸出進行額外masking（雖然Transformer已經處理了，但為了安全）
            mask_for_output = (~symbol_padding_mask).float().unsqueeze(-1)  # [batch, symbols, 1]
            enhanced_transformer_output = enhanced_transformer_output * mask_for_output
            
            # 展平Transformer輸出
            flat_enhanced_transformer = enhanced_transformer_output.view(batch_size, -1)
            
            # 拼接其他特徵（這些特徵已經正確處理了dummy symbols）
            other_features = torch.cat([
                observations["current_positions_nominal_ratio_ac"],  # [batch, symbols]
                observations["unrealized_pnl_ratio_ac"],             # [batch, symbols]
                observations["margin_level"],                        # [batch, 1]
                symbol_padding_mask.float()                          # [batch, symbols] - 作為特徵
            ], dim=1)
            
            # 合併所有特徵
            combined = torch.cat([flat_enhanced_transformer, other_features], dim=1)
            
            # 添加特徵標準化以提高訓練穩定性
            combined = F.layer_norm(combined, combined.shape[1:])
            
            return combined
            
        except Exception as e:
            logger.error(f"特徵提取器前向傳播錯誤: {e}")
            # 返回默認特徵以保持系統穩定
            batch_size = observations["features_from_dataset"].shape[0]
            device = observations["features_from_dataset"].device
            return torch.zeros(batch_size, self._features_dim, device=device)

    def get_model_info(self) -> Dict[str, Any]:
        """獲取模型動態配置信息"""
        if hasattr(self.enhanced_transformer, 'get_dynamic_config'):
            return self.enhanced_transformer.get_dynamic_config()
        else:
            return {
                'model_dim': TRANSFORMER_MODEL_DIM,
                'num_layers': TRANSFORMER_NUM_LAYERS,
                'num_heads': TRANSFORMER_NUM_HEADS,
                'ffn_dim': TRANSFORMER_FFN_DIM,
                'output_dim_per_symbol': TRANSFORMER_OUTPUT_DIM_PER_SYMBOL
            }
    
    def adapt_to_config(self, config: Dict[str, Any]) -> bool:
        """動態適應新的配置
        
        Args:
            config: 包含新配置的字典
            
        Returns:
            是否成功適應新配置
        """
        try:
            current_config = self.get_model_info()
            
            # 檢查是否需要重新初始化模型
            key_params = ['model_dim', 'num_layers', 'num_heads', 'ffn_dim']
            needs_reinit = any(
                config.get(k) != current_config.get(k) 
                for k in key_params if k in config
            )
            
            if needs_reinit:
                logger.warning("⚠️ 檢測到關鍵模型參數變化，建議重新初始化特徵提取器")
                return False
            
            logger.info("✅ 特徵提取器配置已適應")
            return True
            
        except Exception as e:
            logger.error(f"❌ 配置適應失敗: {e}")
            return False


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
