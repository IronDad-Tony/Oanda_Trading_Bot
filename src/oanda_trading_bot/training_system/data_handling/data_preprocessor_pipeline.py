import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

# 導入必要的模組
# 從 mmap_dataset.py 導入 UniversalMemoryMappedDataset
from oanda_trading_bot.training_system.data_manager.mmap_dataset import UniversalMemoryMappedDataset
# 從 enhanced_transformer.py 導入 EnhancedTransformer
from oanda_trading_bot.training_system.models.enhanced_transformer import EnhancedTransformer
# 從 common.config 導入 MAX_SYMBOLS_ALLOWED 和 TIMESTEPS
from oanda_trading_bot.training_system.common.config import MAX_SYMBOLS_ALLOWED, TIMESTEPS
# 從 common.logger_setup 導入 logger
from oanda_trading_bot.training_system.common.logger_setup import logger

class DualTrackDataProcessor(nn.Module):
    """
    雙軌數據預處理器。
    該模組負責從 UniversalMemoryMappedDataset 獲取數據，
    並生成兩種特徵：原始特徵 (Raw Features) 和高維特徵 (Transformed Features)。
    """
    def __init__(self, 
                 enhanced_transformer_config: Dict[str, Any],
                 num_features_per_symbol: int,
                 max_symbols_allowed: int = MAX_SYMBOLS_ALLOWED,
                 timesteps_history: int = TIMESTEPS,
                 device: str = "cpu"):
        super().__init__()
        
        self.num_features_per_symbol = num_features_per_symbol
        self.max_symbols_allowed = max_symbols_allowed
        self.timesteps_history = timesteps_history
        self.device = device
        
        # 實例化 EnhancedTransformer
        # EnhancedTransformer 預期輸入形狀為 [batch_size, num_active_symbols, seq_len, input_dim]
        # 這裡的 input_dim 應該是來自 mmap_dataset 的 features_tensor 的最後一個維度
        self.enhanced_transformer = EnhancedTransformer(
            input_dim=self.num_features_per_symbol, # 來自 mmap_dataset 的特徵數量
            # 這裡需要從 enhanced_transformer_config 中獲取所有 EnhancedTransformer 所需的參數
            # 這些參數在 configs/training/enhanced_model_config.py 中定義
            d_model=enhanced_transformer_config.get('hidden_dim', 256),
            transformer_nhead=enhanced_transformer_config.get('num_heads', 8),
            num_encoder_layers=enhanced_transformer_config.get('num_layers', 6),
            dim_feedforward=enhanced_transformer_config.get('intermediate_dim', 1024),
            dropout=enhanced_transformer_config.get('dropout_rate', 0.1),
            max_seq_len=enhanced_transformer_config.get('max_sequence_length', self.timesteps_history),
            num_symbols=enhanced_transformer_config.get('num_symbols', self.max_symbols_allowed),
            output_dim=enhanced_transformer_config.get('output_dim', 128),
            use_msfe=enhanced_transformer_config.get('use_msfe', True),
            msfe_hidden_dim=enhanced_transformer_config.get('msfe_hidden_dim'),
            msfe_scales=enhanced_transformer_config.get('msfe_scales', [3, 5, 7, 11]),
            use_final_norm=enhanced_transformer_config.get('use_final_norm', True), # 假設 True
            use_adaptive_attention=enhanced_transformer_config.get('use_adaptive_attention', True),
            num_market_states=enhanced_transformer_config.get('num_market_states', 4),
            use_gmm_market_state_detector=enhanced_transformer_config.get('use_gmm_market_state_detector', False),
            gmm_market_state_detector_path=enhanced_transformer_config.get('gmm_market_state_detector_path'),
            gmm_ohlcv_feature_config=enhanced_transformer_config.get('gmm_ohlcv_feature_config'),
            use_cts_fusion=enhanced_transformer_config.get('use_cts_fusion', True),
            cts_time_scales=enhanced_transformer_config.get('cts_time_scales', [1, 3, 5]),
            cts_fusion_type=enhanced_transformer_config.get('cts_fusion_type', "hierarchical_attention"),
            use_symbol_embedding=enhanced_transformer_config.get('use_symbol_embedding', True),
            symbol_embedding_dim=enhanced_transformer_config.get('symbol_embedding_dim', 16), # 新增，確保有預設值
            use_fourier_features=enhanced_transformer_config.get('use_fourier_features', False),
            fourier_num_modes=enhanced_transformer_config.get('fourier_num_modes'), # 假設有預設值或從common.config讀取
            use_wavelet_features=enhanced_transformer_config.get('use_wavelet_features', False),
            wavelet_name=enhanced_transformer_config.get('wavelet_name'), # 假設有預設值或從common.config讀取
            wavelet_levels=enhanced_transformer_config.get('wavelet_levels'), # 假設有預設值或從common.config讀取
            trainable_wavelet_filters=enhanced_transformer_config.get('trainable_wavelet_filters', False),
            use_layer_norm_before=enhanced_transformer_config.get('use_layer_norm_before', True),
            output_activation=enhanced_transformer_config.get('output_activation'),
            positional_encoding_type=enhanced_transformer_config.get('positional_encoding_type', "sinusoidal"),
            use_cross_asset_attention=enhanced_transformer_config.get('use_cross_asset_attention', True),
            num_cross_asset_layers=enhanced_transformer_config.get('num_cross_asset_layers', 4),
            device=self.device
        ).to(self.device)
        
        logger.info(f"DualTrackDataProcessor initialized. Transformer output dim: {self.enhanced_transformer.output_dim}")

    def process_batch(self, 
                      features_batch: torch.Tensor, 
                      raw_prices_batch: torch.Tensor,
                      symbol_ids_batch: Optional[torch.Tensor] = None,
                      padding_mask_batch: Optional[torch.Tensor] = None,
                      raw_ohlcv_data_batch: Optional[List[pd.DataFrame]] = None # For GMM
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        處理一個批次的數據，生成原始特徵和高維特徵。

        Args:
            features_batch (torch.Tensor): 來自 UniversalMemoryMappedDataset 的預處理特徵。
                                          形狀: (batch_size, num_symbols, timesteps_history, num_features_per_symbol)
            raw_prices_batch (torch.Tensor): 來自 UniversalMemoryMappedDataset 的原始價格數據。
                                            形狀: (batch_size, num_symbols, timesteps_history, num_raw_price_features)
            symbol_ids_batch (Optional[torch.Tensor]): 符號 ID，形狀 (batch_size, num_symbols)。
            padding_mask_batch (Optional[torch.Tensor]): 填充遮罩，形狀 (batch_size, num_symbols)。
                                                          True 表示該符號是填充的虛擬符號。
            raw_ohlcv_data_batch (Optional[List[pd.DataFrame]]): 原始 OHLCV 數據，用於 GMM。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - raw_features (torch.Tensor): 作為原始特徵的 features_batch。
                - transformed_features (torch.Tensor): 來自 EnhancedTransformer 的高維特徵。
                                                       形狀: (batch_size, num_symbols, transformer_output_dim)
        """
        # 確保所有輸入張量都在正確的設備上
        features_batch = features_batch.to(self.device)
        raw_prices_batch = raw_prices_batch.to(self.device)
        if symbol_ids_batch is not None:
            symbol_ids_batch = symbol_ids_batch.to(self.device)
        if padding_mask_batch is not None:
            padding_mask_batch = padding_mask_batch.to(self.device)

        # 1. 原始特徵 (Raw Features): 直接使用來自 UniversalMemoryMappedDataset 的 features_batch
        # 計畫中提到 "原始特徵 (Raw Features): 經過標準化和預處理的價量數據"，
        # 這與 UniversalMemoryMappedDataset 輸出的 features_batch 相符。
        raw_features = features_batch

        # 2. 高維特徵 (Transformed Features): 透過 EnhancedTransformer 處理
        # EnhancedTransformer 期望的輸入字典
        transformer_input_dict = {
            "src": features_batch, # 主要輸入
            "symbol_ids": symbol_ids_batch,
            "src_key_padding_mask": padding_mask_batch,
            "raw_ohlcv_data_batch": raw_ohlcv_data_batch # 傳遞給 GMM 使用
        }
        
        # 執行 EnhancedTransformer 的前向傳播
        # EnhancedTransformer 的輸出形狀預期為 (batch_size, num_symbols, output_dim)
        transformed_features = self.enhanced_transformer(transformer_input_dict)
        
        return raw_features, transformed_features
