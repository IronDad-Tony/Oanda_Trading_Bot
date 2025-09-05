import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

# å°Žå…¥å¿…è¦çš„æ¨¡çµ„
# å¾ž mmap_dataset.py å°Žå…¥ UniversalMemoryMappedDataset
from oanda_trading_bot.training_system.data_manager.mmap_dataset import UniversalMemoryMappedDataset
# å¾ž enhanced_transformer.py å°Žå…¥ EnhancedTransformer
from oanda_trading_bot.training_system.models.enhanced_transformer import EnhancedTransformer
# å¾ž common.config å°Žå…¥ MAX_SYMBOLS_ALLOWED å’Œ TIMESTEPS
from oanda_trading_bot.training_system.common.config import MAX_SYMBOLS_ALLOWED, TIMESTEPS
# å¾ž common.logger_setup å°Žå…¥ logger
from oanda_trading_bot.training_system.common.logger_setup import logger

class DualTrackDataProcessor(nn.Module):
    """
    é›™è»Œæ•¸æ“šé è™•ç†å™¨ã€‚
    è©²æ¨¡çµ„è² è²¬å¾ž UniversalMemoryMappedDataset ç²å–æ•¸æ“šï¼Œ
    ä¸¦ç”Ÿæˆå…©ç¨®ç‰¹å¾µï¼šåŽŸå§‹ç‰¹å¾µ (Raw Features) å’Œé«˜ç¶­ç‰¹å¾µ (Transformed Features)ã€‚
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
        
        # å¯¦ä¾‹åŒ– EnhancedTransformer
        # EnhancedTransformer é æœŸè¼¸å…¥å½¢ç‹€ç‚º [batch_size, num_active_symbols, seq_len, input_dim]
        # é€™è£¡çš„ input_dim æ‡‰è©²æ˜¯ä¾†è‡ª mmap_dataset çš„ features_tensor çš„æœ€å¾Œä¸€å€‹ç¶­åº¦
        self.enhanced_transformer = EnhancedTransformer(
            input_dim=self.num_features_per_symbol, # ä¾†è‡ª mmap_dataset çš„ç‰¹å¾µæ•¸é‡
            # é€™è£¡éœ€è¦å¾ž enhanced_transformer_config ä¸­ç²å–æ‰€æœ‰ EnhancedTransformer æ‰€éœ€çš„åƒæ•¸
            # é€™äº›åƒæ•¸åœ¨ configs/training/enhanced_model_config.py ä¸­å®šç¾©
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
            use_final_norm=enhanced_transformer_config.get('use_final_norm', True), # å‡è¨­ True
            use_adaptive_attention=enhanced_transformer_config.get('use_adaptive_attention', True),
            num_market_states=enhanced_transformer_config.get('num_market_states', 4),
            use_gmm_market_state_detector=enhanced_transformer_config.get('use_gmm_market_state_detector', False),
            gmm_market_state_detector_path=enhanced_transformer_config.get('gmm_market_state_detector_path'),
            gmm_ohlcv_feature_config=enhanced_transformer_config.get('gmm_ohlcv_feature_config'),
            use_cts_fusion=enhanced_transformer_config.get('use_cts_fusion', True),
            cts_time_scales=enhanced_transformer_config.get('cts_time_scales', [1, 3, 5]),
            cts_fusion_type=enhanced_transformer_config.get('cts_fusion_type', "hierarchical_attention"),
            use_symbol_embedding=enhanced_transformer_config.get('use_symbol_embedding', True),
            symbol_embedding_dim=enhanced_transformer_config.get('symbol_embedding_dim', 16), # æ–°å¢žï¼Œç¢ºä¿æœ‰é è¨­å€¼
            use_fourier_features=enhanced_transformer_config.get('use_fourier_features', False),
            fourier_num_modes=enhanced_transformer_config.get('fourier_num_modes'), # å‡è¨­æœ‰é è¨­å€¼æˆ–å¾žcommon.configè®€å–
            use_wavelet_features=enhanced_transformer_config.get('use_wavelet_features', False),
            wavelet_name=enhanced_transformer_config.get('wavelet_name'), # å‡è¨­æœ‰é è¨­å€¼æˆ–å¾žcommon.configè®€å–
            wavelet_levels=enhanced_transformer_config.get('wavelet_levels'), # å‡è¨­æœ‰é è¨­å€¼æˆ–å¾žcommon.configè®€å–
            trainable_wavelet_filters=enhanced_transformer_config.get('trainable_wavelet_filters', False),
            use_layer_norm_before=enhanced_transformer_config.get('use_layer_norm_before', True),
            output_activation=enhanced_transformer_config.get('output_activation'),
            positional_encoding_type=enhanced_transformer_config.get('positional_encoding_type', "sinusoidal"),
            use_cross_asset_attention=enhanced_transformer_config.get('use_cross_asset_attention', True),
            num_cross_asset_layers=enhanced_transformer_config.get('num_cross_asset_layers', 4),\n            use_graph_attn=enhanced_transformer_config.get('use_graph_attn', True),\n            graph_layers=enhanced_transformer_config.get('graph_layers', 2),\n            graph_heads=enhanced_transformer_config.get('graph_heads', 4),\n            graph_topk=enhanced_transformer_config.get('graph_topk', 5),\n            device=self.device
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
        è™•ç†ä¸€å€‹æ‰¹æ¬¡çš„æ•¸æ“šï¼Œç”ŸæˆåŽŸå§‹ç‰¹å¾µå’Œé«˜ç¶­ç‰¹å¾µã€‚

        Args:
            features_batch (torch.Tensor): ä¾†è‡ª UniversalMemoryMappedDataset çš„é è™•ç†ç‰¹å¾µã€‚
                                          å½¢ç‹€: (batch_size, num_symbols, timesteps_history, num_features_per_symbol)
            raw_prices_batch (torch.Tensor): ä¾†è‡ª UniversalMemoryMappedDataset çš„åŽŸå§‹åƒ¹æ ¼æ•¸æ“šã€‚
                                            å½¢ç‹€: (batch_size, num_symbols, timesteps_history, num_raw_price_features)
            symbol_ids_batch (Optional[torch.Tensor]): ç¬¦è™Ÿ IDï¼Œå½¢ç‹€ (batch_size, num_symbols)ã€‚
            padding_mask_batch (Optional[torch.Tensor]): å¡«å……é®ç½©ï¼Œå½¢ç‹€ (batch_size, num_symbols)ã€‚
                                                          True è¡¨ç¤ºè©²ç¬¦è™Ÿæ˜¯å¡«å……çš„è™›æ“¬ç¬¦è™Ÿã€‚
            raw_ohlcv_data_batch (Optional[List[pd.DataFrame]]): åŽŸå§‹ OHLCV æ•¸æ“šï¼Œç”¨æ–¼ GMMã€‚

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - raw_features (torch.Tensor): ä½œç‚ºåŽŸå§‹ç‰¹å¾µçš„ features_batchã€‚
                - transformed_features (torch.Tensor): ä¾†è‡ª EnhancedTransformer çš„é«˜ç¶­ç‰¹å¾µã€‚
                                                       å½¢ç‹€: (batch_size, num_symbols, transformer_output_dim)
        """
        # ç¢ºä¿æ‰€æœ‰è¼¸å…¥å¼µé‡éƒ½åœ¨æ­£ç¢ºçš„è¨­å‚™ä¸Š
        features_batch = features_batch.to(self.device)
        raw_prices_batch = raw_prices_batch.to(self.device)
        if symbol_ids_batch is not None:
            symbol_ids_batch = symbol_ids_batch.to(self.device)
        if padding_mask_batch is not None:
            padding_mask_batch = padding_mask_batch.to(self.device)

        # 1. åŽŸå§‹ç‰¹å¾µ (Raw Features): ç›´æŽ¥ä½¿ç”¨ä¾†è‡ª UniversalMemoryMappedDataset çš„ features_batch
        # è¨ˆç•«ä¸­æåˆ° "åŽŸå§‹ç‰¹å¾µ (Raw Features): ç¶“éŽæ¨™æº–åŒ–å’Œé è™•ç†çš„åƒ¹é‡æ•¸æ“š"ï¼Œ
        # é€™èˆ‡ UniversalMemoryMappedDataset è¼¸å‡ºçš„ features_batch ç›¸ç¬¦ã€‚
        raw_features = features_batch

        # 2. é«˜ç¶­ç‰¹å¾µ (Transformed Features): é€éŽ EnhancedTransformer è™•ç†
        # EnhancedTransformer æœŸæœ›çš„è¼¸å…¥å­—å…¸
        transformer_input_dict = {
            "src": features_batch, # ä¸»è¦è¼¸å…¥
            "symbol_ids": symbol_ids_batch,
            "src_key_padding_mask": padding_mask_batch,
            "raw_ohlcv_data_batch": raw_ohlcv_data_batch # å‚³éžçµ¦ GMM ä½¿ç”¨
        }
        
        # åŸ·è¡Œ EnhancedTransformer çš„å‰å‘å‚³æ’­
        # EnhancedTransformer çš„è¼¸å‡ºå½¢ç‹€é æœŸç‚º (batch_size, num_symbols, output_dim)
        transformed_features = self.enhanced_transformer(transformer_input_dict)
        
        return raw_features, transformed_features

