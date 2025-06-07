# src/models/enhanced_transformer.py
"""
增強版通用多資產交易Transformer模型
基於原有UniversalTradingTransformer，大幅提升模型複雜度和學習能力

主要增強：
1. 深度架構：12-16層Transformer
2. 多尺度特徵提取：並行處理不同時間窗口
3. 自適應注意力機制：動態調整注意力權重
4. 跨時間尺度融合：整合多個時間維度信息
5. 殘差連接優化：改善深度網絡訓練
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Optional, Tuple, Union, Dict
import logging

try:
    from src.common.logger_setup import logger
    from src.common.config import DEVICE, MAX_SYMBOLS_ALLOWED, TIMESTEPS
except ImportError:
    logger = logging.getLogger(__name__)
    DEVICE = "cpu"
    MAX_SYMBOLS_ALLOWED = 20
    TIMESTEPS = 128


class MultiScaleFeatureExtractor(nn.Module):
    """多尺度特徵提取器：並行處理不同時間窗口"""
    
    def __init__(self, input_dim: int, hidden_dim: int, scales: List[int] = [3, 5, 7, 11]):
        super().__init__()
        self.scales = scales
        self.hidden_dim = hidden_dim
        
        # 每個尺度的卷積層
        self.scale_convs = nn.ModuleList([
            nn.Conv1d(input_dim, hidden_dim // len(scales), 
                     kernel_size=scale, padding=scale//2, groups=1)
            for scale in scales
        ])
        
        # 特徵融合層
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )
        
        # 時間注意力權重
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim]
        Returns:
            多尺度融合特徵: [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, input_dim = x.shape
        
        # 轉換為卷積格式: [batch, channels, seq_len]
        x_conv = x.transpose(1, 2)
        
        # 並行多尺度特徵提取
        scale_features = []
        for conv in self.scale_convs:
            scale_feat = conv(x_conv)  # [batch, hidden_dim//len(scales), seq_len]
            scale_features.append(scale_feat)
        
        # 拼接所有尺度特徵
        multi_scale = torch.cat(scale_features, dim=1)  # [batch, hidden_dim, seq_len]
        
        # 轉回時序格式
        multi_scale = multi_scale.transpose(1, 2)  # [batch, seq_len, hidden_dim]
        
        # 特徵融合
        fused_features = self.fusion_layer(multi_scale)
        
        # 時間維度自注意力
        attended_features, _ = self.temporal_attention(
            fused_features, fused_features, fused_features
        )
        
        return attended_features + fused_features  # 殘差連接


class MarketStateDetector(nn.Module):
    """市場狀態檢測器：檢測趨勢、波動、均值回歸、突破等市場狀態"""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # 狀態特徵提取器
        self.state_extractors = nn.ModuleDict({
            'trend': nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
                nn.Sigmoid()
            ),
            'volatility': nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
                nn.Sigmoid()
            ),
            'momentum': nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
                nn.Tanh()
            ),
            'regime': nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 4),  # 4種市場機制
                nn.Softmax(dim=-1)
            )
        })
        
        # 狀態融合網絡
        self.state_fusion = nn.Sequential(
            nn.Linear(7, d_model // 4),  # 1+1+1+4 = 7個狀態特徵
            nn.GELU(),
            nn.Linear(d_model // 4, d_model // 8),
            nn.GELU(),
            nn.Linear(d_model // 8, 4),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [batch_size, d_model] 或 [batch_size, seq_len, d_model]
        Returns:
            包含各種市場狀態的字典
        """
        if x.dim() == 3:
            # 如果是序列，取平均
            x = x.mean(dim=1)
        
        # 提取各種狀態特徵
        states = {}
        state_features = []
        
        for state_name, extractor in self.state_extractors.items():
            state_value = extractor(x)
            states[state_name] = state_value
            
            if state_name == 'regime':
                state_features.append(state_value)  # [batch, 4]
            else:
                state_features.append(state_value)  # [batch, 1]
        
        # 拼接所有狀態特徵
        concatenated_states = torch.cat(state_features, dim=-1)  # [batch, 7]
        
        # 融合得到最終市場狀態
        final_market_state = self.state_fusion(concatenated_states)
        states['final_state'] = final_market_state
        
        return states


class AdaptiveAttentionLayer(nn.Module):
    """自適應注意力層：根據市場狀態動態調整注意力模式"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        # 標準多頭注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 市場狀態檢測器
        self.market_state_detector = MarketStateDetector(d_model)
        
        # 狀態特定的注意力權重調製器
        self.attention_modulators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
                nn.Sigmoid()
            ) for _ in range(4)  # 4種市場狀態
        ])
        
        # 自適應溫度參數
        self.temperature_controller = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Softplus()  # 確保溫度為正
        )
        
        # 輸出投影
        self.output_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, 
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            key_padding_mask: [batch_size, seq_len]
        """
        batch_size, seq_len, d_model = x.shape
        
        # 檢測市場狀態
        market_states = self.market_state_detector(x)
        final_state = market_states['final_state']  # [batch, 4]
        
        # 計算自適應溫度
        avg_features = x.mean(dim=1)  # [batch, d_model]
        temperature = self.temperature_controller(avg_features)  # [batch, 1]
        
        # 標準注意力計算
        attn_output, attn_weights = self.attention(
            x, x, x, key_padding_mask=key_padding_mask
        )
        
        # 根據市場狀態調製注意力輸出
        modulated_outputs = []
        for i, modulator in enumerate(self.attention_modulators):
            state_weight = final_state[:, i:i+1, None]  # [batch, 1, 1]
            modulated = modulator(attn_output) * state_weight
            modulated_outputs.append(modulated)
        
        # 加權融合所有狀態的輸出
        adaptive_output = sum(modulated_outputs)
        
        # 應用自適應溫度
        adaptive_output = adaptive_output * temperature.unsqueeze(1)
        
        # 最終投影和dropout
        output = self.output_projection(adaptive_output)
        output = self.dropout(output)
        
        return output + x  # 殘差連接


class EnhancedTransformerLayer(nn.Module):
    """增強版Transformer層：集成自適應注意力和改進的FFN"""
    
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, 
                 dropout: float = 0.1):
        super().__init__()
        
        # 自適應注意力層
        self.adaptive_attention = AdaptiveAttentionLayer(d_model, num_heads, dropout)
        
        # 增強的前饋網絡
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, ffn_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim // 2, d_model),
            nn.Dropout(dropout)
        )
        
        # 層歸一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 門控機制
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, 
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # 自適應注意力子層
        attn_output = self.adaptive_attention(self.norm1(x), key_padding_mask)
        
        # FFN子層
        ffn_input = self.norm2(attn_output)
        ffn_output = self.ffn(ffn_input)
        
        # 門控融合
        gate_input = torch.cat([attn_output, ffn_output], dim=-1)
        gate_weights = self.gate(gate_input)
        
        output = gate_weights * attn_output + (1 - gate_weights) * ffn_output
        
        return output


class CrossTimeScaleFusion(nn.Module):
    """跨時間尺度融合模組：整合不同時間範圍的信息"""
    
    def __init__(self, d_model: int, time_scales: List[int] = [5, 15, 30, 60]):
        super().__init__()
        self.time_scales = time_scales
        self.d_model = d_model
        
        # 每個時間尺度的池化層
        self.scale_poolers = nn.ModuleList([
            nn.AdaptiveAvgPool1d(scale) for scale in time_scales
        ])
        
        # 尺度特定的編碼器
        self.scale_encoders = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=8,
                    dim_feedforward=d_model * 2,
                    dropout=0.1,
                    batch_first=True,
                    norm_first=True
                ),
                num_layers=2
            ) for _ in time_scales
        ])
        
        # 尺度權重計算器
        self.scale_weight_calculator = nn.Sequential(
            nn.Linear(d_model, len(time_scales)),
            nn.Softmax(dim=-1)
        )
        
        # 跨尺度注意力
        self.cross_scale_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            batch_first=True
        )
        
        # 時間一致性約束網絡
        self.consistency_network = nn.Sequential(
            nn.Linear(d_model * len(time_scales), d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # 自適應融合權重
        self.adaptive_fusion = nn.Sequential(
            nn.Linear(d_model * len(time_scales), d_model),
            nn.GELU(),
            nn.Linear(d_model, len(time_scales)),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            跨尺度融合特徵: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # 對每個時間尺度提取特徵
        scale_features = []
        for pooler, encoder in zip(self.scale_poolers, self.scale_encoders):
            # 時間維度池化
            pooled = pooler(x.transpose(1, 2)).transpose(1, 2)  # [batch, scale, d_model]
            
            # 尺度特定編碼
            encoded = encoder(pooled)  # [batch, scale, d_model]
            
            # 上採樣回原始長度
            upsampled = F.interpolate(
                encoded.transpose(1, 2), size=seq_len, mode='linear', align_corners=False
            ).transpose(1, 2)
            
            scale_features.append(upsampled)
        
        # 計算尺度權重
        avg_features = x.mean(dim=1)  # [batch, d_model]
        scale_weights = self.scale_weight_calculator(avg_features)  # [batch, num_scales]
        
        # 加權融合尺度特徵
        weighted_features = []
        for i, feat in enumerate(scale_features):
            weight = scale_weights[:, i:i+1, None]  # [batch, 1, 1]
            weighted_features.append(feat * weight)
        
        # 時間一致性約束
        concatenated = torch.cat(scale_features, dim=-1)  # [batch, seq_len, d_model*num_scales]
        consistency_features = self.consistency_network(concatenated)
        
        # 自適應融合
        fusion_weights = self.adaptive_fusion(concatenated)  # [batch, seq_len, num_scales]
        
        # 最終融合
        final_fused = torch.zeros_like(x)
        for i, feat in enumerate(scale_features):
            weight = fusion_weights[:, :, i:i+1]  # [batch, seq_len, 1]
            final_fused += feat * weight
        
        # 與一致性特徵進行注意力交互
        cross_attended, _ = self.cross_scale_attention(
            consistency_features, final_fused, final_fused
        )
        
        return cross_attended + x  # 殘差連接


class EnhancedUniversalTradingTransformer(nn.Module):
    """
    增強版通用交易Transformer
    
    主要特性：
    1. 深度架構：12-16層
    2. 多尺度特徵提取
    3. 自適應注意力機制
    4. 跨時間尺度融合
    5. 改進的位置編碼
    """
    
    def __init__(self, 
                 num_input_features: int,
                 num_symbols_possible: int = MAX_SYMBOLS_ALLOWED,
                 model_dim: int = 512,  # 從256提升到512
                 num_layers: int = 12,  # 從4提升到12
                 num_heads: int = 16,   # 從8提升到16
                 ffn_dim: int = 2048,   # 從1024提升到2048
                 dropout_rate: float = 0.1,
                 max_seq_len: int = TIMESTEPS,
                 output_dim_per_symbol: int = 32,
                 use_multi_scale: bool = True,
                 use_cross_time_fusion: bool = True):
        
        super().__init__()
        
        self.model_dim = model_dim
        self.num_symbols_possible = num_symbols_possible
        self.num_layers = num_layers
        self.use_multi_scale = use_multi_scale
        self.use_cross_time_fusion = use_cross_time_fusion
        
        logger.info(f"初始化增強版Transformer: "
                   f"layers={num_layers}, heads={num_heads}, dim={model_dim}")
        
        # 輸入投影
        self.input_projection = nn.Linear(num_input_features, model_dim)
        
        # 符號嵌入
        self.symbol_embeddings = nn.Embedding(MAX_SYMBOLS_ALLOWED, model_dim)
        
        # 改進的位置編碼
        self.positional_encoding = self._create_positional_encoding(max_seq_len, model_dim)
        
        # 多尺度特徵提取器
        if use_multi_scale:
            self.multi_scale_extractor = MultiScaleFeatureExtractor(
                model_dim, model_dim, scales=[3, 5, 7, 11]
            )
        
        # 深度Transformer編碼器層
        self.transformer_layers = nn.ModuleList([
            EnhancedTransformerLayer(model_dim, num_heads, ffn_dim, dropout_rate)
            for _ in range(num_layers)
        ])
        
        # 跨時間尺度融合
        if use_cross_time_fusion:
            self.cross_time_fusion = CrossTimeScaleFusion(model_dim)
        
        # 跨資產注意力
        self.cross_asset_attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 輸出投影
        self.output_projection = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(model_dim // 2, output_dim_per_symbol)
        )
        
        # 初始化權重
        self._initialize_weights()
        
        logger.info(f"增強版Transformer初始化完成，總參數量: "
                   f"{sum(p.numel() for p in self.parameters()):,}")
    
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """創建正弦位置編碼"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # [1, max_len, d_model]
    
    def _initialize_weights(self):
        """初始化模型權重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x_features: torch.Tensor, 
                padding_mask_symbols: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向傳播
        
        Args:
            x_features: [batch_size, num_symbols, num_timesteps, num_features]
            padding_mask_symbols: [batch_size, num_symbols] 布爾遮罩
            
        Returns:
            輸出特徵: [batch_size, num_symbols, output_dim_per_symbol]
        """
        batch_size, num_symbols, num_timesteps, num_features = x_features.shape
        
        # 輸入投影
        projected_x = self.input_projection(x_features)  # [B, N, T, D]
        
        # 符號嵌入
        symbol_indices = torch.arange(num_symbols, device=x_features.device)
        symbol_embs = self.symbol_embeddings(symbol_indices)  # [N, D]
        symbol_embs = symbol_embs.unsqueeze(0).unsqueeze(2)  # [1, N, 1, D]
        symbol_embs = symbol_embs.expand(batch_size, -1, num_timesteps, -1)
        
        # 融合特徵和符號嵌入
        x = projected_x + symbol_embs  # [B, N, T, D]
        
        # 重塑為序列格式進行時間建模
        x_reshaped = x.view(batch_size * num_symbols, num_timesteps, self.model_dim)
        
        # 添加位置編碼
        if hasattr(self, 'positional_encoding'):
            pos_enc = self.positional_encoding[:, :num_timesteps, :].to(x.device)
            x_reshaped = x_reshaped + pos_enc
        
        # 多尺度特徵提取
        if self.use_multi_scale:
            x_reshaped = self.multi_scale_extractor(x_reshaped)
        
        # 深度Transformer編碼
        for layer in self.transformer_layers:
            x_reshaped = layer(x_reshaped)
        
        # 跨時間尺度融合
        if self.use_cross_time_fusion:
            x_reshaped = self.cross_time_fusion(x_reshaped)
        
        # 時間維度池化
        time_pooled = x_reshaped[:, -1, :]  # 取最後一個時間步
        
        # 重塑回符號維度
        symbol_features = time_pooled.view(batch_size, num_symbols, self.model_dim)
        
        # 跨資產注意力
        if padding_mask_symbols is not None:
            cross_asset_output, _ = self.cross_asset_attention(
                symbol_features, symbol_features, symbol_features,
                key_padding_mask=padding_mask_symbols
            )
        else:
            cross_asset_output, _ = self.cross_asset_attention(
                symbol_features, symbol_features, symbol_features
            )
        
        # 輸出投影
        output = self.output_projection(cross_asset_output)
        
        # 應用padding遮罩
        if padding_mask_symbols is not None:
            output = output.masked_fill(padding_mask_symbols.unsqueeze(-1), 0.0)
        
        return output
    
    def get_model_info(self) -> Dict[str, Union[int, float]]:
        """獲取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_dim': self.model_dim,
            'num_layers': self.num_layers,
            'memory_usage_mb': total_params * 4 / (1024 * 1024)  # 假設float32
        }


if __name__ == "__main__":
    # 測試增強版模型
    logger.info("開始測試增強版UniversalTradingTransformer...")
    
    # 測試參數
    batch_size = 4
    num_symbols = 5
    num_timesteps = 64
    num_features = 9
    
    # 創建測試數據
    test_features = torch.randn(batch_size, num_symbols, num_timesteps, num_features)
    test_mask = torch.zeros(batch_size, num_symbols, dtype=torch.bool)
    test_mask[:, -1] = True  # 最後一個符號為padding
    
    # 初始化模型
    model = EnhancedUniversalTradingTransformer(
        num_input_features=num_features,
        num_symbols_possible=num_symbols,
        model_dim=256,  # 測試時使用較小的維度
        num_layers=6,   # 測試時使用較少的層數
        num_heads=8,
        ffn_dim=1024,
        use_multi_scale=True,
        use_cross_time_fusion=True
    )
    
    # 前向傳播測試
    try:
        with torch.no_grad():
            output = model(test_features, test_mask)
            
        logger.info(f"測試成功！")
        logger.info(f"輸入形狀: {test_features.shape}")
        logger.info(f"輸出形狀: {output.shape}")
        logger.info(f"模型信息: {model.get_model_info()}")
        
        # 測試梯度計算
        model.train()
        output = model(test_features, test_mask)
        loss = output.mean()
        loss.backward()
        
        logger.info("梯度計算測試通過")
        
    except Exception as e:
        logger.error(f"測試失敗: {e}")
        raise e
