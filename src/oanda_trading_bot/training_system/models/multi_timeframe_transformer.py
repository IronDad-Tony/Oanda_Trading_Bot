"""
跨時間維度Transformer模型
用於融合多個時間維度的金融數據特徵

核心組件:
1. 各時間維度獨立編碼器
2. 跨維度注意力融合層
3. 多頭注意力機制
"""
import torch
import torch.nn as nn
from typing import Dict, List

class TimeSeriesEncoder(nn.Module):
    """
    單一時間維度特徵編碼器
    使用Transformer編碼器提取時間序列特徵
    
    參數:
        input_dim: 輸入特徵維度
        hidden_dim: 隱藏層維度 (默認128)
        num_layers: Transformer層數 (默認2)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Transformer編碼器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 時間位置編碼
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, hidden_dim))
        nn.init.normal_(self.positional_encoding, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        輸入: (batch_size, seq_len, input_dim)
        輸出: (batch_size, hidden_dim)
        """
        # 特徵投影
        x = self.input_proj(x)
        
        # 添加位置編碼
        seq_len = x.size(1)
        x = x + self.positional_encoding[:, :seq_len]
        
        # Transformer處理
        x = x.transpose(0, 1)  # (seq_len, batch_size, hidden_dim)
        x = self.transformer_encoder(x)
        
        # 取最後時間步作為特徵
        return x[-1]

class MultiTimeframeTransformer(nn.Module):
    """
    跨時間維度特徵融合模型
    
    參數:
        input_dims: 各時間維度的輸入特徵維度字典 
            (e.g., {'S5': 10, 'M1': 8, 'H1': 6})
        output_dim: 輸出特徵維度 (默認256)
    """
    def __init__(self, input_dims: Dict[str, int], output_dim: int = 256):
        super().__init__()
        self.input_dims = input_dims
        
        # 各時間維度獨立編碼器
        self.encoders = nn.ModuleDict({
            timeframe: TimeSeriesEncoder(dim) 
            for timeframe, dim in input_dims.items()
        })
        
        # 跨維度注意力融合層
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=8,
            batch_first=True
        )
        
        # 輸出投影層
        self.output_proj = nn.Linear(128 * len(input_dims), output_dim)
    
    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        輸入: 各時間維度數據的字典
            {
                'S5': tensor (batch_size, seq_len, feature_dim),
                'M1': tensor (batch_size, seq_len, feature_dim),
                ...
            }
        輸出: 融合特徵向量 (batch_size, output_dim)
        """
        # 各維度特徵編碼
        encoded_features = {}
        for timeframe, data in x.items():
            if timeframe in self.encoders:
                encoded_features[timeframe] = self.encoders[timeframe](data)
        
        # 堆疊特徵 (batch_size, num_timeframes, 128)
        features_tensor = torch.stack(list(encoded_features.values()), dim=1)
        
        # 跨維度注意力融合
        attn_output, _ = self.cross_attention(
            features_tensor, features_tensor, features_tensor
        )
        
        # 展平特徵
        flattened = attn_output.view(attn_output.size(0), -1)
        
        # 輸出投影
        return self.output_proj(flattened)