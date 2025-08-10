# src/models/enhanced_attention.py
"""
增強注意力機制實現
基於最新研究優化Transformer注意力層

主要創新：
1. 多尺度注意力融合
2. 動態注意力權重
3. 記憶體效率優化
4. 位置編碼增強
5. 跨時間尺度注意力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, Dict, Any
import logging

try:
    from oanda_trading_bot.training_system.common.logger_setup import logger
except ImportError:
    logger = logging.getLogger(__name__)


class MultiScaleAttention(nn.Module):
    """多尺度注意力機制"""
    
    def __init__(self, 
                 d_model: int, 
                 num_heads: int,
                 scales: list = [1, 2, 4, 8],
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scales = scales
        self.dropout = dropout
        
        # 每個尺度的查詢、鍵、值投影
        self.scale_projections = nn.ModuleDict()
        for scale in scales:
            self.scale_projections[f'scale_{scale}'] = nn.ModuleDict({
                'query': nn.Linear(d_model, d_model // len(scales)),
                'key': nn.Linear(d_model, d_model // len(scales)),
                'value': nn.Linear(d_model, d_model // len(scales))
            })
        
        # 尺度融合層
        self.scale_fusion = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 輸出投影
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query, key, value: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, _ = query.shape
        
        scale_outputs = []
        
        for scale in self.scales:
            # 獲取該尺度的投影
            q_proj = self.scale_projections[f'scale_{scale}']['query']
            k_proj = self.scale_projections[f'scale_{scale}']['key']
            v_proj = self.scale_projections[f'scale_{scale}']['value']
            
            # 應用尺度變換
            if scale > 1:
                # 下採樣
                pooled_len = seq_len // scale
                pooled_query = F.adaptive_avg_pool1d(
                    query.transpose(1, 2), pooled_len
                ).transpose(1, 2)
                pooled_key = F.adaptive_avg_pool1d(
                    key.transpose(1, 2), pooled_len
                ).transpose(1, 2)
                pooled_value = F.adaptive_avg_pool1d(
                    value.transpose(1, 2), pooled_len
                ).transpose(1, 2)
            else:
                pooled_query = query
                pooled_key = key
                pooled_value = value
            
            # 計算注意力
            scale_attention = self._scaled_dot_product_attention(
                q_proj(pooled_query),
                k_proj(pooled_key), 
                v_proj(pooled_value),
                mask
            )
            
            # 上採樣回原尺度
            if scale > 1:
                scale_attention = F.interpolate(
                    scale_attention.transpose(1, 2),
                    size=seq_len,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            
            scale_outputs.append(scale_attention)
        
        # 融合所有尺度
        fused_output = torch.cat(scale_outputs, dim=-1)
        fused_output = self.scale_fusion(fused_output)
        
        # 輸出投影
        output = self.out_proj(fused_output)
        output = self.dropout_layer(output)
        
        return output
    
    def _scaled_dot_product_attention(self,
                                    query: torch.Tensor,
                                    key: torch.Tensor, 
                                    value: torch.Tensor,
                                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """縮放點積注意力"""
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout_layer(attention_weights)
        
        return torch.matmul(attention_weights, value)


class DynamicPositionalEncoding(nn.Module):
    """動態位置編碼"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # 可學習的位置編碼
        self.pos_embedding = nn.Parameter(torch.randn(max_len, d_model) * 0.1)
        
        # 相對位置編碼
        self.relative_pos_k = nn.Parameter(torch.randn(2 * max_len - 1, d_model // 2))
        self.relative_pos_v = nn.Parameter(torch.randn(2 * max_len - 1, d_model // 2))
        
        # 時間尺度編碼
        self.time_scale_embedding = nn.Parameter(torch.randn(10, d_model) * 0.1)
        
    def forward(self, x: torch.Tensor, time_scale: int = 1) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            time_scale: 時間尺度 (1, 2, 4, 8, ...)
        """
        batch_size, seq_len, d_model = x.shape
        
        # 絕對位置編碼
        pos_enc = self.pos_embedding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        
        # 時間尺度編碼
        scale_idx = min(int(math.log2(time_scale)), 9)
        time_enc = self.time_scale_embedding[scale_idx].unsqueeze(0).unsqueeze(0)
        time_enc = time_enc.expand(batch_size, seq_len, -1)
        
        return x + pos_enc + time_enc
    
    def get_relative_position_encoding(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """獲取相對位置編碼"""
        positions = torch.arange(seq_len).unsqueeze(1) - torch.arange(seq_len).unsqueeze(0)
        positions = positions + self.max_len - 1
        positions = torch.clamp(positions, 0, 2 * self.max_len - 2)
        
        rel_pos_k = self.relative_pos_k[positions]
        rel_pos_v = self.relative_pos_v[positions]
        
        return rel_pos_k, rel_pos_v


class MemoryEfficientAttention(nn.Module):
    """記憶體高效注意力機制"""
    
    def __init__(self, 
                 d_model: int,
                 num_heads: int,
                 chunk_size: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.chunk_size = chunk_size
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """分塊計算注意力以節省記憶體"""
        batch_size, seq_len, _ = query.shape
        
        # 投影
        Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        if seq_len <= self.chunk_size:
            # 序列較短，直接計算
            attention_output = self._compute_attention(Q, K, V, mask)
        else:
            # 序列較長，分塊計算
            attention_output = self._compute_chunked_attention(Q, K, V, mask)
        
        # 重塑並輸出投影
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.out_proj(attention_output)
    
    def _compute_attention(self, Q, K, V, mask=None):
        """標準注意力計算"""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(1) == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        return torch.matmul(attention_weights, V)
    
    def _compute_chunked_attention(self, Q, K, V, mask=None):
        """分塊注意力計算"""
        batch_size, num_heads, seq_len, d_k = Q.shape
        chunk_size = self.chunk_size
        
        output = torch.zeros_like(Q)
        
        for start_q in range(0, seq_len, chunk_size):
            end_q = min(start_q + chunk_size, seq_len)
            
            chunk_scores = []
            chunk_values = []
            
            for start_k in range(0, seq_len, chunk_size):
                end_k = min(start_k + chunk_size, seq_len)
                
                # 計算當前塊的分數
                scores = torch.matmul(
                    Q[:, :, start_q:end_q], 
                    K[:, :, start_k:end_k].transpose(-2, -1)
                ) / math.sqrt(d_k)
                
                if mask is not None:
                    chunk_mask = mask[start_q:end_q, start_k:end_k]
                    scores = scores.masked_fill(
                        chunk_mask.unsqueeze(0).unsqueeze(0) == 0, -1e9
                    )
                
                chunk_scores.append(scores)
                chunk_values.append(V[:, :, start_k:end_k])
            
            # 在所有塊上計算softmax
            all_scores = torch.cat(chunk_scores, dim=-1)
            attention_weights = F.softmax(all_scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            # 分塊應用注意力權重
            chunk_output = torch.zeros(
                batch_size, num_heads, end_q - start_q, d_k,
                device=Q.device, dtype=Q.dtype
            )
            
            start_idx = 0
            for i, values in enumerate(chunk_values):
                end_idx = start_idx + values.size(-2)
                weights = attention_weights[:, :, :, start_idx:end_idx]
                chunk_output += torch.matmul(weights, values)
                start_idx = end_idx
            
            output[:, :, start_q:end_q] = chunk_output
        
        return output


class CrossTimeScaleAttention(nn.Module):
    """跨時間尺度注意力"""
    
    def __init__(self, 
                 d_model: int,
                 num_heads: int, 
                 time_scales: list = [1, 4, 16, 64],
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.time_scales = time_scales
        
        # 每個時間尺度的注意力層
        self.scale_attentions = nn.ModuleList([
            MultiScaleAttention(d_model, num_heads, [scale], dropout)
            for scale in time_scales
        ])
        
        # 時間尺度融合
        self.temporal_fusion = nn.Sequential(
            nn.Linear(d_model * len(time_scales), d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 門控機制
        self.gate = nn.Sequential(
            nn.Linear(d_model, len(time_scales)),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len]
        """
        scale_outputs = []
        
        # 計算每個時間尺度的注意力
        for attention_layer in self.scale_attentions:
            scale_output = attention_layer(x, x, x, mask)
            scale_outputs.append(scale_output)
        
        # 計算門控權重
        gate_weights = self.gate(x.mean(dim=1, keepdim=True))  # [batch_size, 1, num_scales]
        
        # 加權融合
        weighted_outputs = []
        for i, output in enumerate(scale_outputs):
            weight = gate_weights[:, :, i].unsqueeze(-1)  # [batch_size, 1, 1]
            weighted_outputs.append(output * weight)
        
        # 拼接並融合
        concatenated = torch.cat(weighted_outputs, dim=-1)
        fused_output = self.temporal_fusion(concatenated)
        
        return fused_output + x  # 殘差連接


def create_enhanced_attention_layer(d_model: int,
                                  num_heads: int,
                                  attention_type: str = "multi_scale",
                                  **kwargs) -> nn.Module:
    """創建增強注意力層的便捷函數"""
    
    if attention_type == "multi_scale":
        return MultiScaleAttention(d_model, num_heads, **kwargs)
    elif attention_type == "memory_efficient":
        return MemoryEfficientAttention(d_model, num_heads, **kwargs)
    elif attention_type == "cross_time_scale":
        return CrossTimeScaleAttention(d_model, num_heads, **kwargs)
    else:
        raise ValueError(f"未知的注意力類型: {attention_type}")


if __name__ == "__main__":
    # 測試增強注意力機制
    print("🔍 測試增強注意力機制")
    print("="*50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 測試參數
    batch_size, seq_len, d_model = 4, 128, 768
    num_heads = 12
    
    # 創建測試數據
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    
    # 測試多尺度注意力
    print("🎯 測試多尺度注意力...")
    multi_scale_attn = MultiScaleAttention(d_model, num_heads).to(device)
    output1 = multi_scale_attn(x, x, x)
    print(f"  輸入形狀: {x.shape}")
    print(f"  輸出形狀: {output1.shape}")
    print(f"  參數數量: {sum(p.numel() for p in multi_scale_attn.parameters()):,}")
    
    # 測試記憶體高效注意力
    print("\n💾 測試記憶體高效注意力...")
    memory_eff_attn = MemoryEfficientAttention(d_model, num_heads, chunk_size=64).to(device)
    output2 = memory_eff_attn(x, x, x)
    print(f"  輸入形狀: {x.shape}")
    print(f"  輸出形狀: {output2.shape}")
    print(f"  參數數量: {sum(p.numel() for p in memory_eff_attn.parameters()):,}")
    
    # 測試跨時間尺度注意力
    print("\n⏱️ 測試跨時間尺度注意力...")
    cross_time_attn = CrossTimeScaleAttention(d_model, num_heads).to(device)
    output3 = cross_time_attn(x)
    print(f"  輸入形狀: {x.shape}")
    print(f"  輸出形狀: {output3.shape}")
    print(f"  參數數量: {sum(p.numel() for p in cross_time_attn.parameters()):,}")
    
    print("\n✅ 所有注意力機制測試完成！")
