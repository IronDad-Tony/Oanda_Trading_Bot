# src/models/custom_layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import logging

# 如果 logger 和 DEVICE 在專案中有統一的設定方式, 可以在這裡引入
# 例如: from src.common.logger_setup import logger
# from src.common.config import DEVICE
# 為了模組獨立性, 暫時使用標準 logging
logger = logging.getLogger(__name__)

class HierarchicalAttention(nn.Module):
    """
    分層注意力輔助模組。
    一個標準的 Transformer 注意力區塊 (自註意力/交叉注意力 + FFN)。
    """
    def __init__(self, d_model: int, num_heads: int, dropout_rate: float):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout_rate, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), # FFN 中間層擴展
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor, 
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向傳播。
        Args:
            query: 查詢張量 [batch, seq_len_q, d_model]
            key: 鍵張量 [batch, seq_len_kv, d_model]
            value: 值張量 [batch, seq_len_kv, d_model]
            key_padding_mask: 可選的鍵填充遮罩 [batch, seq_len_kv]
        Returns:
            處理後的張量 [batch, seq_len_q, d_model]
        """
        # 多頭注意力
        attn_output, _ = self.mha(query, key, value, key_padding_mask=key_padding_mask)
        # 殘差連接和層歸一化 (Add & Norm 1)
        out1 = self.norm1(query + self.dropout(attn_output))
        
        # 前饋網絡
        ffn_output = self.ffn(out1)
        # 殘差連接和層歸一化 (Add & Norm 2)
        out2 = self.norm2(out1 + self.dropout(ffn_output))
        return out2

class CrossTimeScaleFusion(nn.Module):
    """
    跨時間尺度融合模組 (Cross-Time-Scale Fusion Module)。
    整合來自不同時間尺度的特徵, 實現分層時間建模和時間一致性。
    """
    def __init__(self, 
                 d_model: int, 
                 time_scales: List[int], # 例如 [1, 3, 5] 表示原始、3倍下採樣、5倍下採樣的相對比例
                 fusion_type: str = "hierarchical_attention", # 可選: "hierarchical_attention", "simple_attention", "concat", "average"
                 dropout_rate: float = 0.1,
                 num_heads_hierarchical: int = 4): # 用於 HierarchicalAttention 的頭數
        super().__init__()
        self.d_model = d_model
        # 確保時間尺度唯一且排序 (通常原始尺度為1)
        self.time_scales = sorted(list(set(ts for ts in time_scales if isinstance(ts, int) and ts > 0)))
        if not self.time_scales:
            logger.warning("CrossTimeScaleFusion: time_scales 為空或無效, 將不執行任何操作。")
            # 設置一個標誌或確保 forward 變為恆等操作
            self._is_noop = True 
            return 
        self._is_noop = False

        self.fusion_type = fusion_type
        self.dropout_rate = dropout_rate

        # 為每個時間尺度（除了原始尺度1）創建下採樣（池化）層
        # 輸入 x 格式: [B*N, T, C]
        # 池化應用於時間維度 T
        self.pooling_layers = nn.ModuleDict()
        for scale in self.time_scales:
            if scale > 1: # 尺度大於1才需要池化
                # AvgPool1d 需要 (N, C, L_in) -> (N, C, L_out)
                # kernel_size=scale, stride=scale 實現按比例下採樣
                self.pooling_layers[str(scale)] = nn.AvgPool1d(kernel_size=scale, stride=scale, ceil_mode=True)
        
        # 分層時間建模組件與融合邏輯
        if self.fusion_type == "hierarchical_attention":
            if len(self.time_scales) > 0: # 只有在有多個尺度時才有意義
                self.hierarchical_attentions = nn.ModuleList()
                for _ in self.time_scales:
                    self.hierarchical_attentions.append(
                        HierarchicalAttention(d_model, num_heads_hierarchical, dropout_rate)
                    )
                # 最終融合來自不同注意力頭的輸出
                self.final_hierarchical_projection = nn.Linear(d_model * len(self.time_scales), d_model) if len(self.time_scales) > 1 else nn.Identity()
            else: # 如果只有一個尺度或沒有尺度, 則此模式無效
                 logger.warning(f"CrossTimeScaleFusion: 'hierarchical_attention' 模式下 time_scales 數量不足 ({len(self.time_scales)}), 將退化。")
                 self.fusion_type = "average" # 退化到平均或首個尺度

        elif self.fusion_type == "simple_attention":
            if len(self.time_scales) > 0:
                self.simple_attention_projection = nn.Linear(d_model * len(self.time_scales), d_model) if len(self.time_scales) > 1 else nn.Identity()
            else:
                logger.warning(f"CrossTimeScaleFusion: 'simple_attention' 模式下 time_scales 數量不足 ({len(self.time_scales)}), 將退化。")
                self.fusion_type = "average"

        elif self.fusion_type == "concat":
            if len(self.time_scales) > 0:
                self.concat_projection = nn.Linear(d_model * len(self.time_scales), d_model) if len(self.time_scales) > 1 else nn.Identity()
            else:
                logger.warning(f"CrossTimeScaleFusion: 'concat' 模式下 time_scales 數量不足 ({len(self.time_scales)}), 將退化。")
                self.fusion_type = "average"
        
        # "average" 模式不需要額外層

        self.output_norm = nn.LayerNorm(d_model)
        self.output_dropout = nn.Dropout(dropout_rate)

        # 時間一致性約束: 通過一個小型時間卷積層來平滑融合後的特徵
        # Depthwise separable convolution for efficiency and per-channel smoothing
        self.temporal_smoother = nn.Conv1d(in_channels=d_model, out_channels=d_model, 
                                           kernel_size=3, padding=1, groups=d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播。
        Args:
            x: 輸入張量, 形狀為 [batch_size_eff, seq_len, d_model]
               (例如 B*N, T, C, 其中 B 是批次大小, N 是符號數量, T 是序列長度, C 是特徵維度)
        Returns:
            融合並平滑後的特徵張量, 形狀與輸入相同。
        """
        if self._is_noop or not self.time_scales:
            # logger.debug("CrossTimeScaleFusion is no-op.")
            return x # 如果初始化時 time_scales 為空, 則不執行任何操作

        batch_size_eff, original_seq_len, _ = x.shape
        
        scaled_features = []
        original_scale_feature = None # 用於 hierarchical_attention 的上下文

        for scale in self.time_scales:
            if scale == 1:
                processed_x = x
                if original_scale_feature is None: # 保存第一個遇到的原始尺度特徵
                    original_scale_feature = processed_x 
            elif str(scale) in self.pooling_layers:
                # 輸入到 pooling_layers 需要是 [N, C, L_in]
                x_permuted = x.permute(0, 2, 1) # [B*N, C, T]
                pooled_x = self.pooling_layers[str(scale)](x_permuted) # [B*N, C, T_scaled]
                processed_x = pooled_x.permute(0, 2, 1) # [B*N, T_scaled, C]
            else: # 尺度為1但不是第一個, 或者 pooling_layer 不存在 (不應發生)
                processed_x = x # 保持原樣

            # 上採樣回原始序列長度以進行融合 (如果長度改變)
            if processed_x.size(1) != original_seq_len:
                # interpolate 需要 [N, C, L]
                processed_x_permuted = processed_x.permute(0, 2, 1) # [B*N, C, T_scaled]
                upsampled_x = F.interpolate(processed_x_permuted, size=original_seq_len, 
                                            mode='linear', align_corners=False)
                processed_x = upsampled_x.permute(0, 2, 1) # [B*N, T, C]
            
            scaled_features.append(processed_x)

        if not scaled_features: # 如果由於某種原因 scaled_features 為空
            logger.warning("CrossTimeScaleFusion: No scaled features generated, returning original input.")
            return x 

        # 融合特徵
        fused_x: torch.Tensor
        if len(self.time_scales) == 1 and scaled_features: # 只有一個尺度, 無需融合
            fused_x = scaled_features[0]
        elif self.fusion_type == "hierarchical_attention" and hasattr(self, 'hierarchical_attentions'):
            attended_outputs = []
            # 如果原始尺度特徵 (scale=1) 未處理或不存在, 使用 scaled_features[0] 作為備用上下文
            context_kv = original_scale_feature if original_scale_feature is not None else scaled_features[0]

            for i, scale_feat_query in enumerate(scaled_features):
                # 每個尺度特徵作為 query, attend to 共同的上下文 (context_kv)
                attended_outputs.append(self.hierarchical_attentions[i](scale_feat_query, context_kv, context_kv))
            
            fused_x_concat = torch.cat(attended_outputs, dim=-1) # [B*N, T, C*num_scales]
            fused_x = self.final_hierarchical_projection(fused_x_concat) # [B*N, T, C]

        elif self.fusion_type == "simple_attention" and hasattr(self, 'simple_attention_projection'):
            fused_x_concat = torch.cat(scaled_features, dim=-1)
            fused_x = self.simple_attention_projection(fused_x_concat)
        
        elif self.fusion_type == "concat" and hasattr(self, 'concat_projection'):
            fused_x_concat = torch.cat(scaled_features, dim=-1)
            fused_x = self.concat_projection(fused_x_concat)   
        
        elif self.fusion_type == "average":
            fused_x = torch.stack(scaled_features, dim=0).mean(dim=0) # [B*N, T, C]
        
        else: # 預設或回退情況: 使用第一個尺度的特徵或原始輸入
            logger.warning(f"CrossTimeScaleFusion: Unknown or invalid fusion_type '{self.fusion_type}' or insufficient scales. Defaulting to first scale or input.")
            fused_x = scaled_features[0] if scaled_features else x

        # 應用時間平滑 (Temporal Consistency)
        # Conv1D expects [N, C_in, L_in]
        smoothed_x_permuted = fused_x.permute(0, 2, 1) # [B*N, C, T]
        smoothed_x_permuted = self.temporal_smoother(smoothed_x_permuted)
        smoothed_x = smoothed_x_permuted.permute(0, 2, 1) # [B*N, T, C]
        
        # 殘差連接: 與 CTS 模塊的原始輸入 x 相加
        output = self.output_norm(x + smoothed_x) 
        output = self.output_dropout(output)
        
        return output
