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
from typing import List, Optional, Tuple, Union, Dict, Any
import logging
import pywt # Added for wavelet functionality

try:
    from src.common.logger_setup import logger
    from src.common.config import DEVICE, MAX_SYMBOLS_ALLOWED, TIMESTEPS
    # Attempt to import wavelet and fourier specific configs if they exist
    # These might be moved to a more general model config or stay here
    try:
        from src.common.config import FOURIER_NUM_MODES, WAVELET_LEVELS, WAVELET_NAME
    except ImportError:
        logger.info("Wavelet/Fourier specific configs (FOURIER_NUM_MODES, WAVELET_LEVELS, WAVELET_NAME) not found in src.common.config, using defaults.")
        FOURIER_NUM_MODES = 32 # Default, from UniversalTradingTransformer
        WAVELET_LEVELS = 3    # Default
        WAVELET_NAME = 'db4'  # Default
except ImportError:
    logger = logging.getLogger(__name__)
    DEVICE = "cpu"
    MAX_SYMBOLS_ALLOWED = 20
    TIMESTEPS = 128
    FOURIER_NUM_MODES = 32 
    WAVELET_LEVELS = 3    
    WAVELET_NAME = 'db4'


# --- Fourier Feature Block (from UniversalTradingTransformer) ---
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_modes = num_modes
        scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(scale * torch.randn(in_channels, out_channels, self.num_modes, dtype=torch.cfloat))

    def forward(self, x_fft: torch.Tensor) -> torch.Tensor:
        # x_fft shape: [batch_size * num_symbols, in_channels, num_freqs]
        out_fft = torch.zeros(x_fft.size(0), self.out_channels, x_fft.size(2), dtype=torch.cfloat, device=x_fft.device)
        
        # Ensure num_modes does not exceed available frequencies
        clamped_num_modes = min(self.num_modes, x_fft.size(2))
        
        selected_modes_data = x_fft[:, :, :clamped_num_modes]
        # weights shape: [in_channels, out_channels, num_modes]
        # selected_modes_data shape: [batch_size * num_symbols, in_channels, clamped_num_modes]
        # We need to match the num_modes dimension. If clamped_num_modes < self.num_modes, slice weights.
        
        current_weights = self.weights[:, :, :clamped_num_modes]
        
        multiplied = torch.einsum("bic,ioc->boc", selected_modes_data, current_weights)
        out_fft[:, :, :clamped_num_modes] = multiplied
        return out_fft

class FourierFeatureBlock(nn.Module):
    def __init__(self, model_dim: int, num_modes: int, activation: str = 'gelu'):
        super().__init__()
        # For EnhancedTransformer, input to FourierBlock is [B*N, T, C_model]
        # SpectralConv1d expects [B*N, C_model, T_freq]
        self.spectral_conv = SpectralConv1d(model_dim, model_dim, num_modes)
        self.act = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.norm = nn.LayerNorm(model_dim) # Added LayerNorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size * num_symbols, seq_len, model_dim]
        B_N, T, C_in = x.shape
        x_perm = x.permute(0, 2, 1)  # [B*N, C_in, T]
        
        x_fft = torch.fft.rfft(x_perm, n=T, dim=2, norm='ortho') # [B*N, C_in, T_freq]
        
        x_filtered_fft = self.spectral_conv(x_fft) # [B*N, C_in, T_freq]
        
        x_time_domain = torch.fft.irfft(x_filtered_fft, n=T, dim=2, norm='ortho') # [B*N, C_in, T]
        
        output = x_time_domain.permute(0, 2, 1) # [B*N, T, C_in]
        output = self.norm(output) # Apply LayerNorm
        return self.act(output)


# --- Wavelet Feature Block (from UniversalTradingTransformer) ---
class DWT1D(nn.Module):
    def __init__(self, wavelet_name: str = 'db4', trainable_filters: bool = False):
        super().__init__()
        self.wavelet = pywt.Wavelet(wavelet_name)
        dec_lo = torch.tensor(self.wavelet.dec_lo[::-1], dtype=torch.float32)
        dec_hi = torch.tensor(self.wavelet.dec_hi[::-1], dtype=torch.float32)
        self.kernel_len = len(dec_lo)
        
        lo_filter = dec_lo.unsqueeze(0).unsqueeze(0) # [1, 1, kernel_len]
        hi_filter = dec_hi.unsqueeze(0).unsqueeze(0) # [1, 1, kernel_len]
        
        if trainable_filters:
            self.lo_filter = nn.Parameter(lo_filter)
            self.hi_filter = nn.Parameter(hi_filter)
        else:
            self.register_buffer('lo_filt', lo_filter)
            self.register_buffer('hi_filt', hi_filter)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: [batch_size * num_symbols, model_dim, seq_len]
        B_N, C, T = x.shape
        
        current_lo_filter = self.lo_filt if hasattr(self, 'lo_filt') else self.lo_filter
        current_hi_filter = self.hi_filt if hasattr(self, 'hi_filt') else self.hi_filter
        
        # Repeat filters for each channel (model_dim) to act as depthwise separable conv
        lo_filt_for_conv = current_lo_filter.repeat(C, 1, 1) # [C, 1, kernel_len]
        hi_filt_for_conv = current_hi_filter.repeat(C, 1, 1) # [C, 1, kernel_len]
        
        # Padding: PyTorch's conv1d padding is applied to both sides if it's a single int.
        # For 'same' like behavior with stride 2, padding needs to be adjusted.
        # PyWavelet's DWT often uses 'symmetric' or other modes.
        # Here, we use padding that attempts to keep dimensions consistent with stride.
        # Effective padding for stride=2: (kernel_len - stride) // 2. Let's use kernel_len // 2 for simplicity.
        padding_size = self.kernel_len // 2 
        
        # Ensure input length is sufficient for convolution with stride
        if T < self.kernel_len:
             # Pad T to be at least kernel_len for the convolution
            padding_diff = self.kernel_len - T
            x = F.pad(x, (padding_diff // 2, padding_diff - padding_diff // 2), mode='replicate')
            T = x.size(2) # Update T

        cA = F.conv1d(x, lo_filt_for_conv, stride=2, padding=padding_size, groups=C)
        cD = F.conv1d(x, hi_filt_for_conv, stride=2, padding=padding_size, groups=C)
        return cA, cD

class MultiLevelWaveletBlock(nn.Module):
    def __init__(self, model_dim: int, wavelet_name: str = 'db4', levels: int = 3, trainable_filters: bool = False):
        super().__init__()
        self.levels = levels
        self.dwt_layers = nn.ModuleList([DWT1D(wavelet_name, trainable_filters) for _ in range(levels)])
        
        # Projections for cA and cD at the final level to match model_dim when concatenated
        # Each will be projected to model_dim, then concatenated and projected back to model_dim
        self.final_cA_proj = nn.Linear(model_dim, model_dim)
        self.final_cD_proj = nn.Linear(model_dim, model_dim)
        self.final_fusion = nn.Linear(model_dim * 2, model_dim) # Fuse projected cA and cD
        self.norm = nn.LayerNorm(model_dim) # Added LayerNorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size * num_symbols, seq_len, model_dim]
        x_perm = x.permute(0, 2, 1)  # [B*N, model_dim, seq_len]
        
        current_signal_cA = x_perm
        all_cD_coeffs = [] # Store cD from all levels if needed, or just use final

        for i in range(self.levels):
            cA, cD = self.dwt_layers[i](current_signal_cA)
            current_signal_cA = cA
            if i == self.levels - 1: # Keep the cD from the last level
                final_cD = cD
        
        # Final cA and cD are at the coarsest scale
        # Upsample them back to the original sequence length
        original_T = x.size(1) # seq_len
        
        # cA_final_upsampled: [B*N, model_dim, original_T]
        cA_final_upsampled = F.interpolate(current_signal_cA, size=original_T, mode='linear', align_corners=False)
        # cD_final_upsampled: [B*N, model_dim, original_T]
        cD_final_upsampled = F.interpolate(final_cD, size=original_T, mode='linear', align_corners=False)
        
        # Permute to [B*N, original_T, model_dim] for Linear layers
        cA_final_upsampled = cA_final_upsampled.permute(0, 2, 1)
        cD_final_upsampled = cD_final_upsampled.permute(0, 2, 1)
        
        projected_cA = self.final_cA_proj(cA_final_upsampled) # [B*N, original_T, model_dim]
        projected_cD = self.final_cD_proj(cD_final_upsampled) # [B*N, original_T, model_dim]
        
        concatenated_coeffs = torch.cat((projected_cA, projected_cD), dim=2) # [B*N, original_T, model_dim*2]
        output = self.final_fusion(concatenated_coeffs) # [B*N, original_T, model_dim]
        output = self.norm(output) # Apply LayerNorm
        return output


class MultiScaleFeatureExtractor(nn.Module):
    """多尺度特徵提取器：並行處理不同時間窗口"""
    
    def __init__(self, input_dim: int, hidden_dim: int, scales: List[int] = [3, 5, 7, 11]):
        super().__init__()
        self.scales = scales
        self.hidden_dim = hidden_dim
        
        # 每個尺度的卷積層
        num_scales = len(scales)
        base_out_channels = hidden_dim // num_scales
        remainder = hidden_dim % num_scales
        
        current_out_channels = []
        for i in range(num_scales):
            channels = base_out_channels + (1 if i < remainder else 0)
            current_out_channels.append(channels)

        self.scale_convs = nn.ModuleList()
        for i, scale in enumerate(scales):
            self.scale_convs.append(
                nn.Conv1d(input_dim, current_out_channels[i],
                         kernel_size=scale, padding=scale//2, groups=1)
            )
        
        # 特徵融合層
        # The input to this linear layer is the sum of current_out_channels, which should now be hidden_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), # This should now be correct
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )
        
        # 時間注意力權重
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8, # 可以考慮也設為可配置
            batch_first=True
        )

        # 自適應池化層
        # 假設我們希望將序列長度池化到 TIMESTEPS // 2，或者一個可配置的固定值
        # 這裡我們暫時使用 TIMESTEPS // 2 作為示例
        self.adaptive_pool_output_size = TIMESTEPS // 2 
        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_size=self.adaptive_pool_output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim]
        Returns:
            多尺度融合特徵: [batch_size, pooled_seq_len, hidden_dim]
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
        
        # 殘差連接
        output_features = attended_features + fused_features

        # 應用自適應池化
        # 轉換為池化層期望的格式: [batch, channels, seq_len]
        # 在這裡, channels 是 hidden_dim, seq_len 是當前的序列長度
        output_features_pooled = output_features.transpose(1, 2) # [batch_size, hidden_dim, seq_len]
        output_features_pooled = self.adaptive_pool(output_features_pooled) # [batch_size, hidden_dim, self.adaptive_pool_output_size]
        
        # 轉回標準格式: [batch_size, pooled_seq_len, hidden_dim]
        output_features_pooled = output_features_pooled.transpose(1, 2)
        
        return output_features_pooled


class MarketStateDetector(nn.Module):
    """市場狀態檢測器：檢測趨勢、波動、均值回歸、突破等市場狀態"""
    
    def __init__(self, d_model: int, num_market_states: int = 4): # Added num_market_states
        super().__init__()
        self.d_model = d_model
        self.num_market_states = num_market_states # Store it
        
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
                nn.Linear(d_model // 2, self.num_market_states),  # Use self.num_market_states
                nn.Softmax(dim=-1)
            )
        })
        
        # 狀態融合網絡
        # Input features: 1 (trend) + 1 (volatility) + 1 (momentum) + num_market_states (regime)
        fusion_input_dim = 3 + self.num_market_states
        self.state_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model // 8),
            nn.GELU(),
            nn.Linear(d_model // 8, self.num_market_states), # Output num_market_states
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]: # Return type changed for clarity, was torch.Tensor
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
                state_features.append(state_value)  # [batch, num_market_states]
            else:
                state_features.append(state_value)  # [batch, 1]
        
        # 拼接所有狀態特徵
        # concatenated_states = torch.cat(state_features, dim=-1)  # [batch, 7]
        # Corrected concatenation logic
        processed_state_features = []
        for feat in state_features:
            if feat.dim() == 1: # if it's [batch_size] from a single output (like trend, volatility, momentum)
                processed_state_features.append(feat.unsqueeze(-1)) # make it [batch_size, 1]
            else: # it's already [batch_size, num_states] for regime
                processed_state_features.append(feat)
        concatenated_states = torch.cat(processed_state_features, dim=-1) # [batch, 3 + num_market_states]
        
        # 融合得到最終市場狀態
        final_market_state = self.state_fusion(concatenated_states)
        states['final_state'] = final_market_state # [batch, num_market_states]
        
        return states # Return the dict


class AdaptiveAttentionLayer(nn.Module):
    """自適應注意力層：根據市場狀態動態調整注意力模式"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, num_market_states: int = 4): # Added num_market_states
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_market_states = num_market_states
        
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.lstm_layer = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model, # Outputting same dimension for easier fusion
            num_layers=1,        # Can be configured later if needed
            batch_first=True
        )

        self.market_state_detector = MarketStateDetector(d_model, num_market_states=self.num_market_states)
        
        self.attention_modulators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
                nn.Sigmoid()
            ) for _ in range(self.num_market_states)
        ])
        
        self.temperature_controller = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Softplus()
        )
        
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
        
        # LSTM Processing
        lstm_out, _ = self.lstm_layer(x) # lstm_out shape: [batch_size, seq_len, d_model]
        
        # Market state detection (uses original x)
        market_states = self.market_state_detector(x)
        final_state = market_states['final_state']  # [batch, num_market_states]
        
        # Adaptive temperature (uses original x)
        avg_features = x.mean(dim=1)  # [batch, d_model]
        temperature = self.temperature_controller(avg_features)  # [batch, 1]
        
        # Standard attention calculation (on original x)
        attn_output, attn_weights = self.attention(
            x, x, x, key_padding_mask=key_padding_mask
        )
        
        # Fuse LSTM output with Attention output
        fused_representation = lstm_out + attn_output # Simple addition for fusion
        
        # Modulate fused representation based on market state
        modulated_outputs = []
        for i, modulator in enumerate(self.attention_modulators):
            state_weight = final_state[:, i:i+1, None]  # [batch, 1, 1]
            modulated = modulator(fused_representation) * state_weight 
            modulated_outputs.append(modulated)
        
        # Weighted fusion of state-specific outputs
        adaptive_output = sum(modulated_outputs)
        
        # Apply adaptive temperature
        adaptive_output = adaptive_output * temperature.unsqueeze(1)
        
        # Final projection and dropout
        output = self.output_projection(adaptive_output)
        output = self.dropout(output)
        
        return output + x  # Residual connection with original x


class EnhancedTransformerLayer(nn.Module):
    """增強版Transformer層：集成自適應注意力和改進的FFN"""
    
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, 
                 dropout: float = 0.1, use_adaptive_attention: bool = True, num_market_states: int = 4): # Added use_adaptive_attention and num_market_states
        super().__init__()
        self.use_adaptive_attention = use_adaptive_attention
        
        # 自適應注意力層 or 標準多頭注意力
        if self.use_adaptive_attention:
            self.attention_layer = AdaptiveAttentionLayer(d_model, num_heads, dropout, num_market_states=num_market_states)
        else:
            # Fallback to standard MultiheadAttention if adaptive is not used
            self.attention_layer = nn.MultiheadAttention(
                embed_dim=d_model, 
                num_heads=num_heads, 
                dropout=dropout, 
                batch_first=True
            )

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
                key_padding_mask: Optional[torch.Tensor] = None,
                market_states: Optional[torch.Tensor] = None) -> torch.Tensor: # Added market_states for external control if needed
        
        # 自適應注意力子層
        normed_x = self.norm1(x)
        if self.use_adaptive_attention:
            # AdaptiveAttentionLayer handles market state detection internally
            # However, if we want to pass pre-computed market_states (e.g. from a global detector)
            # the AdaptiveAttentionLayer's forward method would need to accept it.
            # For now, assuming internal detection as per its original design.
            # If market_states is passed to this layer, it implies the internal detector of AdaptiveAttentionLayer might be bypassed or augmented.
            # Current AdaptiveAttentionLayer.forward does not take market_states.
            # Let's stick to its current API.
            attn_output = self.attention_layer(normed_x, key_padding_mask=key_padding_mask)
        else:
            # Standard nn.MultiheadAttention returns (attn_output, attn_output_weights)
            attn_output, _ = self.attention_layer(normed_x, normed_x, normed_x, key_padding_mask=key_padding_mask)
        
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


class PositionalEncoding(nn.Module):
    """位置編碼"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim] if batch_first=False
               Tensor, shape [batch_size, seq_len, embedding_dim] if batch_first=True
        """
        # 假設 x 是 [batch_size, seq_len, embedding_dim]
        x = x + self.pe[:x.size(1)].transpose(0,1) # self.pe is [max_len, 1, d_model], transpose to [1, max_len, d_model]
        return self.dropout(x)


class EnhancedTransformer(nn.Module):
    """
    增強版通用多資產交易Transformer模型
    """
    def __init__(self, 
                 input_dim: int, 
                 d_model: int, 
                 num_heads: int, 
                 num_layers: int, 
                 ffn_dim: int,
                 output_dim: int, # This is output_dim_per_symbol
                 max_seq_len: int = TIMESTEPS,
                 max_symbols: int = MAX_SYMBOLS_ALLOWED, # Max number of symbols
                 msfe_scales: List[int] = [3, 5, 7, 11],
                 cts_time_scales: List[int] = [5, 15, 30, 60],
                 dropout: float = 0.1,
                 use_msfe: bool = True,
                 use_cts_fusion: bool = True,
                 use_symbol_embedding: bool = True,
                 use_fourier_features: bool = True, # Added
                 fourier_num_modes: int = FOURIER_NUM_MODES, # Added
                 use_wavelet_features: bool = True,  # Added
                 wavelet_levels: int = WAVELET_LEVELS, # Added
                 wavelet_name: str = WAVELET_NAME,   # Added
                 wavelet_trainable_filters: bool = False, # Added
                 # Parameters for Adaptive Attention, if controlled at top level
                 use_adaptive_attention: bool = True, # Default to True for Enhanced version
                 num_market_states: int = 4           # Default number of market states
                 ):
        super().__init__()
        self.d_model = d_model
        self.use_msfe = use_msfe
        self.use_cts_fusion = use_cts_fusion
        self.use_symbol_embedding = use_symbol_embedding
        self.max_symbols = max_symbols
        self.use_fourier_features = use_fourier_features
        self.use_wavelet_features = use_wavelet_features
        self.use_adaptive_attention = use_adaptive_attention # Store this
        self.num_market_states = num_market_states # Store this

        # 0. Symbol Embedding (可選)
        if self.use_symbol_embedding:
            # Embedding for symbol ID (0 to max_symbols-1)
            self.symbol_embed = nn.Embedding(self.max_symbols, self.d_model)
            # Embedding for symbol position/order (0 to max_symbols-1)
            self.symbol_pos_embed = nn.Embedding(self.max_symbols, self.d_model)
            # LayerNorm for combined symbol embeddings
            self.symbol_embed_norm = nn.LayerNorm(self.d_model)

        # 1. 輸入嵌入層
        # The actual input_dim to this projection might be d_model if features are already projected
        # or if symbol embeddings are added to an initial feature projection.
        # Let's assume `input_dim` is the raw feature dim per symbol per timestep.
        self.input_feature_projection = nn.Linear(input_dim, d_model)
        
        # Feature fusion layer if multiple feature extractors (Fourier, Wavelet) are used before MSFE/Transformer
        # The number of features to fuse depends on which ones are active.
        num_early_features = 1 # Base features (after input_feature_projection)
        if self.use_fourier_features:
            self.fourier_block = FourierFeatureBlock(d_model, num_modes=fourier_num_modes)
            num_early_features += 1
        else:
            self.fourier_block = None
            
        if self.use_wavelet_features:
            self.wavelet_block = MultiLevelWaveletBlock(d_model, wavelet_name=wavelet_name, levels=wavelet_levels, trainable_filters=wavelet_trainable_filters)
            num_early_features += 1
        else:
            self.wavelet_block = None

        if num_early_features > 1:
            self.early_feature_fusion_layer = nn.Sequential(
                nn.Linear(d_model * num_early_features, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model)
            )
        else:
            self.early_feature_fusion_layer = nn.Identity()


        # 2. 多尺度特徵提取器 (可選)
        if self.use_msfe:
            self.msfe = MultiScaleFeatureExtractor(d_model, d_model, scales=msfe_scales)
            # MSFE的輸出序列長度是固定的 (TIMESTEPS // 2), 更新max_seq_len給後續層
            self.processed_seq_len = self.msfe.adaptive_pool_output_size
        else:
            self.msfe = None
            self.processed_seq_len = max_seq_len
        
        # 3. 位置編碼 (Temporal Positional Encoding)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=self.processed_seq_len)
        
        # 4. 增強版Transformer層堆疊
        self.transformer_layers = nn.ModuleList([
            EnhancedTransformerLayer(d_model, num_heads, ffn_dim, dropout, 
                                     use_adaptive_attention=self.use_adaptive_attention, 
                                     num_market_states=self.num_market_states)
            for _ in range(num_layers)
        ])
        
        # 5. 跨時間尺度融合 (可選)
        if self.use_cts_fusion:
            # CrossTimeScaleFusion 期望的輸入是 [batch, seq_len, d_model]
            # 這裡的 seq_len 應該是 Transformer 層處理後的序列長度，即 self.processed_seq_len
            self.cts_fusion = CrossTimeScaleFusion(d_model, time_scales=cts_time_scales)
        else:
            self.cts_fusion = None
            
        # 6. 輸出層
        self.output_projection = nn.Linear(d_model, output_dim)
        
        # 初始化權重
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, 
                src: torch.Tensor, 
                symbol_ids: Optional[torch.Tensor] = None, # New: [batch_size, num_active_symbols]
                src_key_padding_mask: Optional[torch.Tensor] = None # Mask for padded SYMBOLS [batch_size, num_active_symbols]
               ) -> torch.Tensor:
        """
        Args:
            src: 輸入序列, shape [batch_size, num_active_symbols, seq_len, input_dim]
            symbol_ids: 每個活躍符號的ID, shape [batch_size, num_active_symbols].
                        用於 symbol_embed. 如果 None 且 use_symbol_embedding=True, 則自動生成.
            src_key_padding_mask: Mask for indicating padded symbols. 
                                  Shape [batch_size, num_active_symbols]. True for padded symbols.
                                  This will be converted to a mask for Transformer layers.
        Returns:
            輸出序列, shape [batch_size, num_active_symbols, output_dim_per_symbol] 
        """
        batch_size, num_active_symbols, seq_len, _ = src.shape
        
        # 1. 輸入特徵投影
        # src: [B, N_active, T, F_in] -> projected_features: [B, N_active, T, D_model]
        projected_features = self.input_feature_projection(src)

        # Reshape for per-symbol processing by Fourier/Wavelet/MSFE if any
        # x_symbol_batched: [B * N_active, T, D_model]
        x_symbol_batched = projected_features.reshape(batch_size * num_active_symbols, seq_len, self.d_model)

        # 1.a Early Feature Extraction (Fourier, Wavelet)
        early_features_list = [x_symbol_batched] # Start with base projected features
        if self.fourier_block is not None:
            fourier_out = self.fourier_block(x_symbol_batched.clone()) # Use clone if x_symbol_batched is modified in-place by other blocks
            early_features_list.append(fourier_out)
        
        if self.wavelet_block is not None:
            wavelet_out = self.wavelet_block(x_symbol_batched.clone())
            early_features_list.append(wavelet_out)

        if len(early_features_list) > 1:
            concatenated_early_features = torch.cat(early_features_list, dim=-1)
            x_processed_early = self.early_feature_fusion_layer(concatenated_early_features)
        else:
            x_processed_early = x_symbol_batched # Or early_features_list[0]

        # x_processed_early is [B * N_active, T, D_model]

        # 2. 多尺度特徵提取 (如果啟用)
        if self.msfe is not None:
            # MSFE expects [batch, seq_len, features]
            # Input is x_processed_early: [B*N_active, T, D_model]
            x_after_msfe = self.msfe(x_processed_early) 
            # x_after_msfe: [B*N_active, msfe_output_seq_len, D_model]
            current_seq_len = self.processed_seq_len # This is msfe_output_seq_len
        else:
            x_after_msfe = x_processed_early
            # current_seq_len is the original seq_len (or TIMESTEPS if not using MSFE)
            current_seq_len = seq_len # or self.processed_seq_len which is max_seq_len here

        # x_after_msfe is [B*N_active, current_seq_len, D_model]

        # 3. 時間位置編碼 (Temporal Positional Encoding)
        # Applied to each symbol's time series independently
        x_temp_pos_encoded = self.pos_encoder(x_after_msfe) # pos_encoder expects [B', T', D]

        # x_temp_pos_encoded is [B*N_active, current_seq_len, D_model]

        # 4. Symbol Embedding and Symbol Positional Embedding (可選)
        # These are added *after* temporal processing (like MSFE, pos_encoder)
        # but *before* the main Transformer layers that might do cross-symbol attention.
        # Reshape back to [B, N_active, current_seq_len, D_model] to add symbol-level embeddings
        x_before_sym_embed = x_temp_pos_encoded.view(batch_size, num_active_symbols, current_seq_len, self.d_model)

        if self.use_symbol_embedding:
            if symbol_ids is None:
                # Create default symbol_ids if not provided: 0, 1, 2... for each active symbol
                symbol_ids = torch.arange(num_active_symbols, device=src.device).unsqueeze(0).expand(batch_size, -1)
            
            # Symbol ID embedding: [B, N_active] -> [B, N_active, D_model]
            sym_id_embed = self.symbol_embed(symbol_ids)
            # Symbol Positional (order) embedding: [B, N_active] -> [B, N_active, D_model]
            # Assuming symbol_ids themselves can serve as order if they are 0..N-1, or use a separate arange
            sym_pos_ids = torch.arange(num_active_symbols, device=src.device).unsqueeze(0).expand(batch_size, -1)
            sym_pos_embed_vec = self.symbol_pos_embed(sym_pos_ids)
            
            # Combine symbol ID and positional embeddings
            combined_symbol_embeddings = sym_id_embed + sym_pos_embed_vec # [B, N_active, D_model]
            combined_symbol_embeddings = self.symbol_embed_norm(combined_symbol_embeddings)

            # Expand for seq_len and add to features:
            # [B, N_active, D_model] -> [B, N_active, 1, D_model] -> [B, N_active, current_seq_len, D_model]
            expanded_symbol_embeddings = combined_symbol_embeddings.unsqueeze(2).expand(-1, -1, current_seq_len, -1)
            
            x = x_before_sym_embed + expanded_symbol_embeddings
        else:
            x = x_before_sym_embed
        
        # x is now [B, N_active, current_seq_len, D_model]

        # Prepare for Transformer encoder layers.
        # The Transformer layers will operate over the `num_active_symbols` dimension if we want cross-asset attention,
        # or over the `current_seq_len` dimension for temporal attention per symbol.
        # The current EnhancedTransformerLayer is designed for temporal attention: input [B', T', D_model]
        # So, we need to reshape x: [B, N_active, T_current, D_model] -> [B*N_active, T_current, D_model]
        
        x_for_transformer = x.reshape(batch_size * num_active_symbols, current_seq_len, self.d_model)

        # src_key_padding_mask is [B, N_active]. It masks entire symbols.
        # The EnhancedTransformerLayer expects a mask of shape [B', T'] for padded timesteps within a sequence.
        # If a symbol is padded (masked by src_key_padding_mask), all its timesteps should be masked.
        
        # `current_key_padding_mask_for_transformer` should be [B*N_active, T_current]
        current_key_padding_mask_for_transformer: Optional[torch.Tensor] = None
        if src_key_padding_mask is not None:
            # src_key_padding_mask [B, N_active] -> expanded to [B, N_active, T_current]
            expanded_symbol_mask = src_key_padding_mask.unsqueeze(-1).expand(-1, -1, current_seq_len)
            # Reshape to [B*N_active, T_current]
            current_key_padding_mask_for_transformer = expanded_symbol_mask.reshape(batch_size * num_active_symbols, current_seq_len)

        # If MSFE changed sequence length, the original src_key_padding_mask (for symbols) is still valid for symbols,
        # but if there was a per-timestep mask, it would need adjustment.
        # Here, src_key_padding_mask is for symbols, so it's fine.
        # The `EnhancedTransformerLayer`'s `key_padding_mask` is for timesteps.
        # So, if a symbol is padded, all its timesteps are masked.

        # 5. Transformer 編碼層
        for layer in self.transformer_layers:
            x_for_transformer = layer(x_for_transformer, key_padding_mask=current_key_padding_mask_for_transformer) 
            
        # x_for_transformer is [B*N_active, current_seq_len, D_model]

        # 6. 跨時間尺度融合 (如果啟用)
        # CTS Fusion also expects [B', T', D_model]
        if self.cts_fusion is not None:
            x_for_transformer = self.cts_fusion(x_for_transformer) 
        
        # After temporal processing, we might want to pool or select features across time
        # For a SAC agent, we typically need one feature vector per symbol.
        # Let's take the output of the last timestep (or average pool).
        # Using last timestep:
        # x_final_time_step = x_for_transformer[:, -1, :] # [B*N_active, D_model]
        # Or average pool across time:
        x_pooled_time = x_for_transformer.mean(dim=1) # [B*N_active, D_model]


        # Reshape back to [B, N_active, D_model]
        x_output_per_symbol = x_pooled_time.view(batch_size, num_active_symbols, self.d_model)
        
        # 7. 輸出投影
        # Output per symbol: [B, N_active, D_model] -> [B, N_active, output_dim_per_symbol]
        output = self.output_projection(x_output_per_symbol) 
        
        # Apply symbol padding mask to the final output
        if src_key_padding_mask is not None:
            # Ensure mask is boolean
            if src_key_padding_mask.dtype != torch.bool:
                src_key_padding_mask = src_key_padding_mask.bool()
            # Expand mask for the output_dim dimension
            output_mask = src_key_padding_mask.unsqueeze(-1).expand_as(output)
            output = output.masked_fill(output_mask, 0.0) # Fill padded symbols' outputs with 0

        return output

# Example Usage (for testing purposes)
if __name__ == '__main__':
    device = DEVICE
    # Test parameters
    batch_size = 2
    num_active_symbols_test = MAX_SYMBOLS_ALLOWED # Test with max symbols
    # num_active_symbols_test = 5 # Test with fewer symbols than MAX_SYMBOLS_ALLOWED
    
    seq_len_test = TIMESTEPS # 128
    input_dim_test = 10 # e.g., 10 features per timestep

    d_model_test = 64
    num_heads_test = 4 # Reduced for faster testing
    num_layers_test = 2 # Reduced for faster testing
    ffn_dim_test = d_model_test * 2 # Reduced
    output_dim_per_symbol_test = 3 # e.g., predict buy, sell, hold probabilities

    logger.info(f"Device: {device}")
    logger.info(f"TIMESTEPS: {TIMESTEPS}, MSFE adaptive pool output size: {TIMESTEPS // 2 if TIMESTEPS > 0 else 'N/A'}")
    logger.info(f"MAX_SYMBOLS_ALLOWED: {MAX_SYMBOLS_ALLOWED}")
    logger.info(f"Fourier Modes: {FOURIER_NUM_MODES}, Wavelet Levels: {WAVELET_LEVELS}, Wavelet Name: {WAVELET_NAME}")

    configs_to_test = [
        {"name": "Full (MSFE, Wavelet, Fourier, SymEmb, CTS)", "use_msfe": True, "use_wavelet_features": True, "use_fourier_features": True, "use_symbol_embedding": True, "use_cts_fusion": True},
        {"name": "MSFE only", "use_msfe": True, "use_wavelet_features": False, "use_fourier_features": False, "use_symbol_embedding": False, "use_cts_fusion": False},
        {"name": "Wavelet only", "use_msfe": False, "use_wavelet_features": True, "use_fourier_features": False, "use_symbol_embedding": False, "use_cts_fusion": False},
        {"name": "Fourier only", "use_msfe": False, "use_wavelet_features": False, "use_fourier_features": True, "use_symbol_embedding": False, "use_cts_fusion": False},
        {"name": "Basic (No MSFE, No Wavelet, No Fourier, No SymEmb, No CTS)", "use_msfe": False, "use_wavelet_features": False, "use_fourier_features": False, "use_symbol_embedding": False, "use_cts_fusion": False},
        {"name": "SymEmb + CTS (No MSFE, Wavelet, Fourier)", "use_msfe": False, "use_wavelet_features": False, "use_fourier_features": False, "use_symbol_embedding": True, "use_cts_fusion": True},
    ]

    for config in configs_to_test:
        print(f"\\n--- Testing Model Configuration: {config['name']} ---")
        model = EnhancedTransformer(
            input_dim=input_dim_test,
            d_model=d_model_test,
            num_heads=num_heads_test,
            num_layers=num_layers_test,
            ffn_dim=ffn_dim_test,
            output_dim=output_dim_per_symbol_test,
            max_seq_len=seq_len_test,
            max_symbols=MAX_SYMBOLS_ALLOWED, # Max capacity
            use_msfe=config["use_msfe"],
            use_cts_fusion=config["use_cts_fusion"],
            use_symbol_embedding=config["use_symbol_embedding"],
            use_fourier_features=config["use_fourier_features"],
            fourier_num_modes=FOURIER_NUM_MODES // 2 or 16, # Adjust for test
            use_wavelet_features=config["use_wavelet_features"],
            wavelet_levels=WAVELET_LEVELS -1 or 1, # Adjust for test
            wavelet_name=WAVELET_NAME
        ).to(device)
        model.eval()

        # Input source data: [batch_size, num_active_symbols, seq_len, input_dim]
        src_for_mask_test = torch.randn(batch_size, num_active_symbols_test, seq_len_test, input_dim_test).to(device)
        
        # Symbol IDs for embedding: [batch_size, num_active_symbols]
        # Simulate that the first `num_active_symbols_test` are active
        symbol_ids_test = torch.arange(num_active_symbols_test, device=device).unsqueeze(0).expand(batch_size, -1)

        # Padding mask for symbols: [batch_size, num_active_symbols]
        # True means padded. Let's pad half of the symbols for testing.
        symbol_padding_mask_test = torch.zeros(batch_size, num_active_symbols_test, dtype=torch.bool).to(device)
        num_to_pad = num_active_symbols_test // 2
        if num_to_pad > 0:
            symbol_padding_mask_test[:, -num_to_pad:] = True
        
        print(f"Input shape (src_for_mask_test): {src_for_mask_test.shape}")
        if config["use_symbol_embedding"]:
             print(f"Symbol IDs shape: {symbol_ids_test.shape}")
        print(f"Symbol mask shape: {symbol_padding_mask_test.shape}, Num masked symbols per batch item: {num_to_pad if num_to_pad > 0 else 0}")


        output = model(src_for_mask_test, symbol_ids=symbol_ids_test if config["use_symbol_embedding"] else None, src_key_padding_mask=symbol_padding_mask_test)
        
        print(f"Output shape: {output.shape}")
        expected_output_shape = (batch_size, num_active_symbols_test, output_dim_per_symbol_test)
        assert output.shape == expected_output_shape, \
            f"Output shape mismatch for config '{config['name']}'. Expected {expected_output_shape}, Got {output.shape}"

        # Check if outputs for padded symbols are zero
        if num_to_pad > 0:
            padded_outputs = output[symbol_padding_mask_test] # Selects elements where mask is True
            # Expected shape of padded_outputs: (batch_size * num_to_pad, output_dim_per_symbol_test)
            # Or more simply, check the sum of absolute values of the masked parts
            masked_sum = 0
            for i in range(batch_size):
                for j in range(num_active_symbols_test):
                    if symbol_padding_mask_test[i,j]:
                        masked_sum += output[i,j,:].abs().sum()

            assert torch.allclose(masked_sum, torch.tensor(0.0, device=device), atol=1e-6), \
                f"Output for padded symbols is not zero for config '{config['name']}'. Sum of abs values: {masked_sum.item()}"
            print(f"Masked symbol output check passed (sum: {masked_sum.item()}).")
        else:
            print("No symbols were padded in this test configuration.")

        print(f"Configuration '{config['name']}' passed shape and mask checks.")

    print("\\nEnhancedTransformer all configured tests completed.")
