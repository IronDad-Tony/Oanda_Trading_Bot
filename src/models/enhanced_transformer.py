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
import os # Added for os.path.exists
from src.features.market_state_detector import GMMMarketStateDetector # Added for GMM integration

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
                key_padding_mask: Optional[torch.Tensor] = None,
                external_market_state_probs: Optional[torch.Tensor] = None) -> torch.Tensor: # MODIFIED SIGNATURE
        """
        Args:
            x: [batch_size, seq_len, d_model] (Note: batch_size here can be B*N from EnhancedTransformer)
            key_padding_mask: [batch_size, seq_len]
            external_market_state_probs: [batch_size, num_market_states], optional. (Note: batch_size here can be B*N)
        """
        batch_size_eff, seq_len, d_model = x.shape # batch_size_eff could be B or B*N
        
        # LSTM Processing
        lstm_out, _ = self.lstm_layer(x) # lstm_out shape: [batch_size_eff, seq_len, d_model]
        
        if external_market_state_probs is not None:
            final_state = external_market_state_probs  # Shape: [batch_size_eff, num_market_states]
        else:
            # Market state detection (uses original x)
            # MarketStateDetector's forward takes [B, D_model] or [B, T, D_model].
            # If x is [B_eff, T, D_model], it will average over T if not handled by detector.
            # The current MarketStateDetector averages over dim=1 if x.dim() == 3.
            market_states_internal = self.market_state_detector(x) 
            final_state = market_states_internal['final_state']  # Shape: [batch_size_eff, num_market_states]
        
        # Adaptive temperature (uses original x)
        avg_features = x.mean(dim=1)  # [batch_size_eff, d_model]
        temperature = self.temperature_controller(avg_features)  # [batch_size_eff, 1]
        
        # Standard attention calculation (on original x)
        attn_output, attn_weights = self.attention(
            x, x, x, key_padding_mask=key_padding_mask
        )
        
        # Fuse LSTM output with Attention output
        fused_representation = lstm_out + attn_output # Simple addition for fusion
        
        # Modulate fused representation based on market state
        modulated_outputs = []
        for i, modulator in enumerate(self.attention_modulators):
            # final_state is [batch_size_eff, num_market_states]
            # state_weight needs to be [batch_size_eff, 1, 1] for broadcasting with fused_representation [batch_size_eff, seq_len, d_model]
            state_weight = final_state[:, i:i+1].unsqueeze(-1)  # [batch_size_eff, 1, 1]
            modulated = modulator(fused_representation) * state_weight 
            modulated_outputs.append(modulated)
        
        # Weighted fusion of state-specific outputs
        adaptive_output = sum(modulated_outputs)
        
        # Apply adaptive temperature
        adaptive_output = adaptive_output * temperature.unsqueeze(1) # temperature is [B_eff, 1], unsqueeze to [B_eff, 1, 1]
        
        # Final projection and dropout
        output = self.output_projection(adaptive_output)
        output = self.dropout(output)
        
        return output + x  # Residual connection with original x


class EnhancedTransformerLayer(nn.Module):
    """增強版Transformer層：集成自適應注意力和改進的FFN"""
    
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, 
                 dropout: float = 0.1, use_adaptive_attention: bool = True, num_market_states: int = 4):
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
                external_market_state_probs: Optional[torch.Tensor] = None) -> torch.Tensor: # MODIFIED PARAMETER
        
        # 自適應注意力子層
        normed_x = self.norm1(x)
        if self.use_adaptive_attention:
            # Pass external_market_state_probs to AdaptiveAttentionLayer
            attn_output = self.attention_layer(normed_x, 
                                               key_padding_mask=key_padding_mask, 
                                               external_market_state_probs=external_market_state_probs)
        else:
            # Standard nn.MultiheadAttention returns (attn_output, attn_output_weights)
            # It does not add residual connection itself. This layer's structure assumes attn_output is the full sublayer output.
            # A residual connection might be needed here if not using adaptive attention: x_res = normed_x + attn_output_raw
            attn_output_raw, _ = self.attention_layer(normed_x, normed_x, normed_x, key_padding_mask=key_padding_mask)
            # The current structure directly uses attn_output. If standard MHA, this lacks residual.
            # For now, maintaining existing structure, only passing probs.
            # Consider adding residual: attn_output = normed_x + self.dropout_attn(attn_output_raw) if a dropout layer for attention is added
            attn_output = attn_output_raw # This might be an issue if residual is expected by subsequent ops.
                                          # However, AdaptiveAttentionLayer *does* include a residual.
                                          # This means behavior differs based on use_adaptive_attention.
        
        # FFN子層
        ffn_input = self.norm2(attn_output) # If attn_output doesn't have residual, this norm is on raw attention.
        ffn_output = self.ffn(ffn_input)
