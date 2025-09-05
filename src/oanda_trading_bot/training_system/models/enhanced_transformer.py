# src/models/enhanced_transformer.py
"""
å¢žå¼·ç‰ˆé€šç”¨å¤šè³‡ç”¢äº¤æ˜“Transformeræ¨¡åž‹
åŸºæ–¼åŽŸæœ‰UniversalTradingTransformerï¼Œå¤§å¹…æå‡æ¨¡åž‹è¤‡é›œåº¦å’Œå­¸ç¿’èƒ½åŠ›

ä¸»è¦å¢žå¼·ï¼š
1. æ·±åº¦æž¶æ§‹ï¼š12-16å±¤Transformer
2. å¤šå°ºåº¦ç‰¹å¾µæå–ï¼šä¸¦è¡Œè™•ç†ä¸åŒæ™‚é–“çª—å£
3. è‡ªé©æ‡‰æ³¨æ„åŠ›æ©Ÿåˆ¶ï¼šå‹•æ…‹èª¿æ•´æ³¨æ„åŠ›æ¬Šé‡
4. è·¨æ™‚é–“å°ºåº¦èžåˆï¼šæ•´åˆå¤šå€‹æ™‚é–“ç¶­åº¦ä¿¡æ¯
5. æ®˜å·®é€£æŽ¥å„ªåŒ–ï¼šæ”¹å–„æ·±åº¦ç¶²çµ¡è¨“ç·´
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
from oanda_trading_bot.training_system.features.market_state_detector import GMMMarketStateDetector # Added for GMM integration
from oanda_trading_bot.training_system.models.transformer_model import PositionalEncoding
from oanda_trading_bot.training_system.models.custom_layers import CrossTimeScaleFusion
from oanda_trading_bot.training_system.models.graph_layers import GraphBlock

try:
    from oanda_trading_bot.training_system.common.logger_setup import logger
    from oanda_trading_bot.training_system.common.config import DEVICE, MAX_SYMBOLS_ALLOWED, TIMESTEPS
    # Attempt to import wavelet and fourier specific configs if they exist
    # These might be moved to a more general model config or stay here
    try:
        from oanda_trading_bot.training_system.common.config import FOURIER_NUM_MODES, WAVELET_LEVELS, WAVELET_NAME
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


# Define LearnedPositionalEncoding if not available elsewhere
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, d_model] or [seq_len, batch_size, d_model] if batch_first=False
        # Assuming batch_first=True for consistency with Transformer layers
        # Or, more generally, ensure slicing is correct for the input shape.
        # If x is [B, S, E], self.pos_embedding is [1, max_len, E]. We need [1, S, E].
        # If x is [S, B, E], self.pos_embedding.transpose(0,1) is [max_len, 1, E]. We need [S, 1, E].
        
        # Assuming x is [B, S, E] (batch_first=True for nn.TransformerEncoderLayer)
        # Or if the input to this module is already permuted to [S, B, E] as expected by some Transformer impls
        
        # Let's assume x is [Batch, SeqLen, EmbDim] as is common for batch_first=True
        # Or, if it's [SeqLen, Batch, EmbDim], the addition still works due to broadcasting if seq_len matches.
        # The nn.TransformerEncoderLayer by default expects [SeqLen, Batch, EmbDim].
        # Our PositionalEncoding in transformer_model.py produces [1, max_len, d_model] and adds to x,
        # assuming x is [batch_size, seq_len, d_model] and then slices pe as self.pe[:, :x.size(1)]
        # which results in [1, seq_len, d_model] that broadcasts over batch_size.

        # For LearnedPositionalEncoding, self.pos_embedding is [1, max_len, d_model]
        # If x is [B, S, E], we need to add self.pos_embedding[:, :x.size(1), :]
        # If x is [S, B, E], we need to add self.pos_embedding.squeeze(0)[:x.size(0), :].unsqueeze(1)
        
        # Given the context of EnhancedTransformer, x is likely [B*N, T, C] before Transformer layers,
        # and then permuted to [T, B*N, C] for the nn.TransformerEncoder.
        # Let's assume this PE is applied when x is [T, B*N, C] or [B*N, T, C] and handle accordingly.
        # The PositionalEncoding in transformer_model.py is applied when x is [B, S, E] (batch_first=True like).

        # If this PE is used like the sinusoidal one (applied to [B,S,E] or [S,B,E] and sliced):
        if x.size(0) == self.pos_embedding.size(1) and len(x.shape) == 3 and x.size(2) == self.pos_embedding.size(2):
            # x is likely [SeqLen, Batch, EmbDim]
            x = x + self.pos_embedding.squeeze(0)[:x.size(0), :].unsqueeze(1)
        elif x.size(1) == self.pos_embedding.size(1) and len(x.shape) == 3 and x.size(2) == self.pos_embedding.size(2):
             # This case is unlikely if max_len is large and x.size(1) is the actual sequence length.
             # More likely, x.size(1) is seq_len, and self.pos_embedding needs slicing.
             x = x + self.pos_embedding[:, :x.size(1), :]
        elif len(x.shape) == 3 and x.shape[1] <= self.pos_embedding.shape[1]: # Common case: x is [B, S, E]
            x = x + self.pos_embedding[:, :x.shape[1], :]
        elif len(x.shape) == 3 and x.shape[0] <= self.pos_embedding.shape[1]: # Common case: x is [S, B, E]
             x = x + self.pos_embedding.squeeze(0)[:x.shape[0], :].unsqueeze(1)
        else:
            # Fallback or error, this indicates a shape mismatch not handled by simple slicing.
            # This might happen if pe_seq_len used for PE init is different from actual seq_len at forward pass
            # and the PE is not designed to handle dynamic seq len by just slicing (e.g. if it was [SeqLen, EmbDim])
            logger.warning(f"LearnedPositionalEncoding shape mismatch: x.shape={x.shape}, pos_embedding.shape={self.pos_embedding.shape}. Ensure correct permutation and slicing.")
            # Attempt a robust slice assuming x.size(1) is seq_len for [B,S,E] or x.size(0) for [S,B,E]
            if len(x.shape) == 3:
                if x.shape[1] <= self.pos_embedding.shape[1]: # [B, S, E]
                    x = x + self.pos_embedding[:, :x.shape[1], :]
                elif x.shape[0] <= self.pos_embedding.shape[1]: # [S, B, E]
                    x = x + self.pos_embedding.squeeze(0)[:x.shape[0], :].unsqueeze(1)


        return self.dropout(x)

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
    """å¤šå°ºåº¦ç‰¹å¾µæå–å™¨ï¼šä¸¦è¡Œè™•ç†ä¸åŒæ™‚é–“çª—å£"""
    
    def __init__(self, input_dim: int, hidden_dim: int, scales: List[int] = [3, 5, 7, 11]):
        super().__init__()
        self.scales = scales
        self.hidden_dim = hidden_dim
        
        # æ¯å€‹å°ºåº¦çš„å·ç©å±¤
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
        
        # ç‰¹å¾µèžåˆå±¤
        # The input to this linear layer is the sum of current_out_channels, which should now be hidden_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), # This should now be correct
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )
        
        # æ™‚é–“æ³¨æ„åŠ›æ¬Šé‡
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8, # å¯ä»¥è€ƒæ…®ä¹Ÿè¨­ç‚ºå¯é…ç½®
            batch_first=True
        )

        # è‡ªé©æ‡‰æ± åŒ–å±¤
        # å‡è¨­æˆ‘å€‘å¸Œæœ›å°‡åºåˆ—é•·åº¦æ± åŒ–åˆ° TIMESTEPS // 2ï¼Œæˆ–è€…ä¸€å€‹å¯é…ç½®çš„å›ºå®šå€¼
        # é€™è£¡æˆ‘å€‘æš«æ™‚ä½¿ç”¨ TIMESTEPS // 2 ä½œç‚ºç¤ºä¾‹
        self.adaptive_pool_output_size = TIMESTEPS // 2 
        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_size=self.adaptive_pool_output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim]
        Returns:
            å¤šå°ºåº¦èžåˆç‰¹å¾µ: [batch_size, pooled_seq_len, hidden_dim]
        """
        batch_size, seq_len, input_dim = x.shape
        
        # è½‰æ›ç‚ºå·ç©æ ¼å¼: [batch, channels, seq_len]
        x_conv = x.transpose(1, 2)
        
        # ä¸¦è¡Œå¤šå°ºåº¦ç‰¹å¾µæå–
        scale_features = []
        for conv in self.scale_convs:
            scale_feat = conv(x_conv)  # [batch, hidden_dim//len(scales), seq_len]
            scale_features.append(scale_feat)
        
        # æ‹¼æŽ¥æ‰€æœ‰å°ºåº¦ç‰¹å¾µ
        multi_scale = torch.cat(scale_features, dim=1)  # [batch, hidden_dim, seq_len]
        
        # è½‰å›žæ™‚åºæ ¼å¼
        multi_scale = multi_scale.transpose(1, 2)  # [batch, seq_len, hidden_dim]
        
        # ç‰¹å¾µèžåˆ
        fused_features = self.fusion_layer(multi_scale)
        
        # æ™‚é–“ç¶­åº¦è‡ªæ³¨æ„åŠ›
        attended_features, _ = self.temporal_attention(
            fused_features, fused_features, fused_features
        )
        
        # æ®˜å·®é€£æŽ¥
        output_features = attended_features + fused_features

        # æ‡‰ç”¨è‡ªé©æ‡‰æ± åŒ–
        # è½‰æ›ç‚ºæ± åŒ–å±¤æœŸæœ›çš„æ ¼å¼: [batch, channels, seq_len]
        # åœ¨é€™è£¡, channels æ˜¯ hidden_dim, seq_len æ˜¯ç•¶å‰çš„åºåˆ—é•·åº¦
        output_features_pooled = output_features.transpose(1, 2) # [batch_size, hidden_dim, seq_len]
        output_features_pooled = self.adaptive_pool(output_features_pooled) # [batch_size, hidden_dim, self.adaptive_pool_output_size]
        
        # è½‰å›žæ¨™æº–æ ¼å¼: [batch_size, pooled_seq_len, hidden_dim]
        output_features_pooled = output_features_pooled.transpose(1, 2)
        
        return output_features_pooled


class MarketStateDetector(nn.Module):
    """å¸‚å ´ç‹€æ…‹æª¢æ¸¬å™¨ï¼šæª¢æ¸¬è¶¨å‹¢ã€æ³¢å‹•ã€å‡å€¼å›žæ­¸ã€çªç ´ç­‰å¸‚å ´ç‹€æ…‹"""
    
    def __init__(self, d_model: int, num_market_states: int = 4): # Added num_market_states
        super().__init__()
        self.d_model = d_model
        self.num_market_states = num_market_states # Store it
        
        # ç‹€æ…‹ç‰¹å¾µæå–å™¨
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
        
        # ç‹€æ…‹èžåˆç¶²çµ¡
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
            x: [batch_size, d_model] æˆ– [batch_size, seq_len, d_model]
        Returns:
            åŒ…å«å„ç¨®å¸‚å ´ç‹€æ…‹çš„å­—å…¸
        """
        if x.dim() == 3:
            # å¦‚æžœæ˜¯åºåˆ—ï¼Œå–å¹³å‡
            x = x.mean(dim=1)
        
        # æå–å„ç¨®ç‹€æ…‹ç‰¹å¾µ
        states = {}
        state_features = []
        
        for state_name, extractor in self.state_extractors.items():
            state_value = extractor(x)
            states[state_name] = state_value
            
            if state_name == 'regime':
                state_features.append(state_value)  # [batch, num_market_states]
            else:
                state_features.append(state_value)  # [batch, 1]
        
        # æ‹¼æŽ¥æ‰€æœ‰ç‹€æ…‹ç‰¹å¾µ
        # concatenated_states = torch.cat(state_features, dim=-1)  # [batch, 7]
        # Corrected concatenation logic
        processed_state_features = []
        for feat in state_features:
            if feat.dim() == 1: # if it's [batch_size] from a single output (like trend, volatility, momentum)
                processed_state_features.append(feat.unsqueeze(-1)) # make it [batch_size, 1]
            else: # it's already [batch_size, num_states] for regime
                processed_state_features.append(feat)
        concatenated_states = torch.cat(processed_state_features, dim=-1) # [batch, 3 + num_market_states]
        
        # èžåˆå¾—åˆ°æœ€çµ‚å¸‚å ´ç‹€æ…‹
        final_market_state = self.state_fusion(concatenated_states)
        states['final_state'] = final_market_state # [batch, num_market_states]
        
        return states # Return the dict


class AdaptiveAttentionLayer(nn.Module):
    """è‡ªé©æ‡‰æ³¨æ„åŠ›å±¤ï¼šæ ¹æ“šå¸‚å ´ç‹€æ…‹å‹•æ…‹èª¿æ•´æ³¨æ„åŠ›æ¨¡å¼"""
    
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
    """å¢žå¼·ç‰ˆTransformerå±¤ï¼šé›†æˆè‡ªé©æ‡‰æ³¨æ„åŠ›å’Œæ”¹é€²çš„FFN"""
    
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, 
                 dropout: float = 0.1, use_adaptive_attention: bool = True, num_market_states: int = 4,
                 use_layer_norm_before: bool = True): # num_heads is the correct param name here
        super().__init__()
        self.use_adaptive_attention = use_adaptive_attention
        self.use_layer_norm_before = use_layer_norm_before # Store the parameter
        
        # è‡ªé©æ‡‰æ³¨æ„åŠ›å±¤ or æ¨™æº–å¤šé ­æ³¨æ„åŠ›
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

        # å¢žå¼·çš„å‰é¥‹ç¶²çµ¡
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
        
        # å±¤æ­¸ä¸€åŒ–
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # é–€æŽ§æ©Ÿåˆ¶
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, 
                key_padding_mask: Optional[torch.Tensor] = None,
                external_market_state_probs: Optional[torch.Tensor] = None) -> torch.Tensor: # MODIFIED PARAMETER
        
        # è‡ªé©æ‡‰æ³¨æ„åŠ›å­å±¤
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
            attn_output = attn_output_raw # This might be an issue if residual is expected by subsequent ops.
                                          # However, AdaptiveAttentionLayer *does* include a residual.
                                          # This means behavior differs based on use_adaptive_attention.
        
        # FFNå­å±¤
        ffn_input = self.norm2(attn_output) # If attn_output doesn't have residual, this norm is on raw attention.
        ffn_output = self.ffn(ffn_input)
        
        return ffn_output + x  # Residual connection with input x


class EnhancedTransformer(nn.Module):
    """
    Enhanced Universal Trading Transformer
    Integrates various advanced features like multi-scale feature extraction,
    adaptive attention, cross-time-scale fusion, and GMM-based market state detection.
    """
    def __init__(self,
                 input_dim: int,
                 d_model: int,
                 transformer_nhead: int, 
                 num_encoder_layers: int, 
                 dim_feedforward: int,
                 dropout: float,
                 max_seq_len: int,
                 num_symbols: Optional[int], 
                 output_dim: int,
                 use_msfe: bool = True,
                 msfe_hidden_dim: Optional[int] = None, 
                 msfe_scales: List[int] = [3, 5, 7, 11], 
                 use_final_norm: bool = True,
                 use_adaptive_attention: bool = True,
                 num_market_states: int = 4, 
                 use_gmm_market_state_detector: bool = False,
                 gmm_market_state_detector_path: Optional[str] = None,
                 gmm_ohlcv_feature_config: Optional[Dict[str, Any]] = None,
                 use_cts_fusion: bool = False, # CrossTimeScaleFusion
                 cts_time_scales: Optional[List[int]] = None, # e.g., [1, 3, 5]
                 cts_fusion_type: str = "hierarchical_attention", # "hierarchical_attention", "simple_attention", "concat", "average"
                 use_symbol_embedding: bool = True,
                 symbol_embedding_dim: int = 16,
                 use_cross_asset_attention: bool = False,
                 num_cross_asset_layers: int = 0,
                 use_fourier_features: bool = False,
                 fourier_num_modes: int = FOURIER_NUM_MODES, 
                 use_wavelet_features: bool = False,
                 wavelet_name: str = WAVELET_NAME, 
                 wavelet_levels: int = WAVELET_LEVELS, 
                 trainable_wavelet_filters: bool = False, 
                 use_layer_norm_before: bool = True, 
                 output_activation: Optional[str] = None, 
                 positional_encoding_type: str = "sinusoidal", 
                 device: str = DEVICE,
                 use_graph_attn: bool = False,
                 graph_layers: int = 0,
                 graph_heads: int = 4,
                 graph_topk: int = 5
                ):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_symbols = num_symbols
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.transformer_nhead = transformer_nhead

        self.use_symbol_embedding = use_symbol_embedding 
        self.use_msfe = use_msfe
        self.use_adaptive_attention = use_adaptive_attention
        self.use_gmm_market_state_detector = use_gmm_market_state_detector
        self.use_cts_fusion = use_cts_fusion
        self.use_fourier_features = use_fourier_features
        self.use_wavelet_features = use_wavelet_features
        self.positional_encoding_type = positional_encoding_type
        self.use_cross_asset_attention = bool(use_cross_asset_attention)
        self.num_cross_asset_layers = int(num_cross_asset_layers) if num_cross_asset_layers is not None else 0
        
        self.num_market_states = num_market_states
        self.gmm_detector: Optional[GMMMarketStateDetector] = None
        self.gmm_ohlcv_feature_config = gmm_ohlcv_feature_config

        if self.use_gmm_market_state_detector:
            if gmm_market_state_detector_path and os.path.exists(gmm_market_state_detector_path):
                try:
                    self.gmm_detector = GMMMarketStateDetector.load_model(gmm_market_state_detector_path)
                    if self.gmm_detector and self.gmm_detector.fitted: 
                        logger.info(f"Successfully loaded fitted GMMMarketStateDetector from {gmm_market_state_detector_path}.")
                        self.num_market_states = self.gmm_detector.n_states 
                        logger.info(f"Updated num_market_states to {self.num_market_states} based on loaded GMM.")
                    else:
                        logger.warning(f"Loaded GMM model from {gmm_market_state_detector_path} is not fitted. Will use configured num_market_states: {self.num_market_states}.")
                        self.gmm_detector = None 
                except Exception as e:
                    logger.error(f"Failed to load GMMMarketStateDetector from {gmm_market_state_detector_path}: {e}. Will use configured num_market_states: {self.num_market_states}.")
                    self.gmm_detector = None
            else:
                logger.warning(f"GMMMarketStateDetector path '{gmm_market_state_detector_path}' not found or not provided, but use_gmm_market_state_detector is True. Will use configured num_market_states: {self.num_market_states}.")

        # --- Input Processing Chain ---
        # 1. MSFE (optional)
        if self.use_msfe:
            msfe_eff_hidden_dim = msfe_hidden_dim if msfe_hidden_dim is not None else d_model
            self.msfe = MultiScaleFeatureExtractor(
                input_dim=input_dim,
                hidden_dim=msfe_eff_hidden_dim,
                scales=msfe_scales
            )
            current_transformer_input_dim = msfe_eff_hidden_dim
        else:
            self.msfe = None
            current_transformer_input_dim = input_dim

        # 2. Input Projection (if MSFE not used or its output dim doesn't match d_model)
        if current_transformer_input_dim != d_model:
            self.input_proj = nn.Linear(current_transformer_input_dim, d_model)
        else:
            self.input_proj = nn.Identity()
        
        # --- Symbol Embedding ---
        if self.use_symbol_embedding:
            effective_num_symbols = self.num_symbols if self.num_symbols is not None else MAX_SYMBOLS_ALLOWED
            self.symbol_embed = nn.Embedding(effective_num_symbols + 1, symbol_embedding_dim, padding_idx=0)
            if symbol_embedding_dim != d_model:
                self.symbol_embed_proj = nn.Linear(symbol_embedding_dim, d_model)
            else:
                self.symbol_embed_proj = nn.Identity()
        else:
            self.symbol_embed = None
            self.symbol_embed_proj = None

        # --- Positional Encoding ---
        pe_seq_len = TIMESTEPS // 2 if use_msfe and self.msfe is not None else max_seq_len

        if self.positional_encoding_type == "sinusoidal":
            self.pos_encoder = PositionalEncoding(d_model, dropout, pe_seq_len)
            self.pos_embed = None 
        elif self.positional_encoding_type == "learned":
            self.pos_embed = LearnedPositionalEncoding(d_model, dropout, pe_seq_len) 
            self.pos_encoder = self.pos_embed 
        else: # None or other
            self.pos_encoder = nn.Identity()
            self.pos_embed = None


        # --- Feature Blocks (Fourier, Wavelet) applied after projection to d_model ---
        self.fourier_block = FourierFeatureBlock(d_model, fourier_num_modes) if use_fourier_features else None
        self.wavelet_block = MultiLevelWaveletBlock(d_model, wavelet_name, wavelet_levels, trainable_wavelet_filters) if use_wavelet_features else None
        
        # --- Cross-Time-Scale Fusion (CTS) Initialization ---
        if self.use_cts_fusion:
            # å¦‚æžœæœªæä¾› cts_time_scales, ä½¿ç”¨é è¨­å€¼
            if cts_time_scales is None:
                cts_time_scales = [1, 3, 5] # é è¨­æ™‚é–“å°ºåº¦: åŒ…å«åŽŸå§‹å°ºåº¦(1), çŸ­æœŸ(3), ä¸­æœŸ(5)
                logger.info(f"CrossTimeScaleFusion: 'cts_time_scales' æœªæä¾›, ä½¿ç”¨é è¨­å€¼: {cts_time_scales}")
            
            self.cts_fusion_module = CrossTimeScaleFusion(
                d_model=d_model,
                time_scales=cts_time_scales,
                fusion_type=cts_fusion_type, # å¾žé…ç½®ä¸­è®€å–èžåˆé¡žåž‹
                dropout_rate=dropout # ä½¿ç”¨æ¨¡åž‹ä¸» dropout çŽ‡
            )
            logger.info(f"CrossTimeScaleFusion æ¨¡çµ„å·²å•Ÿç”¨, é¡žåž‹: {cts_fusion_type}, å°ºåº¦: {cts_time_scales}.")
        else:
            self.cts_fusion_module = None
            logger.info("CrossTimeScaleFusion æ¨¡çµ„æœªå•Ÿç”¨ã€‚")

        # --- Cross-Asset Attention Initialization (optional) ---
        if self.use_cross_asset_attention and self.num_cross_asset_layers > 0:
            try:
                cross_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=transformer_nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True,
                )
                self.cross_asset_encoder = nn.TransformerEncoder(cross_layer, num_layers=self.num_cross_asset_layers)
                logger.info(f"Cross-asset attention enabled: layers={self.num_cross_asset_layers}, d_model={d_model}, heads={transformer_nhead}")
            except Exception as e:
                logger.error(f"Failed to initialize cross-asset attention: {e}")
                self.cross_asset_encoder = None
                self.use_cross_asset_attention = False
        else:
            self.cross_asset_encoder = None

        # --- Transformer Layers Initialization --- (existing code)
        self.transformer_layers = nn.ModuleList([
            EnhancedTransformerLayer(
                d_model=d_model,
                num_heads=transformer_nhead, # Use the passed transformer_nhead
                ffn_dim=dim_feedforward,
                dropout=dropout,
                use_adaptive_attention=self.use_adaptive_attention,
                num_market_states=self.num_market_states, # This will be from GMM or config
                # adaptive_dim_reduction_factor, adaptive_activation are internal to AdaptiveAttentionLayer
                use_layer_norm_before=use_layer_norm_before
            ) for _ in range(num_encoder_layers) # Use num_encoder_layers
        ])

        # --- Output Processing ---
        self.final_norm = nn.LayerNorm(d_model) if use_final_norm else nn.Identity()
        self.output_projection = nn.Linear(d_model, output_dim)
        
        if output_activation == "softmax":
            self.output_activation_fn = nn.Softmax(dim=-1)
        elif output_activation == "sigmoid":
            self.output_activation_fn = nn.Sigmoid()
        else:
            self.output_activation_fn = nn.Identity()
            
        self.dropout_layer = nn.Dropout(dropout) # é€šç”¨ dropout å±¤

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """ç”Ÿæˆè‡ªå›žæ­¸ä»»å‹™æ‰€éœ€çš„æ–¹å½¢å¾ŒçºŒé®ç½©"""
        mask = (torch.triu(torch.ones(sz, sz), diagonal=1) == 1).transpose(0, 1)
        return mask

    def _extract_ohlcv_for_gmm(self, raw_ohlcv_src: torch.Tensor) -> torch.Tensor:
        # This method seems to be defined but not used in the provided forward pass.
        # It might be intended for a different GMM integration path or was part of an older version.
        # For now, keeping it as is. If it's needed, ensure self.ohlcv_feature_index_in_src is set.
        if not hasattr(self, 'ohlcv_feature_index_in_src') or not self.ohlcv_feature_index_in_src:
            raise ValueError("ohlcv_feature_index_in_src is not defined for GMM when using _extract_ohlcv_for_gmm.")
        gmm_expected_order = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
        indices = []
        for key in gmm_expected_order:
            if key not in self.ohlcv_feature_index_in_src:
                 raise ValueError(f"Missing OHLCV key for GMM: {key}. Required: {gmm_expected_order}")
            indices.append(self.ohlcv_feature_index_in_src[key])
        return raw_ohlcv_src[..., indices]

    def forward(self, x_dict: Dict[str, Optional[torch.Tensor]], return_full_sequence: bool = False) -> torch.Tensor:
        """
        Args:
            x_dict: Dictionary containing input tensors:
                "src": [batch_size, num_active_symbols, seq_len, input_dim] - Main input features.
                "symbol_ids": Optional[torch.Tensor] - [batch_size, num_active_symbols] - Symbol identifiers for embedding.
                "src_key_padding_mask": Optional[torch.Tensor] - [batch_size, num_active_symbols] - Mask for padded symbols.
                                         True indicates a padded symbol that should be ignored.
                "raw_ohlcv_data_batch": Optional[List[pd.DataFrame]] - Batch of raw OHLCV data for GMM.
                                          Each DataFrame should have OHLCV columns and a DatetimeIndex.
                                          Length of list is batch_size. Each df is [seq_len, num_features_ohlcv].
            return_full_sequence: If True, return [batch, num_symbols, seq_len, d_model] for downstream modules needing full sequence.
        Returns:
            Output tensor: [batch_size, num_active_symbols, output_dim] or [batch_size, num_active_symbols, seq_len, d_model] if return_full_sequence
        """
        src = x_dict["src"]
        symbol_ids = x_dict.get("symbol_ids")
        # --- SHAPE ASSERT ---
        assert src.shape[-1] == self.input_dim, f"[ShapeError] src last dim ({src.shape[-1]}) != input_dim ({self.input_dim})"
        symbol_padding_mask = x_dict.get("src_key_padding_mask") 
        raw_ohlcv_data_batch = x_dict.get("raw_ohlcv_data_batch")

        batch_size, num_active_symbols, seq_len, _ = src.shape
        
        # Reshape for processing: [B * N, S, C_in]
        x = src.reshape(batch_size * num_active_symbols, seq_len, self.input_dim)

        # --- GMM Market State Detection (early, if used) ---
        # GMM state can influence other parts like adaptive attention
        # The GMM detector expects a list of DataFrames (one per batch item)
        # If we have B*N items, we need to decide how to pass this.
        # Assuming GMM state is per-symbol, we might need to run GMM N times per batch item,
        # or adapt GMM to handle batch of symbols.
        # For now, if raw_ohlcv_data_batch is provided, it's List[DataFrame] of length B.
        # We need to expand or select for B*N.
        # Let's assume for now GMM state is calculated per batch item (averaging over symbols if needed by GMM internally)
        # or that raw_ohlcv_data_batch is already structured for B*N if GMM is symbol-specific.
        # The current GMMMarketStateDetector takes a list of DFs.
        
        external_market_state_probs_for_transformer = None # [B*N, num_market_states]
        if self.use_gmm_market_state_detector and self.gmm_detector and raw_ohlcv_data_batch:
            if len(raw_ohlcv_data_batch) == batch_size: # We have B dataframes
                # We need to get states for B*N. We can repeat the states for each symbol within a batch item.
                gmm_states_list_batch = [] # List of [num_market_states] tensors
                for i in range(batch_size):
                    # GMM expects a single DataFrame for predict_proba_for_sequence
                    # If raw_ohlcv_data_batch[i] is a list of DFs for symbols, this needs adjustment.
                    # Assuming raw_ohlcv_data_batch[i] is one DF for that batch item.
                    # The GMMMarketStateDetector._calculate_features needs a DataFrame.
                    # And predict_proba_for_sequence expects a pre-calculated feature tensor.
                    
                    # Let's assume gmm_detector.predict_proba_for_batch_sequences can take List[pd.DataFrame]
                    # and returns something like [B, num_market_states] or [B, T, num_market_states]
                    # For now, let's assume we get one state vector per batch item.
                    # This is a simplification. A more robust way would be to get per-symbol states if GMM supports it,
                    # or pass the raw_ohlcv_data_batch appropriately if it's already per-symbol.

                    # The GMMMarketStateDetector.predict_proba_for_batch_sequences is not defined.
                    # Let's use predict_proba_for_sequence if we assume one GMM state per batch item.
                    # This requires calculating features first.
                    try:
                        # Ensure gmm_ohlcv_feature_config is available
                        if not self.gmm_ohlcv_feature_config:
                            logger.warning("gmm_ohlcv_feature_config is None, cannot calculate GMM features.")
                            gmm_features_df = None
                        else:
                            # The GMMMarketStateDetector instance (self.gmm_detector) uses its own
                            # self.feature_config, which should be set up during its initialization
                            # or when the EnhancedTransformer loads/configures it.
                            # Thus, we don't need to pass 'config' argument here.
                            gmm_features_df = self.gmm_detector._calculate_features(raw_ohlcv_data_batch[i])
                        
                        if gmm_features_df is not None and not gmm_features_df.empty:
                            # predict_proba_for_sequence expects features for a single sequence
                            # It returns [n_samples, n_components] where n_samples is num timesteps
                            # We need one state vector, so we average probabilities over time
                            gmm_state_probs_seq = self.gmm_detector.predict_proba_for_sequence(gmm_features_df) # [T, num_market_states]
                            gmm_state_probs_item = torch.tensor(gmm_state_probs_seq, dtype=torch.float, device=self.device).mean(dim=0) # [num_market_states]
                            gmm_states_list_batch.append(gmm_state_probs_item)
                        else:
                            # Handle case where features could not be calculated (e.g., insufficient data)
                            # Fallback to a uniform distribution or zeros, or let AdaptiveAttention handle it
                            logger.warning(f"Could not calculate GMM features for batch item {i}. Market state might be unreliable.")
                            # Using a uniform distribution as a fallback for this item
                            uniform_probs = torch.ones(self.num_market_states, device=self.device) / self.num_market_states
                            gmm_states_list_batch.append(uniform_probs)

                    except Exception as e:
                        logger.error(f"Error during GMM state calculation for batch item {i}: {e}")
                        uniform_probs = torch.ones(self.num_market_states, device=self.device) / self.num_market_states
                        gmm_states_list_batch.append(uniform_probs)


                if gmm_states_list_batch:
                    batch_gmm_states = torch.stack(gmm_states_list_batch) # [B, num_market_states]
                    # Expand for B*N for transformer layers
                    external_market_state_probs_for_transformer = batch_gmm_states.unsqueeze(1).repeat(1, num_active_symbols, 1)
                    external_market_state_probs_for_transformer = external_market_state_probs_for_transformer.reshape(batch_size * num_active_symbols, self.num_market_states)
                else: # No GMM states could be computed
                    external_market_state_probs_for_transformer = None # Let AdaptiveAttention use its internal one

            else:
                logger.warning(f"raw_ohlcv_data_batch length ({len(raw_ohlcv_data_batch)}) does not match batch_size ({batch_size}). Skipping GMM state usage.")
                external_market_state_probs_for_transformer = None
        else: # GMM not used or detector not loaded or no raw data
            external_market_state_probs_for_transformer = None


        # --- Feature Processing Chain ---
        # 1. MultiScaleFeatureExtractor (optional)
        if self.msfe is not None:
            x = self.msfe(x) # Output: [B*N, pooled_seq_len, msfe_hidden_dim]
            # Update seq_len if MSFE changed it (due to pooling)
            seq_len = x.size(1) 
        # 2. Input Projection (to d_model)
        x = self.input_proj(x) # Output: [B*N, seq_len, d_model]

        # 3. Add Symbol Embedding (if used)
        if self.use_symbol_embedding and symbol_ids is not None:
            # symbol_ids: [B, N_active] -> needs to be [B*N_active]
            symbol_ids_flat = symbol_ids.reshape(batch_size * num_active_symbols)
            sym_embeds = self.symbol_embed(symbol_ids_flat) # [B*N, symbol_embedding_dim]
            sym_embeds_proj = self.symbol_embed_proj(sym_embeds) # [B*N, d_model]
            # Add to each time step of x: unsqueeze to [B*N, 1, d_model] and broadcast
            x = x + sym_embeds_proj.unsqueeze(1)

        # 4. Add Positional Encoding
        # self.pos_encoder is either Sinusoidal or Learned or Identity
        if self.positional_encoding_type == 'learned' and self.pos_embed is not None:
            # LearnedPositionalEncoding expects [B*N, S, E] or [S, B*N, E]
            # If self.pos_embed is an instance of LearnedPositionalEncoding
            x = self.pos_embed(x) # Assuming LearnedPositionalEncoding handles [B*N, S, E]
        elif self.positional_encoding_type == 'sinusoidal' and self.pos_encoder is not None and not isinstance(self.pos_encoder, nn.Identity):
            x = self.pos_encoder(x)

        x = self.dropout_layer(x) 

        temporal_key_padding_mask = None 

        for layer in self.transformer_layers:
            x = layer(x, 
                      key_padding_mask=temporal_key_padding_mask, 
                      external_market_state_probs=external_market_state_probs_for_transformer)
        
        x = self.final_norm(x) 

        if return_full_sequence:
            # å›žå‚³å®Œæ•´åºåˆ— [batch, num_symbols, seq_len, d_model]
            x_full = x.reshape(batch_size, num_active_symbols, seq_len, self.d_model)
            if symbol_padding_mask is not None:
                mask_expanded = symbol_padding_mask.unsqueeze(-1).unsqueeze(-1).expand_as(x_full)
                x_full = x_full.masked_fill(mask_expanded, 0.0)
            return x_full

        x_agg = x[:, -1, :]  # [B*N, d_model]
        if self.use_cross_asset_attention and self.cross_asset_encoder is not None:
            x_sym = x_agg.reshape(batch_size, num_active_symbols, self.d_model)
            try:
                x_sym = self.cross_asset_encoder(x_sym, src_key_padding_mask=symbol_padding_mask)
            except Exception as e:
                logger.error(f"Cross-asset forward failed: {e}")
            x_agg = x_sym.reshape(batch_size * num_active_symbols, self.d_model)

        output = self.output_projection(x_agg) # [B*N, output_dim]
        output = self.output_activation_fn(output)
        output = output.reshape(batch_size, num_active_symbols, self.output_dim)
        if symbol_padding_mask is not None:
            mask_expanded = symbol_padding_mask.unsqueeze(-1).expand_as(output)
            output = output.masked_fill(mask_expanded, 0.0)
        return output

    def get_dynamic_config(self) -> Dict[str, Any]:
        """Get the dynamic configuration of the Transformer."""
        try:
            # Get transformer layer information safely
            num_layers = len(self.transformer_layers) if hasattr(self, 'transformer_layers') and self.transformer_layers else 0
            
            # Try to get dim_feedforward from transformer layers
            dim_feedforward = None
            if hasattr(self, 'transformer_layers') and self.transformer_layers:
                try:
                    first_layer = self.transformer_layers[0]
                    if hasattr(first_layer, 'ffn') and hasattr(first_layer.ffn, '__len__') and len(first_layer.ffn) > 0:
                        dim_feedforward = first_layer.ffn[0].out_features
                except (AttributeError, IndexError):
                    dim_feedforward = getattr(self, 'dim_feedforward', None)
            
            # Get dropout value safely
            dropout_val = None
            if hasattr(self, 'dropout_layer') and hasattr(self.dropout_layer, 'p'):
                dropout_val = self.dropout_layer.p
            
            config = {
                "input_dim": getattr(self, 'input_dim', None),
                "d_model": getattr(self, 'd_model', None),
                "transformer_nhead": getattr(self, 'transformer_nhead', None),
                "num_encoder_layers": num_layers,
                "dim_feedforward": dim_feedforward,
                "dropout": dropout_val,
                "max_seq_len": getattr(self, 'max_seq_len', None),
                "num_symbols": getattr(self, 'num_symbols', None),
                "output_dim": getattr(self, 'output_dim', None),
                "use_msfe": getattr(self, 'use_msfe', False),
                "use_adaptive_attention": getattr(self, 'use_adaptive_attention', False),
                "num_market_states": getattr(self, 'num_market_states', None),
                "use_gmm_market_state_detector": getattr(self, 'use_gmm_market_state_detector', False),
                "gmm_market_state_detector_path": None,
                "gmm_ohlcv_feature_config": getattr(self, 'gmm_ohlcv_feature_config', None) is not None,
                "use_cts_fusion": getattr(self, 'use_cts_fusion', False),
                "use_symbol_embedding": getattr(self, 'use_symbol_embedding', False),
                "use_fourier_features": getattr(self, 'use_fourier_features', False),
                "use_wavelet_features": getattr(self, 'use_wavelet_features', False),
                "positional_encoding_type": getattr(self, 'positional_encoding_type', None),
                "output_activation": self.output_activation_fn.__class__.__name__ if hasattr(self, 'output_activation_fn') and not isinstance(self.output_activation_fn, nn.Identity) else None,
            }
              # Safely get GMM detector path
            if hasattr(self, 'gmm_detector') and self.gmm_detector and hasattr(self.gmm_detector, 'model_path'):
                config["gmm_market_state_detector_path"] = self.gmm_detector.model_path
                
            # Add additional configuration details if available
            if self.use_msfe and hasattr(self, 'msfe') and self.msfe is not None:
                config['msfe_hidden_dim'] = getattr(self.msfe, 'hidden_dim', None)
                config['msfe_scales'] = getattr(self.msfe, 'scales', None)

            if self.use_cts_fusion and hasattr(self, 'cts_fusion_module') and self.cts_fusion_module is not None:
                config['cts_time_scales'] = getattr(self.cts_fusion_module, 'time_scales', None)
                config['cts_fusion_type'] = getattr(self.cts_fusion_module, 'fusion_type', None)

            if self.use_fourier_features and hasattr(self, 'fourier_block') and self.fourier_block is not None:
                if hasattr(self.fourier_block, 'spectral_conv'):
                    config['fourier_num_modes'] = getattr(self.fourier_block.spectral_conv, 'num_modes', None)

            if self.use_wavelet_features and hasattr(self, 'wavelet_block') and self.wavelet_block is not None:
                if hasattr(self.wavelet_block, 'dwt_layers') and self.wavelet_block.dwt_layers:
                    wavelet_layer = self.wavelet_block.dwt_layers[0]
                    if hasattr(wavelet_layer, 'wavelet') and hasattr(wavelet_layer.wavelet, 'name'):
                        config['wavelet_name'] = wavelet_layer.wavelet.name
                config['wavelet_levels'] = getattr(self.wavelet_block, 'levels', None)
                
        except Exception as e:
            # If anything goes wrong, return a basic config
            config = {
                "input_dim": getattr(self, 'input_dim', None),
                "d_model": getattr(self, 'd_model', None),
                "transformer_nhead": getattr(self, 'transformer_nhead', None),
                "num_encoder_layers": 0,
                "dim_feedforward": None,
                "dropout": None,
                "max_seq_len": getattr(self, 'max_seq_len', None),
                "num_symbols": getattr(self, 'num_symbols', None),
                "output_dim": getattr(self, 'output_dim', None),
                "error": f"Config retrieval error: {str(e)}"
            }
            
        return config

# Ensure this class is defined after all its dependencies like EnhancedTransformerLayer, etc.


