# src/models/transformer_model.py
"""
增強版通用多資產交易Transformer模型。
集成Symbol Embedding, 時域處理 (Transformer Encoder),
頻域處理 (FFT頻譜卷積和小波分解), 跨資產注意力。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Optional, Tuple, Union
import pywt
import sys
from pathlib import Path
import logging # 導入 logging 以便後備 logger 使用
import warnings # 導入 warnings 模組

# --- 過濾特定的 UserWarning ---
warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True",
    category=UserWarning,
    module="torch.nn.modules.transformer" # 限定來源模組
)

# --- Simplified Import Block ---
try:
    from src.common.logger_setup import logger
    logger.debug("transformer_model.py: Successfully imported logger from src.common.logger_setup.")
except ImportError:
    logger = logging.getLogger("transformer_model_fallback") # type: ignore
    logger.setLevel(logging.DEBUG)
    _ch_fallback = logging.StreamHandler(sys.stdout)
    _ch_fallback.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    if not logger.handlers: logger.addHandler(_ch_fallback)
    logger.warning("transformer_model.py: Failed to import logger from src.common.logger_setup. Using fallback logger.")

try:
    from src.common.config import (
        TIMESTEPS, MAX_SYMBOLS_ALLOWED,
        TRANSFORMER_MODEL_DIM, TRANSFORMER_NUM_LAYERS, TRANSFORMER_NUM_HEADS,
        TRANSFORMER_FFN_DIM, TRANSFORMER_DROPOUT_RATE, TRANSFORMER_LAYER_NORM_EPS,
        TRANSFORMER_MAX_SEQ_LEN_POS_ENCODING, TRANSFORMER_OUTPUT_DIM_PER_SYMBOL,
        DEVICE
    )
    logger.info("transformer_model.py: Successfully imported common.config.") # type: ignore
except ImportError as e:
    logger.error(f"transformer_model.py: Failed to import common.config: {e}. Using fallback values.", exc_info=True) # type: ignore
    TIMESTEPS=128; MAX_SYMBOLS_ALLOWED=20; TRANSFORMER_MODEL_DIM=128;
    TRANSFORMER_NUM_LAYERS=2; TRANSFORMER_NUM_HEADS=2; TRANSFORMER_FFN_DIM=256;
    TRANSFORMER_DROPOUT_RATE=0.1; TRANSFORMER_LAYER_NORM_EPS=1e-5;
    TRANSFORMER_MAX_SEQ_LEN_POS_ENCODING=5000; TRANSFORMER_OUTPUT_DIM_PER_SYMBOL=32;
    DEVICE=torch.device("cpu")
    logger.warning("transformer_model.py: Using fallback values for config due to import error.") # type: ignore


# --- 位置編碼 (Positional Encoding) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = TRANSFORMER_DROPOUT_RATE, max_len: int = TRANSFORMER_MAX_SEQ_LEN_POS_ENCODING):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1: pe[0, :, 1::2] = torch.cos(position * div_term[:-1])
        else: pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe, persistent=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# --- 頻域處理: 傅立葉部分 (類似FEDformer) ---
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_modes: int):
        super().__init__()
        self.in_channels = in_channels; self.out_channels = out_channels; self.num_modes = num_modes
        scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(scale * torch.randn(in_channels, out_channels, self.num_modes, dtype=torch.cfloat))
        # logger.debug(f"SpectralConv1d initialized: in={in_channels}, out={out_channels}, modes={num_modes}")
    def forward(self, x_fft: torch.Tensor) -> torch.Tensor:
        out_fft = torch.zeros(x_fft.size(0), self.out_channels, x_fft.size(2), dtype=torch.cfloat, device=x_fft.device)
        selected_modes_data = x_fft[:, :, :self.num_modes]
        multiplied = torch.einsum("bim,iom->bom", selected_modes_data, self.weights)
        out_fft[:, :, :self.num_modes] = multiplied
        return out_fft

class FourierFeatureBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_modes: int, activation: str = 'gelu'):
        super().__init__()
        self.spectral_conv = SpectralConv1d(in_features, out_features, num_modes)
        self.act = nn.GELU() if activation == 'gelu' else nn.ReLU()
        # logger.debug(f"FourierFeatureBlock initialized: in_feat={in_features}, out_feat={out_features}, modes={num_modes}")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_N, T, C_in = x.shape; x_perm = x.permute(0, 2, 1)
        x_fft = torch.fft.rfft(x_perm, n=T, dim=2, norm='ortho')
        x_filtered_fft = self.spectral_conv(x_fft)
        x_time_domain = torch.fft.irfft(x_filtered_fft, n=T, dim=2, norm='ortho')
        output = x_time_domain.permute(0, 2, 1)
        return self.act(output)

# --- 頻域處理: 小波部分 ---
class DWT1D(nn.Module):
    def __init__(self, wavelet_name: str = 'db4', trainable_filters: bool = False):
        super().__init__()
        self.wavelet = pywt.Wavelet(wavelet_name)
        dec_lo = torch.tensor(self.wavelet.dec_lo[::-1], dtype=torch.float32)
        dec_hi = torch.tensor(self.wavelet.dec_hi[::-1], dtype=torch.float32)
        self.kernel_len = len(dec_lo)
        lo_filter = dec_lo.unsqueeze(0).unsqueeze(0); hi_filter = dec_hi.unsqueeze(0).unsqueeze(0)
        if trainable_filters: self.lo_filter = nn.Parameter(lo_filter); self.hi_filter = nn.Parameter(hi_filter)
        else: self.register_buffer('lo_filt', lo_filter); self.register_buffer('hi_filt', hi_filter)
        # logger.debug(f"DWT1D initialized with wavelet: {wavelet_name}, kernel_len: {self.kernel_len}, trainable: {trainable_filters}")
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B_N, C, T = x.shape
        current_lo_filter = self.lo_filt if hasattr(self, 'lo_filt') else self.lo_filter
        current_hi_filter = self.hi_filt if hasattr(self, 'hi_filt') else self.hi_filter
        lo_filt_for_conv = current_lo_filter.repeat(C, 1, 1); hi_filt_for_conv = current_hi_filter.repeat(C, 1, 1)
        padding_size = (self.kernel_len - 1) // 2
        cA = F.conv1d(x, lo_filt_for_conv, stride=2, padding=padding_size, groups=C)
        cD = F.conv1d(x, hi_filt_for_conv, stride=2, padding=padding_size, groups=C)
        return cA, cD

class MultiLevelWaveletBlock(nn.Module):
    def __init__(self, model_dim: int, wavelet_name: str = 'db4', levels: int = 3, trainable_filters: bool = False):
        super().__init__()
        self.levels = levels
        self.dwt_layers = nn.ModuleList([DWT1D(wavelet_name, trainable_filters) for _ in range(levels)])
        self.coef_projections = nn.ModuleList()
        output_dim_per_coef = model_dim // 2 # 為簡化，cA和cD各自投影到 model_dim/2
        self.final_cA_proj = nn.Linear(model_dim, output_dim_per_coef)
        self.final_cD_proj = nn.Linear(model_dim, model_dim - output_dim_per_coef) # 確保總和是model_dim
        # logger.debug(f"MultiLevelWaveletBlock: levels={levels}, output_dim_per_coef for cD approx {output_dim_per_coef}")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_perm = x.permute(0, 2, 1) # (B*N, C, T)
        current_signal = x_perm; cA_final = x_perm; cD_final = x_perm # 初始化
        for i in range(self.levels):
            cA, cD = self.dwt_layers[i](current_signal)
            current_signal = cA
            if i == self.levels - 1: cA_final, cD_final = cA, cD # 保存最後一層的結果
        original_T = x.size(1)
        cA_final_upsampled = F.interpolate(cA_final, size=original_T, mode='linear', align_corners=False).permute(0, 2, 1)
        cD_final_upsampled = F.interpolate(cD_final, size=original_T, mode='linear', align_corners=False).permute(0, 2, 1)
        projected_cA = self.final_cA_proj(cA_final_upsampled)
        projected_cD = self.final_cD_proj(cD_final_upsampled)
        return torch.cat((projected_cA, projected_cD), dim=2) # (B*N, T, D_model)

# --- 主 Transformer 模型 ---
class UniversalTradingTransformer(nn.Module):
    def __init__(self, num_input_features: int, num_symbols_possible: int = MAX_SYMBOLS_ALLOWED,
                 model_dim: int = TRANSFORMER_MODEL_DIM, num_time_encoder_layers: int = TRANSFORMER_NUM_LAYERS // 2,
                 num_cross_asset_layers: int = TRANSFORMER_NUM_LAYERS // 2, num_heads: int = TRANSFORMER_NUM_HEADS,
                 ffn_dim: int = TRANSFORMER_FFN_DIM, dropout_rate: float = TRANSFORMER_DROPOUT_RATE,
                 use_fourier_block: bool = True, fourier_num_modes: int = 32,
                 use_wavelet_block: bool = True, wavelet_levels: int = 3, wavelet_name: str = 'db4',
                 output_dim_per_symbol: int = TRANSFORMER_OUTPUT_DIM_PER_SYMBOL):
        super().__init__()
        logger.info(f"Initializing UniversalTradingTransformer: input_feat={num_input_features}, max_symbols={num_symbols_possible}, model_dim={model_dim}")
        self.model_dim = model_dim; self.num_symbols_possible = num_symbols_possible
        self.input_projection = nn.Linear(num_input_features, self.model_dim)
        self.slot_embeddings = nn.Embedding(MAX_SYMBOLS_ALLOWED, self.model_dim) # 使用MAX_SYMBOLS_ALLOWED
        self.temporal_pos_encoder = PositionalEncoding(self.model_dim, dropout_rate)
        time_encoder_layer = nn.TransformerEncoderLayer(d_model=self.model_dim, nhead=num_heads, dim_feedforward=ffn_dim, dropout=dropout_rate, activation='gelu', batch_first=True, norm_first=True)
        self.time_transformer_encoder = nn.TransformerEncoder(time_encoder_layer, num_layers=num_time_encoder_layers)
        self.use_fourier_block = use_fourier_block; self.use_wavelet_block = use_wavelet_block
        if self.use_fourier_block:
            actual_fourier_modes = min(fourier_num_modes, TIMESTEPS // 2 +1 if TIMESTEPS > 0 else 16) # 確保TIMESTEPS有效
            self.fourier_block = FourierFeatureBlock(self.model_dim, self.model_dim, actual_fourier_modes)
            logger.info(f"Fourier Block enabled with num_modes={actual_fourier_modes}")
        if self.use_wavelet_block:
            self.wavelet_block = MultiLevelWaveletBlock(self.model_dim, wavelet_name, wavelet_levels)
            logger.info(f"Wavelet Block enabled with wavelet='{wavelet_name}', levels={wavelet_levels}")
        fusion_input_dims = self.model_dim
        if self.use_fourier_block: fusion_input_dims += self.model_dim
        if self.use_wavelet_block: fusion_input_dims += self.model_dim
        if fusion_input_dims > self.model_dim: self.fusion_layer = nn.Linear(fusion_input_dims, self.model_dim); logger.info(f"Fusion layer will map from {fusion_input_dims} to {self.model_dim}")
        else: self.fusion_layer = nn.Identity()
        cross_asset_layer = nn.TransformerEncoderLayer(d_model=self.model_dim, nhead=num_heads, dim_feedforward=ffn_dim, dropout=dropout_rate, activation='gelu', batch_first=True, norm_first=True)
        self.cross_asset_encoder = nn.TransformerEncoder(cross_asset_layer, num_layers=num_cross_asset_layers)
        self.output_projection = nn.Linear(self.model_dim, output_dim_per_symbol)
        self.output_activation = nn.GELU()
        logger.info("UniversalTradingTransformer structure initialized.")

    def forward(self, x_features: torch.Tensor, padding_mask_symbols: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, num_slots, num_timesteps, _ = x_features.shape
        projected_x = self.input_projection(x_features)
        # slot_indices應該基於num_slots，而不是固定的self.num_symbols_possible
        slot_indices = torch.arange(num_slots, device=x_features.device).unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, num_timesteps)
        slot_embs = self.slot_embeddings(slot_indices.flatten(0,1)).view(batch_size, num_slots, num_timesteps, self.model_dim)
        x = projected_x + slot_embs
        x_reshaped = x.view(batch_size * num_slots, num_timesteps, self.model_dim)
        time_x = self.temporal_pos_encoder(x_reshaped)
        time_features = self.time_transformer_encoder(time_x)
        features_to_fuse = [time_features]
        if self.use_fourier_block:
            fourier_features = self.fourier_block(x_reshaped) # x_reshaped (B*N, T, C)
            features_to_fuse.append(fourier_features)
        if self.use_wavelet_block:
            wavelet_features = self.wavelet_block(x_reshaped) # x_reshaped (B*N, T, C)
            features_to_fuse.append(wavelet_features)
        if len(features_to_fuse) > 1: fused_x = torch.cat(features_to_fuse, dim=2); fused_x = self.fusion_layer(fused_x)
        else: fused_x = time_features
        pooled_x = fused_x[:, -1, :]
        pooled_x_sym_dim = pooled_x.view(batch_size, num_slots, self.model_dim)
        if padding_mask_symbols is not None and padding_mask_symbols.dtype != torch.bool: padding_mask_symbols = padding_mask_symbols.bool()
        contextual_repr = self.cross_asset_encoder(pooled_x_sym_dim, src_key_padding_mask=padding_mask_symbols)
        output = self.output_projection(contextual_repr); output = self.output_activation(output)
        if padding_mask_symbols is not None: output = output.masked_fill(padding_mask_symbols.unsqueeze(-1), 0.0)
        return output

# --- if __name__ == "__main__": 測試塊 (與之前版本類似) ---
if __name__ == "__main__":
    # 確保 logger 和從 config 導入的變量在此作用域可用
    # 頂部的 try-except 結構應該已經處理了
    logger.info("正在直接運行 (增強版) UniversalTradingTransformer.py 進行測試...")
    bs = 4; num_s = MAX_SYMBOLS_ALLOWED; num_t = TIMESTEPS
    num_f_in = 9
    # 使用後備配置中較小的值進行測試，或者確保 common.config 可導入
    model_dim_test = TRANSFORMER_MODEL_DIM
    num_layers_test = TRANSFORMER_NUM_LAYERS // 2 if TRANSFORMER_NUM_LAYERS > 1 else 1 # 確保不為0
    num_heads_test = TRANSFORMER_NUM_HEADS
    ffn_dim_test = TRANSFORMER_FFN_DIM
    output_dim_test = TRANSFORMER_OUTPUT_DIM_PER_SYMBOL
    fourier_modes_test = min(32, num_t//2 + 1 if num_t > 0 else 16)


    logger.info(f"Test params: model_dim={model_dim_test}, layers={num_layers_test*2}, heads={num_heads_test}, ffn={ffn_dim_test}, out_dim={output_dim_test}")

    model = UniversalTradingTransformer(
        num_input_features=num_f_in, num_symbols_possible=num_s, model_dim=model_dim_test,
        num_time_encoder_layers=num_layers_test, num_cross_asset_layers=num_layers_test,
        num_heads=num_heads_test, ffn_dim=ffn_dim_test, dropout_rate=TRANSFORMER_DROPOUT_RATE,
        use_fourier_block=True, fourier_num_modes=fourier_modes_test,
        use_wavelet_block=True, wavelet_levels=2, wavelet_name='db4',
        output_dim_per_symbol=output_dim_test
    ).to(DEVICE)
    model.eval()
    dummy_features = torch.randn(bs, num_s, num_t, num_f_in).to(DEVICE)
    dummy_padding_mask = torch.zeros(bs, num_s, dtype=torch.bool).to(DEVICE)
    if num_s > 1: dummy_padding_mask[:, -1] = True
    logger.info(f"輸入特徵 shape: {dummy_features.shape}, Padding mask shape: {dummy_padding_mask.shape if dummy_padding_mask is not None else 'None'}")
    try:
        with torch.no_grad(): output = model(dummy_features, dummy_padding_mask)
        logger.info(f"模型輸出 shape: {output.shape}")
        expected_shape = (bs, num_s, output_dim_test)
        assert output.shape == expected_shape, f"輸出形狀不匹配! 預期 {expected_shape}, 得到 {output.shape}"
        if dummy_padding_mask is not None and num_s > 1:
            padded_output_sum = output[:, -1, :].abs().sum()
            assert torch.allclose(padded_output_sum, torch.tensor(0.0, device=DEVICE), atol=1e-6), f"Padding槽位輸出不為零: sum={padded_output_sum.item()}"
            logger.info("Padding槽位輸出已驗證為零。")
        l2_norm = sum(p.data.norm(2).item() ** 2 for p in model.parameters() if p.requires_grad) ** 0.5
        logger.info(f"模型總可訓練參數L2範數: {l2_norm}")
        logger.info("增強版 UniversalTradingTransformer 模型基本測試通過！")
    except Exception as e:
        logger.error(f"模型測試過程中發生錯誤: {e}", exc_info=True)