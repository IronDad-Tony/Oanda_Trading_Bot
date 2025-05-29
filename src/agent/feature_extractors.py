# src/agent/feature_extractors.py
"""
為 Stable Baselines3 定義自定義特徵提取器。
這裡我們將包裝 UniversalTradingTransformer 模型。
"""
import torch
import torch.nn as nn
from gymnasium import spaces # 確保導入
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict
import sys # 確保導入
from pathlib import Path # 確保導入
import numpy as np # <--- 在文件頂部導入 numpy

try:
    from models.transformer_model import UniversalTradingTransformer
    from common.config import (
        TIMESTEPS, MAX_SYMBOLS_ALLOWED,
        TRANSFORMER_MODEL_DIM, TRANSFORMER_NUM_LAYERS, TRANSFORMER_NUM_HEADS,
        TRANSFORMER_FFN_DIM, TRANSFORMER_DROPOUT_RATE,
        TRANSFORMER_OUTPUT_DIM_PER_SYMBOL, DEVICE # <--- 確保 DEVICE 從 config 導入
    )
    from common.logger_setup import logger
except ImportError:
    # project_root_fe = Path(__file__).resolve().parent.parent.parent # 移除
    # src_path_fe = project_root_fe / "src" # 移除
    # if str(project_root_fe) not in sys.path: # 移除
    #     sys.path.insert(0, str(project_root_fe)) # 移除
    try:
        # 假設 PYTHONPATH 已設定，這些導入應該能工作
        from src.models.transformer_model import UniversalTradingTransformer
        from src.common.config import (
            TIMESTEPS, MAX_SYMBOLS_ALLOWED,
            TRANSFORMER_MODEL_DIM, TRANSFORMER_NUM_LAYERS, TRANSFORMER_NUM_HEADS,
            TRANSFORMER_FFN_DIM, TRANSFORMER_DROPOUT_RATE,
            TRANSFORMER_OUTPUT_DIM_PER_SYMBOL, DEVICE # <--- 確保 DEVICE 從 config 導入
        )
        from src.common.logger_setup import logger
        logger.info("Direct run FeatureExtractor: Successfully re-imported common modules.")
    except ImportError as e_retry_fe:
        import logging
        logger = logging.getLogger("feature_extractor_fallback") # type: ignore
        logger.error(f"Direct run FeatureExtractor: Critical import error: {e_retry_fe}", exc_info=True)
        UniversalTradingTransformer = None # type: ignore
        TIMESTEPS=128; MAX_SYMBOLS_ALLOWED=20; TRANSFORMER_OUTPUT_DIM_PER_SYMBOL=64
        DEVICE = torch.device("cpu") # <--- 後備 DEVICE 定義


class AdvancedTransformerFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict,
                 transformer_output_dim_per_symbol: int = TRANSFORMER_OUTPUT_DIM_PER_SYMBOL,
                 model_dim: int = TRANSFORMER_MODEL_DIM, num_time_encoder_layers: int = TRANSFORMER_NUM_LAYERS // 2,
                 num_cross_asset_layers: int = TRANSFORMER_NUM_LAYERS // 2, num_heads: int = TRANSFORMER_NUM_HEADS,
                 ffn_dim: int = TRANSFORMER_FFN_DIM, dropout_rate: float = TRANSFORMER_DROPOUT_RATE,
                 use_fourier_block: bool = True, fourier_num_modes: int = 32,
                 use_wavelet_block: bool = True, wavelet_levels: int = 3, wavelet_name: str = 'db4',
                 use_amp: bool = False): # 控制是否對Transformer啟用AMP
        transformer_input_shape = observation_space.spaces["features_from_dataset"].shape
        num_input_features_for_transformer = transformer_input_shape[2]
        num_pos_ratio_features = observation_space.spaces["current_positions_nominal_ratio_ac"].shape[0]
        num_pnl_ratio_features = observation_space.spaces["unrealized_pnl_ratio_ac"].shape[0]
        num_margin_level_features = observation_space.spaces["margin_level"].shape[0]
        _features_dim = (MAX_SYMBOLS_ALLOWED * transformer_output_dim_per_symbol) + \
                        num_pos_ratio_features + num_pnl_ratio_features + num_margin_level_features
        super().__init__(observation_space, _features_dim)
        
        self.use_amp = use_amp
        # 考慮從 common.config 導入 USE_AMP 作為 self.use_amp 的預設值
        # from common.config import USE_AMP as global_use_amp
        # self.use_amp = use_amp if use_amp is not None else global_use_amp # 如果參數未提供則使用全局配置

        logger.info(f"AdvancedTransformerFeatureExtractor initialized. Input feature dim for Transformer: {num_input_features_for_transformer}")
        logger.info(f"Total flattened output features_dim for SAC: {_features_dim}")
        logger.info(f"AMP (autocast) will be enabled for Transformer forward pass if CUDA is available: {self.use_amp}")

        if UniversalTradingTransformer is None: raise RuntimeError("UniversalTradingTransformer未能正確導入。")
        self.transformer = UniversalTradingTransformer(
            num_input_features=num_input_features_for_transformer, num_symbols_possible=MAX_SYMBOLS_ALLOWED,
            model_dim=model_dim, num_time_encoder_layers=num_time_encoder_layers,
            num_cross_asset_layers=num_cross_asset_layers, num_heads=num_heads, ffn_dim=ffn_dim,
            dropout_rate=dropout_rate, use_fourier_block=use_fourier_block,
            fourier_num_modes=min(fourier_num_modes, TIMESTEPS // 2 + 1 if TIMESTEPS > 0 else 16),
            use_wavelet_block=use_wavelet_block, wavelet_levels=wavelet_levels, wavelet_name=wavelet_name,
            output_dim_per_symbol=transformer_output_dim_per_symbol)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        transformer_input_data = observations["features_from_dataset"]
        symbol_padding_mask = observations["padding_mask"]
        current_positions_ratio = observations["current_positions_nominal_ratio_ac"]
        unrealized_pnl_ratio = observations["unrealized_pnl_ratio_ac"]
        margin_level = observations["margin_level"]        # 確定是否為此特定前向傳播啟用AMP
        # 只有當 use_amp 為 True 且 CUDA 可用時才啟用
        amp_enabled_for_this_forward = self.use_amp and torch.cuda.is_available()
        
        # 使用 autocast 上下文僅包裹 Transformer 的調用
        with torch.amp.autocast('cuda', enabled=amp_enabled_for_this_forward):
            # 確保輸入給 Transformer 的數據是 float32，autocast 會處理後續轉換
            transformer_output_internal = self.transformer(transformer_input_data.to(torch.float32), symbol_padding_mask)
            # transformer_output_internal 的 dtype 可能是 float16 (如果 amp_enabled) 或 float32

        # 將 Transformer 的輸出明確轉換為 float32，以便與其他 float32 特徵拼接
        # 並且確保後續網絡接收的是 float32
        transformer_output_float32 = transformer_output_internal.to(torch.float32)
        
        if amp_enabled_for_this_forward and transformer_output_internal.dtype == torch.float16:
            logger.debug(f"Transformer output was float16 (due to AMP), converted to float32 before concatenation.")

        batch_size = transformer_output_float32.size(0)
        flattened_transformer_output = transformer_output_float32.reshape(batch_size, -1)
        
        # 拼接其他特徵，它們預期是 float32
        # 確保所有參與拼接的張量都是 float32
        final_features = torch.cat([flattened_transformer_output,
                                    current_positions_ratio.to(torch.float32),
                                    unrealized_pnl_ratio.to(torch.float32),
                                    margin_level.to(torch.float32)], dim=1)
        
        # 最終再次確認輸出是 float32 (儘管上述步驟已確保)
        if final_features.dtype != torch.float32:
            logger.warning(f"Final features dtype was {final_features.dtype}, forcing to float32. This should not happen if inputs are handled correctly.")
            final_features = final_features.to(torch.float32)
            
        return final_features


if __name__ == "__main__":
    logger.info("正在直接運行 AdvancedTransformerFeatureExtractor.py 進行測試...")
    # --- 確保 np 和 DEVICE 在此作用域可用 ---
    # np 應該在文件頂部導入
    # DEVICE 應該從 common.config 導入，如果失敗則使用後備
    if 'UniversalTradingTransformer' not in globals() or UniversalTradingTransformer is None:
        logger.error("UniversalTradingTransformer is None (likely import error). Test cannot proceed.")
        sys.exit(1)
    if 'DEVICE' not in globals(): # 再次檢查 DEVICE 是否已定義
        logger.warning("DEVICE is not defined in __main__ scope, attempting to re-import or use fallback.")
        try:
            from src.common.config import DEVICE as DEVICE_FROM_CONFIG # 嘗試再次導入
            DEVICE = DEVICE_FROM_CONFIG
            logger.info("Successfully re-imported DEVICE for __main__.")
        except ImportError:
            DEVICE = torch.device("cpu") # 最終後備
            logger.error("Failed to re-import DEVICE, using cpu as fallback for __main__ test.")


    num_test_input_features = 9
    test_obs_space = spaces.Dict({
        "features_from_dataset": spaces.Box(low=-np.inf, high=np.inf, shape=(MAX_SYMBOLS_ALLOWED, TIMESTEPS, num_test_input_features), dtype=np.float32),
        "current_positions_nominal_ratio_ac": spaces.Box(low=-1.0, high=1.0, shape=(MAX_SYMBOLS_ALLOWED,), dtype=np.float32),
        "unrealized_pnl_ratio_ac": spaces.Box(low=-1.0, high=1.0, shape=(MAX_SYMBOLS_ALLOWED,), dtype=np.float32),
        "margin_level": spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32),
        "padding_mask": spaces.Box(low=0, high=1, shape=(MAX_SYMBOLS_ALLOWED,), dtype=np.bool_)
    })
    try:
        feature_extractor = AdvancedTransformerFeatureExtractor(
            observation_space=test_obs_space, transformer_output_dim_per_symbol=64, model_dim=128,
            num_time_encoder_layers=2, num_cross_asset_layers=1, num_heads=4, ffn_dim=256,
            fourier_num_modes=16, wavelet_levels=2
        ).to(DEVICE) # 使用已確保定義的 DEVICE
        feature_extractor.eval()
        logger.info(f"測試用 Feature Extractor 初始化完成。期望輸出維度: {feature_extractor.features_dim}")
        batch_size_test = 4
        dummy_observation = test_obs_space.sample()
        dummy_observation_torch: Dict[str, torch.Tensor] = {}
        for key, value in dummy_observation.items():
            dtype = torch.bool if key == "padding_mask" else torch.float32
            dummy_observation_torch[key] = torch.from_numpy(value).unsqueeze(0).expand(batch_size_test, *value.shape).to(DEVICE).to(dtype)
        logger.info(f"創建的假觀察樣本 (batch_size={batch_size_test}):")
        for key, tensor in dummy_observation_torch.items(): logger.info(f"  '{key}' shape: {tensor.shape}, dtype: {tensor.dtype}")
        with torch.no_grad(): extracted_features = feature_extractor(dummy_observation_torch)
        logger.info(f"提取的特徵形狀: {extracted_features.shape}")
        expected_output_dim_flat = feature_extractor.features_dim
        assert extracted_features.shape == (batch_size_test, expected_output_dim_flat), f"提取特徵形狀不匹配! 預期 ({batch_size_test}, {expected_output_dim_flat}), 得到 {extracted_features.shape}"
        logger.info("AdvancedTransformerFeatureExtractor 基本測試通過！")
    except Exception as e:
        logger.error(f"特徵提取器測試過程中發生錯誤: {e}", exc_info=True)