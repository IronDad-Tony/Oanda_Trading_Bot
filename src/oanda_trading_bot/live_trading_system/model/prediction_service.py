# 匯入 PyTorch 與 Numpy
import torch
import numpy as np
import logging
from typing import Optional, Dict, Any
# 匯入 stable_baselines3 的 SAC 類別，用於正確載入 SAC 模型
from stable_baselines3 import SAC

logger = logging.getLogger("LiveTradingSystem")

class PredictionService:
    """
    處理已訓練模型的載入和預測生成。
    能夠處理多個交易對的批次預測，並管理 padding/masking。
    """
    def __init__(self, config: Dict[str, Any]):
        """
        初始化 PredictionService。
        
        Args:
            config (Dict[str, Any]): 系統設定檔。
        """
        self.model: Optional[object] = None  # 可能為 torch.nn.Module 或 stable_baselines3.SAC
        self.device: str = 'cpu'
        self.current_model_path: Optional[str] = None
        self.max_symbols = config.get('max_symbols_allowed', 10) # 從設定檔讀取
        self.model_lookback_window = config.get('model_lookback_window', 128)
        # 新增：記錄目前模型是否為 SAC
        self.is_sac_model: bool = False
        logger.info(f"PredictionService initialized. Max symbols: {self.max_symbols}, Lookback: {self.model_lookback_window}")

    def load_model(self, model_path: str, device: str = 'cpu'):
        """
        從指定路徑載入模型，支援 PyTorch 及 stable_baselines3 SAC。
        若檔名或副檔名包含 'sac' 字樣，則以 stable_baselines3.SAC.load 載入。
        其餘則以 torch.load 載入。
        """
        if self.current_model_path == model_path and self.model is not None:
            logger.info(f"Model from {model_path} is already loaded.")
            return
        
        self.device = device
        try:
            # 判斷是否為 SAC 模型（可依實際命名規則調整）
            if 'sac' in model_path.lower():
                # 使用 stable_baselines3 SAC 方式載入
                # 設定 is_sac_model 為 True
                self.model = SAC.load(model_path, device=self.device)
                self.is_sac_model = True
                self.current_model_path = model_path
                logger.info(f"SAC 模型已使用 stable_baselines3.SAC.load 從 {model_path} 載入到 {self.device}。")
            else:
                # 預設為 PyTorch 類型模型
                self.model = torch.load(model_path, map_location=self.device)
                self.model.to(self.device)
                self.model.eval()
                self.is_sac_model = False
                self.current_model_path = model_path
                logger.info(f"PyTorch 模型成功從 {model_path} 載入到 {self.device}。")
        except FileNotFoundError:
            logger.error(f"錯誤: 模型檔案不存在於 {model_path}")
            self.model = None
            self.current_model_path = None
            self.is_sac_model = False
            raise
        except Exception as e:
            logger.error(f"從 {model_path} 載入模型時發生錯誤: {e}", exc_info=True)
            self.model = None
            self.current_model_path = None
            self.is_sac_model = False
            raise

    def predict(self, processed_data_map: Dict[str, np.ndarray]) -> Dict[str, int]:
        """
        根據一批預處理後的數據生成預測。
        處理多個交易對的數據，進行必要的填充和掩碼，然後進行批次預測。
    
        Args:
            processed_data_map (Dict[str, np.ndarray]):
                一個字典，鍵是交易對名稱，值是預處理後的數據 (T, C)，
                其中 T 是時間步長，C 是特徵數。
    
        Returns:
            Dict[str, int]: 一個字典，鍵是交易對名稱，值是預測的動作。
            
        Raises:
            ValueError: 如果模型未載入或輸入數據有問題。
        """
        if self.model is None:
            logger.error("模型未載入，無法預測。")
            raise ValueError("模型未載入。請先呼叫 load_model()。")
    
        if not processed_data_map:
            logger.warning("收到的 processed_data_map 為空，不進行預測。")
            return {}
    
        instruments = list(processed_data_map.keys())
        if len(instruments) > self.max_symbols:
            logger.warning(f"提供的交易對數量 ({len(instruments)}) 超過模型最大容量 ({self.max_symbols})。將只使用前 {self.max_symbols} 個。")
            instruments = instruments[:self.max_symbols]
    
        # 獲取特徵維度
        first_instrument_data = next(iter(processed_data_map.values()))
        if first_instrument_data.ndim != 2:
            raise ValueError(f"數據維度不正確。預期為 2D (T, C)，但收到 {first_instrument_data.ndim}D。")
        _, num_features = first_instrument_data.shape
    
        # 準備批次張量和 padding mask
        # Tensor shape: (batch_size, num_symbols, sequence_length, num_features)
        batch_tensor = torch.zeros(1, self.max_symbols, self.model_lookback_window, num_features, dtype=torch.float32)
        # Mask shape: (batch_size, num_symbols) -> True 表示 padded
        padding_mask = torch.ones(1, self.max_symbols, dtype=torch.bool)
    
        active_instruments = []
        for i, instrument in enumerate(instruments):
            data = processed_data_map[instrument]
            
            # 處理長度不足的數據 (在時間序列開頭填充0)
            if data.shape[0] < self.model_lookback_window:
                pad_width = self.model_lookback_window - data.shape[0]
                # 在第一個維度 (時間) 的開頭填充
                padded_data = np.pad(data, ((pad_width, 0), (0, 0)), 'constant', constant_values=0)
            else:
                # 確保數據長度正確
                padded_data = data[-self.model_lookback_window:]
    
            batch_tensor[0, i, :, :] = torch.from_numpy(padded_data)
            padding_mask[0, i] = False  # 標記為非填充
            active_instruments.append(instrument)
    
        batch_tensor = batch_tensor.to(self.device)
        padding_mask = padding_mask.to(self.device)
    
        try:
            # 判斷目前模型是否為 SAC
            if self.is_sac_model:
                # ========== SAC 推論流程 ==========
                # 針對每個 active_instrument，組裝 observation dict 並進行推論
                results = {}
                for i, instrument in enumerate(active_instruments):
                    # 取得 observation_space 結構
                    obs_space = self.model.observation_space

                    # 取得 processed_data，shape: (lookback_window, num_features)
                    data = processed_data_map[instrument]
                    # 若長度不足，需與前面一致進行 padding
                    if data.shape[0] < self.model_lookback_window:
                        pad_width = self.model_lookback_window - data.shape[0]
                        padded_data = np.pad(data, ((pad_width, 0), (0, 0)), 'constant', constant_values=0)
                    else:
                        padded_data = data[-self.model_lookback_window:]

                    # 組裝 observation dict，需包含 'context_features', 'market_features', 'symbol_id'
                    # 取得各 key 的 shape
                    obs_dict = {}
                    for key in obs_space.spaces:
                        space = obs_space.spaces[key]
                        # 依照 key 分類處理
                        if key == 'market_features':
                            # market_features 填入 processed_data（如有 context_features，需分割）
                            # 若 processed_data 包含多於一組特徵，需根據 shape 對應
                            # 取得目標 shape
                            target_shape = space.shape
                            # 若 processed_data 特徵數與 target_shape 不符，則補 0
                            if padded_data.shape[-1] >= target_shape[-1]:
                                obs_dict[key] = padded_data[:, :target_shape[-1]].astype(np.float32)
                            else:
                                # 特徵數不足時補 0
                                temp = np.zeros((padded_data.shape[0], target_shape[-1]), dtype=np.float32)
                                temp[:, :padded_data.shape[-1]] = padded_data
                                obs_dict[key] = temp
                        elif key == 'context_features':
                            # context_features 若 processed_data 有包含，則分割出來，否則補 0
                            target_shape = space.shape
                            # 假設 context_features 在 processed_data 最後幾個欄位（如有），否則補 0
                            if padded_data.shape[-1] >= target_shape[-1]:
                                obs_dict[key] = padded_data[:, -target_shape[-1]:].astype(np.float32)
                            else:
                                obs_dict[key] = np.zeros((padded_data.shape[0], target_shape[-1]), dtype=np.float32)
                        elif key == 'symbol_id':
                            # symbol_id 直接填入 instrument index
                            obs_dict[key] = np.array([i], dtype=np.int32)
                        else:
                            # 其他 key 一律補 0
                            if len(space.shape) == 0:
                                obs_dict[key] = np.array(0, dtype=space.dtype if hasattr(space, 'dtype') else np.float32)
                            else:
                                obs_dict[key] = np.zeros(space.shape, dtype=space.dtype if hasattr(space, 'dtype') else np.float32)

                    # ==========================
                    # 進行 SAC 模型推論
                    # ==========================
                    try:
                        # SAC 模型預測，需傳入 observation dict
                        # 取得動作（action），通常為 int
                        action, _ = self.model.predict(obs_dict, deterministic=True)
                        results[instrument] = int(action)
                    except Exception as e:
                        logger.error(f"SAC模型推論失敗: {e}", exc_info=True)
                        results[instrument] = None

                logger.info(f"SAC 模型成功生成預測: {results}")
                return results
            else:
                # ========== 原本 PyTorch 模型推論流程 ==========
                with torch.no_grad():
                    # 假設模型 forward 接受 (features, padding_mask_symbols)
                    output = self.model(batch_tensor, padding_mask_symbols=padding_mask)
                    
                    # 假設輸出是分類的 logits，形狀為 (batch_size, num_symbols, num_actions)
                    predictions_tensor = torch.argmax(output, dim=2) # (1, max_symbols)
                    
                    results = {}
                    for i, instrument in enumerate(active_instruments):
                        results[instrument] = predictions_tensor[0, i].item()
                    
                    logger.info(f"成功生成預測: {results}")
                    return results
    
        except Exception as e:
            logger.error(f"預測期間發生錯誤: {e}", exc_info=True)
            # 返回空字典表示失敗
            return {}
