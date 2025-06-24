import torch
import numpy as np
import logging
from typing import Optional, Dict, Any

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
        self.model: Optional[torch.nn.Module] = None
        self.device: str = 'cpu'
        self.current_model_path: Optional[str] = None
        self.max_symbols = config.get('max_symbols_allowed', 10) # 從設定檔讀取
        self.model_lookback_window = config.get('model_lookback_window', 128)
        logger.info(f"PredictionService initialized. Max symbols: {self.max_symbols}, Lookback: {self.model_lookback_window}")

    def load_model(self, model_path: str, device: str = 'cpu'):
        """
        從指定路徑載入 PyTorch 模型。
        """
        if self.current_model_path == model_path and self.model is not None:
            logger.info(f"Model from {model_path} is already loaded.")
            return

        self.device = device
        try:
            # 假設模型是 UniversalTradingTransformer 或類似結構
            self.model = torch.load(model_path, map_location=self.device)
            self.model.to(self.device)
            self.model.eval()
            self.current_model_path = model_path
            logger.info(f"模型成功從 {model_path} 載入到 {self.device}。")
        except FileNotFoundError:
            logger.error(f"錯誤: 模型檔案不存在於 {model_path}")
            self.model = None
            self.current_model_path = None
            raise
        except Exception as e:
            logger.error(f"從 {model_path} 載入模型時發生錯誤: {e}", exc_info=True)
            self.model = None
            self.current_model_path = None
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
