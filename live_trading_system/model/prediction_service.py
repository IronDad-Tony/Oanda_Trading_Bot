import torch
import numpy as np
from typing import Optional

class PredictionService:
    """
    處理已訓練模型的載入和預測生成。
    """
    def __init__(self):
        """
        初始化 PredictionService。
        """
        self.model: Optional[torch.nn.Module] = None
        self.device: str = 'cpu'
        self.current_model_path: Optional[str] = None

    def load_model(self, model_path: str, device: str = 'cpu'):
        """
        從指定路徑載入 PyTorch 模型。
        支援動態模型載入。如果請求的模型已經載入，則不會執行任何操作。

        Args:
            model_path (str): PyTorch 模型 (.pth) 的檔案路徑。
            device (str): 載入模型的設備 ('cpu' 或 'cuda')。
        """
        if self.current_model_path == model_path and self.model is not None:
            # 模型已載入
            print(f"Model from {model_path} is already loaded.")
            return

        self.device = device
        try:
            # 為了簡單起見，這裡使用 torch.load() 直接載入整個模型。
            # 如果儲存的是 state_dict，則需要先實例化模型結構再載入。
            self.model = torch.load(model_path, map_location=self.device)
            self.model.to(self.device)
            self.model.eval()  # 將模型設置為評估模式
            self.current_model_path = model_path
            print(f"模型成功從 {model_path} 載入到 {self.device}。")
        except FileNotFoundError:
            print(f"錯誤: 模型檔案不存在於 {model_path}")
            self.model = None
            self.current_model_path = None
            raise
        except Exception as e:
            print(f"從 {model_path} 載入模型時發生錯誤: {e}")
            self.model = None
            self.current_model_path = None
            raise

    def predict(self, processed_data: np.ndarray) -> int:
        """
        根據預處理後的數據生成預測。

        Args:
            processed_data (np.ndarray): 輸入數據，已經過預處理和標準化，準備好輸入模型。

        Returns:
            int: 預測的動作 (例如: 0 for HOLD, 1 for BUY, 2 for SELL)。
        
        Raises:
            ValueError: 如果模型未載入。
        """
        if self.model is None:
            raise ValueError("模型未載入。請先呼叫 load_model()。")

        try:
            # 將 numpy 陣列轉換為 torch 張量
            # 數據應具有 (batch_size, sequence_length, num_features) 的形狀
            # 對於單一預測，batch_size 為 1。
            input_tensor = torch.from_numpy(processed_data).float().to(self.device)

            # 如果沒有 batch 維度，則添加一個
            if input_tensor.ndim == 2:
                input_tensor = input_tensor.unsqueeze(0)

            with torch.no_grad():
                output = self.model(input_tensor)
                # 假設輸出是分類的 logits
                prediction = torch.argmax(output, dim=1).item()
            
            return prediction
        except Exception as e:
            print(f"預測期間發生錯誤: {e}")
            # 根據所需的穩健性，您可以返回預設動作（如 HOLD (0)）或重新引發異常。
            raise
