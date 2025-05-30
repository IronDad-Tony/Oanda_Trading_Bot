"""
多時間維度記憶映射數據集
支援同時加載多個時間維度的金融數據（5S, 1M, 15M, 1H）
並提供統一的時間對齊接口

主要功能:
1. 自動聚合不同時間維度數據
2. 確保各時間維度數據在時間軸上對齊
3. 提供統一的數據訪問接口
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from .mmap_dataset import UniversalMemoryMappedDataset

class MultiTimeframeDataset:
    """
    管理多時間維度數據的統一容器
    
    參數:
        symbols: 交易品種列表 (e.g., ['EUR_USD', 'AUD_USD'])
        timeframes: 時間維度列表 (e.g., ['S5', 'M1', 'H1'])
        start_iso: 起始時間 (ISO格式)
        end_iso: 結束時間 (ISO格式)
        base_granularity: 基礎時間粒度 (默認'S5')
    """
    def __init__(self, symbols: List[str], timeframes: List[str], 
                 start_iso: str, end_iso: str, base_granularity: str = 'S5'):
        self.symbols = symbols
        self.timeframes = timeframes
        self.start_iso = start_iso
        self.end_iso = end_iso
        self.base_granularity = base_granularity
        
        # 初始化各時間維度數據集
        self.datasets: Dict[str, UniversalMemoryMappedDataset] = {}
        for tf in timeframes:
            self.datasets[tf] = UniversalMemoryMappedDataset(
                symbols=symbols,
                start_time_iso=start_iso,
                end_time_iso=end_iso,
                granularity=tf,
                timesteps_history=0  # 由外部控制
            )
    
    def get_multi_frame_data(self, timestamp: pd.Timestamp) -> Dict[str, Any]:
        """
        獲取特定時間點的多維度數據
        
        參數:
            timestamp: 精確時間戳記 (需包含時區資訊)
            
        返回:
            包含各時間維度數據的字典:
            {
                'S5': 5秒數據,
                'M1': 1分鐘數據,
                'H1': 1小時數據
            }
        """
        result = {}
        for tf, dataset in self.datasets.items():
            try:
                # 獲取最接近的時間點數據
                idx = dataset.time_index.get_loc(timestamp, method='nearest')
                result[tf] = dataset[idx]
            except KeyError:
                # 處理時間點不存在的情況
                result[tf] = None
        return result
    
    def close(self):
        """關閉所有數據集釋放資源"""
        for dataset in self.datasets.values():
            dataset.close()
    
    def __len__(self):
        """返回基礎時間維度的數據點數量"""
        return len(self.datasets[self.base_granularity])
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """通過索引獲取多維度數據"""
        base_timestamp = self.datasets[self.base_granularity].time_index[idx]
        return self.get_multi_frame_data(base_timestamp)
