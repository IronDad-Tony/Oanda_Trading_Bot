# src/agent/quantum_strategy_layer.py
"""
完整的量子啟發式交易策略層實現
基於技術文檔中的設計，實現三個核心組件：
1. QuantumEncoder - 量子編碼器
2. StrategySuperposition - 策略疊加
3. HamiltonianObserver - 哈密頓觀察者
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from abc import ABC, abstractmethod

try:
    from oanda_trading_bot.training_system.common.logger_setup import logger
    from oanda_trading_bot.training_system.common.config import DEVICE, MAX_SYMBOLS_ALLOWED
except ImportError:
    logger = logging.getLogger(__name__)
    DEVICE = "cpu"
    MAX_SYMBOLS_ALLOWED = 5

class QuantumEncoder(nn.Module):
    """
    量子編碼器：將市場狀態編碼到量子啟發的表示空間
    使用相位偏移和非線性變換來創建量子態的類比
    """
    
    def __init__(self, input_dim: int, latent_dim: int, num_qubits: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_qubits = num_qubits
        
        # 主要編碼網絡
        self.encoding_layers = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 2),
            nn.BatchNorm1d(latent_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.BatchNorm1d(latent_dim),
        )
        
        # 量子相位參數（可學習）
        self.phase_shift = nn.Parameter(torch.randn(latent_dim) * 0.1)
        self.frequency_scale = nn.Parameter(torch.ones(latent_dim))
        
        # 量子糾纏模擬：相互作用項
        self.entanglement_matrix = nn.Parameter(torch.randn(latent_dim, latent_dim) * 0.01)
        
        # 量子測量概率編碼
        self.measurement_projection = nn.Linear(latent_dim, latent_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播：將市場狀態編碼為量子啟發的表示
        
        Args:
            x: 市場狀態張量 [batch_size, input_dim]
            
        Returns:
            量子編碼的狀態 [batch_size, latent_dim]
        """
        batch_size = x.size(0)
        
        # 基礎編碼
        encoded = self.encoding_layers(x)
        
        # 量子相位編碼：使用正弦和餘弦函數模擬量子相位
        phase_component = torch.sin(encoded * self.frequency_scale + self.phase_shift)
        amplitude_component = torch.cos(encoded * self.frequency_scale + self.phase_shift)
        
        # 量子糾纏效應：通過矩陣乘法模擬量子態之間的相互作用
        entangled_state = torch.matmul(phase_component, self.entanglement_matrix)
        
        # 疊加態：結合相位和振幅分量
        superposition = phase_component + 0.5 * entangled_state
        
        # 量子測量：投影到觀測空間
        quantum_state = torch.tanh(self.measurement_projection(superposition))
        
        return quantum_state
    
    def get_quantum_probabilities(self, quantum_state: torch.Tensor) -> torch.Tensor:
        """
        計算量子測量概率（玻恩規則的類比）
        
        Args:
            quantum_state: 量子編碼狀態
            
        Returns:
            測量概率分佈
        """
        probabilities = torch.softmax(torch.abs(quantum_state) ** 2, dim=-1)
        return probabilities


class BaseStrategy(nn.Module, ABC):
    """基礎策略抽象類"""
    
    @abstractmethod
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        pass


class ArbitrageStrategy(BaseStrategy):
    """套利策略網絡"""
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.strategy_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.strategy_net(state)
    
    def get_strategy_name(self) -> str:
        return "Arbitrage"


class TrendFollowingStrategy(BaseStrategy):
    """趨勢跟隨策略網絡"""
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.strategy_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.strategy_net(state) * 1.2  # 趨勢策略的放大因子
    
    def get_strategy_name(self) -> str:
        return "TrendFollowing"


class MeanReversionStrategy(BaseStrategy):
    """均值回歸策略網絡"""
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.strategy_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Swish(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.Swish(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return -self.strategy_net(state) * 0.8  # 反向信號，較小的放大因子
    
    def get_strategy_name(self) -> str:
        return "MeanReversion"


class StrategySuperposition(nn.Module):
    """
    策略疊加層：管理多個基礎策略的量子疊加
    動態調整策略權重（振幅），根據市場波動率進行調節
    """
    
    def __init__(self, state_dim: int, action_dim: int, num_strategies: Optional[int] = None, 
                 custom_strategies: Optional[List[nn.Module]] = None, device: Optional[torch.device] = None):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 策略池長度以 custom_strategies 為主
        if custom_strategies is not None:
            self.strategies = nn.ModuleList([s.to(self.device) for s in custom_strategies])
            self.num_strategies = len(custom_strategies)
        else:
            self.strategies = nn.ModuleList([
                TrendFollowingStrategy(state_dim, action_dim).to(self.device),
                MeanReversionStrategy(state_dim, action_dim).to(self.device)
            ])
            self.num_strategies = len(self.strategies)
        # 可學習的量子振幅參數
        self.base_amplitudes = nn.Parameter(torch.ones(self.num_strategies, device=self.device) / self.num_strategies)
        # 波動率調節網絡
        self.volatility_modulator = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_strategies),
            nn.Sigmoid()
        ).to(self.device)
        # 自適應權重調整
        self.adaptive_weight_net = nn.Sequential(
            nn.Linear(state_dim + 1, 64),  # state + volatility
            nn.ReLU(),
            nn.Linear(64, self.num_strategies),
            nn.Softmax(dim=-1)
        ).to(self.device)
        # 策略相關性學習
        self.strategy_correlation = nn.Parameter(torch.eye(self.num_strategies, device=self.device) * 0.1)
        
    def forward(self, state: torch.Tensor, volatility: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播：計算策略疊加的輸出
        
        Args:
            state: 量子編碼的市場狀態 [batch, num_strategies, latent_dim] 或 [batch, MAX_SYMBOLS_ALLOWED, latent_dim]
            volatility: 市場波動率 [batch, 1] 或 [batch, num_strategies, 1]
            
        Returns:
            Tuple[疊加策略輸出, 調整後的權重]
        """
        # 保證 device 對齊
        state = state.to(self.device)
        volatility = volatility.to(self.device)
        
        # 動態 shape 適配，保證 shape 一致且資訊不丟失
        if state.ndim == 2:
            # [batch, latent_dim] -> [batch, num_strategies, latent_dim]
            state = state.unsqueeze(1).expand(-1, self.num_strategies, -1)
        elif state.ndim == 3 and state.shape[1] != self.num_strategies:
            # [batch, MAX_SYMBOLS_ALLOWED, latent_dim] -> [batch, num_strategies, latent_dim] (取前 num_strategies 個 symbol)
            state = state[:, :self.num_strategies, :]
        batch_size = state.size(0)
        
        # 執行所有策略
        strategy_outputs = []
        for i, strategy in enumerate(self.strategies):
            output = strategy(state[:, i, :].to(self.device))  # [batch, action_dim]
            strategy_outputs.append(output)
        strategy_tensor = torch.stack(strategy_outputs, dim=1)  # [batch, num_strategies, action_dim]
        
        # 基礎振幅正規化
        base_amps = F.softmax(self.base_amplitudes, dim=0)
        
        # 波動率調節
        volatility_factor = self.volatility_modulator(volatility.unsqueeze(-1))

        # 自適應權重
        # 修正：將 volatility broadcast 成 [batch, num_strategies, 1]
        if volatility.dim() == 2 and volatility.shape[1] == 1:
            vol_broadcast = volatility.unsqueeze(1).expand(-1, state.shape[1], 1)  # [batch, num_strategies, 1]
        elif volatility.dim() == 3 and volatility.shape[1] == state.shape[1]:
            vol_broadcast = volatility  # 已經是 [batch, num_strategies, 1]
        else:
            raise ValueError(f"volatility shape {volatility.shape} 不符預期")
        state_vol_concat = torch.cat([state, vol_broadcast], dim=-1)
        adaptive_weights = self.adaptive_weight_net(state_vol_concat)
        
        # 組合所有權重因子
        combined_weights = base_amps.unsqueeze(0) * volatility_factor * adaptive_weights
        combined_weights = F.softmax(combined_weights, dim=-1)
        
        # 策略相關性調整
        correlation_adjusted_weights = torch.matmul(combined_weights, self.strategy_correlation)
        correlation_adjusted_weights = F.softmax(correlation_adjusted_weights, dim=-1)
        
        # 計算加權疊加
        final_weights = correlation_adjusted_weights.unsqueeze(-1)  # [batch, num_strategies, 1]
        superposed_output = torch.sum(strategy_tensor * final_weights, dim=1)  # [batch, action_dim]
        
        return superposed_output, correlation_adjusted_weights
    
    def get_strategy_contributions(self, state: torch.Tensor, volatility: torch.Tensor) -> Dict[str, torch.Tensor]:
        """獲取每個策略的貢獻度"""
        _, weights = self.forward(state, volatility)
        
        contributions = {}
        for i, strategy in enumerate(self.strategies):
            strategy_name = strategy.get_strategy_name()
            contributions[strategy_name] = weights[:, i].mean().item()
            
        return contributions


class HamiltonianObserver(nn.Module):
    """
    哈密頓觀察者：基於量子力學的哈密頓算子進行觀測
    實現量子測量和行動選擇的最終決策層
    """
    
    def __init__(self, action_dim: int, num_energy_levels: int = 8):
        super().__init__()
        self.action_dim = action_dim
        self.num_energy_levels = num_energy_levels
        
        # 能級參數（可學習的本徵值）
        self.energy_levels = nn.Parameter(torch.linspace(0.1, 2.0, num_energy_levels))
        
        # 哈密頓算子矩陣（可學習）
        self.hamiltonian_matrix = nn.Parameter(torch.randn(action_dim, action_dim) * 0.1)
        
        # 觀測算子
        self.observation_operators = nn.ParameterList([
            nn.Parameter(torch.randn(action_dim, action_dim) * 0.1) 
            for _ in range(num_energy_levels)
        ])
        
        # 最終決策網絡
        self.decision_net = nn.Sequential(
            nn.Linear(action_dim, action_dim * 2),
            nn.GELU(),
            nn.LayerNorm(action_dim * 2),
            nn.Dropout(0.1),
            nn.Linear(action_dim * 2, action_dim),
        )
        
        # 量子測量概率網絡
        self.measurement_net = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_energy_levels),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, strategy_output: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        哈密頓觀測和最終行動決策
        
        Args:
            strategy_output: 策略疊加的輸出
            training: 是否處於訓練模式
            
        Returns:
            Tuple[最終行動, 觀測信息字典]
        """
        batch_size = strategy_output.size(0)
        
        # 應用哈密頓算子（類比時間演化）
        hamiltonian_evolution = torch.matmul(strategy_output, self.hamiltonian_matrix)
        
        # 計算測量概率
        measurement_probs = self.measurement_net(strategy_output)
        
        # 能級加權觀測
        observed_states = []
        for i, operator in enumerate(self.observation_operators):
            # 每個能級的觀測結果
            observed = torch.matmul(hamiltonian_evolution, operator)
            energy_weight = self.energy_levels[i]
            prob_weight = measurement_probs[:, i:i+1]
            
            weighted_observed = observed * energy_weight * prob_weight
            observed_states.append(weighted_observed)
        
        # 疊加所有能級的觀測結果
        total_observed = torch.stack(observed_states, dim=1).sum(dim=1)
        
        # 最終決策
        final_action = self.decision_net(total_observed)
        
        # 在訓練模式下添加探索噪聲
        if training:
            exploration_noise = torch.randn_like(final_action) * 0.1
            final_action = final_action + exploration_noise
        
        # 動作約束到[-1, 1]
        final_action = torch.tanh(final_action)
        
        # 觀測信息
        observation_info = {
            'measurement_probabilities': measurement_probs,
            'energy_levels': self.energy_levels.detach(),
            'hamiltonian_evolution': hamiltonian_evolution,
            'total_observed': total_observed
        }
        
        return final_action, observation_info


class QuantumTradingLayer(nn.Module):
    """
    完整的量子啟發式交易策略層
    整合量子編碼器、策略疊加和哈密頓觀察者
    """
    
    def __init__(self, input_dim: int, action_dim: int, latent_dim: int = 256, 
                 num_strategies: int = 3, num_energy_levels: int = 8,
                 custom_strategies: Optional[List[BaseStrategy]] = None):
        super().__init__()
        
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        
        # 三個核心組件
        self.quantum_encoder = QuantumEncoder(input_dim, latent_dim)
        self.strategy_superposition = StrategySuperposition(
            latent_dim, action_dim, num_strategies, custom_strategies
        )
        self.hamiltonian_observer = HamiltonianObserver(action_dim, num_energy_levels)
        
        # 波動率估計網絡
        self.volatility_estimator = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 量子退火參數
        self.annealing_temperature = nn.Parameter(torch.tensor(1.0))
        self.annealing_schedule = 0.999  # 退火衰減率
        
    def forward(self, market_state: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        完整的量子策略層前向傳播
        
        Args:
            market_state: 原始市場狀態
            training: 是否處於訓練模式
            
        Returns:
            Tuple[最終交易行動, 詳細信息字典]
        """
        # 1. 量子編碼
        quantum_state = self.quantum_encoder(market_state)
        
        # 2. 波動率估計
        volatility = self.volatility_estimator(market_state).squeeze(-1)
        
        # 3. 策略疊加
        strategy_output, strategy_weights = self.strategy_superposition(quantum_state, volatility)
        
        # 4. 哈密頓觀測
        final_action, observation_info = self.hamiltonian_observer(strategy_output, training)
        
        # 量子退火更新（僅在訓練時）
        if training:
            self.annealing_temperature.data *= self.annealing_schedule
            self.annealing_temperature.data = torch.clamp(self.annealing_temperature.data, min=0.1)
        
        # 詳細信息匯總
        detailed_info = {
            'quantum_state': quantum_state,
            'volatility': volatility,
            'strategy_weights': strategy_weights,
            'strategy_contributions': self.strategy_superposition.get_strategy_contributions(quantum_state, volatility),
            'annealing_temperature': self.annealing_temperature.item(),
            'observation_info': observation_info
        }
        
        return final_action, detailed_info
    
    # 向後兼容屬性和方法（為了與舊代碼介面兼容）
    @property
    def amplitudes(self):
        """兼容舊版本的amplitudes屬性"""
        return self.strategy_superposition.base_amplitudes
    
    @property 
    def optimizer(self):
        """兼容舊版本的optimizer屬性"""
        if not hasattr(self, '_optimizer'):
            self._optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, value):
        """設置優化器"""
        self._optimizer = value
    
    def forward_compatible(self, state: torch.Tensor, volatility: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        向後兼容的前向傳播方法
        返回格式與舊版本QuantumPolicyLayer一致
        
        Args:
            state: 市場狀態
            volatility: 波動率
            
        Returns:
            Tuple[最終動作, 策略權重批次]
        """
        final_action, detailed_info = self.forward(state, training=self.training)
        strategy_weights = detailed_info['strategy_weights']
        
        # 將策略權重格式化為與舊版本兼容的格式
        batch_size = state.size(0)
        amplitudes_batch = strategy_weights  # 已經是正確的格式 [batch_size, num_strategies]
        
        return final_action, amplitudes_batch
    
    def quantum_annealing_step(self, rewards: torch.Tensor):
        """
        兼容舊版本的量子退火步驟
        
        Args:
            rewards: 獎勵張量 [batch_size, num_strategies]
        """
        if len(rewards.shape) == 1:
            rewards = rewards.unsqueeze(0)
            
        probs = F.softmax(self.amplitudes, dim=0)
        loss = -torch.mean(torch.sum(torch.log(probs) * rewards, dim=1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新退火溫度
        self.annealing_temperature.data *= self.annealing_schedule
        self.annealing_temperature.data = torch.clamp(self.annealing_temperature.data, min=0.1)
    
    def set_training_mode(self, mode: bool):
        """兼容舊版本的訓練模式設置方法"""
        self.train(mode)

    def get_quantum_metrics(self) -> Dict[str, float]:
        """獲取量子系統的性能指標"""
        metrics = {
            'annealing_temperature': self.annealing_temperature.item(),
            'quantum_entanglement_strength': torch.norm(self.quantum_encoder.entanglement_matrix).item(),
            'energy_level_spread': torch.std(self.hamiltonian_observer.energy_levels).item(),
            'strategy_diversity': torch.std(self.strategy_superposition.base_amplitudes).item()
        }
        return metrics
    
    def reset_annealing(self):
        """重置量子退火溫度"""
        self.annealing_temperature.data = torch.tensor(1.0)
        logger.info("量子退火溫度已重置")


# 自定義激活函數
class Swish(nn.Module):
    """Swish激活函數：x * sigmoid(x)"""
    
    def forward(self, x):
        return x * torch.sigmoid(x)

# 註冊Swish激活函數
nn.Swish = Swish


if __name__ == "__main__":
    # 測試量子策略層
    logger.info("開始測試量子交易策略層...")
    
    # 測試參數
    batch_size = 32
    input_dim = 128
    action_dim = 5  # 假設5個交易對
    
    # 創建測試數據
    test_market_state = torch.randn(batch_size, input_dim)
    
    # 初始化量子策略層
    quantum_layer = QuantumTradingLayer(
        input_dim=input_dim,
        action_dim=action_dim,
        latent_dim=256,
        num_strategies=3,
        num_energy_levels=8
    )
    
    logger.info(f"量子策略層參數數量: {sum(p.numel() for p in quantum_layer.parameters()):,}")
    
    # 前向傳播測試
    with torch.no_grad():
        final_action, detailed_info = quantum_layer(test_market_state, training=False)
        
        logger.info(f"輸入形狀: {test_market_state.shape}")
        logger.info(f"輸出動作形狀: {final_action.shape}")
        logger.info(f"策略權重: {detailed_info['strategy_weights'][0]}")
        logger.info(f"策略貢獻度: {detailed_info['strategy_contributions']}")
        logger.info(f"量子指標: {quantum_layer.get_quantum_metrics()}")
    
    # 梯度計算測試
    quantum_layer.train()
    final_action, _ = quantum_layer(test_market_state, training=True)
    loss = final_action.mean()
    loss.backward()
    
    total_grad_norm = 0
    for param in quantum_layer.parameters():
        if param.grad is not None:
            total_grad_norm += param.grad.data.norm(2) ** 2
    total_grad_norm = total_grad_norm ** 0.5
    
    logger.info(f"總梯度範數: {total_grad_norm:.6f}")
    logger.info("量子交易策略層測試完成！")
