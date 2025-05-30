import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class QuantumPolicyLayer(nn.Module):
    """量子啟發式交易策略層 (整合時間序列與波動率)"""
    
    def __init__(self, state_dim, action_dim, num_strategies=3, nhead=4, n_layers=2):
        super().__init__()
        self.num_strategies = num_strategies
        
        # 時間序列處理層
        self.time_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            TransformerEncoder(
                TransformerEncoderLayer(d_model=256, nhead=nhead, batch_first=True),
                num_layers=n_layers
            )
        )
        
        # 波動率特徵處理
        self.volatility_net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 256)
        )
        
        # 策略網絡
        self.strategy_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.LayerNorm(256),
                nn.Linear(256, action_dim)
            ) for _ in range(num_strategies)
        ])
        
        # 量子振幅參數
        self.amplitudes = nn.Parameter(torch.ones(num_strategies) / num_strategies)
        self.time_pool = nn.AdaptiveAvgPool1d(1)
        self.quantum_optimizer = torch.optim.Adam([self.amplitudes], lr=1e-3)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
    def set_training_mode(self, mode: bool):
        """設置訓練/評估模式"""
        self.train(mode)
        
    def forward(self, state, volatility):
        # 處理時間序列
        time_features = self.time_encoder(state)
        time_features = self.time_pool(time_features.permute(0, 2, 1)).squeeze(-1)
        
        # 處理波動率
        vol_features = self.volatility_net(volatility)
        
        # 合併特徵
        combined = torch.cat([time_features, vol_features], dim=1)
        
        # 生成策略動作
        strategy_actions = torch.stack([net(combined) for net in self.strategy_nets], dim=1)
        
        # 應用量子振幅
        weights = F.softmax(self.amplitudes, dim=0).view(1, -1, 1)
        final_action = torch.sum(strategy_actions * weights, dim=1)
        
        amplitudes_batch = weights.squeeze(0).squeeze(-1).expand(state.size(0), -1)
        return final_action, amplitudes_batch
        
    def quantum_annealing_step(self, rewards):
        """量子退火優化步驟"""
        probs = F.softmax(self.amplitudes, dim=0)
        loss = -torch.mean(torch.sum(torch.log(probs) * rewards, dim=1))
        
        self.quantum_optimizer.zero_grad()
        loss.backward()
        self.quantum_optimizer.step()
        
        self.temperature.data = torch.clamp(self.temperature * 0.99, min=0.1)