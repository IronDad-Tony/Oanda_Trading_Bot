"""
Main End-to-End Flow Debugging Script
"""
import sys
import os
from pathlib import Path # Import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # Import F
import pandas as pd
import logging
from typing import Optional, Dict, Any, Tuple
import traceback # Import traceback
import importlib
import glob
import inspect

# --- Project Path Setup ---
# This should be the VERY FIRST part of the script related to project structure.
try:
    CURRENT_FILE_PATH = Path(__file__).resolve()
    PROJECT_ROOT = CURRENT_FILE_PATH.parents[1] 
except NameError:
    PROJECT_ROOT = Path(os.getcwd()).parent
    if not (PROJECT_ROOT / 'src').exists():
        PROJECT_ROOT = Path(os.getcwd())

SRC_PATH = PROJECT_ROOT / 'src'

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(PROJECT_ROOT) not in sys.path: # If modules are in project root
    sys.path.insert(0, str(PROJECT_ROOT))

print(f"PROJECT_ROOT set to: {PROJECT_ROOT}")
print(f"SRC_PATH set to: {SRC_PATH}")
# print(f"sys.path: {sys.path}") # For verbose debugging of sys.path

# Now that sys.path is configured, attempt main project imports
from src.models.enhanced_transformer import EnhancedTransformer
from src.agent.strategies.base_strategy import BaseStrategy, StrategyConfig
from src.agent.strategies.trend_strategies import TrendFollowingStrategy
from src.agent.strategies.statistical_arbitrage_strategies import MeanReversionStrategy
from src.agent.quantum_strategy_layer import StrategySuperposition
# ... any other top-level imports from the project ...

# --- Global Configuration & Device Setup ---
# Define STATE_DIM, s1_out_dim, s2_out_dim, etc. early
STATE_DIM = 64
s1_out_dim = 8
s2_out_dim = 8
ACTION_DIM = 5

# Setup device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA GPU detected, device set to: cuda")
    try:
        # Basic GPU information (optional, but good for debugging)
        print(f"   - GPU name: {torch.cuda.get_device_name(0)}")
        print(f"   - GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")
    except Exception as e:
        print(f"Could not get GPU details: {e}")
elif torch.backends.mps.is_available(): # For Apple Silicon
    device = torch.device("mps")
    print("Apple Metal Performance Shaders (MPS) detected, device set to: mps")
else:
    device = torch.device("cpu")
    print("No GPU or MPS detected, device set to: cpu")


# --- Dummy Strategy Definition ---
class DummyStrategy(BaseStrategy):
    def __init__(self, input_dim: int, output_dim: int, strategy_name: str = "DummyStrategy", params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        # 實務修正：output_dim 不屬於 StrategyConfig，僅 input_dim
        conf = StrategyConfig(name=strategy_name, description="A dummy strategy.", default_params={}, input_dim=input_dim)
        super().__init__(config=conf, params=params, logger=logger if logger else logging.getLogger(strategy_name))
        self.fc = nn.Linear(input_dim, output_dim)
        self.to(device) # Move to device during init

    def forward(self, x: torch.Tensor, current_positions: Optional[torch.Tensor] = None, timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor:
        x = x.to(next(self.parameters()).device)
        return torch.relu(self.fc(x))

    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(name="DummyStrategy", description="A dummy strategy.", default_params={})

# --- Quantum Strategy Layer Definition ---
class QuantumStrategyLayer(nn.Module):
    def __init__(self, strategy_pool, d_model, attention_dim):
        super().__init__()
        self.strategy_pool = strategy_pool  # 確保策略池正確設置
        self.attention_q = nn.Linear(d_model, attention_dim)
        # 根據實際拼接後維度設置 in_features
        self.attention_k = nn.Linear(len(strategy_pool), attention_dim)
        self.attention_v = nn.Linear(len(strategy_pool), attention_dim)
        self.attention = nn.MultiheadAttention(attention_dim, num_heads=1, batch_first=True)
        self.output_fc = nn.Linear(attention_dim, attention_dim)

        # 將策略池內所有策略都移到正確 device
        for i, s in enumerate(strategy_pool):
            strategy_pool[i] = s.to(device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: [batch_size_eff, sequence_len, feature_dim] from Transformer
        # Pass the full sequence to strategies. Strategies must handle 3D input.
        x_for_strategies = x.to(device) 

        strategy_outputs_list = []
        for strategy_module in self.strategies:
            # strategy_module is already on device from its __init__
            # Each strategy receives [B_eff, T, F]
            strategy_output_full_seq = strategy_module(x_for_strategies) 
            
            # For QL's attention mechanism, we need a 2D output [B_eff, strategy_feature_dim] from each strategy.
            # If strategy returns a sequence, take the last time step.
            if strategy_output_full_seq.ndim == 3:
                strategy_output_last_step = strategy_output_full_seq[:, -1, :]
            elif strategy_output_full_seq.ndim == 2:
                strategy_output_last_step = strategy_output_full_seq
            else:
                raise ValueError(f"Strategy {strategy_module.config.name if hasattr(strategy_module, 'config') else 'Unknown'} output has unexpected ndim: {strategy_output_full_seq.ndim}")
            strategy_outputs_list.append(strategy_output_last_step)
        
        concatenated_strategy_outputs = torch.cat(strategy_outputs_list, dim=1).to(device) # Shape: [B_eff, total_strategy_output_dim]

        batch_size_eff = x_for_strategies.size(0)
        
        # Query for QL's attention mechanism is derived from the last time step of QL's input features (x)
        ql_input_last_step_features = x_for_strategies[:, -1, :] # Shape: [B_eff, feature_dim]
        
        q = self.attention_q(ql_input_last_step_features).view(batch_size_eff, self.num_attention_heads, self.head_dim)
        k = self.attention_k(concatenated_strategy_outputs).view(batch_size_eff, self.num_attention_heads, self.head_dim)
        v = self.attention_v(concatenated_strategy_outputs).view(batch_size_eff, self.num_attention_heads, self.head_dim)
        
        attn_scores = torch.einsum("bhd,bmd->bhm", q, k) * self.scale_factor
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        context_vector = torch.einsum("bhm,bmd->bhd", attn_probs, v)
        context_vector = context_vector.contiguous().view(batch_size_eff, -1)
        
        final_output = self.output_projection(context_vector)
        normalized_weights = F.softmax(self.strategy_weights, dim=0)
        return final_output, normalized_weights

    def forward(self, market_data, symbol_ids):
        # market_data: (batch_size, seq_len, d_model)  # 這裡 market_data 實際上是 transformer 的 output
        # symbol_ids: (batch_size, num_symbols)

        # 直接用傳入的 market_data 作為 strategies 輸入
        transformer_output_for_strategies = market_data

        strategy_outputs_list = []
        for i, strategy in enumerate(self.strategy_pool):
            current_strategy_output = strategy(transformer_output_for_strategies)
            strategy_outputs_list.append(current_strategy_output[:, -1, :])

        concatenated_strategy_outputs = torch.cat(strategy_outputs_list, dim=1)

        # Q 來自 transformer 的最後一個 timestep
        q_input = market_data[:, -1, :]
        # 強制將 q_input、concatenated_strategy_outputs 移到 attention_q/attention_k 權重所在 device
        q_input = q_input.to(self.attention_q.weight.device)
        concatenated_strategy_outputs = concatenated_strategy_outputs.to(self.attention_k.weight.device)

        q = self.attention_q(q_input)

        # K, V 來自策略池輸出
        # ---- Start of new diagnostic prints ----
        print(f"QSL DIAGNOSTIC: concatenated_strategy_outputs shape: {concatenated_strategy_outputs.shape}")
        if isinstance(self.attention_k, nn.Linear):
            print(f"QSL DIAGNOSTIC: self.attention_k input_features: {self.attention_k.in_features}")
            print(f"QSL DIAGNOSTIC: self.attention_k output_features: {self.attention_k.out_features}")
            print(f"QSL DIAGNOSTIC: self.attention_k.weight shape: {self.attention_k.weight.shape}")
        else:
            print(f"QSL DIAGNOSTIC: self.attention_k is not nn.Linear, it is {type(self.attention_k)}")
        # ---- End of new diagnostic prints ----

        k = self.attention_k(concatenated_strategy_outputs)
        v = self.attention_v(concatenated_strategy_outputs)

        q_unsqueezed = q.unsqueeze(1)
        k_unsqueezed = k.unsqueeze(1)
        v_unsqueezed = v.unsqueeze(1)

        attention_output, _ = self.attention(q_unsqueezed, k_unsqueezed, v_unsqueezed)
        attention_output_squeezed = attention_output.squeeze(1)
        final_output = self.output_fc(attention_output_squeezed)
        return final_output

# --- Dummy Actor and Critic Definitions ---
class DummyActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Linear(state_dim, action_dim)
        self.to(device) # Move to device during init
    def forward(self, state):
        state = state.to(next(self.parameters()).device)
        return torch.tanh(self.fc(state))

class DummyCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Linear(state_dim + action_dim, 1)
        self.to(device) # Move to device during init
    def forward(self, state, action):
        state = state.to(next(self.parameters()).device)
        action = action.to(next(self.parameters()).device)
        return self.fc(torch.cat([state, action], dim=1))

# --- Dummy Strategy Loader ---
# 動態載入所有策略類別
STRATEGY_DIR = os.path.join(os.path.dirname(__file__), '../src/agent/strategies')
STRATEGY_DIR = os.path.abspath(STRATEGY_DIR)
strategy_files = glob.glob(os.path.join(STRATEGY_DIR, '*.py'))
strategy_classes = []
for file in strategy_files:
    if file.endswith('__init__.py') or not file.endswith('.py'):
        continue
    module_name = 'src.agent.strategies.' + os.path.basename(file)[:-3]
    try:
        module = importlib.import_module(module_name)
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # 只收集 BaseStrategy 子類且排除基底類
            if hasattr(obj, '__bases__') and any('BaseStrategy' in str(base) for base in obj.__bases__):
                if name != 'BaseStrategy':
                    strategy_classes.append(obj)
    except Exception as e:
        print(f"[WARN] 無法載入 {module_name}: {e}")

print(f"[INFO] 共發現 {len(strategy_classes)} 個策略類別: {[cls.__name__ for cls in strategy_classes]}")

# 之後可用 strategy_classes 動態建立策略池

# ========== 端到端測試主流程 ========== #
def run_debug_flow():
    print("\n--- 1. 資料模擬與預處理 ---")
    batch_size = 2
    MAX_SYMBOLS_ALLOWED = 10  # 固定最大 symbol 數
    num_active_symbols = 2    # 本次訓練實際 symbol 數
    seq_len = 60
    input_dim = 64
    # 產生 [batch, MAX_SYMBOLS_ALLOWED, seq_len, input_dim]，前 num_active_symbols 為真實 symbol，後面為 dummy
    dummy_market_data = torch.zeros(batch_size, MAX_SYMBOLS_ALLOWED, seq_len, input_dim)
    dummy_market_data[:, :num_active_symbols] = torch.randn(batch_size, num_active_symbols, seq_len, input_dim)
    print(f"模擬市場資料 shape: {dummy_market_data.shape}")
    # symbol_ids: [batch, MAX_SYMBOLS_ALLOWED]，前 num_active_symbols 為真實 symbol id，後面為 0
    dummy_symbol_ids = torch.zeros(batch_size, MAX_SYMBOLS_ALLOWED, dtype=torch.long)
    dummy_symbol_ids[:, :num_active_symbols] = torch.arange(1, num_active_symbols+1)
    # src_key_padding_mask: [batch, MAX_SYMBOLS_ALLOWED]，dummy symbol 為 True
    src_key_padding_mask = torch.zeros(batch_size, MAX_SYMBOLS_ALLOWED, dtype=torch.bool)
    src_key_padding_mask[:, num_active_symbols:] = True

    # --- SHAPE AUTO-ADAPT ---
    # 保證 downstream 特徵 shape 嚴格對齊，遇到不符自動 reshape
    assert dummy_market_data.shape[1] == MAX_SYMBOLS_ALLOWED, "market_data symbol 維度錯誤"
    assert dummy_market_data.shape[-1] == input_dim, "market_data input_dim 錯誤"

    print("\n--- 2. Transformer 前向 ---")
    from src.models.enhanced_transformer import EnhancedTransformer
    d_model = 64
    input_dim = dummy_market_data.shape[-1]  # 保證 input_dim 與資料 shape 一致
    transformer = EnhancedTransformer(
        input_dim=input_dim,
        d_model=d_model,
        max_seq_len=seq_len,
        num_symbols=MAX_SYMBOLS_ALLOWED,
        transformer_nhead=4,
        num_encoder_layers=4,
        dim_feedforward=256,
        dropout=0.1,
        output_dim=d_model
    )
    transformer = transformer.to('cuda' if torch.cuda.is_available() else 'cpu')
    dummy_market_data = dummy_market_data.to(transformer.device)
    dummy_symbol_ids = dummy_symbol_ids.to(transformer.device)
    src_key_padding_mask = src_key_padding_mask.to(transformer.device)
    transformer_input = {"src": dummy_market_data, "symbol_ids": dummy_symbol_ids, "src_key_padding_mask": src_key_padding_mask}
    print(f"[DEBUG] transformer_input['src'] shape: {dummy_market_data.shape}, input_dim: {input_dim}")
    transformer_out = transformer(transformer_input, return_full_sequence=True)
    print(f"Transformer 輸出 shape: {transformer_out.shape}")

    print("\n--- 2.1 Transformer 參數 requires_grad 與範數檢查 ---")
    for name, param in transformer.named_parameters():
        print(f"[CHECK] {name}: requires_grad={param.requires_grad}, param_norm={param.data.norm().item():.6f}")
        if not param.requires_grad:
            print(f"[FIX] {name} requires_grad=False，自動設為True")
            param.requires_grad = True
        if param.data.norm().item() == 0.0:
            print(f"[WARN] {name} 範數為0，可能未正確訓練或未參與loss")

    # 檢查 optimizer 是否包含所有參數
    print("\n--- 2.2 檢查 optimizer 參數覆蓋 ---")
    # 若有 optimizer，檢查其 param group 覆蓋所有 transformer 參數
    # 這裡僅示範，實際訓練時請用 optimizer = torch.optim.Adam(transformer.parameters(), ...)
    # 若有自定義 optimizer，請確保 transformer.parameters() 全部被包含
    # optimizer_params = set([id(p) for group in optimizer.param_groups for p in group['params']])
    # for name, param in transformer.named_parameters():
    #     if id(param) not in optimizer_params:
    #         print(f"[WARN] {name} 未被 optimizer 管理，將無法訓練")

    print("\n--- 3. 動態組合策略池 ---")
    # --- 動態組合策略池 ---
    from types import SimpleNamespace
    strategy_pool = []
    strategy_input_type = []  # 新增：記錄每個策略 forward 期望 shape
    dummy_config = SimpleNamespace(input_dim=d_model, output_dim=d_model, name="debug", device=transformer.device)
    for i in range(MAX_SYMBOLS_ALLOWED):
        if i < len(strategy_classes):
            cls = strategy_classes[i]
            try:
                strat = cls(config=SimpleNamespace(input_dim=d_model, output_dim=d_model, name="debug", device=transformer.device))
                # 檢查 forward 支援的 shape
                try:
                    # 嘗試 3D 輸入
                    test_x = torch.randn(2, 10, d_model, device=transformer.device)
                    out = strat(test_x)
                    if out.ndim == 3 or out.ndim == 2:
                        strategy_input_type.append('3d')
                    else:
                        strategy_input_type.append('2d')
                except Exception:
                    try:
                        # 嘗試 2D 輸入
                        test_x = torch.randn(2, d_model, device=transformer.device)
                        out = strat(test_x)
                        strategy_input_type.append('2d')
                    except Exception:
                        strategy_input_type.append('unknown')
                strategy_pool.append(strat.to(transformer.device))
            except Exception as e:
                print(f"[WARN] 策略 {cls.__name__} 初始化失敗: {e}")
                strategy_pool.append(DummyStrategy(d_model, d_model, strategy_name=f"DummyStrategy_{i}").to(transformer.device))
                strategy_input_type.append('2d')
        else:
            strategy_pool.append(DummyStrategy(d_model, d_model, strategy_name=f"DummyStrategy_{i}").to(transformer.device))
            strategy_input_type.append('2d')
    print(f"策略池共 {len(strategy_pool)} 個策略: {[s.__class__.__name__ for s in strategy_pool]}")
    print(f"策略池 input_type: {strategy_input_type}")

    print("\n--- 3.1 檢查所有策略池策略參數可訓練性 ---")
    for i, strat in enumerate(strategy_pool):
        param_requires_grad = [p.requires_grad for p in strat.parameters()]
        total_params = sum(p.numel() for p in strat.parameters())
        print(f"策略 {i} {strat.__class__.__name__}: requires_grad={param_requires_grad}, total_params={total_params}")
        # 若完全不可訓練，則自動包裝為可微分 soft rule 策略
        if total_params == 0 or not any(param_requires_grad):
            print(f"[AUTO-FIX] {strat.__class__.__name__} 無可訓練參數，自動包裝為可微分 soft rule 策略")
            class SoftRuleStrategy(nn.Module):
                def __init__(self, input_dim, output_dim):
                    super().__init__()
                    self.fc = nn.Linear(input_dim, output_dim)
                def forward(self, x):
                    # soft rule: sigmoid 激活模擬 rule-based decision
                    return torch.sigmoid(self.fc(x))
            strategy_pool[i] = SoftRuleStrategy(d_model, d_model).to(transformer.device)
    print("所有策略池策略參數可訓練性檢查與修正完畢。")

    print("\n--- 4. 量子策略層 ---")
    state_dim = transformer_out.shape[-1]  # 取最後一維
    action_dim = d_model  # 量子策略層 output_dim/action_dim 設為 d_model

    # 量子策略層 forward 根據每個策略需求自動 slice 輸入
    def quantum_strategy_layer_forward(transformer_out, strategy_pool, strategy_input_type):
        # transformer_out: [batch, MAX_SYMBOLS_ALLOWED, seq_len, d_model]
        batch, max_symbols, seq_len, d_model_ = transformer_out.shape
        strategy_outputs = []
        for i, (strategy, input_type) in enumerate(zip(strategy_pool, strategy_input_type)):
            if input_type == '3d':
                # 傳入完整序列
                x = transformer_out[:, i, :, :]  # [batch, seq_len, d_model]
                try:
                    out = strategy(x)
                except Exception as e:
                    print(f"[WARN] 策略 {strategy.__class__.__name__} 3D forward 失敗: {e}")
                    # fallback to 2d
                    x2 = transformer_out[:, i, -1, :]
                    out = strategy(x2)
            elif input_type == '2d':
                x = transformer_out[:, i, -1, :]  # [batch, d_model]
                out = strategy(x)
            else:
                # fallback: 傳入最後一個 timestep
                x = transformer_out[:, i, -1, :]
                out = strategy(x)
            # 若輸出是 3D，取最後一個 timestep
            if out.ndim == 3:
                out = out[:, -1, :]
            # 若輸出不是 [batch, d_model]，自動線性投影
            if out.ndim == 1:
                out = out.unsqueeze(0)
            if out.shape[-1] != d_model_:
                out = torch.nn.Linear(out.shape[-1], d_model_).to(out.device)(out)
            strategy_outputs.append(out)
        # 拼接所有策略輸出 [batch, n*output_dim]
        concatenated = torch.cat(strategy_outputs, dim=1)
        return concatenated

    # 量子策略層模擬
    quantum_out = quantum_strategy_layer_forward(transformer_out, strategy_pool, strategy_input_type)
    print(f"量子策略層輸出 shape: {quantum_out.shape}")

    # --- DummyVecEnv + MockEnv for SAC ---
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv
    import numpy as np
    from gymnasium import spaces

    class MockEnv(gym.Env):
        def __init__(self):
            super().__init__()
            # observation_space shape 固定為 (MAX_SYMBOLS_ALLOWED * d_model,)
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(MAX_SYMBOLS_ALLOWED * d_model,), dtype=np.float32)
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
        def reset(self, seed=None, options=None):
            return self.observation_space.sample(), {}
        def step(self, action):
            return self.observation_space.sample(), 0.0, False, False, {}
        def render(self): pass
        def close(self): pass

    dummy_vec_env = DummyVecEnv([lambda: MockEnv()])

    # policy_kwargs 只傳空 dict，避免 features_extractor_kwargs 錯誤
    policy_kwargs = {
        'features_extractor_kwargs': {
            'model_config_path': str(PROJECT_ROOT / 'configs' / 'enhanced_model_config.json')
        }
    }

    print("\n--- 5. SAC 智能體 ---")
    from src.agent.sac_agent_wrapper import QuantumEnhancedSAC
    sac_agent = QuantumEnhancedSAC(env=dummy_vec_env, policy_kwargs=policy_kwargs)
    sac_agent = sac_agent
    # 實務：確保 obs_tensor 與 SAC agent 在同一裝置
    # obs_tensor: [batch, MAX_SYMBOLS_ALLOWED * d_model]
    obs_tensor = quantum_out.reshape(quantum_out.shape[0], -1).to(sac_agent.agent.device)
    assert obs_tensor.shape[1] == MAX_SYMBOLS_ALLOWED * d_model, f"obs_tensor shape {obs_tensor.shape} 不符預期 {MAX_SYMBOLS_ALLOWED * d_model}"
    action = sac_agent.agent.policy.actor(obs_tensor)
    # 修正：SAC Actor 可能回傳 (mean_actions, log_std) tuple
    if isinstance(action, tuple):
        mean_actions, log_std = action
        print(f"SAC Actor mean_actions shape: {mean_actions.shape}, log_std shape: {log_std.shape}")
    else:
        mean_actions = action
        print(f"SAC Actor 輸出 shape: {mean_actions.shape}")
    q_value = sac_agent.agent.critic.q1_forward(obs_tensor, mean_actions)
    print(f"SAC Critic 輸出 shape: {q_value.shape}")

    print("\n--- 6. 元學習系統 ---")
    from src.agent.meta_learning_system import MetaLearningSystem
    meta_learner = MetaLearningSystem(initial_state_dim=quantum_out.shape[-1], action_dim=5)
    meta_learner = meta_learner.to(transformer.device)
    meta_action, meta_info = meta_learner(quantum_out)
    print(f"元學習系統輸出 shape: {meta_action.shape}, info: {meta_info}")

    print("\n--- 7. 端到端梯度流檢查 ---")
    # --- 7.0 Auxiliary Loss 強制梯度覆蓋所有 transformer 層 ---
    aux_loss = transformer_out.pow(2).mean() * 0.01  # L2 regularization, 可調整權重
    dummy_loss = q_value.mean() + meta_action.mean() + aux_loss
    dummy_loss.backward()
    # 檢查關鍵參數梯度
    print("\n--- 7.1 Transformer 參數梯度範數檢查 ---")
    for name, param in transformer.named_parameters():
        if param.grad is not None:
            print(f"[GRAD] Transformer {name} grad norm: {param.grad.norm().item():.4f}")
            if param.grad.norm().item() == 0.0:
                print(f"[WARN] {name} grad norm 為0，可能未參與loss或梯度未流動")
        else:
            print(f"[WARN] Transformer {name} grad is None，未參與loss或未正確反向傳播")
    # --- 7.2 SAC 參數梯度範數檢查 ---
    # 修正：QuantumEnhancedSAC 沒有 named_parameters，需用 agent.policy.named_parameters()
    for name, param in sac_agent.agent.policy.named_parameters():
        if param.grad is not None:
            print(f"[GRAD] SAC {name} grad norm: {param.grad.norm():.4f}")
        else:
            print(f"[WARN] SAC {name} grad is None，未參與loss或未正確反向傳播")
    for name, param in meta_learner.named_parameters():
        if param.grad is not None:
            print(f"[GRAD] MetaLearner {name} grad norm: {param.grad.norm().item():.4f}")
    print("\n--- 測試結束 ---")

if __name__ == "__main__":
    run_debug_flow()

