# src/trainer/callbacks.py
"""
自定義 Stable Baselines3 回調函數 (重寫版本 V5)。
用於模型保存、評估、早停、TensorBoard日誌記錄等。
"""
import os
import gym
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv # VecEnv 用於類型提示
from stable_baselines3.common.base_class import BaseAlgorithm # 用於類型提示 model
from typing import Optional, Any, Dict, Union, List, Tuple # Tuple for predict
from pathlib import Path
import time
import sys
import logging
import shutil # 用於 __main__ 中的清理
from gymnasium import spaces # 用於 __main__ 測試

# --- Logger 和 Config 導入 ---
# (與之前 V4.8 版本類似的導入邏輯，確保 logger 和必要config可用)
_logger_cb_v5: logging.Logger
_config_cb_v5: Dict[str, Any] = {}
try:
    from common.logger_setup import logger as common_logger_cb_v5; _logger_cb_v5 = common_logger_cb_v5; logger = _logger_cb_v5
    logger.debug("callbacks.py (V5): Successfully imported logger from common.logger_setup.")
    from common.config import WEIGHTS_DIR as _WEIGHTS_DIR, ACCOUNT_CURRENCY as _ACCOUNT_CURRENCY
    _config_cb_v5 = {"WEIGHTS_DIR": _WEIGHTS_DIR, "ACCOUNT_CURRENCY": _ACCOUNT_CURRENCY}
    logger.info("callbacks.py (V5): Successfully imported common.config.")
except ImportError:
    logger_temp_cb_v5 = logging.getLogger("callbacks_v5_fallback_initial"); logger_temp_cb_v5.addHandler(logging.StreamHandler(sys.stdout)); logger_temp_cb_v5.setLevel(logging.DEBUG)
    _logger_cb_v5 = logger_temp_cb_v5; logger = _logger_cb_v5
    logger.warning(f"callbacks.py (V5): Initial import failed. Attempting path adjustment...")
    project_root_cb_v5 = Path(__file__).resolve().parent.parent.parent
    if str(project_root_cb_v5) not in sys.path: sys.path.insert(0, str(project_root_cb_v5)); logger.info(f"callbacks.py (V5): Added project root to sys.path: {project_root_cb_v5}")
    try:
        from src.common.logger_setup import logger as common_logger_retry_cb_v5; logger = common_logger_retry_cb_v5
        logger.info("callbacks.py (V5): Successfully re-imported common_logger after path adj.")
        from src.common.config import WEIGHTS_DIR as _WEIGHTS_DIR_R, ACCOUNT_CURRENCY as _ACCOUNT_CURRENCY_R
        _config_cb_v5 = {"WEIGHTS_DIR": _WEIGHTS_DIR_R, "ACCOUNT_CURRENCY": _ACCOUNT_CURRENCY_R}
        logger.info("callbacks.py (V5): Successfully re-imported common.config after path adjustment.")
    except ImportError as e_retry_cb_v5_critical:
        logger.error(f"callbacks.py (V5): Critical import error after path adjustment: {e_retry_cb_v5_critical}", exc_info=True)
        logger.warning("callbacks.py (V5): Using fallback values for config.")
        _config_cb_v5 = {"WEIGHTS_DIR": Path("./weights_fallback"), "ACCOUNT_CURRENCY": "AUD"}
WEIGHTS_DIR = _config_cb_v5.get("WEIGHTS_DIR", Path("./weights_fallback_default"))
ACCOUNT_CURRENCY = _config_cb_v5.get("ACCOUNT_CURRENCY", "AUD")


class UniversalCheckpointCallback(BaseCallback):
    def __init__(self,
                 save_freq: int,
                 save_path: Union[str, Path],
                 name_prefix: str = "sac_model",
                 eval_env: Optional[VecEnv] = None,
                 eval_freq: int = 10000,
                 n_eval_episodes: int = 3, # 增加默認評估episodes
                 deterministic_eval: bool = True,
                 best_model_save_path: Optional[Union[str, Path]] = None,
                 early_stopping_patience: int = 10,
                 early_stopping_min_delta_abs: float = 100.0, # 絕對值改善，例如淨值至少增加100 AUD
                 early_stopping_metric: str = "eval/mean_final_portfolio_value", # 用於早停的指標
                 early_stopping_min_evals: int = 10, # 減少最小評估次數以便更快觸發測試
                 log_transformer_norm_freq: int = 1000,
                 verbose: int = 1,
                 streamlit_session_state=None):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic_eval = deterministic_eval
        
        self.save_path.mkdir(parents=True, exist_ok=True)
        if best_model_save_path is None:
            self.best_model_save_path = self.save_path / "best_model"
        else:
            self.best_model_save_path = Path(best_model_save_path)
        self.best_model_save_path.parent.mkdir(parents=True, exist_ok=True)

        self.best_metric_val = -np.inf # 用於最佳模型保存
        self.last_eval_trigger_step = 0 # 用於控制評估頻率的n_calls

        # 早停相關
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta_abs = early_stopping_min_delta_abs # 使用絕對值
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_min_evals = early_stopping_min_evals
        self.es_eval_count = 0 # 早停的評估計數器
        self.es_no_improvement_count = 0
        self.es_best_metric_val = -np.inf

        self.log_transformer_norm_freq = log_transformer_norm_freq
        self.last_norm_log_ncalls = 0

        self.interrupted = False
        self.streamlit_session_state = streamlit_session_state  # 用於更新Streamlit UI
        
        # 嘗試設置signal處理器，但在非主線程中會失敗
        try:
            import signal
            import threading
            if threading.current_thread() is threading.main_thread():
                signal.signal(signal.SIGINT, self._handle_interrupt)
                logger.info("Signal handler registered for Ctrl+C interruption.")
            else:
                logger.info("Running in non-main thread, signal handler not registered.")
        except Exception as e:
            logger.warning(f"Could not register signal handler: {e}")
        
        logger.info(f"UniversalCheckpointCallback initialized. Save path: {self.save_path}, Best model path: {self.best_model_save_path}")

    def _handle_interrupt(self, signum, frame):
        logger.warning("接收到中斷信號 (Ctrl+C)！將在當前步驟結束後嘗試安全保存並退出。")
        self.interrupted = True

    def _on_training_start(self) -> None:
        logger.info("訓練開始 (UniversalCheckpointCallback)。")
        # 可以在這裡加載已有的最佳指標值，如果需要斷點續比較的話
        # 例如，從一個文件中讀取 self.best_metric_val 和 self.es_best_metric_val

    def _run_evaluation(self) -> float:
        """在評估環境中運行評估並返回主要的評估指標。"""
        if self.eval_env is None:
            logger.warning("未提供評估環境，跳過評估。")
            return -np.inf # 或其他合適的默認值

        logger.info(f"開始評估 (n_calls={self.n_calls}, num_timesteps={self.num_timesteps})...")
        all_episode_rewards: List[float] = []
        all_episode_lengths: List[int] = []
        all_final_portfolio_values_ac: List[float] = []

        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            dones = np.array([False] * self.eval_env.num_envs) # 支持 VecEnv
            episode_reward_sum = np.zeros(self.eval_env.num_envs)
            episode_length = 0
            
            max_steps = getattr(self.eval_env.envs[0], "max_episode_steps", 1000) # 從環境獲取或默認

            for step in range(max_steps):
                action, _ = self.model.predict(obs, deterministic=self.deterministic_eval)
                obs, rewards, dones, infos = self.eval_env.step(action)
                episode_reward_sum += rewards
                episode_length += 1
                if np.any(dones): # 如果VecEnv中的任何一個環境結束了
                    for i_env in range(self.eval_env.num_envs):
                        if dones[i_env]: # 記錄結束的環境信息
                            all_episode_rewards.append(episode_reward_sum[i_env])
                            all_episode_lengths.append(episode_length) # 這裡的長度是第一個done的，如果多env可以分別記
                            if infos[i_env] and "portfolio_value_ac" in infos[i_env]:
                                all_final_portfolio_values_ac.append(infos[i_env]["portfolio_value_ac"])
                    # 為了簡化，如果任何一個env done，我們就結束這個eval episode
                    break 
            else: # for循環正常結束 (沒有break，即沒達到done)
                 all_episode_infos_from_eval_no_done = infos # type: ignore
                 for i_env in range(self.eval_env.num_envs):
                     all_episode_rewards.append(episode_reward_sum[i_env])
                     all_episode_lengths.append(episode_length)
                     if all_episode_infos_from_eval_no_done[i_env] and "portfolio_value_ac" in all_episode_infos_from_eval_no_done[i_env]:
                         all_final_portfolio_values_ac.append(all_episode_infos_from_eval_no_done[i_env]["portfolio_value_ac"])


        mean_reward = np.mean(all_episode_rewards) if all_episode_rewards else -np.inf
        mean_portfolio_value = np.mean(all_final_portfolio_values_ac) if all_final_portfolio_values_ac else -np.inf
        
        logger.info(f"評估完成: 平均獎勵={mean_reward:.3f}, 平均最終淨值={mean_portfolio_value:.2f} {ACCOUNT_CURRENCY}")
        
        if self.logger is not None: # SB3 logger
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/mean_episode_length", np.mean(all_episode_lengths) if all_episode_lengths else 0)
            self.logger.record("eval/mean_final_portfolio_value", mean_portfolio_value)
            self.logger.dump(step=self.num_timesteps) # 確保寫入TensorBoard
        
        # 更新到Streamlit session state
        if self.streamlit_session_state is not None and hasattr(self.streamlit_session_state, 'training_metrics'):
            metrics = self.streamlit_session_state.training_metrics
            # 確保有對應的步數記錄
            if len(metrics['steps']) > 0 and metrics['steps'][-1] == self.num_timesteps:
                # 更新最後一個記錄的獎勵和投資組合價值
                if len(metrics['rewards']) == len(metrics['steps']):
                    metrics['rewards'][-1] = mean_reward
                else:
                    metrics['rewards'].append(mean_reward)
                
                if len(metrics['portfolio_values']) == len(metrics['steps']):
                    metrics['portfolio_values'][-1] = mean_portfolio_value
                else:
                    metrics['portfolio_values'].append(mean_portfolio_value)

        # 返回用於早停和最佳模型判斷的指標
        if self.early_stopping_metric == "eval/mean_reward":
            return float(mean_reward)
        return float(mean_portfolio_value) # 默認使用淨值

    def _on_step(self) -> bool:
        if self.model is None or self.logger is None: logger.error("Callback model/logger未初始化"); return False

        # 1. 定期保存 - 使用統一的命名策略，覆蓋同一個檢查點文件
        if self.n_calls > 0 and self.n_calls % self.save_freq == 0:
            # 使用模型標識符作為檢查點名稱，每次覆蓋
            intermediate_path = self.save_path / f"{self.name_prefix}.zip"
            self.model.save(intermediate_path)
            logger.info(f"定期保存模型到: {intermediate_path} (步數: {self.num_timesteps})")

        # 2. 定期評估
        if self.eval_env is not None and self.n_calls > 0 and self.n_calls % self.eval_freq == 0:
            current_metric = self._run_evaluation()
            self.es_eval_count +=1

            # 保存最佳模型 - 使用統一命名策略
            if current_metric > self.best_metric_val:
                logger.info(f"新最佳評估指標: {current_metric:.3f} (舊: {self.best_metric_val:.3f})")
                self.best_metric_val = current_metric
                # 最佳模型也使用相同的基礎名稱，只是保存在不同位置
                best_model_path = self.best_model_save_path / f"{self.name_prefix}_best.zip"
                self.model.save(best_model_path)
                logger.info(f"最佳模型已更新並保存到: {best_model_path}")
            
            # 早停邏輯
            if self.es_eval_count >= self.early_stopping_min_evals:
                if current_metric < self.es_best_metric_val + self.early_stopping_min_delta_abs:
                    self.es_no_improvement_count += 1
                    logger.info(f"早停檢查: 指標 ({current_metric:.3f}) 未比最佳 ({self.es_best_metric_val:.3f}) 改善至少 {self.early_stopping_min_delta_abs}. 連續未改善: {self.es_no_improvement_count}/{self.early_stopping_patience}")
                else:
                    logger.info(f"早停檢查: 指標改善 ({current_metric:.3f}). 重置未改善計數。")
                    self.es_best_metric_val = current_metric # 更新早停比較的最佳值
                    self.es_no_improvement_count = 0
                
                if self.es_no_improvement_count >= self.early_stopping_patience:
                    logger.warning("早停觸發！訓練將停止。")
                    self.interrupted = True # 觸發中斷保存邏輯
                    return False # 停止訓練
            elif current_metric > self.es_best_metric_val : # 即使未達到最小評估次數，也更新最佳指標
                 self.es_best_metric_val = current_metric


        # 3. 記錄Transformer範數
        if self.n_calls > 0 and self.n_calls % self.log_transformer_norm_freq == 0:
            if hasattr(self.model.policy, 'features_extractor') and \
               self.model.policy.features_extractor is not None and \
               hasattr(self.model.policy.features_extractor, 'transformer'):
                try:
                    transformer_params = self.model.policy.features_extractor.transformer.parameters() # type: ignore
                    l2_norm = sum(p.data.norm(2).item() ** 2 for p in transformer_params if p.requires_grad) ** 0.5
                    self.logger.record("train/transformer_l2_norm", l2_norm)
                    logger.debug(f"Transformer L2 Norm @{self.num_timesteps}: {l2_norm:.4f}")
                    
                    # 更新範數到Streamlit
                    if self.streamlit_session_state is not None and hasattr(self.streamlit_session_state, 'training_metrics'):
                        metrics = self.streamlit_session_state.training_metrics
                        if 'norms' in metrics:
                            # 找到對應的步數索引
                            if len(metrics['steps']) > 0 and metrics['steps'][-1] >= self.num_timesteps - self.log_transformer_norm_freq:
                                if len(metrics['norms']) < len(metrics['steps']):
                                    metrics['norms'].append({'l2_norm': l2_norm})
                                else:
                                    # 更新最後一個記錄
                                    if len(metrics['norms']) > 0:
                                        metrics['norms'][-1]['l2_norm'] = l2_norm
                                    
                except Exception as e_norm: logger.warning(f"計算Transformer範數出錯: {e_norm}")
        
        if self.interrupted:
            logger.info("檢測到中斷請求，準備停止訓練...")
            return False # 告訴SB3停止訓練
        return True

    def _on_training_end(self) -> None:
        logger.info(f"訓練結束 (UniversalCheckpointCallback)。總步數: {self.num_timesteps}, 總調用次數: {self.n_calls}")
        # 保存最終模型，覆蓋同一個文件
        if self.model is not None:
            final_path = self.save_path / f"{self.name_prefix}.zip"
            self.model.save(final_path)
            logger.info(f"最終模型已保存到: {final_path}")


if __name__ == "__main__":
    logger.info("正在直接運行 callbacks.py 進行測試...")
    
    class MockAlgo(BaseAlgorithm): # 需要實現抽象方法
        def __init__(self, policy, env, learning_rate=0.0003, verbose=0):
            super().__init__(policy=policy, env=env, learning_rate=learning_rate, verbose=verbose)
            self.num_timesteps = 0; self.logger = None; self.policy = policy # policy可以是字符串或類
            if env: self.observation_space = env.observation_space; self.action_space = env.action_space
        def _setup_model(self): pass
        def learn(self, total_timesteps, callback=None, **kwargs):
            if callback: callback.on_training_start(locals(), globals())
            for i in range(1, total_timesteps + 1):
                self.num_timesteps = self.num_timesteps + 1 # SB3內部會這樣做
                if callback:
                    callback.n_calls = self.num_timesteps # 確保n_calls被更新
                    if not callback.on_step(): break
            if callback: callback.on_training_end()
        def predict(self, observation, state=None, episode_start=None, deterministic=False): return self.action_space.sample(), None
        def save(self, path, exclude=None, include=None): logger.info(f"MockAlgo: save() to {path}")
        @classmethod
        def load(cls, path, env=None, device='auto', **kwargs): return cls(policy="MlpPolicy", env=env, learning_rate=0.0003) # type: ignore

    class MockSingleEnv(gym.Env[np.ndarray, np.ndarray]):
        def __init__(self, obs_s, act_s, max_steps=50):
            super().__init__(); self.observation_space = obs_s; self.action_space = act_s
            self.current_step = 0; self.max_episode_steps = max_steps
        def reset(self, seed=None, options=None): super().reset(seed=seed); self.current_step=0; return self.observation_space.sample().astype(np.float32), {}
        def step(self, action):
            self.current_step += 1; obs = self.observation_space.sample().astype(np.float32)
            reward = float(np.random.rand()); terminated = self.current_step >= self.max_episode_steps
            info = {"portfolio_value_ac": 10000 + self.current_step * 10}; return obs, reward, terminated, False, info
        def render(self): pass;
        def close(self): pass

    from stable_baselines3.common.vec_env import DummyVecEnv
    obs_sp = spaces.Box(-1, 1, (10,), np.float32); act_sp = spaces.Box(-1, 1, (2,), np.float32)
    eval_env_main = DummyVecEnv([lambda: MockSingleEnv(obs_sp, act_sp)])
    
    # 創建一個假的logger給MockAlgo
    from stable_baselines3.common.logger import Logger, HumanOutputFormat
    import tempfile
    temp_log_folder = tempfile.mkdtemp()
    mock_sb3_logger = Logger(folder=temp_log_folder, output_formats=[HumanOutputFormat(sys.stdout)])

    model_to_train = MockAlgo(policy="MlpPolicy", env=eval_env_main, learning_rate=lambda x: 0.001)
    model_to_train.logger = mock_sb3_logger # 賦值logger

    cb = UniversalCheckpointCallback(save_freq=20, save_path="./cb_test_saves", eval_env=eval_env_main, eval_freq=15, n_eval_episodes=1, early_stopping_min_evals=1, early_stopping_patience=2, log_transformer_norm_freq=10)
    
    logger.info("開始模擬訓練循環以測試回調...")
    try:
        model_to_train.learn(total_timesteps=100, callback=cb)
    except Exception as e_learn:
        logger.error(f"模擬訓練 learn() 出錯: {e_learn}", exc_info=True)

    logger.info("callbacks.py 測試執行完畢。")
    if Path("./cb_test_saves").exists(): shutil.rmtree("./cb_test_saves"); logger.info("已清理測試保存目錄。")
    if Path(temp_log_folder).exists(): shutil.rmtree(temp_log_folder); logger.info("已清理臨時日誌目錄。")