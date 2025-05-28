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
from datetime import datetime, timezone # Added for timestamp

# --- Simplified Import Block ---
try:
    from src.common.logger_setup import logger
    logger.debug("callbacks.py (V5): Successfully imported logger from src.common.logger_setup.")
except ImportError:
    logger = logging.getLogger("callbacks_v5_fallback") # type: ignore
    logger.setLevel(logging.DEBUG)
    _ch_fallback_cb = logging.StreamHandler(sys.stdout)
    _ch_fallback_cb.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    if not logger.handlers: logger.addHandler(_ch_fallback_cb)
    logger.warning("callbacks.py (V5): Failed to import logger from src.common.logger_setup. Using fallback logger.")

_config_cb_v5: Dict[str, Any] = {}
try:
    from src.common.config import LOGS_DIR as _LOGS_DIR, ACCOUNT_CURRENCY as _ACCOUNT_CURRENCY, INITIAL_CAPITAL as _INITIAL_CAPITAL
    _config_cb_v5 = {"LOGS_DIR": _LOGS_DIR, "ACCOUNT_CURRENCY": _ACCOUNT_CURRENCY, "INITIAL_CAPITAL": _INITIAL_CAPITAL}
    logger.info("callbacks.py (V5): Successfully imported common.config.") # type: ignore
except ImportError as e:
    logger.error(f"callbacks.py (V5): Failed to import common.config: {e}. Using fallback values.", exc_info=True) # type: ignore
    _config_cb_v5 = {"LOGS_DIR": Path("./logs"), "ACCOUNT_CURRENCY": "AUD", "INITIAL_CAPITAL": 100000.0}
    logger.warning("callbacks.py (V5): Using fallback values for config due to import error.") # type: ignore

LOGS_DIR = _config_cb_v5.get("LOGS_DIR", Path("./logs"))
ACCOUNT_CURRENCY = _config_cb_v5.get("ACCOUNT_CURRENCY", "AUD")
INITIAL_CAPITAL = _config_cb_v5.get("INITIAL_CAPITAL", 100000.0) # Added INITIAL_CAPITAL


class UniversalCheckpointCallback(BaseCallback):
    def __init__(self,
                 save_freq: int,
                 save_path: Union[str, Path],
                 name_prefix: str = "sac_model",
                 eval_env: Optional[VecEnv] = None,
                 eval_freq: int = 10000,
                 n_eval_episodes: int = 3,
                 deterministic_eval: bool = True,
                 best_model_save_path: Optional[Union[str, Path]] = None,
                 early_stopping_patience: int = 10,
                 early_stopping_min_delta_abs: float = 100.0,
                 early_stopping_metric: str = "eval/mean_final_portfolio_value",
                 early_stopping_min_evals: int = 10,
                 log_transformer_norm_freq: int = 1000,
                 verbose: int = 1,
                 streamlit_session_state=None,
                 shared_data_manager=None): # Added shared_data_manager
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

        self.best_metric_val = -np.inf
        self.last_eval_trigger_step = 0

        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta_abs = early_stopping_min_delta_abs
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_min_evals = early_stopping_min_evals
        self.es_eval_count = 0
        self.es_no_improvement_count = 0
        self.es_best_metric_val = -np.inf

        self.log_transformer_norm_freq = log_transformer_norm_freq
        self.last_norm_log_ncalls = 0

        self.interrupted = False
        self.streamlit_session_state = streamlit_session_state
        self.shared_data_manager = shared_data_manager # Store shared_data_manager
        
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

    def _run_evaluation(self) -> float:
        if self.eval_env is None:
            logger.warning("未提供評估環境，跳過評估。")
            return -np.inf

        logger.info(f"開始評估 (n_calls={self.n_calls}, num_timesteps={self.num_timesteps})...")
        all_episode_rewards: List[float] = []
        all_episode_lengths: List[int] = []
        all_final_portfolio_values_ac: List[float] = []

        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            dones = np.array([False] * self.eval_env.num_envs)
            episode_reward_sum = np.zeros(self.eval_env.num_envs)
            episode_length = 0
            
            max_steps = getattr(self.eval_env.envs[0], "max_episode_steps", 1000)

            for step in range(max_steps):
                action, _ = self.model.predict(obs, deterministic=self.deterministic_eval)
                obs, rewards, dones, infos = self.eval_env.step(action)
                episode_reward_sum += rewards
                episode_length += 1
                if np.any(dones):
                    for i_env in range(self.eval_env.num_envs):
                        if dones[i_env]:
                            all_episode_rewards.append(episode_reward_sum[i_env])
                            all_episode_lengths.append(episode_length)
                            if infos[i_env] and "portfolio_value_ac" in infos[i_env]:
                                all_final_portfolio_values_ac.append(infos[i_env]["portfolio_value_ac"])
                    break 
            else: 
                 all_episode_infos_from_eval_no_done = infos # type: ignore
                 for i_env in range(self.eval_env.num_envs):
                     all_episode_rewards.append(episode_reward_sum[i_env])
                     all_episode_lengths.append(episode_length)
                     if all_episode_infos_from_eval_no_done[i_env] and "portfolio_value_ac" in all_episode_infos_from_eval_no_done[i_env]:
                         all_final_portfolio_values_ac.append(all_episode_infos_from_eval_no_done[i_env]["portfolio_value_ac"])

        mean_reward = np.mean(all_episode_rewards) if all_episode_rewards else -np.inf
        mean_portfolio_value = np.mean(all_final_portfolio_values_ac) if all_final_portfolio_values_ac else -np.inf
        
        logger.info(f"評估完成: 平均獎勵={mean_reward:.3f}, 平均最終淨值={mean_portfolio_value:.2f} {ACCOUNT_CURRENCY}")
        
        if self.logger is not None:
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/mean_episode_length", np.mean(all_episode_lengths) if all_episode_lengths else 0)
            self.logger.record("eval/mean_final_portfolio_value", mean_portfolio_value)
            self.logger.dump(step=self.num_timesteps)
        
        if self.streamlit_session_state is not None and hasattr(self.streamlit_session_state, 'training_metrics'):
            metrics = self.streamlit_session_state.training_metrics
            if len(metrics['steps']) > 0 and metrics['steps'][-1] == self.num_timesteps:
                if len(metrics['rewards']) == len(metrics['steps']):
                    metrics['rewards'][-1] = mean_reward
                else:
                    metrics['rewards'].append(mean_reward)
                if len(metrics['portfolio_values']) == len(metrics['steps']):
                    metrics['portfolio_values'][-1] = mean_portfolio_value
                else:
                    metrics['portfolio_values'].append(mean_portfolio_value)

        if self.early_stopping_metric == "eval/mean_reward":
            return float(mean_reward)
        return float(mean_portfolio_value)

    def _on_step(self) -> bool:
        if self.verbose > 0 and self.n_calls % 50 == 0: # Log every 50 calls
            print(f"Callback Checkpoint: n_calls={self.n_calls}, num_timesteps={self.num_timesteps}, save_freq={self.save_freq}")

        # Check for stop request
        if self.shared_data_manager and self.shared_data_manager.is_stop_requested():
            if self.verbose > 0:
                print(f"Stop requested at step {self.num_timesteps}. Stopping training and saving final model.")
            # Save the model before stopping
            final_model_path = os.path.join(self.save_path, f"{self.name_prefix}_final_step_{self.num_timesteps}.zip")
            self.model.save(final_model_path)
            if self.verbose > 0:
                print(f"Saved final model to {final_model_path}")
            self.shared_data_manager.update_training_status("Training stopped by user. Final model saved.")
            self.shared_data_manager.reset_stop_request() # Reset the stop request
            self.interrupted = True # Mark as interrupted
            return False # Stop training

        current_time = datetime.now(timezone.utc)
        # Update shared data manager with current training metrics
        if self.shared_data_manager is not None:
            step_count = self.num_timesteps

            # Safely get metrics from SB3 logger; provide defaults if not found
            actor_loss = self.model.logger.name_to_value.get('train/actor_loss', 0.0)
            critic_loss = self.model.logger.name_to_value.get('train/critic_loss', 0.0)
            current_step_reward = 0.0
            # self.locals often contains 'rewards' for the last step
            if 'rewards' in self.locals and isinstance(self.locals['rewards'], np.ndarray) and self.locals['rewards'].size > 0:
                current_step_reward = self.locals['rewards'][0] 

            portfolio_value = INITIAL_CAPITAL # Default to initial capital
            # self.locals 'infos' is a list of dicts, one per env
            if 'infos' in self.locals and isinstance(self.locals['infos'], list) and len(self.locals['infos']) > 0:
                info_dict = self.locals['infos'][0] # Assuming single env or primary env's info
                portfolio_value = info_dict.get('portfolio_value', info_dict.get('portfolio_value_ac', INITIAL_CAPITAL))
            
            l2_norm_val = self.model.logger.name_to_value.get('train/transformer_l2_norm', 0.0) # Will be updated later if transformer exists
            grad_norm_val = self.model.logger.name_to_value.get('train/gradient_norm', self.model.logger.name_to_value.get('actor_grad_norm', 0.0))

            self.shared_data_manager.add_training_metric(
                step=step_count,
                reward=current_step_reward,
                portfolio_value=portfolio_value,
                actor_loss=actor_loss,
                critic_loss=critic_loss,
                l2_norm=l2_norm_val, 
                grad_norm=grad_norm_val
            )
            
            # Update training progress for UI
            # self.model._total_timesteps is the total_timesteps passed to learn()
            if hasattr(self.model, '_total_timesteps') and self.model._total_timesteps > 0:
                progress = (step_count / self.model._total_timesteps) * 100
                self.shared_data_manager.update_training_status('running', progress)

        # 1. 定期保存 - Use self.num_timesteps for the check
        if self.num_timesteps > 0 and self.num_timesteps % self.save_freq == 0:
            intermediate_path = self.save_path / f"{self.name_prefix}.zip"
            self.model.save(intermediate_path)
            logger.info(f"定期保存模型到: {intermediate_path} (步數: {self.num_timesteps})")

        # 2. 定期評估
        if self.eval_env is not None and self.n_calls > 0 and self.n_calls % self.eval_freq == 0:
            current_metric = self._run_evaluation()
            self.es_eval_count +=1

            if current_metric > self.best_metric_val:
                logger.info(f"新最佳評估指標: {current_metric:.3f} (舊: {self.best_metric_val:.3f})")
                self.best_metric_val = current_metric
                best_model_path = self.best_model_save_path / f"{self.name_prefix}_best.zip"
                self.model.save(best_model_path)
                logger.info(f"最佳模型已更新並保存到: {best_model_path}")
            
            if self.es_eval_count >= self.early_stopping_min_evals:
                if current_metric < self.es_best_metric_val + self.early_stopping_min_delta_abs:
                    self.es_no_improvement_count += 1
                    logger.info(f"早停檢查: 指標 ({current_metric:.3f}) 未比最佳 ({self.es_best_metric_val:.3f}) 改善至少 {self.early_stopping_min_delta_abs}. 連續未改善: {self.es_no_improvement_count}/{self.early_stopping_patience}")
                else:
                    logger.info(f"早停檢查: 指標改善 ({current_metric:.3f}). 重置未改善計數。")
                    self.es_best_metric_val = current_metric
                    self.es_no_improvement_count = 0
                
                if self.es_no_improvement_count >= self.early_stopping_patience:
                    logger.warning("早停觸發！訓練將停止。")
                    self.interrupted = True
                    return False 
            elif current_metric > self.es_best_metric_val :
                 self.es_best_metric_val = current_metric

        # 3. 記錄Transformer範數
        if self.n_calls > 0 and self.n_calls % self.log_transformer_norm_freq == 0:
            # Corrected line continuation and attribute access
            if hasattr(self.model.policy, 'features_extractor') and \
               self.model.policy.features_extractor is not None and \
               hasattr(self.model.policy.features_extractor, 'transformer'):
                try:
                    transformer_module = getattr(self.model.policy.features_extractor, 'transformer', None)
                    if transformer_module is not None:
                        transformer_params = transformer_module.parameters()
                        l2_norm = sum(p.data.norm(2).item() ** 2 for p in transformer_params if p.requires_grad) ** 0.5
                        self.logger.record("train/transformer_l2_norm", l2_norm) # Log to SB3 logger
                        logger.debug(f"Transformer L2 Norm @{self.num_timesteps}: {l2_norm:.4f}")
                        
                        # If shared_data_manager is available and has metrics_data, update the l2_norm of the last entry
                        if self.shared_data_manager and self.shared_data_manager.metrics_data:
                            # This is a simplified update; ideally, metrics are structured to accommodate this.
                            # For now, we assume the main add_training_metric has a placeholder or this is logged separately for UI.
                            # To directly update the last metric:
                            # self.shared_data_manager.metrics_data[-1]['l2_norm'] = l2_norm
                            pass # Placeholder for more sophisticated update if needed
                                    
                except Exception as e_norm: 
                    logger.warning(f"計算Transformer範數出錯: {e_norm}")
        
        # Check for stop request from shared_data_manager (e.g., Streamlit button)
        if self.shared_data_manager is not None and self.shared_data_manager.is_stop_requested():
            logger.info("Stop requested via shared data manager, stopping training...")
            self.interrupted = True 

        if self.interrupted:
            logger.info("檢測到中斷請求，準備停止訓練...")
            return False 
        return True

    def _on_training_end(self) -> None:
        logger.info(f"訓練結束 (UniversalCheckpointCallback)。總步數: {self.num_timesteps}, 總調用次數: {self.n_calls}")
        if self.model is not None:
            final_path = self.save_path / f"{self.name_prefix}.zip" # Standard final save
            self.model.save(final_path)
            logger.info(f"最終模型已保存到: {final_path}")

            # If training was interrupted, also consider saving a specific "interrupted" model
            # or ensure the current `trainer.save_current_model()` in streamlit_app_complete.py handles this.
            # The current logic in streamlit_app_complete.py calls trainer.save_current_model() which saves a "_checkpoint.zip"
            # This _on_training_end save will be the "final" state, even if interrupted.