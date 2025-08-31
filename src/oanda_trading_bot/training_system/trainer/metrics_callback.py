import os
import json
import time
from collections import deque
from typing import Optional, Dict, Any

import torch
from stable_baselines3.common.callbacks import BaseCallback


class SACMetricsCallback(BaseCallback):
    """
    Collects runtime metrics during training:
    - GPU VRAM/utilization (NVML if available, else torch.cuda stats)
    - Step time (ms/step over a sliding window)
    - Reward stability (mean/std over recent episodes)
    - Policy entropy proxy (actor log_std mean) and ent_coef (alpha)

    Writes JSON lines to `log_path` at a fixed step interval.
    """

    def __init__(
        self,
        log_path: str,
        log_interval_steps: int = 200,
        recent_ep_window: int = 20,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.log_path = log_path
        self.log_interval_steps = max(1, int(log_interval_steps))
        self.recent_ep_window = max(1, int(recent_ep_window))
        self.extra_metadata = extra_metadata or {}

        self._last_log_step = 0
        self._last_time = None
        self._step_time_window = deque(maxlen=512)

        # Episode tracking for vectorized envs
        self._ep_returns = None  # type: ignore
        self._recent_ep_returns = deque(maxlen=self.recent_ep_window)

        # NVML state
        self._nvml = None
        self._nvml_handle = None

        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _on_training_start(self) -> None:
        # Initialize per-env episode returns
        n_envs = getattr(self.training_env, 'num_envs', 1)
        self._ep_returns = [0.0 for _ in range(n_envs)]
        self._last_time = time.perf_counter()
        self._try_init_nvml()
        self._write_json({
            'event': 'training_start',
            'timestamp': time.time(),
            'metadata': self.extra_metadata,
        })

    def _on_step(self) -> bool:
        # Track time/step
        now = time.perf_counter()
        if self._last_time is not None:
            dt = now - self._last_time
            # steps may be vectorized; assume one env step per callback
            self._step_time_window.append(dt)
        self._last_time = now

        # Track rewards and episode ends (vectorized env compatible)
        try:
            rewards = self.locals.get('rewards')
            dones = self.locals.get('dones')
        except Exception:
            rewards, dones = None, None

        if rewards is not None:
            # If vectorized, rewards is ndarray of shape (n_envs,)
            if hasattr(rewards, '__len__'):
                for i, r in enumerate(rewards):
                    self._ep_returns[i] += float(r)
            else:
                self._ep_returns[0] += float(rewards)

        if dones is not None:
            if hasattr(dones, '__len__'):
                for i, d in enumerate(dones):
                    if d:
                        self._recent_ep_returns.append(self._ep_returns[i])
                        self._ep_returns[i] = 0.0
            else:
                if dones:
                    self._recent_ep_returns.append(self._ep_returns[0])
                    self._ep_returns[0] = 0.0

        # Periodic logging
        if (self.num_timesteps - self._last_log_step) >= self.log_interval_steps:
            self._last_log_step = self.num_timesteps
            self._emit_metrics()

        return True

    def _emit_metrics(self) -> None:
        # GPU stats (NVML preferred)
        nvml_util = None
        nvml_mem_used = None
        if self._nvml and self._nvml_handle is not None:
            try:
                util = self._nvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                meminfo = self._nvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                nvml_util = int(util.gpu)
                nvml_mem_used = int(meminfo.used // (1024 * 1024))
            except Exception:
                pass

        torch_cuda = torch.cuda.is_available()
        torch_mem_alloc = int(torch.cuda.memory_allocated() // (1024 * 1024)) if torch_cuda else None
        torch_mem_reserved = int(torch.cuda.memory_reserved() // (1024 * 1024)) if torch_cuda else None
        torch_mem_max = int(torch.cuda.max_memory_allocated() // (1024 * 1024)) if torch_cuda else None

        # Step time avg (ms)
        if self._step_time_window:
            avg_step_s = sum(self._step_time_window) / len(self._step_time_window)
            avg_step_ms = avg_step_s * 1000.0
        else:
            avg_step_ms = None

        # Policy entropy proxy: mean log_std of actor (if available)
        log_std_mean = None
        try:
            actor = getattr(self.model.policy, 'actor', None)
            if actor is not None:
                log_std = getattr(actor, 'log_std', None)
                if log_std is not None:
                    with torch.no_grad():
                        log_std_mean = float(log_std.mean().cpu().item())
        except Exception:
            pass

        # Entropy coefficient alpha (if present)
        ent_coef = None
        try:
            # In SB3 SAC, log_ent_coef exists when ent_coef='auto'
            lec = getattr(self.model, 'log_ent_coef', None)
            if lec is not None:
                with torch.no_grad():
                    ent_coef = float(lec.exp().cpu().item())
        except Exception:
            pass

        # Optimizer LR (actor)
        lr = None
        try:
            actor = getattr(self.model.policy, 'actor', None)
            if actor is not None and hasattr(actor, 'optimizer') and actor.optimizer is not None:
                lr = float(actor.optimizer.param_groups[0].get('lr', 0.0))
        except Exception:
            pass

        # Reward stats over recent episodes
        rew_mean = None
        rew_std = None
        if len(self._recent_ep_returns) > 0:
            vals = list(self._recent_ep_returns)
            m = sum(vals) / len(vals)
            v = sum((x - m) ** 2 for x in vals) / max(1, len(vals) - 1)
            rew_mean, rew_std = float(m), float(v ** 0.5)

        rec = {
            'event': 'metrics',
            'timestamp': time.time(),
            'timesteps': int(self.num_timesteps),
            'avg_step_ms': avg_step_ms,
            'nvml_util_gpu_pct': nvml_util,
            'nvml_mem_used_mb': nvml_mem_used,
            'torch_mem_alloc_mb': torch_mem_alloc,
            'torch_mem_reserved_mb': torch_mem_reserved,
            'torch_mem_max_alloc_mb': torch_mem_max,
            'actor_log_std_mean': log_std_mean,
            'alpha_ent_coef': ent_coef,
            'actor_lr': lr,
            'recent_ep_rew_mean': rew_mean,
            'recent_ep_rew_std': rew_std,
            'metadata': self.extra_metadata,
        }
        self._write_json(rec)

    def _on_training_end(self) -> None:
        self._emit_metrics()
        self._write_json({'event': 'training_end', 'timestamp': time.time(), 'timesteps': int(self.num_timesteps)})

    # --- helpers ---
    def _try_init_nvml(self) -> None:
        if not torch.cuda.is_available():
            return
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._nvml = pynvml
            self._nvml_handle = handle
        except Exception:
            self._nvml = None
            self._nvml_handle = None

    def _write_json(self, obj: Dict[str, Any]) -> None:
        try:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        except Exception:
            pass

