#!/usr/bin/env python3
"""
Phase-wise capacity expansion test runner.

Implements Phases A/B/C with short validation runs to check:
- Parameter shapes, memory footprint, and speed
- GPU utilization / VRAM
- Per-step time and reward stability (mean/std)
- Policy entropy proxy (actor log_std mean) and entropy coefficient alpha

Outputs JSONL metrics under `reports/phase_tests/`.
"""
import os
import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import torch
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

# --- Make local imports work ---
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = REPO_ROOT / 'src'
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from oanda_trading_bot.training_system.data_manager.mmap_dataset import UniversalMemoryMappedDataset
from oanda_trading_bot.training_system.data_manager.currency_manager import ensure_currency_data_for_trading
from oanda_trading_bot.training_system.common.config import ACCOUNT_CURRENCY
from oanda_trading_bot.common.instrument_info_manager import InstrumentInfoManager
from oanda_trading_bot.training_system.environment.trading_env import UniversalTradingEnvV4
from oanda_trading_bot.training_system.agent.sac_agent_wrapper import QuantumEnhancedSAC
from oanda_trading_bot.training_system.common.config import (
    DEFAULT_TRAIN_START_ISO,
    DEFAULT_TRAIN_END_ISO,
    DEFAULT_SYMBOLS,
    MAX_SYMBOLS_ALLOWED,
    LOGS_DIR,
)
from oanda_trading_bot.training_system.trainer.metrics_callback import SACMetricsCallback


def setup_logger(level=logging.INFO, log_file: str = None) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
        handlers=[logging.StreamHandler()] + ([logging.FileHandler(log_file)] if log_file else [])
    )
    return logging.getLogger("phase_test_runner")


def create_training_env(symbols: List[str], start_time_iso: str, end_time_iso: str, trading_granularity: str = "S5", env_params: Dict[str, Any] = None):
    logger = logging.getLogger("phase_test_runner")
    env_params = env_params or {}

    # Ensure currency conversion pairs are present in DB for accurate training conversion
    try:
        ok, all_syms = ensure_currency_data_for_trading(
            trading_symbols=symbols,
            account_currency=ACCOUNT_CURRENCY,
            start_time_iso=start_time_iso,
            end_time_iso=end_time_iso,
            granularity=trading_granularity,
        )
        if ok:
            logger.info(f"Currency conversion pairs ensured. Total symbols fetched: {len(all_syms)}")
    except Exception as e:
        logger.warning(f"ensure_currency_data_for_trading failed: {e}")

    dataset = UniversalMemoryMappedDataset(
        symbols=symbols,
        start_time_iso=start_time_iso,
        end_time_iso=end_time_iso,
    )

    info_mgr = InstrumentInfoManager()
    env = UniversalTradingEnvV4(
        dataset=dataset,
        instrument_info_manager=info_mgr,
        active_symbols_for_episode=symbols,
        **env_params,
    )
    return DummyVecEnv([lambda: env])


def run_one(
    logger: logging.Logger,
    phase_name: str,
    model_config: Dict[str, Any],
    sac_params: Dict[str, Any],
    total_timesteps: int,
    model_save_dir: Path,
    reports_dir: Path,
    symbols: List[str],
    start_iso: str,
    end_iso: str,
    trading_granularity: str = "S5",
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = f"{phase_name}_hd{model_config.get('hidden_dim')}_L{model_config.get('num_layers')}_H{model_config.get('num_heads')}_BS{sac_params.get('batch_size')}" \
              f"_GS{sac_params.get('gradient_steps', 'NA')}_{timestamp}"

    # Paths
    model_save_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = model_save_dir / f"sac_universal_{run_tag}"
    metrics_path = reports_dir / f"metrics_{run_tag}.jsonl"

    # Env
    env = create_training_env(symbols, start_iso, end_iso, trading_granularity)

    # Compose callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=max(1000, total_timesteps // 4),
        save_path=str(model_save_dir / "checkpoints"),
        name_prefix=f"{run_tag}",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    metrics_cb = SACMetricsCallback(
        log_path=str(metrics_path),
        log_interval_steps=max(100, total_timesteps // 20),
        recent_ep_window=20,
        extra_metadata={
            'phase': phase_name,
            'model_config': model_config,
            'sac_params': sac_params,
        },
    )
    cbs = CallbackList([checkpoint_cb, metrics_cb])

    # Ensure model_dim divisible by num_heads for PyTorch MultiheadAttention
    try:
        hd = int(model_config.get('hidden_dim'))
        nh = int(model_config.get('num_heads'))
        if hd % max(1, nh) != 0:
            adj_hd = hd - (hd % nh)
            if adj_hd <= 0:
                adj_hd = nh  # minimal valid
            logging.getLogger("phase_test_runner").warning(
                f"Adjusting hidden_dim from {hd} to {adj_hd} to be divisible by num_heads={nh}."
            )
            model_config['hidden_dim'] = adj_hd
    except Exception:
        pass

    # Policy kwargs override to inject model_config and optionally optimizer tweaks
    policy_kwargs = {
        'features_extractor_kwargs': {
            'model_config': model_config,
        },
    }
    if 'optimizer_kwargs' in model_config:
        policy_kwargs['optimizer_kwargs'] = model_config['optimizer_kwargs']

    # Build agent
    logger.info(f"[RUN] {run_tag} | timesteps={total_timesteps} | policy_kwargs override")
    device_override = os.getenv('FORCE_DEVICE', '').strip().lower() or None
    build_kwargs = dict(
        env=env,
        policy_kwargs=policy_kwargs,
        tensorboard_log_path=str(REPO_ROOT / 'logs' / 'tb'),
        **sac_params,
    )
    if device_override in ('cpu', 'cuda'):
        build_kwargs['device'] = device_override
        logger.info(f"Overriding device to: {device_override}")

    agent = QuantumEnhancedSAC(**build_kwargs)

    # Train
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        agent.agent.learn(total_timesteps=total_timesteps, callback=cbs, log_interval=10)
        # Final save
        agent.agent.save(str(model_save_path) + "_final.zip")
    except RuntimeError as e:
        msg = str(e)
        if 'no kernel image is available' in msg and (device_override is None or device_override == 'cuda'):
            logger.warning("CUDA kernel image not available for this GPU with current PyTorch. Falling back to CPU and retrying once.")
            try:
                # rebuild on CPU and retry briefly
                env.close()
                env = create_training_env(symbols, start_iso, end_iso, trading_granularity)
                build_kwargs['env'] = env
                build_kwargs['device'] = 'cpu'
                agent = QuantumEnhancedSAC(**build_kwargs)
                agent.agent.learn(total_timesteps=min(total_timesteps, 1000), callback=cbs, log_interval=10)
                agent.agent.save(str(model_save_path) + "_final_cpu.zip")
            finally:
                env.close()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            raise
    finally:
        try:
            env.close()
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    # Logging
    logs_dir = Path(LOGS_DIR)
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"phase_test_runner_{ts}.log"
    logger = setup_logger(logging.INFO, str(log_file))
    logger.info(f"Logging to {log_file}")

    # Defaults
    symbols = DEFAULT_SYMBOLS[:MAX_SYMBOLS_ALLOWED]
    start_iso, end_iso = DEFAULT_TRAIN_START_ISO, DEFAULT_TRAIN_END_ISO
    model_dir = REPO_ROOT / "trained_models"
    reports_dir = REPO_ROOT / "reports" / "phase_tests"
    total_timesteps_short = int(os.getenv('PHASE_TEST_STEPS', '2000'))  # short validation

    # Phases definition
    phases: Dict[str, Dict[str, Any]] = {
        'PhaseA': {
            'model_variants': [
                {'hidden_dim': 384, 'num_layers': 6, 'num_heads': 8,  'use_symbol_embedding': True},
                {'hidden_dim': 384, 'num_layers': 6, 'num_heads': 12, 'use_symbol_embedding': True},
            ],
            'sac_variants': [
                {'batch_size': 256},
                {'batch_size': 512},
            ],
            'gradient_steps_delta': 1,
        },
        'PhaseB': {
            'model_variants': [
                {'hidden_dim': 512, 'num_layers': 8, 'num_heads': 12, 'use_symbol_embedding': True, 'use_gradient_checkpointing': True},
            ],
            'sac_variants': [
                {'batch_size': 256},
            ],
            'gradient_steps_delta': 0,
        },
        'PhaseC': {
            'model_variants': [
                # No checkpoint for speed/VRAM saturation
                {'hidden_dim': 768, 'num_layers': 12, 'num_heads': 16, 'use_symbol_embedding': True, 'use_gradient_checkpointing': False},
                {'hidden_dim': 768, 'num_layers': 16, 'num_heads': 16, 'use_symbol_embedding': True, 'use_gradient_checkpointing': False},
                # Larger width option (divisible by heads)
                {'hidden_dim': 1024, 'num_layers': 16, 'num_heads': 16, 'use_symbol_embedding': True, 'use_gradient_checkpointing': False},
            ],
            'sac_variants': [
                {'batch_size': 512},
                {'batch_size': 768},
                {'batch_size': 1024},
                {'batch_size': 1536},
                {'batch_size': 2048},
            ],
            'gradient_steps_delta': 0,
        },
    }

    # Base SAC params from common config defaults, overridden per variant
    from oanda_trading_bot.training_system.common.config import (
        SAC_LEARNING_RATE, SAC_BATCH_SIZE, SAC_BUFFER_SIZE_PER_SYMBOL_FACTOR, SAC_LEARNING_STARTS_FACTOR,
        SAC_GAMMA, SAC_ENT_COEF, SAC_TRAIN_FREQ_STEPS, SAC_GRADIENT_STEPS, SAC_TAU,
    )

    # Optional optimizer tweaks (used in Phase C if desired)
    phase_c_optimizer_kwargs = {'weight_decay': float(os.getenv('PHASE_C_WEIGHT_DECAY', '0.0'))}

    # Allow selecting specific phases via env var
    sel = os.getenv('PHASES')
    selected_phases = None
    if sel:
        selected_phases = {p.strip() for p in sel.split(',') if p.strip()}

    # Execute phases sequentially (filtered if requested)
    for phase_name, phase_cfg in phases.items():
        if selected_phases is not None and phase_name not in selected_phases:
            logger.info(f"Skipping {phase_name} (not in PHASES filter)")
            continue
        logger.info(f"===== {phase_name} =====")
        for mv in phase_cfg['model_variants']:
            for sv in phase_cfg['sac_variants']:
                sac_params = dict(
                    learning_rate=SAC_LEARNING_RATE,
                    batch_size=sv.get('batch_size', SAC_BATCH_SIZE),
                    buffer_size=None,  # wrapper computes from factor and symbols
                    buffer_size_factor=SAC_BUFFER_SIZE_PER_SYMBOL_FACTOR,
                    learning_starts=None,
                    learning_starts_factor=SAC_LEARNING_STARTS_FACTOR,
                    gamma=SAC_GAMMA,
                    ent_coef=SAC_ENT_COEF,
                    train_freq_steps=SAC_TRAIN_FREQ_STEPS,
                    gradient_steps=SAC_GRADIENT_STEPS + int(phase_cfg.get('gradient_steps_delta', 0)),
                    tau=SAC_TAU,
                    verbose=0,
                )

                # Augment model config with optimizer tweaks for Phase C if requested
                model_config = mv.copy()
                if phase_name == 'PhaseC' and phase_c_optimizer_kwargs.get('weight_decay', 0.0) > 0.0:
                    model_config['optimizer_kwargs'] = phase_c_optimizer_kwargs

                try:
                    run_one(
                        logger=logger,
                        phase_name=phase_name,
                        model_config=model_config,
                        sac_params=sac_params,
                        total_timesteps=total_timesteps_short,
                        model_save_dir=model_dir,
                        reports_dir=reports_dir,
                        symbols=symbols,
                        start_iso=start_iso,
                        end_iso=end_iso,
                    )
                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        logger.error(f"OOM in {phase_name} with config={model_config}, sac={sac_params}. Skipping.")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        logger.exception(f"Error in run for {phase_name} | config={model_config} | sac={sac_params}")


if __name__ == "__main__":
    # Try to hint-install pynvml if missing (non-fatal)
    try:
        import pynvml  # noqa: F401
    except Exception:
        try:
            import subprocess, sys as _sys
            subprocess.run([_sys.executable, '-m', 'pip', 'install', 'pynvml'], check=False)
        except Exception:
            pass
    main()
