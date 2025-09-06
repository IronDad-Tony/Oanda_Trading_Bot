import os
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

from oanda_trading_bot.training_system.sb3_mini_env import MiniDictTradingEnv


def main():
    env = MiniDictTradingEnv(symbols=5, feat=8, seq=16, ctx=5)
    check_env(env, warn=True)

    model = SAC("MultiInputPolicy", env, verbose=0, tensorboard_log=None, learning_starts=100, train_freq=1, gradient_steps=1)
    model.learn(total_timesteps=2000)

    # Save to weights path expected by live runner
    # parents[3] resolves to repo root: .../Oanda_Trading_Bot
    root = Path(__file__).resolve().parents[3]
    weights_dir = root / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)
    out = weights_dir / 'sac_model_symbols5.zip'
    model.save(str(out))
    print(f"Saved SAC model to: {out}")


if __name__ == '__main__':
    main()
