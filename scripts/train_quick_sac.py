import os
import sys
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

SRC_PATH = os.path.join(os.path.dirname(__file__), '..', 'src')
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from oanda_trading_bot.training_system.sb3_mini_env import MiniDictTradingEnv


def main():
    # Configure a small but GPU-friendly SAC with deeper MLP to exercise VRAM
    symbols = 5
    feat = 64
    seq = 256
    ctx = 16

    env = DummyVecEnv([lambda: MiniDictTradingEnv(symbols=symbols, feat=feat, seq=seq, ctx=ctx)])

    policy_kwargs = dict(
        net_arch=[1024, 1024, 1024],
        activation_fn=torch.nn.ReLU,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SAC(
        'MultiInputPolicy', env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=50_000,
        batch_size=512,
        train_freq=64,
        gradient_steps=64,
        tau=0.02,
        gamma=0.98,
        target_update_interval=1,
        policy_kwargs=policy_kwargs,
        device=device,
    )

    total_timesteps = 10_000
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    # Save to the expected live model path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    out_dir = os.path.join(project_root, 'weights')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'sac_model_symbols{symbols}.zip')
    model.save(out_path)
    print(f"Saved model to {out_path}")


if __name__ == '__main__':
    main()
