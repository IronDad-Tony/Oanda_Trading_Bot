import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
from oanda_trading_bot.training_system.environment.trading_env import UniversalTradingEnvV4
from stable_baselines3 import SAC

logger = logging.getLogger("BacktestEngine")

class BacktestEngine:
    """
    A dedicated engine for backtesting trained models using the UniversalTradingEnvV4.
    Ensures deterministic execution and detailed reporting.
    """
    def __init__(self, env: UniversalTradingEnvV4, model: SAC):
        self.env = env
        self.model = model
        self.results: Dict[str, Any] = {}

    def run(self, deterministic: bool = True) -> Dict[str, Any]:
        """
        Executes the backtest over the entire environment dataset.
        """
        logger.info("Starting backtest...")
        obs, _ = self.env.reset()
        done = False
        truncated = False
        
        portfolio_values = []
        rewards = []
        actions_history = []
        
        pbar = tqdm(total=self.env.max_episode_steps, desc="Backtesting")
        
        while not (done or truncated):
            action, _states = self.model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = self.env.step(action)
            
            portfolio_values.append(float(self.env.portfolio_value_ac))
            rewards.append(float(reward))
            actions_history.append(action)
            pbar.update(1)
            
        pbar.close()
        
        # Compile results
        self.results = {
            "portfolio_values": portfolio_values,
            "rewards": rewards,
            "final_portfolio_value": portfolio_values[-1] if portfolio_values else 0.0,
            "total_return_pct": ((portfolio_values[-1] / self.env.initial_capital) - 1) * 100 if portfolio_values else 0.0,
            "trade_log": self.env.trade_log,
            "max_drawdown": float(self.env.max_drawdown_episode)
        }
        
        logger.info(f"Backtest completed. Final Value: {self.results['final_portfolio_value']:.2f} ({self.results['total_return_pct']:.2f}%)")
        return self.results

    def generate_report(self, output_path: str = "backtest_report.png"):
        """
        Generates a visual report of the backtest.
        """
        if not self.results:
            logger.warning("No results to report. Run backtest first.")
            return

        plt.figure(figsize=(12, 8))
        
        # Portfolio Value
        plt.subplot(2, 1, 1)
        plt.plot(self.results["portfolio_values"], label="Portfolio Value")
        plt.title(f"Backtest Performance (Return: {self.results['total_return_pct']:.2f}%)")
        plt.ylabel("Account Currency")
        plt.legend()
        plt.grid(True)
        
        # Drawdown or Rewards could go here, keeping it simple for now
        plt.subplot(2, 1, 2)
        plt.plot(self.results["rewards"], label="Step Reward", color='orange', alpha=0.6)
        plt.title("Reward History")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path)
        logger.info(f"Backtest report saved to {output_path}")
