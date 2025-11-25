import os
import logging
from stable_baselines3.common.callbacks import BaseCallback
try:
    import wandb
except ImportError:
    wandb = None

logger = logging.getLogger("WandbCallback")

class WandbCallback(BaseCallback):
    """
    Custom callback for logging to Weights & Biases.
    """
    def __init__(self, verbose=0):
        super(WandbCallback, self).__init__(verbose)
        self.wandb_available = wandb is not None
        if not self.wandb_available:
            logger.warning("WandB not installed. Logging will be skipped.")

    def _on_step(self) -> bool:
        if not self.wandb_available:
            return True
            
        # Log standard metrics
        logs = {}
        
        # Extract values from locals if available (e.g. from the logger)
        # SB3 logger is a bit complex, but we can access some info
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                if "episode" in info:
                    logs["episode_reward"] = info["episode"]["r"]
                    logs["episode_length"] = info["episode"]["l"]
        
        # Log custom metrics from the environment if exposed
        # (Assuming single env for now)
        env = self.training_env.envs[0].unwrapped
        if hasattr(env, "portfolio_value_ac"):
            logs["portfolio_value"] = float(env.portfolio_value_ac)
        if hasattr(env, "max_drawdown_episode"):
            logs["max_drawdown"] = float(env.max_drawdown_episode)

        if logs:
            wandb.log(logs)
            
        return True
