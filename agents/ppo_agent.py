"""
PPO Agent - Month 1-2 primary agent.
Wraps stable-baselines3 PPO with clean interface.
"""
from pathlib import Path
from typing import Optional
from loguru import logger

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

from agents.base_agent import BaseAgent


class PPOAgent(BaseAgent):
    """
    PPO Agent for Forex Trading.
    
    Usage:
        agent = PPOAgent(train_env, eval_env, pair="EURUSD")
        agent.build()
        agent.train(total_timesteps=500_000)
        agent.save("models/EURUSD/final")
    """

    def __init__(
        self,
        train_env,
        eval_env,
        pair: str = "EURUSD",
        config_path: str = "configs/agent_config.yaml",
        model_dir: str = "models",
    ):
        super().__init__(train_env, config_path, model_dir)

        self.train_env = train_env
        self.eval_env = eval_env
        self.pair = pair
        self.ppo_config = self.config["ppo"]
        self.training_config = self.config["training"]

    def build(self) -> None:
        """Initialize PPO model with config hyperparameters."""
        logger.info(f"Building PPO agent for {self.pair}")

        self.model = PPO(
            policy="MlpPolicy",
            env=self.train_env,
            learning_rate=self.ppo_config["learning_rate"],
            n_steps=self.ppo_config["n_steps"],
            batch_size=self.ppo_config["batch_size"],
            n_epochs=self.ppo_config["n_epochs"],
            gamma=self.ppo_config["gamma"],
            gae_lambda=self.ppo_config["gae_lambda"],
            clip_range=self.ppo_config["clip_range"],
            ent_coef=self.ppo_config["ent_coef"],
            vf_coef=self.ppo_config["vf_coef"],
            max_grad_norm=self.ppo_config["max_grad_norm"],
            verbose=1,
            tensorboard_log=f"results/{self.pair}/tensorboard",
            device="auto",
        )

        self.log_model_info()

    def build_callbacks(self) -> CallbackList:
        """Build training callbacks."""

        # Stop training if no improvement
        no_improve_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=self.training_config["early_stopping_patience"],
            min_evals=10,
            verbose=1,
        )

        # Evaluate and save best model
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=str(self.model_dir / self.pair / "best"),
            log_path=f"results/{self.pair}/eval_logs",
            eval_freq=self.training_config["eval_freq"],
            n_eval_episodes=self.training_config["n_eval_episodes"],
            deterministic=True,
            verbose=1,
            callback_after_eval=no_improve_callback,
        )

        # Save checkpoints periodically
        checkpoint_callback = CheckpointCallback(
            save_freq=self.training_config["save_freq"],
            save_path=str(self.model_dir / self.pair / "checkpoints"),
            name_prefix=f"ppo_{self.pair.lower()}",
            verbose=1,
        )

        return CallbackList([eval_callback, checkpoint_callback])

    def train(
        self,
        total_timesteps: Optional[int] = None,
        callbacks: Optional[list] = None,
    ) -> None:
        """Train the PPO agent."""
        if self.model is None:
            raise RuntimeError("Model not built. Call agent.build() first.")

        timesteps = total_timesteps or self.training_config["total_timesteps"]
        cbs = callbacks or self.build_callbacks()

        logger.info(f"Training PPO on {self.pair} for {timesteps:,} timesteps")

        self.model.learn(
            total_timesteps=timesteps,
            callback=cbs,
            progress_bar=True,
            reset_num_timesteps=True,
        )

        logger.info("Training complete!")

    def predict(
        self,
        observation,
        deterministic: bool = True,
    ):
        """
        Predict action from observation.
        deterministic=True for evaluation/live trading.
        deterministic=False for exploration during training.
        """
        if self.model is None:
            raise RuntimeError("Model not built or loaded.")

        action, _states = self.model.predict(
            observation, deterministic=deterministic
        )
        return action, _states

    def save(self, path: Optional[str] = None) -> None:
        """Save model to disk."""
        if self.model is None:
            raise RuntimeError("No model to save.")

        save_path = path or str(self.get_model_path(self.pair, "final"))
        self.model.save(save_path)
        logger.info(f"Model saved → {save_path}")

    def load(self, path: Optional[str] = None) -> None:
        """Load model from disk."""
        load_path = path or str(self.get_model_path(self.pair, "final"))

        if not Path(f"{load_path}.zip").exists():
            raise FileNotFoundError(f"No model found at {load_path}.zip")

        self.model = PPO.load(load_path, env=self.train_env, device="auto")
        logger.info(f"Model loaded ← {load_path}")
        self.log_model_info()

    def load_best(self) -> None:
        """Load the best model saved during training."""
        best_path = str(self.model_dir / self.pair / "best" / "best_model")
        self.load(best_path)
        logger.info("Loaded best model from evaluation checkpoints")
