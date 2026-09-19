"""
Base agent class - all agents inherit from this.
Makes it easy to swap PPO for SAC/TD3 later.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from loguru import logger
import yaml


class BaseAgent(ABC):
    """
    Abstract base class for all RL trading agents.
    """

    def __init__(
        self,
        env,
        config_path: str = "configs/agent_config.yaml",
        model_dir: str = "models"
    ):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)["agent"]

        self.env = env
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model = None

    @abstractmethod
    def build(self) -> None:
        """Build/initialize the model."""
        pass

    @abstractmethod
    def train(self, total_timesteps: int, callbacks: list) -> None:
        """Train the agent."""
        pass

    @abstractmethod
    def predict(self, observation, deterministic: bool = True):
        """Predict action from observation."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk."""
        pass

    def get_model_path(self, pair: str, suffix: str = "final") -> Path:
        """Generate consistent model save path."""
        path = self.model_dir / pair
        path.mkdir(parents=True, exist_ok=True)
        return path / f"ppo_{pair.lower()}_{suffix}"

    def log_model_info(self) -> None:
        """Log model parameter count and architecture."""
        if self.model is None:
            logger.warning("Model not built yet.")
            return

        total_params = sum(
            p.numel() for p in self.model.policy.parameters()
        )
        trainable_params = sum(
            p.numel()
            for p in self.model.policy.parameters()
            if p.requires_grad
        )

        logger.info(f"Model Architecture: {self.model.policy.__class__.__name__}")
        logger.info(f"Total parameters:     {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
