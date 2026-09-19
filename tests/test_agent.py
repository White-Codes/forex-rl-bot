import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from stable_baselines3.common.monitor import Monitor

from agents.ppo_agent import PPOAgent
from env.trading_env import ForexTradingEnv
from env.features import FeatureEngineer
import yaml


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2023-01-01", periods=n, freq="5min")
    price = 1.1
    prices = [price]
    for _ in range(n - 1):
        price *= 1 + np.random.normal(0, 0.0002)
        prices.append(price)

    return pd.DataFrame({
        "open": prices, "high": [p * 1.0005 for p in prices],
        "low": [p * 0.9995 for p in prices],
        "close": prices,
        "volume": np.random.randint(100, 1000, n).astype(float)
    }, index=dates)


@pytest.fixture
def env_config():
    with open("configs/env_config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def trading_env(sample_data, env_config):
    feature_eng = FeatureEngineer(env_config)
    features = feature_eng.build_features(sample_data)
    env = ForexTradingEnv(df=sample_data, features_df=features, training=False)
    return Monitor(env)


@pytest.fixture
def ppo_agent(trading_env):
    agent = PPOAgent(
        train_env=trading_env,
        eval_env=trading_env,
        pair="EURUSD"
    )
    agent.build()
    return agent


class TestPPOAgent:

    def test_agent_builds(self, ppo_agent):
        assert ppo_agent.model is not None

    def test_predict_returns_valid_action(self, ppo_agent, trading_env):
        obs, _ = trading_env.reset()
        action, _ = ppo_agent.predict(obs, deterministic=True)
        assert action in [0, 1, 2, 3]

    def test_predict_before_build_raises(self, trading_env):
        agent = PPOAgent(train_env=trading_env, eval_env=trading_env)
        with pytest.raises(RuntimeError):
            agent.predict(np.zeros(10))

    def test_save_and_load(self, ppo_agent, trading_env, tmp_path):
        save_path = str(tmp_path / "test_model")
        ppo_agent.save(save_path)

        new_agent = PPOAgent(train_env=trading_env, eval_env=trading_env)
        new_agent.build()
        new_agent.load(save_path)

        assert new_agent.model is not None

    def test_short_training_runs(self, ppo_agent):
        """Sanity check - short training should not crash."""
        ppo_agent.train(total_timesteps=1000, callbacks=[])

    def test_callbacks_built(self, ppo_agent):
        callbacks = ppo_agent.build_callbacks()
        assert callbacks is not None
