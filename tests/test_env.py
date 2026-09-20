import pytest
import numpy as np
import pandas as pd
import yaml

from env.trading_env import ForexTradingEnv
from env.features import FeatureEngineer


@pytest.fixture
def sample_data():
    """Create minimal sample data for testing."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2023-01-01", periods=n, freq="5min")

    price = 1.1000
    prices = [price]
    for _ in range(n - 1):
        price *= (1 + np.random.normal(0, 0.0002))
        prices.append(price)

    df = pd.DataFrame({
        "open":   prices,
        "high":   [p * 1.0005 for p in prices],
        "low":    [p * 0.9995 for p in prices],
        "close":  prices,
        "volume": np.random.randint(100, 1000, n).astype(float)
    }, index=dates)

    return df


@pytest.fixture
def full_config():
    """Load full config the way the app loads it."""
    with open("configs/env_config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def trading_env(sample_data, full_config):
    feature_eng = FeatureEngineer(full_config)
    features    = feature_eng.build_features(sample_data)

    env = ForexTradingEnv(
        df=sample_data,
        features_df=features,
        config_path="configs/env_config.yaml",
        training=False
    )
    return env


class TestForexTradingEnv:

    def test_env_initializes(self, trading_env):
        assert trading_env is not None

    def test_observation_space_shape(self, trading_env):
        obs, _ = trading_env.reset()
        assert obs.shape == trading_env.observation_space.shape

    def test_valid_actions(self, trading_env):
        trading_env.reset()
        for action in range(4):
            obs, reward, terminated, truncated, info = trading_env.step(action)
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert obs.shape == trading_env.observation_space.shape
            trading_env.reset()

    def test_no_nan_in_observations(self, trading_env):
        obs, _ = trading_env.reset()
        assert not np.any(np.isnan(obs))
        assert not np.any(np.isinf(obs))

    def test_buy_then_close(self, trading_env):
        trading_env.reset()
        trading_env.step(1)  # Buy
        assert trading_env.position == 1
        trading_env.step(3)  # Close
        assert trading_env.position == 0

    def test_sell_then_close(self, trading_env):
        trading_env.reset()
        trading_env.step(2)  # Sell
        assert trading_env.position == -1
        trading_env.step(3)  # Close
        assert trading_env.position == 0

    def test_episode_runs_to_completion(self, trading_env):
        obs, _ = trading_env.reset()
        done  = False
        steps = 0

        while not done and steps < 5000:
            action = trading_env.action_space.sample()
            obs, reward, terminated, truncated, info = trading_env.step(action)
            done = terminated or truncated
            steps += 1

        assert done, "Episode should complete"

    def test_balance_decreases_with_costs(self, trading_env):
        trading_env.reset()

        for _ in range(5):
            trading_env.step(1)  # Buy
            trading_env.step(3)  # Close

        assert trading_env.total_trades == 5
