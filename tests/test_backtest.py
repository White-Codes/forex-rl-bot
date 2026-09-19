import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
import yaml

from backtest.engine import BacktestEngine
from backtest.metrics import MetricsCalculator
from env.trading_env import ForexTradingEnv
from env.features import FeatureEngineer


@pytest.fixture
def sample_data():
    np.random.seed(99)
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
    return ForexTradingEnv(df=sample_data, features_df=features, training=False)


@pytest.fixture
def mock_agent(trading_env):
    """Mock agent that always holds - simplest case."""
    agent = MagicMock()
    agent.predict.return_value = (0, None)  # Always hold
    return agent


@pytest.fixture
def mock_trading_agent(trading_env):
    """Mock agent that cycles through actions."""
    agent = MagicMock()
    actions = [1, 0, 0, 3, 2, 0, 0, 3]  # Buy, hold, hold, close, sell...
    agent.predict.side_effect = [(a, None) for a in actions * 1000]
    return agent


class TestMetricsCalculator:

    def test_empty_trade_log_returns_zeros(self):
        calc = MetricsCalculator(initial_balance=10_000)
        metrics = calc.calculate_all(
            trade_log=[],
            equity_curve=[10_000, 10_000],
            total_steps=100
        )
        assert metrics["total_trades"] == 0
        assert metrics["win_rate_pct"] == 0.0

    def test_profit_factor_no_losses(self):
        calc = MetricsCalculator()
        trades = [{"pnl": 10, "duration": 5}, {"pnl": 20, "duration": 3}]
        df = pd.DataFrame(trades)
        pf = calc._profit_factor(df)
        assert pf == float("inf")

    def test_sharpe_flat_equity_returns_zero(self):
        calc = MetricsCalculator()
        equity = [10_000.0] * 100
        sharpe = calc._sharpe_ratio(np.array(equity))
        assert sharpe == 0.0

    def test_max_drawdown_calculation(self):
        calc = MetricsCalculator(initial_balance=10_000)
        # Goes up 20% then drops 10%
        equity = np.array([10_000, 11_000, 12_000, 10_800])
        dd = calc._max_drawdown(equity)
        expected = ((12_000 - 10_800) / 12_000) * 100
        assert abs(dd - expected) < 0.01

    def test_win_rate_calculation(self):
        calc = MetricsCalculator()
        trades = pd.DataFrame({"pnl": [10, -5, 20, -3, 15]})
        wr = calc._win_rate(trades)
        assert wr == 60.0  # 3 winners out of 5


class TestBacktestEngine:

    def test_backtest_runs_to_completion(self, mock_agent, trading_env, tmp_path):
        engine = BacktestEngine(
            agent=mock_agent,
            env=trading_env,
            pair="EURUSD",
            results_dir=str(tmp_path)
        )
        results = engine.run()
        assert "metrics" in results
        assert "equity_curve" in results
        assert len(results["equity_curve"]) > 0

    def test_equity_curve_starts_at_initial_balance(self, mock_agent, trading_env, tmp_path):
        engine = BacktestEngine(mock_agent, trading_env, results_dir=str(tmp_path))
        results = engine.run()
        assert results["equity_curve"][0] == trading_env.initial_balance

    def test_results_save_without_error(self, mock_agent, trading_env, tmp_path):
        engine = BacktestEngine(mock_agent, trading_env, results_dir=str(tmp_path))
        results = engine.run()
        engine.save_results(results, tag="test")
        assert (tmp_path / "EURUSD" / "test_metrics.json").exists()

    def test_plot_generates_image(self, mock_trading_agent, trading_env, tmp_path):
        engine = BacktestEngine(
            mock_trading_agent, trading_env,
            pair="EURUSD", results_dir=str(tmp_path)
        )
        results = engine.run()
        engine.plot(results, tag="test", show=False)
        assert (tmp_path / "EURUSD" / "test_report.png").exists()
