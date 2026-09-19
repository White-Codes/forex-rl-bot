"""
Performance metrics for backtesting.
Industry standard metrics used by professional traders.
"""
import numpy as np
import pandas as pd
from typing import Optional
from loguru import logger


class MetricsCalculator:
    """
    Calculates all trading performance metrics from trade log.
    """

    def __init__(self, initial_balance: float = 10_000.0, risk_free_rate: float = 0.02):
        self.initial_balance = initial_balance
        self.risk_free_rate = risk_free_rate  # Annual

    def calculate_all(
        self,
        trade_log: list[dict],
        equity_curve: list[float],
        total_steps: int,
    ) -> dict:
        """
        Master method - calculates all metrics.
        Returns dict of all performance metrics.
        """
        if not trade_log:
            logger.warning("Empty trade log - returning zero metrics")
            return self._empty_metrics()

        equity = np.array(equity_curve)
        trades = pd.DataFrame(trade_log)

        metrics = {}

        # --- Return Metrics ---
        metrics["total_return_pct"] = self._total_return(equity)
        metrics["final_balance"] = float(equity[-1])
        metrics["initial_balance"] = self.initial_balance

        # --- Risk Metrics ---
        metrics["max_drawdown_pct"] = self._max_drawdown(equity)
        metrics["sharpe_ratio"] = self._sharpe_ratio(equity)
        metrics["calmar_ratio"] = self._calmar_ratio(
            metrics["total_return_pct"],
            metrics["max_drawdown_pct"]
        )
        metrics["sortino_ratio"] = self._sortino_ratio(equity)

        # --- Trade Metrics ---
        metrics["total_trades"] = len(trades)
        metrics["winning_trades"] = int((trades["pnl"] > 0).sum())
        metrics["losing_trades"] = int((trades["pnl"] <= 0).sum())
        metrics["win_rate_pct"] = self._win_rate(trades)
        metrics["avg_win"] = float(trades[trades["pnl"] > 0]["pnl"].mean()) if len(trades[trades["pnl"] > 0]) > 0 else 0.0
        metrics["avg_loss"] = float(trades[trades["pnl"] <= 0]["pnl"].mean()) if len(trades[trades["pnl"] <= 0]) > 0 else 0.0
        metrics["profit_factor"] = self._profit_factor(trades)
        metrics["avg_trade_duration"] = float(trades["duration"].mean())
        metrics["expectancy"] = self._expectancy(trades)

        # --- Risk/Reward ---
        metrics["avg_rr_ratio"] = self._avg_rr_ratio(trades)
        metrics["trades_per_100_steps"] = (len(trades) / total_steps) * 100

        self._log_summary(metrics)

        return metrics

    # --- Individual Metric Methods ---

    def _total_return(self, equity: np.ndarray) -> float:
        return float(((equity[-1] - self.initial_balance) / self.initial_balance) * 100)

    def _max_drawdown(self, equity: np.ndarray) -> float:
        """Maximum peak-to-trough drawdown as percentage."""
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        return float(np.max(drawdown) * 100)

    def _sharpe_ratio(self, equity: np.ndarray, periods_per_year: int = 26280) -> float:
        """
        Sharpe ratio annualized.
        26280 = 5min bars per year (252 days * 6.5 hours * 12 bars)
        """
        returns = np.diff(equity) / equity[:-1]
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        daily_rf = self.risk_free_rate / periods_per_year
        excess_returns = returns - daily_rf
        sharpe = np.mean(excess_returns) / (np.std(excess_returns) + 1e-8)
        return float(sharpe * np.sqrt(periods_per_year))

    def _sortino_ratio(self, equity: np.ndarray, periods_per_year: int = 26280) -> float:
        """Sortino ratio - only penalizes downside volatility."""
        returns = np.diff(equity) / equity[:-1]
        if len(returns) == 0:
            return 0.0

        daily_rf = self.risk_free_rate / periods_per_year
        excess_returns = returns - daily_rf
        downside = excess_returns[excess_returns < 0]

        if len(downside) == 0 or np.std(downside) == 0:
            return 0.0

        sortino = np.mean(excess_returns) / (np.std(downside) + 1e-8)
        return float(sortino * np.sqrt(periods_per_year))

    def _calmar_ratio(self, total_return: float, max_drawdown: float) -> float:
        """Calmar ratio = Annual Return / Max Drawdown."""
        if max_drawdown == 0:
            return 0.0
        return float(total_return / (max_drawdown + 1e-8))

    def _win_rate(self, trades: pd.DataFrame) -> float:
        if len(trades) == 0:
            return 0.0
        return float((trades["pnl"] > 0).mean() * 100)

    def _profit_factor(self, trades: pd.DataFrame) -> float:
        """Gross profit / Gross loss."""
        gross_profit = trades[trades["pnl"] > 0]["pnl"].sum()
        gross_loss = abs(trades[trades["pnl"] <= 0]["pnl"].sum())
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return float(gross_profit / gross_loss)

    def _expectancy(self, trades: pd.DataFrame) -> float:
        """Average expected PnL per trade."""
        return float(trades["pnl"].mean()) if len(trades) > 0 else 0.0

    def _avg_rr_ratio(self, trades: pd.DataFrame) -> float:
        """Average reward/risk ratio of trades."""
        winners = trades[trades["pnl"] > 0]["pnl"]
        losers = trades[trades["pnl"] < 0]["pnl"].abs()

        if len(winners) == 0 or len(losers) == 0:
            return 0.0

        return float(winners.mean() / (losers.mean() + 1e-8))

    def _empty_metrics(self) -> dict:
        """Return zero metrics when no trades taken."""
        return {
            "total_return_pct": 0.0,
            "final_balance": self.initial_balance,
            "initial_balance": self.initial_balance,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "calmar_ratio": 0.0,
            "sortino_ratio": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate_pct": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "avg_trade_duration": 0.0,
            "expectancy": 0.0,
            "avg_rr_ratio": 0.0,
            "trades_per_100_steps": 0.0,
        }

    def _log_summary(self, metrics: dict) -> None:
        """Log a clean summary of metrics."""
        logger.info("=" * 50)
        logger.info("BACKTEST RESULTS SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total Return:    {metrics['total_return_pct']:+.2f}%")
        logger.info(f"Sharpe Ratio:    {metrics['sharpe_ratio']:.3f}")
        logger.info(f"Sortino Ratio:   {metrics['sortino_ratio']:.3f}")
        logger.info(f"Calmar Ratio:    {metrics['calmar_ratio']:.3f}")
        logger.info(f"Max Drawdown:    {metrics['max_drawdown_pct']:.2f}%")
        logger.info(f"Total Trades:    {metrics['total_trades']}")
        logger.info(f"Win Rate:        {metrics['win_rate_pct']:.1f}%")
        logger.info(f"Profit Factor:   {metrics['profit_factor']:.2f}")
        logger.info(f"Expectancy:      {metrics['expectancy']:.4f} pips")
        logger.info(f"Avg RR Ratio:    {metrics['avg_rr_ratio']:.2f}")
        logger.info("=" * 50)
