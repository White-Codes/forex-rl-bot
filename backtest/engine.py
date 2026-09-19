"""
Backtesting engine - runs trained agent on historical data
and collects all performance data.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from loguru import logger
from typing import Optional

from backtest.metrics import MetricsCalculator
from env.trading_env import ForexTradingEnv


class BacktestEngine:
    """
    Runs a trained agent through historical data
    and produces full performance report.

    Usage:
        engine = BacktestEngine(agent, env, pair="EURUSD")
        results = engine.run()
        engine.save_results(results)
        engine.plot(results)
    """

    def __init__(
        self,
        agent,
        env: ForexTradingEnv,
        pair: str = "EURUSD",
        results_dir: str = "results",
    ):
        self.agent = agent
        self.env = env
        self.pair = pair
        self.results_dir = Path(results_dir) / pair
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_calc = MetricsCalculator(
            initial_balance=env.initial_balance
        )

    def run(self, deterministic: bool = True) -> dict:
        """
        Run full backtest episode.
        Returns dict with metrics, equity curve, trade log.
        """
        logger.info(f"Starting backtest | Pair: {self.pair} | Deterministic: {deterministic}")

        obs, info = self.env.reset()
        done = False
        step = 0

        # Tracking
        equity_curve = [self.env.initial_balance]
        balance_curve = [self.env.initial_balance]
        action_log = []
        reward_log = []

        while not done:
            # Agent predicts action
            action, _ = self.agent.predict(obs, deterministic=deterministic)

            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(int(action))
            done = terminated or truncated

            # Track equity (balance + unrealized pnl)
            equity = info["balance"] + info["unrealized_pnl"]
            equity_curve.append(equity)
            balance_curve.append(info["balance"])
            action_log.append(int(action))
            reward_log.append(reward)

            step += 1

        logger.info(f"Backtest complete | Steps: {step} | Trades: {info['total_trades']}")

        # Calculate metrics
        metrics = self.metrics_calc.calculate_all(
            trade_log=self.env.trade_log,
            equity_curve=equity_curve,
            total_steps=step,
        )

        results = {
            "metrics": metrics,
            "equity_curve": equity_curve,
            "balance_curve": balance_curve,
            "action_log": action_log,
            "reward_log": reward_log,
            "trade_log": self.env.trade_log,
            "total_steps": step,
            "pair": self.pair,
        }

        return results

    def save_results(self, results: dict, tag: str = "backtest") -> None:
        """Save metrics to JSON and curves to parquet."""

        # Save metrics JSON
        metrics_path = self.results_dir / f"{tag}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(results["metrics"], f, indent=2)
        logger.info(f"Metrics saved → {metrics_path}")

        # Save equity curve
        equity_df = pd.DataFrame({
            "equity": results["equity_curve"],
            "balance": results["balance_curve"],
            "action": results["action_log"] + [0],  # Pad last step
            "reward": results["reward_log"] + [0],
        })
        equity_path = self.results_dir / f"{tag}_equity.parquet"
        equity_df.to_parquet(equity_path)
        logger.info(f"Equity curve saved → {equity_path}")

        # Save trade log
        if results["trade_log"]:
            trade_df = pd.DataFrame(results["trade_log"])
            trade_path = self.results_dir / f"{tag}_trades.parquet"
            trade_df.to_parquet(trade_path)
            logger.info(f"Trade log saved → {trade_path}")

    def plot(self, results: dict, tag: str = "backtest", show: bool = False) -> None:
        """
        Generate comprehensive backtest visualization.
        Saves to results directory.
        """
        metrics = results["metrics"]
        equity = results["equity_curve"]
        actions = results["action_log"]
        trades = pd.DataFrame(results["trade_log"]) if results["trade_log"] else pd.DataFrame()

        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(
            f"{self.pair} Backtest Results | "
            f"Return: {metrics['total_return_pct']:+.2f}% | "
            f"Sharpe: {metrics['sharpe_ratio']:.2f} | "
            f"Win Rate: {metrics['win_rate_pct']:.1f}%",
            fontsize=14,
            fontweight="bold"
        )

        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

        # --- Plot 1: Equity Curve ---
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(equity, color="royalblue", linewidth=1.5, label="Equity")
        ax1.axhline(
            self.env.initial_balance,
            color="gray",
            linestyle="--",
            alpha=0.7,
            label="Initial Balance"
        )
        ax1.fill_between(
            range(len(equity)),
            self.env.initial_balance,
            equity,
            where=[e >= self.env.initial_balance for e in equity],
            alpha=0.2, color="green", label="Profit"
        )
        ax1.fill_between(
            range(len(equity)),
            self.env.initial_balance,
            equity,
            where=[e < self.env.initial_balance for e in equity],
            alpha=0.2, color="red", label="Loss"
        )
        ax1.set_title("Equity Curve")
        ax1.set_ylabel("Balance ($)")
        ax1.legend(loc="upper left")
        ax1.grid(alpha=0.3)

        # --- Plot 2: Drawdown ---
        ax2 = fig.add_subplot(gs[1, 0])
        eq_arr = np.array(equity)
        peak = np.maximum.accumulate(eq_arr)
        drawdown = ((peak - eq_arr) / peak) * 100
        ax2.fill_between(range(len(drawdown)), 0, -drawdown, color="red", alpha=0.6)
        ax2.set_title(f"Drawdown (Max: {metrics['max_drawdown_pct']:.2f}%)")
        ax2.set_ylabel("Drawdown (%)")
        ax2.grid(alpha=0.3)

        # --- Plot 3: Action Distribution ---
        ax3 = fig.add_subplot(gs[1, 1])
        action_labels = ["Hold", "Buy", "Sell", "Close"]
        action_counts = [actions.count(i) for i in range(4)]
        colors = ["gray", "green", "red", "orange"]
        ax3.bar(action_labels, action_counts, color=colors, alpha=0.8)
        ax3.set_title("Action Distribution")
        ax3.set_ylabel("Count")
        ax3.grid(alpha=0.3, axis="y")

        # --- Plot 4: Trade PnL Distribution ---
        ax4 = fig.add_subplot(gs[2, 0])
        if not trades.empty:
            colors_pnl = ["green" if p > 0 else "red" for p in trades["pnl"]]
            ax4.bar(range(len(trades)), trades["pnl"], color=colors_pnl, alpha=0.7)
            ax4.axhline(0, color="black", linewidth=0.8)
            ax4.set_title(f"Trade PnL | Total: {len(trades)} trades")
            ax4.set_ylabel("PnL (pips)")
            ax4.set_xlabel("Trade #")
        else:
            ax4.text(0.5, 0.5, "No Trades", ha="center", va="center")
        ax4.grid(alpha=0.3)

        # --- Plot 5: Metrics Summary Table ---
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis("off")
        table_data = [
            ["Metric", "Value"],
            ["Total Return", f"{metrics['total_return_pct']:+.2f}%"],
            ["Sharpe Ratio", f"{metrics['sharpe_ratio']:.3f}"],
            ["Sortino Ratio", f"{metrics['sortino_ratio']:.3f}"],
            ["Calmar Ratio", f"{metrics['calmar_ratio']:.3f}"],
            ["Max Drawdown", f"{metrics['max_drawdown_pct']:.2f}%"],
            ["Win Rate", f"{metrics['win_rate_pct']:.1f}%"],
            ["Profit Factor", f"{metrics['profit_factor']:.2f}"],
            ["Total Trades", str(metrics['total_trades'])],
            ["Avg RR Ratio", f"{metrics['avg_rr_ratio']:.2f}"],
        ]
        table = ax5.table(
            cellText=table_data[1:],
            colLabels=table_data[0],
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        ax5.set_title("Performance Metrics")

        # Save
        plot_path = self.results_dir / f"{tag}_report.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        logger.info(f"Plot saved → {plot_path}")

        if show:
            plt.show()

        plt.close()
