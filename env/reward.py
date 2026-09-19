import numpy as np


class RewardCalculator:
    """
    Modular reward calculation.
    Easy to swap/tune reward components.
    """
    
    def __init__(self, config: dict):
        self.weights = config["environment"]["reward"]
        self.pnl_history = []
        self.trade_count = 0
        
    def reset(self):
        self.pnl_history = []
        self.trade_count = 0
    
    def calculate(
        self,
        pnl: float,
        transaction_cost: float,
        current_drawdown: float,
        position_changed: bool,
        balance: float,
        initial_balance: float,
    ) -> tuple[float, dict]:
        """
        Calculate total reward from components.
        Returns: (reward, info_dict)
        """
        
        # Track history
        net_pnl = pnl - transaction_cost
        self.pnl_history.append(net_pnl)
        if position_changed:
            self.trade_count += 1
        
        # --- Component 1: PnL (normalized by balance) ---
        pnl_reward = net_pnl / initial_balance
        
        # --- Component 2: Sharpe-like (risk-adjusted) ---
        if len(self.pnl_history) >= 10:
            returns = np.array(self.pnl_history[-50:])  # Last 50 steps
            mean_return = np.mean(returns)
            std_return = np.std(returns) + 1e-8
            sharpe_reward = mean_return / std_return
        else:
            sharpe_reward = 0.0
        
        # --- Component 3: Drawdown penalty ---
        drawdown_penalty = -abs(current_drawdown) * 2.0
        
        # --- Component 4: Overtrading penalty ---
        # Penalize excessive trading (costs money in real life)
        trade_penalty = -self.trade_count * 0.0001 if position_changed else 0.0
        
        # --- Total Reward ---
        total_reward = (
            self.weights["pnl_weight"] * pnl_reward +
            self.weights["sharpe_weight"] * sharpe_reward +
            self.weights["drawdown_weight"] * drawdown_penalty +
            self.weights["trade_penalty_weight"] * trade_penalty
        )
        
        # Clip reward to prevent extreme values destabilizing training
        total_reward = np.clip(total_reward, -1.0, 1.0)
        
        info = {
            "pnl_reward": pnl_reward,
            "sharpe_reward": sharpe_reward,
            "drawdown_penalty": drawdown_penalty,
            "trade_penalty": trade_penalty,
            "total_reward": total_reward,
            "net_pnl": net_pnl,
        }
        
        return float(total_reward), info
