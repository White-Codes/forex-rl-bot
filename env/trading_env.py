import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from loguru import logger
from typing import Optional
import yaml

from env.features import FeatureEngineer
from env.reward import RewardCalculator


class ForexTradingEnv(gym.Env):
    """
    Forex Trading RL Environment - Month 1-2
    Strategy: Mean Reversion on EURUSD M5
    
    Actions:
        0: Hold (do nothing)
        1: Buy (go long)
        2: Sell (go short)
        3: Close position
    
    Observation:
        Market features + position state
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(
        self,
        df: pd.DataFrame,
        features_df: pd.DataFrame,
        config_path: str = "configs/env_config.yaml",
        render_mode: Optional[str] = None,
        training: bool = True
    ):
        super().__init__()
        
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        env_cfg = self.config["environment"]
        
        # Data
        self.df = df
        self.features_df = features_df
        self.feature_names = list(features_df.columns)
        self.n_features = len(self.feature_names)
        
        # Align df and features_df indices
        common_idx = df.index.intersection(features_df.index)
        self.df = df.loc[common_idx]
        self.features_df = features_df.loc[common_idx]
        
        # Account settings
        self.initial_balance = env_cfg["initial_balance"]
        self.transaction_cost = env_cfg["transaction_cost"]
        self.slippage = env_cfg["slippage"]
        self.max_steps = env_cfg["max_steps"]
        self.max_drawdown = env_cfg["max_drawdown"]
        
        # Reward calculator
        self.reward_calculator = RewardCalculator(self.config)
        
        # Mode
        self.render_mode = render_mode
        self.training = training
        
        # --- Action Space ---
        # Discrete: Hold, Buy, Sell, Close
        self.action_space = spaces.Discrete(4)
        
        # --- Observation Space ---
        # Market features + [position, unrealized_pnl, time_in_trade, balance_ratio]
        n_obs = self.n_features + 4
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_obs,),
            dtype=np.float32
        )
        
        logger.info(
            f"Environment initialized | "
            f"Features: {self.n_features} | "
            f"Obs space: {n_obs} | "
            f"Data rows: {len(self.df)}"
        )
        
        # State variables (initialized in reset)
        self._init_state_vars()
    
    def _init_state_vars(self):
        """Initialize/reset all state variables."""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0           # -1: short, 0: none, 1: long
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.time_in_trade = 0
        self.max_balance = self.initial_balance
        self.current_drawdown = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.episode_rewards = []
        self.trade_log = []
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        
        self._init_state_vars()
        self.reward_calculator.reset()
        
        # Random start point during training (data augmentation)
        if self.training:
            max_start = len(self.df) - self.max_steps - 1
            self.current_step = self.np_random.integers(0, max(1, max_start))
        else:
            self.current_step = 0
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment.
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        current_price = self._get_current_price()
        position_changed = False
        transaction_cost = 0.0
        
        # --- Execute Action ---
        if action == 0:  # Hold
            pass
        
        elif action == 1:  # Buy
            if self.position == 0:
                self._open_position(1, current_price)
                transaction_cost = self._calculate_transaction_cost(current_price)
                position_changed = True
            elif self.position == -1:
                # Close short, open long
                self._close_position(current_price)
                transaction_cost = self._calculate_transaction_cost(current_price)
                self._open_position(1, current_price)
                transaction_cost += self._calculate_transaction_cost(current_price)
                position_changed = True
        
        elif action == 2:  # Sell
            if self.position == 0:
                self._open_position(-1, current_price)
                transaction_cost = self._calculate_transaction_cost(current_price)
                position_changed = True
            elif self.position == 1:
                # Close long, open short
                self._close_position(current_price)
                transaction_cost = self._calculate_transaction_cost(current_price)
                self._open_position(-1, current_price)
                transaction_cost += self._calculate_transaction_cost(current_price)
                position_changed = True
        
        elif action == 3:  # Close
            if self.position != 0:
                self._close_position(current_price)
                transaction_cost = self._calculate_transaction_cost(current_price)
                position_changed = True
        
        # Deduct transaction costs
        self.balance -= transaction_cost
        
        # Move to next step
        self.current_step += 1
        if self.position != 0:
            self.time_in_trade += 1
        
        # Update unrealized PnL
        next_price = self._get_current_price()
        self._update_unrealized_pnl(next_price)
        
        # Update drawdown
        current_equity = self.balance + self.unrealized_pnl
        self.max_balance = max(self.max_balance, current_equity)
        self.current_drawdown = (self.max_balance - current_equity) / self.max_balance
        
        # --- Calculate Reward ---
        reward, reward_info = self.reward_calculator.calculate(
            pnl=self.unrealized_pnl,
            transaction_cost=transaction_cost,
            current_drawdown=self.current_drawdown,
            position_changed=position_changed,
            balance=self.balance,
            initial_balance=self.initial_balance,
        )
        
        self.episode_rewards.append(reward)
        
        # --- Check Termination ---
        terminated = self._check_terminated()
        truncated = self.current_step >= self.max_steps
        
        # Force close on episode end
        if (terminated or truncated) and self.position != 0:
            self._close_position(self._get_current_price())
        
        obs = self._get_observation()
        info = self._get_info()
        info.update(reward_info)
        
        return obs, reward, terminated, truncated, info
    
    def _open_position(self, direction: int, price: float):
        """Open a position with slippage."""
        slippage = self.slippage * (1 if direction == 1 else -1)
        self.entry_price = price + slippage
        self.position = direction
        self.time_in_trade = 0
        self.total_trades += 1
    
    def _close_position(self, price: float):
        """Close current position and realize PnL."""
        if self.position == 0:
            return
        
        slippage = self.slippage * (-1 if self.position == 1 else 1)
        close_price = price + slippage
        
        pnl = self.position * (close_price - self.entry_price) * 10000  # Pips
        
        self.realized_pnl += pnl
        self.balance += pnl
        
        if pnl > 0:
            self.winning_trades += 1
        
        # Log trade
        self.trade_log.append({
            "entry_price": self.entry_price,
            "exit_price": close_price,
            "direction": self.position,
            "pnl": pnl,
            "duration": self.time_in_trade,
        })
        
        # Reset position
        self.position = 0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.time_in_trade = 0
    
    def _update_unrealized_pnl(self, current_price: float):
        """Update unrealized PnL for open position."""
        if self.position != 0:
            self.unrealized_pnl = (
                self.position * (current_price - self.entry_price) * 10000
            )
        else:
            self.unrealized_pnl = 0.0
    
    def _calculate_transaction_cost(self, price: float) -> float:
        """Calculate transaction cost (spread simulation)."""
        return self.transaction_cost * price * 10000
    
    def _get_current_price(self) -> float:
        """Get current close price."""
        idx = min(self.current_step, len(self.df) - 1)
        return float(self.df.iloc[idx]["close"])
    
    def _get_observation(self) -> np.ndarray:
        """Build observation vector."""
        idx = min(self.current_step, len(self.features_df) - 1)
        
        # Market features
        market_features = self.features_df.iloc[idx].values.astype(np.float32)
        
        # Position state features
        position_features = np.array([
            float(self.position),                              # -1, 0, 1
            float(self.unrealized_pnl) / self.initial_balance,  # Normalized PnL
            float(self.time_in_trade) / 100.0,                # Normalized time
            float(self.balance) / self.initial_balance,        # Balance ratio
        ], dtype=np.float32)
        
        obs = np.concatenate([market_features, position_features])
        
        # Replace NaN/Inf with 0
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return obs
    
    def _check_terminated(self) -> bool:
        """Check if episode should terminate early."""
        # Blown account
        if self.balance <= self.initial_balance * 0.5:
            logger.debug("Episode terminated: 50% account loss")
            return True
        
        # Max drawdown hit
        if self.current_drawdown >= self.max_drawdown:
            logger.debug(f"Episode terminated: max drawdown {self.current_drawdown:.2%}")
            return True
        
        # Out of data
        if self.current_step >= len(self.df) - 1:
            return True
        
        return False
    
    def _get_info(self) -> dict:
        """Return info dictionary."""
        win_rate = (
            self.winning_trades / self.total_trades
            if self.total_trades > 0 else 0.0
        )
        
        return {
            "step": self.current_step,
            "balance": self.balance,
            "position": self.position,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "total_trades": self.total_trades,
            "win_rate": win_rate,
            "drawdown": self.current_drawdown,
            "total_reward": sum(self.episode_rewards),
        }
    
    def render(self):
        """Simple text render."""
        if self.render_mode == "human":
            info = self._get_info()
            print(
                f"Step: {info['step']:5d} | "
                f"Balance: ${info['balance']:,.2f} | "
                f"Position: {info['position']:+d} | "
                f"PnL: {info['unrealized_pnl']:+.2f} | "
                f"Trades: {info['total_trades']} | "
                f"Win Rate: {info['win_rate']:.1%} | "
                f"Drawdown: {info['drawdown']:.2%}"
            )
