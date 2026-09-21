import pandas as pd
import numpy as np
import ta
from loguru import logger


class FeatureEngineer:
    """
    Builds feature matrix from raw OHLCV data.
    Works with both intraday and daily timeframes.
    """

    def __init__(self, config: dict):
        self.config  = config
        env_cfg      = config.get("environment", config)
        self.feature_cfg = env_cfg.get("features", {})
        self.lookback    = self.feature_cfg.get("lookback_window", 20)

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build all features from OHLCV data."""
        features = pd.DataFrame(index=df.index)

        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]

        # --- Price Action ---
        features["return_1"]     = close.pct_change(1)
        features["return_5"]     = close.pct_change(5)
        features["return_20"]    = close.pct_change(20)
        features["log_return_1"] = np.log(close / close.shift(1))
        features["log_return_5"] = np.log(close / close.shift(5))

        # --- RSI ---
        if self.feature_cfg.get("use_rsi", True):
            features["rsi_7"]  = ta.momentum.RSIIndicator(close, window=7).rsi()  / 100
            features["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi() / 100
            features["rsi_21"] = ta.momentum.RSIIndicator(close, window=21).rsi() / 100

        # --- MACD ---
        if self.feature_cfg.get("use_macd", True):
            macd_ind = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
            features["macd_histogram"]    = macd_ind.macd_diff() / close
            features["macd_signal_cross"] = np.sign(macd_ind.macd_diff())

        # --- Bollinger Bands ---
        if self.feature_cfg.get("use_bollinger", True):
            bb       = ta.volatility.BollingerBands(close, window=20, window_dev=2)
            bb_upper = bb.bollinger_hband()
            bb_lower = bb.bollinger_lband()
            bb_mid   = bb.bollinger_mavg()
            bb_range = (bb_upper - bb_lower).replace(0, np.nan)
            features["bb_position"] = (
                (close - bb_mid) / (bb_range / 2)
            ).clip(-1, 1)
            features["bb_width"] = bb_range / bb_mid

        # --- ATR ---
        if self.feature_cfg.get("use_atr", True):
            atr = ta.volatility.AverageTrueRange(
                high, low, close, window=14
            ).average_true_range()
            features["atr_normalized"] = atr / close

        # --- ADX ---
        if self.feature_cfg.get("use_adx", True):
            adx_ind = ta.trend.ADXIndicator(high, low, close, window=14)
            features["adx"] = adx_ind.adx()     / 100
            features["dmp"] = adx_ind.adx_pos() / 100
            features["dmn"] = adx_ind.adx_neg() / 100

        # --- Session Features (only for intraday data) ---
        if self.feature_cfg.get("use_session", False):
            try:
                hour = df.index.hour
                features["hour_sin"]        = np.sin(2 * np.pi * hour / 24)
                features["hour_cos"]        = np.cos(2 * np.pi * hour / 24)
                features["london_session"]  = ((hour >= 8)  & (hour < 16)).astype(float)
                features["ny_session"]      = ((hour >= 13) & (hour < 21)).astype(float)
                features["overlap_session"] = ((hour >= 13) & (hour < 16)).astype(float)
            except AttributeError:
                logger.warning("Session features skipped — index has no 'hour' (daily data?)")

        # --- Market Structure ---
        rolling_high = high.rolling(self.lookback)
        rolling_low  = low.rolling(self.lookback)
        features["dist_from_high"] = (close - rolling_high.max()) / close
        features["dist_from_low"]  = (close - rolling_low.min())  / close

        # --- Volume ---
        vol_ma = volume.rolling(20).mean()
        features["volume_ratio"] = (volume / vol_ma).clip(0, 5) / 5

        # Drop NaN rows
        features.dropna(inplace=True)

        logger.info(f"Built {len(features.columns)} features, {len(features)} rows")

        return features

    def get_feature_names(self) -> list[str]:
        return [
            "return_1", "return_5", "return_20",
            "log_return_1", "log_return_5",
            "rsi_7", "rsi_14", "rsi_21",
            "macd_histogram", "macd_signal_cross",
            "bb_position", "bb_width",
            "atr_normalized",
            "adx", "dmp", "dmn",
            "dist_from_high", "dist_from_low",
            "volume_ratio",
        ]
