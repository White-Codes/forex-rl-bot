import pandas as pd
import numpy as np
import pandas_ta as ta
from loguru import logger


class FeatureEngineer:
    """
    Builds feature matrix from raw OHLCV data.
    All features normalized to [-1, 1] or [0, 1] range.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.lookback = config["features"]["lookback_window"]
    
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main method - builds all features.
        Returns DataFrame with same index as input.
        """
        features = pd.DataFrame(index=df.index)
        
        # --- Price Action Features ---
        features["return_1"] = df["close"].pct_change(1)
        features["return_5"] = df["close"].pct_change(5)
        features["return_20"] = df["close"].pct_change(20)
        
        # Log returns (more normally distributed)
        features["log_return_1"] = np.log(df["close"] / df["close"].shift(1))
        features["log_return_5"] = np.log(df["close"] / df["close"].shift(5))
        
        # --- RSI ---
        if self.config["features"]["use_rsi"]:
            features["rsi_7"] = ta.rsi(df["close"], length=7) / 100
            features["rsi_14"] = ta.rsi(df["close"], length=14) / 100
            features["rsi_21"] = ta.rsi(df["close"], length=21) / 100
        
        # --- MACD ---
        if self.config["features"]["use_macd"]:
            macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
            if macd is not None:
                # Normalize by price
                features["macd_histogram"] = macd["MACDh_12_26_9"] / df["close"]
                features["macd_signal_cross"] = np.sign(macd["MACDh_12_26_9"])
        
        # --- Bollinger Bands ---
        if self.config["features"]["use_bollinger"]:
            bb = ta.bbands(df["close"], length=20, std=2)
            if bb is not None:
                # Position within bands: -1 (at lower) to 1 (at upper)
                bb_range = bb["BBU_20_2.0"] - bb["BBL_20_2.0"]
                bb_range = bb_range.replace(0, np.nan)
                features["bb_position"] = (
                    (df["close"] - bb["BBM_20_2.0"]) / (bb_range / 2)
                ).clip(-1, 1)
                # Band width (volatility proxy)
                features["bb_width"] = bb_range / bb["BBM_20_2.0"]
        
        # --- ATR (Volatility) ---
        if self.config["features"]["use_atr"]:
            atr = ta.atr(df["high"], df["low"], df["close"], length=14)
            features["atr_normalized"] = atr / df["close"]
        
        # --- ADX (Trend Strength) ---
        if self.config["features"]["use_adx"]:
            adx = ta.adx(df["high"], df["low"], df["close"], length=14)
            if adx is not None:
                features["adx"] = adx["ADX_14"] / 100
                features["dmp"] = adx["DMP_14"] / 100  # +DI
                features["dmn"] = adx["DMN_14"] / 100  # -DI
        
        # --- Session Features (Forex-specific) ---
        if self.config["features"]["use_session"]:
            hour = df.index.hour
            # Encode time cyclically
            features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
            features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
            # Session flags
            features["london_session"] = (
                ((hour >= 8) & (hour < 16)).astype(float)
            )
            features["ny_session"] = (
                ((hour >= 13) & (hour < 21)).astype(float)
            )
            features["overlap_session"] = (
                ((hour >= 13) & (hour < 16)).astype(float)
            )
        
        # --- Market Structure ---
        rolling_high = df["high"].rolling(self.lookback)
        rolling_low = df["low"].rolling(self.lookback)
        
        features["dist_from_high"] = (
            (df["close"] - rolling_high.max()) / df["close"]
        )
        features["dist_from_low"] = (
            (df["close"] - rolling_low.min()) / df["close"]
        )
        
        # Volume features
        vol_ma = df["volume"].rolling(20).mean()
        features["volume_ratio"] = (df["volume"] / vol_ma).clip(0, 5) / 5
        
        # Drop NaN rows (from indicator lookback periods)
        features.dropna(inplace=True)
        
        logger.info(f"Built {len(features.columns)} features, {len(features)} rows")
        
        return features
    
    def get_feature_names(self) -> list[str]:
        """Returns list of feature names for reference."""
        return [
            "return_1", "return_5", "return_20",
            "log_return_1", "log_return_5",
            "rsi_7", "rsi_14", "rsi_21",
            "macd_histogram", "macd_signal_cross",
            "bb_position", "bb_width",
            "atr_normalized",
            "adx", "dmp", "dmn",
            "hour_sin", "hour_cos",
            "london_session", "ny_session", "overlap_session",
            "dist_from_high", "dist_from_low",
            "volume_ratio"
        ]
