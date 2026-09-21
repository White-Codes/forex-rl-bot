import time
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import yaml


class DataLoader:
    """
    Handles data fetching, caching, and preparation.
    Designed to swap yfinance for MT5 later.
    """

    PAIR_MAP = {
        "EURUSD": "EURUSD=X",
        "GBPUSD": "GBPUSD=X",
        "USDJPY": "USDJPY=X",
        "AUDUSD": "AUDUSD=X",
        "USDCAD": "USDCAD=X",
    }

    # yfinance supported intervals and their max lookback
    TIMEFRAME_LIMITS = {
        "1m":  7,    # days
        "5m":  60,   # days
        "15m": 60,   # days
        "1h":  730,  # days
        "1d":  None, # unlimited
        "1wk": None, # unlimited
    }

    def __init__(self, config_path: str = "configs/data_config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)["data"]

        self.raw_dir       = Path("data/raw")
        self.processed_dir = Path("data/processed")
        self.splits_dir    = Path("data/splits")

        for d in [self.raw_dir, self.processed_dir, self.splits_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def fetch_data(
        self,
        pair: str,
        timeframe: str,
        start: str,
        end: str,
        use_cache: bool = True,
        max_retries: int = 3,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data with retries and caching.
        """
        cache_file = self.raw_dir / f"{pair}_{timeframe}_{start}_{end}.parquet"

        if use_cache and cache_file.exists():
            logger.info(f"Loading cached data: {cache_file}")
            df = pd.read_parquet(cache_file)
            logger.info(f"Loaded {len(df)} rows from cache")
            return df

        ticker = self.PAIR_MAP.get(pair)
        if not ticker:
            raise ValueError(f"Unknown pair: {pair}. Available: {list(self.PAIR_MAP.keys())}")

        # Validate timeframe limit
        limit_days = self.TIMEFRAME_LIMITS.get(timeframe)
        if limit_days is not None:
            logger.warning(
                f"Timeframe {timeframe} is limited to {limit_days} days history on yfinance. "
                f"Consider using '1d' for full history."
            )

        logger.info(f"Fetching {pair} ({ticker}) {timeframe} data from {start} to {end}")

        df = None
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Download attempt {attempt}/{max_retries}...")

                raw = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    interval=timeframe,
                    progress=False,
                    auto_adjust=True,
                    timeout=30,
                )

                if raw is None or raw.empty:
                    raise ValueError(f"Empty data returned for {pair} on attempt {attempt}")

                # Flatten MultiIndex columns if present
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)

                # Standardize columns to lowercase
                raw.columns = [c.lower() for c in raw.columns]

                # Keep only OHLCV
                available = [c for c in ["open", "high", "low", "close", "volume"] if c in raw.columns]
                if len(available) < 4:
                    raise ValueError(f"Missing columns. Got: {list(raw.columns)}")

                df = raw[available].copy()
                df.index.name = "datetime"

                # Drop rows where close is NaN
                df.dropna(subset=["close"], inplace=True)

                # Fill remaining NaNs in volume with 0
                if "volume" in df.columns:
                    df["volume"] = df["volume"].fillna(0)

                if len(df) < 10:
                    raise ValueError(f"Too few rows returned: {len(df)}")

                logger.info(f"Successfully fetched {len(df)} rows for {pair}")
                break

            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt} failed: {e}")
                if attempt < max_retries:
                    wait = attempt * 5
                    logger.info(f"Waiting {wait}s before retry...")
                    time.sleep(wait)

        if df is None or df.empty:
            raise ValueError(
                f"Failed to fetch {pair} after {max_retries} attempts. "
                f"Last error: {last_error}"
            )

        # Cache the data
        df.to_parquet(cache_file)
        logger.info(f"Data cached → {cache_file}")

        return df

    def fetch_all_pairs(self) -> dict[str, pd.DataFrame]:
        """Fetch data for all configured pairs."""
        data   = {}
        config = self.config

        for pair in config["pairs"]:
            try:
                df = self.fetch_data(
                    pair=pair,
                    timeframe=config["timeframe"],
                    start=config["train_start"],
                    end=config["test_end"]
                )
                data[pair] = df
            except Exception as e:
                logger.error(f"Failed to fetch {pair}: {e}")

        return data

    def split_data(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Time-based split — NO random shuffling for time series!
        """
        config = self.config

        # Make index tz-naive for safe slicing
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        train = df[config["train_start"]:config["train_end"]]
        val   = df[config["val_start"]:config["val_end"]]
        test  = df[config["test_start"]:config["test_end"]]

        logger.info(
            f"Data splits — "
            f"Train: {len(train)} | "
            f"Val: {len(val)} | "
            f"Test: {len(test)}"
        )

        if len(train) == 0:
            raise ValueError("Train split is empty — check date ranges in data_config.yaml")
        if len(val) == 0:
            raise ValueError("Val split is empty — check date ranges in data_config.yaml")
        if len(test) == 0:
            raise ValueError("Test split is empty — check date ranges in data_config.yaml")

        return train, val, test

    def save_splits(
        self,
        pair: str,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame
    ) -> None:
        """Save splits to disk."""
        for name, df in [("train", train), ("val", val), ("test", test)]:
            path = self.splits_dir / f"{pair}_{name}.parquet"
            df.to_parquet(path)
            logger.info(f"Saved {name} split → {path} ({len(df)} rows)")

    def load_splits(
        self, pair: str
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load pre-saved splits."""
        splits = []
        for name in ["train", "val", "test"]:
            path = self.splits_dir / f"{pair}_{name}.parquet"
            if not path.exists():
                raise FileNotFoundError(
                    f"Split not found: {path}. "
                    f"Run prepare_data.py first."
                )
            df = pd.read_parquet(path)
            logger.info(f"Loaded {name} split ← {path} ({len(df)} rows)")
            splits.append(df)

        return tuple(splits)
