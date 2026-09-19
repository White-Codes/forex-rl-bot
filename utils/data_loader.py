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
    
    # yfinance ticker mapping for forex pairs
    PAIR_MAP = {
        "EURUSD": "EURUSD=X",
        "GBPUSD": "GBPUSD=X",
        "USDJPY": "USDJPY=X",
        "AUDUSD": "AUDUSD=X",
        "USDCAD": "USDCAD=X",
    }
    
    TIMEFRAME_MAP = {
        "1m":  "1m",
        "5m":  "5m",
        "15m": "15m",
        "1h":  "1h",
        "1d":  "1d",
    }
    
    def __init__(self, config_path: str = "configs/data_config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)["data"]
        
        self.raw_dir = Path("data/raw")
        self.processed_dir = Path("data/processed")
        self.splits_dir = Path("data/splits")
        
        # Create directories
        for d in [self.raw_dir, self.processed_dir, self.splits_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def fetch_data(
        self,
        pair: str,
        timeframe: str,
        start: str,
        end: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data. Uses cache if available.
        """
        cache_file = self.raw_dir / f"{pair}_{timeframe}_{start}_{end}.parquet"
        
        if use_cache and cache_file.exists():
            logger.info(f"Loading cached data: {cache_file}")
            return pd.read_parquet(cache_file)
        
        ticker = self.PAIR_MAP.get(pair)
        if not ticker:
            raise ValueError(f"Unknown pair: {pair}. Available: {list(self.PAIR_MAP.keys())}")
        
        logger.info(f"Fetching {pair} {timeframe} data from {start} to {end}")
        
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval=timeframe,
            progress=False
        )
        
        if df.empty:
            raise ValueError(f"No data returned for {pair}")
        
        # Standardize columns
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        df.index.name = "datetime"
        
        # Drop NaN rows
        df.dropna(inplace=True)
        
        logger.info(f"Fetched {len(df)} rows for {pair}")
        
        # Cache it
        df.to_parquet(cache_file)
        
        return df
    
    def fetch_all_pairs(self) -> dict[str, pd.DataFrame]:
        """Fetch data for all configured pairs."""
        data = {}
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
    
    def split_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/validation/test sets.
        Time-based split (NO random shuffling for time series!)
        """
        config = self.config
        
        train = df[config["train_start"]:config["train_end"]]
        val = df[config["val_start"]:config["val_end"]]
        test = df[config["test_start"]:config["test_end"]]
        
        logger.info(f"Data splits - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        
        return train, val, test
    
    def save_splits(self, pair: str, train: pd.DataFrame, 
                    val: pd.DataFrame, test: pd.DataFrame):
        """Save splits to disk."""
        for name, df in [("train", train), ("val", val), ("test", test)]:
            path = self.splits_dir / f"{pair}_{name}.parquet"
            df.to_parquet(path)
            logger.info(f"Saved {name} split to {path}")
    
    def load_splits(self, pair: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load pre-saved splits."""
        splits = []
        for name in ["train", "val", "test"]:
            path = self.splits_dir / f"{pair}_{name}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"Split not found: {path}. Run data pipeline first.")
            splits.append(pd.read_parquet(path))
        return tuple(splits)
