"""Fetch and split data - run before training."""
import argparse
from utils.data_loader import DataLoader
from utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default="EURUSD")
    args = parser.parse_args()
    
    setup_logger()
    
    loader = DataLoader()
    
    # Fetch all data
    df = loader.fetch_data(
        pair=args.pair,
        timeframe=loader.config["timeframe"],
        start=loader.config["train_start"],
        end=loader.config["test_end"],
        use_cache=True
    )
    
    # Split
    train, val, test = loader.split_data(df)
    
    # Save
    loader.save_splits(args.pair, train, val, test)
    

if __name__ == "__main__":
    main()
