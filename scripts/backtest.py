"""Run backtest on a trained agent."""
import argparse
import yaml
from utils.logger import setup_logger
from utils.data_loader import DataLoader
from env.features import FeatureEngineer
from env.trading_env import ForexTradingEnv
from agents.ppo_agent import PPOAgent
from backtest.engine import BacktestEngine


def parse_args():
    parser = argparse.ArgumentParser(description="Backtest trained RL agent")
    parser.add_argument("--pair", type=str, default="EURUSD")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--config-env", default="configs/env_config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logger()

    with open(args.config_env) as f:
        env_config = yaml.safe_load(f)

    # --- Load Data ---
    loader = DataLoader()
    splits = loader.load_splits(args.pair)
    split_map = {"train": 0, "val": 1, "test": 2}
    df = splits[split_map[args.split]]

    # --- Build Features ---
    feature_eng = FeatureEngineer(env_config)
    features = feature_eng.build_features(df)

    # --- Create Environment ---
    env = ForexTradingEnv(
        df=df,
        features_df=features,
        config_path=args.config_env,
        training=False,
    )

    # --- Load Agent ---
    agent = PPOAgent(
        train_env=env,
        eval_env=env,
        pair=args.pair,
    )
    agent.build()

    model_path = args.model_path or f"models/{args.pair}/best/best_model"
    agent.load(model_path)

    # --- Run Backtest ---
    engine = BacktestEngine(agent=agent, env=env, pair=args.pair)
    results = engine.run(deterministic=True)
    engine.save_results(results, tag=f"backtest_{args.split}")
    engine.plot(results, tag=f"backtest_{args.split}")


if __name__ == "__main__":
    main()
