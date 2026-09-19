"""
Main training script.
Run locally or via GitHub Actions.
"""
import argparse
import yaml
from pathlib import Path
from loguru import logger

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
    StopTrainingOnNoModelImprovement
)
from stable_baselines3.common.monitor import Monitor

from utils.data_loader import DataLoader
from utils.logger import setup_logger
from env.features import FeatureEngineer
from env.trading_env import ForexTradingEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Train RL Forex Agent")
    parser.add_argument("--pair", type=str, default="EURUSD")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--config-env", default="configs/env_config.yaml")
    parser.add_argument("--config-agent", default="configs/agent_config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logger()
    
    # Create output dirs
    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # Load configs
    with open(args.config_env) as f:
        env_config = yaml.safe_load(f)
    with open(args.config_agent) as f:
        agent_config = yaml.safe_load(f)["agent"]["ppo"]
    
    logger.info(f"Starting training | Pair: {args.pair} | Steps: {args.timesteps:,}")
    
    # --- Load Data ---
    loader = DataLoader()
    train_df, val_df, _ = loader.load_splits(args.pair)
    
    # --- Build Features ---
    feature_eng = FeatureEngineer(env_config)
    train_features = feature_eng.build_features(train_df)
    val_features = feature_eng.build_features(val_df)
    
    logger.info(f"Train: {len(train_df)} rows | Val: {len(val_df)} rows")
    
    # --- Create Environments ---
    def make_train_env():
        env = ForexTradingEnv(
            df=train_df,
            features_df=train_features,
            config_path=args.config_env,
            training=True
        )
        return Monitor(env, "logs/train_monitor")
    
    def make_val_env():
        env = ForexTradingEnv(
            df=val_df,
            features_df=val_features,
            config_path=args.config_env,
            training=False
        )
        return Monitor(env, "logs/val_monitor")
    
    train_env = make_vec_env(make_train_env, n_envs=1)
    eval_env = make_val_env()
    
    # --- Callbacks ---
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"models/{args.pair}/best",
        log_path=f"results/{args.pair}/eval_logs",
        eval_freq=agent_config.get("eval_freq", 10_000),
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
        callback_after_eval=StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=5,
            min_evals=10,
            verbose=1
        )
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=f"models/{args.pair}/checkpoints",
        name_prefix=f"ppo_{args.pair.lower()}",
        verbose=1
    )
    
    callbacks = CallbackList([eval_callback, checkpoint_callback])
    
    # --- Create Agent ---
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=agent_config["learning_rate"],
        n_steps=agent_config["n_steps"],
        batch_size=agent_config["batch_size"],
        n_epochs=agent_config["n_epochs"],
        gamma=agent_config["gamma"],
        gae_lambda=agent_config["gae_lambda"],
        clip_range=agent_config["clip_range"],
        ent_coef=agent_config["ent_coef"],
        vf_coef=agent_config["vf_coef"],
        max_grad_norm=agent_config["max_grad_norm"],
        verbose=1,
        tensorboard_log=f"results/{args.pair}/tensorboard",
        device="auto"
    )
    
    logger.info("Model architecture:")
    logger.info(f"  Policy: {model.policy}")
    logger.info(f"  Total params: {sum(p.numel() for p in model.policy.parameters()):,}")
    
    # --- Train ---
    logger.info("Starting training...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        progress_bar=True
    )
    
    # --- Save Final Model ---
    final_path = f"models/{args.pair}/final_model"
    model.save(final_path)
    logger.info(f"Final model saved to {final_path}")
    
    # --- Quick Eval Summary ---
    logger.info("Training complete!")
    

if __name__ == "__main__":
    main()
