#!/usr/bin/env python3

"""Load data, train DDPG, and train the trading agent."""

# Standard library
import os
# Disable TensorFlow info and warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 3rd party packages
import tensorflow as tf
import pandas as pd
import numpy as np
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines.td3.policies import MlpPolicy, LnMlpPolicy
from stable_baselines import TD3
from stable_baselines.common.callbacks import EvalCallback

# Local source
from src.environment.stock_market import StockMarketEnv


def main():
    # Disable TensorFlow warning and info messages
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # Reading data
    training_file_name = "data/processed/AAPL_training.csv"
    validation_file_name = "data/processed/AAPL_validation.csv"

    # Output files
    model_file_name = "models/td3_aapl_action_pm_large_punishment_1m"
    tensorboard_log = "./reports/td3_aapl_tensorboard/"
    validation_best_model = "./models/td3_aapl_pm_5_punishment_best_1m"

    dataframe = pd.read_csv(training_file_name)
    validation_dataframe = pd.read_csv(validation_file_name)

    # Creating environments
    env = StockMarketEnv(dataframe)
    validation_env = StockMarketEnv(validation_dataframe)

    # Ensuring compliance with OpenAI Gym API
    check_env(env, warn=True)
    check_env(validation_env, warn=True)

    # wrap it
    env = DummyVecEnv([lambda: env])
    validation_env = DummyVecEnv([lambda: validation_env])

    # Automatically normalize the input features and reward
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    validation_env = VecNormalize(validation_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # The noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.5 * np.ones(n_actions))

    # Train the agent
    model = TD3(MlpPolicy, env, action_noise=action_noise, verbose=1,
                buffer_size=50000,
                # learning_starts=10000,
                learning_rate=0.001,
                random_exploration=0.1,
                tensorboard_log=tensorboard_log,
                )
    model.learn(total_timesteps=400000)
    model.save(model_file_name)

    # Saving best model based on validation set performance
    # Callback for best model
    validation_callback = EvalCallback(validation_env, best_model_save_path=validation_best_model,
                                       eval_freq=1000, deterministic=True, verbose=1,
                                       n_eval_episodes=10)
    model.set_env(validation_env)
    model.learn(total_timesteps=100000, callback=validation_callback)

    stats_path = "reports/vec_normalize.pkl"
    env.save(stats_path)


if __name__ == "__main__":
    main()
