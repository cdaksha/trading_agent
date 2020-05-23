#!/usr/bin/env python3

"""Load test data and see how the trading agent performs."""

# Standard library
import os
# Disable TensorFlow info and warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 3rd party packages
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.td3 import TD3
from stable_baselines.common.evaluation import evaluate_policy

# Local source
from src.environment.stock_market import StockMarketEnv


def main():
    # Disable TensorFlow warning and info messages
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # Reading data
    test_file_name = "data/processed/AAPL_test.csv"
    test_dataframe = pd.read_csv(test_file_name)

    # Creating environment
    test_env = DummyVecEnv([lambda: StockMarketEnv(test_dataframe)])

    # Load saved statistics
    stats_path = "reports/vec_normalize.pkl"
    # test_env = VecNormalize.load(stats_path, test_env)
    #  do not update them at test time
    # test_env.training = False
    # reward normalization is not needed at test time
    # test_env.norm_reward = False

    # Load saved agent
    model_file_path = "models/td3_aapl_action_pm_large_punishment_1m.zip"
    model = TD3.load(model_file_path)
    model.set_env(test_env)

    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f'Mean Reward: {mean_reward}; Standard Deviation: {std_reward}')

    # Test (Enjoy) the trained agent
    obs = test_env.reset()
    n_steps = 1000
    all_info = []
    for i in range(n_steps):
        action, _states = model.predict(obs)
        print(action)
        obs, reward, done, info = test_env.step(action)
        all_info.append(info[0])

        if i % 50 == 0:
            test_env.render(mode='console')
        if done:
            # Note that the VecEnv resets automatically
            # when a done signal is encountered
            print("Goal reached!")
            all_info[-1].pop('terminal_observation')
            break

    final_df = pd.DataFrame.from_records(all_info)

    # Buy and hold strategy
    buy_and_hold_df = buy_and_hold_strategy(test_dataframe)

    # Plotting net returns
    plt.rc('font', family='serif')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(final_df.index, final_df.net_worth, color='k', label='Agent')
    ax.plot(buy_and_hold_df.index, buy_and_hold_df.net_worth, 'k--', label='Buy & Hold')

    ax.legend(loc='lower right')
    ax.set_xlabel('Time')
    ax.set_ylabel('Net Worth ($)')

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('# Stocks Held')  # we already handled the x-label with ax1
    ax2.plot(final_df.index, final_df.shares_held, 'k--')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.show()
    plt.savefig("reports/figures/net_worth.jpg", transparent=True, bbox_inches='tight')

    # NOTE: Buy & Hold strategy for AAPL from 5/21/2019 to 5/21/2020 gives a return of ~$7,200
    # NOTE: Buy & Hold for training set has initial worth = $10,000 and final worth = ~$52,000
    # training buy & hold net return ~ $42,000


def buy_and_hold_strategy(dataframe) -> pd.DataFrame:
    """Return dataframe with datetime and net worth for a buy-and-hold stock strategy."""
    total_initial_balance = 10000
    initial_price = dataframe.loc[0, 'price']
    num_stocks_bought = total_initial_balance // initial_price
    remaining_balance = total_initial_balance - initial_price * num_stocks_bought
    net_worth = dataframe.price * num_stocks_bought + remaining_balance
    df = pd.concat([dataframe.date, net_worth], axis=1)
    df.columns = ['time', 'net_worth']
    return df


if __name__ == "__main__":
    main()
