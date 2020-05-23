#!/usr/bin/env python3

"""Stock market custom environment implementation using the gym (OpenAI sourced) API."""

# Standard library
from typing import Dict

# 3rd party packages
import gym
from gym import spaces
import numpy as np
import pandas as pd

# Local source
from gym.utils import seeding


class StockMarketEnv(gym.Env):
    """Stock trading environment that follows the OpenAI gym interface."""

    metadata = {'render.modes': ['human']}

    initial_account_balance = 10000  # Initial account balance in dollars
    number_of_stocks = 1  # Current compatibility only supports trading with one stock

    def __init__(self, dataframe: pd.DataFrame):
        super(StockMarketEnv, self).__init__()

        self.training_data = dataframe
        self.balance = self.initial_account_balance
        self.current_time = 0  # Starting time step = 0
        self.maximum_time = dataframe.shape[0] - 1  # Maximum possible time step

        # Tuple containing (Stock Price, # Shares Held, Remaining Account Balance)
        # TODO: NEED BETTER ENVIRONMENTAL OBSERVATIONS
        self.current_state = np.append(np.array(dataframe.loc[self.current_time].price),  # Stock price
                                       [0,  # Amount of shares held
                                        self.initial_account_balance  # Remaining balance
                                        ])

        # All actions possible for agent to take in the environment
        # buy or sell maximum 5 shares
        # shape parameter = # stocks = D (= 1 for one stock)
        # negative = sell; positive = buy
        # self.action_space = spaces.Box(low=-5, high=5, shape=(self.number_of_stocks,), dtype=np.int8)
        # TODO: NEED BETTER ACTION SPACE
        # TODO: NEED BETTER HANDLING FOR BUY/HOLD
        # TODO: IDEA. GIVE A NEGATIVE REWARD FOR AN INVALID ACTION.
        self.action_space = spaces.Box(low=-self.initial_account_balance, high=self.initial_account_balance,
                                       shape=(self.number_of_stocks,), dtype=np.int8)

        # All of the environment's data to be observed by the agent
        # In this case, we are observing the price of stocks (# = D),
        # amount of holdings in stocks (# = D), and the remaining holding balance
        # shape = (D + D + 1) = 2D + 1 (= 3 for one stock)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(2 * self.number_of_stocks + 1,))

    def step(self, action):
        # Execute one time step within the environment
        stock_price = self.current_state[0]
        number_shares_held = self.current_state[1]
        remaining_balance = self.current_state[2]

        initial_portfolio_value = stock_price * number_shares_held + remaining_balance
        reward = 0
        if action < 0:  # Sell order
            if number_shares_held == 0:
                # Punishment for trying to sell a stock when you can't sell one
                reward -= self.initial_account_balance
            self._sell_stock(action)
        elif action > 0:  # Buy order
            if remaining_balance < stock_price:
                # Punishment for trying to buy a stock when you can't buy one
                reward -= self.initial_account_balance
            self._buy_stock(action)

        self.current_time += 1

        self.current_state = np.append(np.array(self.training_data.loc[self.current_time].price),  # Stock price
                                       [self.current_state[1:]])  # Amount of shares held & remaining balance
        final_portfolio_value = self.current_state[0] * self.current_state[1] + self.current_state[2]

        self.balance = final_portfolio_value
        reward += final_portfolio_value - initial_portfolio_value

        terminal_state = self.current_time >= self.maximum_time
        if terminal_state:
            print(f'Final Portfolio Value: {final_portfolio_value}')
            print(f'Final Total Reward: {final_portfolio_value - self.initial_account_balance}')

        info = self._generate_statistics(action)

        return self.current_state, reward, terminal_state, info

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = self.initial_account_balance
        self.current_time = 0
        self.current_state = np.append(np.array(self.training_data.loc[self.current_time].price),  # Stock price
                                       [0,  # Amount of shares held
                                        self.initial_account_balance  # Remaining balance
                                        ])

        return self.current_state

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.balance - self.initial_account_balance
        print(f'Step: {self.current_time}')
        print(f'Balance: {self.balance}')
        print(f'Shares Held: {self.current_state[1]}')
        print(f'Profit: {profit}')

    def _generate_statistics(self, action) -> Dict:
        """To be used for outputting dictionary of statistics at each training step."""
        price = self.current_state[0]
        shares_held = self.current_state[1]
        remaining_cash = self.current_state[2]
        net_worth = price * shares_held + remaining_cash
        info = {
            'time': self.training_data.loc[self.current_time].date,
            'net_worth': net_worth,
            'profit': (net_worth - self.initial_account_balance),
            'action': action[0],
            'shares_held': shares_held
        }
        return info

    def _sell_stock(self, action):
        current_price = self.current_state[0]
        number_shares_held = self.current_state[1]
        maximum_sellable_shares = min(abs(action), number_shares_held)
        # Shares are sold & holding balance is increased
        self.current_state[2] += current_price * maximum_sellable_shares
        # Number shares held is correspondingly decreased
        self.current_state[1] -= maximum_sellable_shares

    def _buy_stock(self, action):
        current_price = self.current_state[0]
        remaining_balance = self.current_state[2]
        maximum_buyable_shares = min(remaining_balance // current_price, action)
        # Shares are bought & holding balance is decreased
        self.current_state[2] -= current_price * maximum_buyable_shares
        # Number shares held is correspondingly increased
        self.current_state[1] += maximum_buyable_shares

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
