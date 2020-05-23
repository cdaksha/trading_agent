# Bonus Project
### Chad Daksha

## Program Details

The stock-trading process is modeled as a Markov Decision Process (MDP). The trading goal, then, is to maximize
the potential stock returns and minimize the risk associated with those returns.
The problem is, we have a continuous search space. However, there are multiple deep reinforcement learning
methods that attempt to solve this issue. One of them is **Deep Deterministic Policy Gradient (DDPG)**, which
concurrently learns a Q-function and a policy using a neural network for both an "actor" and "critic".

**HOWEVER**, the problem with DDPG is that it is **frequently brittle with respect to hyperparameters**.
Given the scope of this work, there isn't enough time to tune all of the hyperparameters associated with a deep learning problem;
thus, I will **consider the improvement to DDPG, known as Twin Delayed DDPG (TD3)**. TD3 attempts to augment DDPG by 
introducing three tricks. The actor/critic networks employed are standard, multi-layer perceptron (MLP) networks.

The current implementation supports agent trading for ONE stock only. At any timestep (currently, one timestep = one day), 
the agent has the choice to either buy, sell, or hold. The agent is restricted to buying/selling only one stock to restrict
the action space. The reward function is the change in the total value of the agent's portfolio going from *t* to *t + 1*.

The total value of the agent's portfolio at any given time is equal to *stock_price \* number_stocks_held + remaining_balance*.
The agent is started out with an initial balance of $10,000. 

The TD3 algorithm is trained for 1,000,000 iterations. An Ornstein-Uhlenbeck action noise is added. Most of the hyperparameters used are the reasonable defaults provided by the original TD3 author and in the OpenAI stable baselines package. Based on the best reward, the corresponding best model is saved (based on the validation data set) using an evaluation callback.

## Dependencies

The project dependencies are in the requirements.txt file.
The notable dependencies are as follows:
	- TensorFlow 1.15
	- OpenAI Gym
	- OpenAI Stable Baselines

OpenAI Gym is a nice package, as it allows standardization of reinforcement learning projects. I have implemented my own
stock trading environment by using its API. Meanwhile, Stable Baselines provides an implementation for DDPG and TD3.

The requirements can be installed by using
```$pip install -r requirements.txt```
Otherwise, you can do what I did and run the following commands individually:
```
$pip install tensorflow==1.15
$pip install gym
$pip install stable-baselines[mpi]
```

I recommend creating your own virtual environment for this.

The DDPG algorithm requires MPI, and so may the TD3 algorithm. If you are having troubles with a Python "module not found error",
make sure you have the relevant MPI version installed for your system. For windows 10, MPI is available 
[here](https://www.microsoft.com/en-us/download/details.aspx?id=57467).

## Usage

First of all, make sure that you are in the project directory:

```
$pwd
> PATH/TO/trading_agent
```

### Model Training

To train the TD3 model, you can run

```
$python3 -m src.models.train_model
```

The model is then saved at `models/`.

**Note that training the model is somewhat time intensive. A trained model is already provided in case you wish not to perform this step.**

### Model Prediction

To run predictions based on the trained model existing in `models/`, you can run

```
$python3 -m src.models.predict_model
```

A plot of the net worth as a function of time is created in `figures/`. The net worth as a function of time based on the agent's
strategy is compared with a baseline buy-and-hold strategy by plotting them both in the same graph.

## Input

The input data currently used is the stock for Apple (stock ticker = 'AAPL'). The data starts from 2010 and ends in 2020.
For proper training, **the data is split into training, validation, and test sets**. There are 2614 data points total in this
time span, with the following split:
	- **Training Data**: 1888 points, ranging from 1/4/2010 \- 6/30/2017 
	- **Validation Data**: 473 points, ranging from 7/3/2017 \- 5/20/2019
	- **Test Data**: 254 points, ranging from 5/21/2019 \- 5/21/2020
Note that the split is heavy towards the training set, as I wish to maximize the data used in training the model.
The test data is used for seeing the model's performance, which is compared to a baseline buy-and-hold stock strategy.

The raw data is provided [here](data/raw/AAPL_since_1980.csv). The training, validation, and test data are available
at `data/processed/`.

## Output

After running the prediction from the model, a plot of the net worth as a function of time is created in `figures/`. The net worth as a function of time based on the agent's strategy is compared with a baseline buy-and-hold strategy by plotting them both in the same graph.

It is interesting to note that, based on the current prediction and training parameters, the agent is underperforming a 
standard buy-and-hold strategy! This is most likely because the agent, itself, seems to be pursuing a buy-and-hold strategy, evident
from the secondary (right) y-axis in the generated graph. However, since a maximum number of stocks is placed as a trading limit,
it takes many days for the agent to achieve the maximum amount of stocks holdable. 

Another reason the agent could be underperforming a buy-and-hold strategy could be due to the nature of AAPL's stock growth.
The training data didn't really contain many drops in the price, while the test data contains drops - which the agent hasn't seen yet.

Nonetheless, the trading agent still achieves a high compounded return of about 60%.
