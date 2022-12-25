"""
A python implementation of reinforcement learning to balance a pole on a cart.
Made possible thanks to the OpenAI gym: https://www.gymlibrary.dev/

This code is based on the article by John Joo. The article also provides some very nice theoretical knowledge
of how Deep Q Networks and reinforcement learning works in general.
https://www.dominodatalab.com/blog/deep-reinforcement-learning

All faults in this code are my own and all great ideas are taken from his article.
"""

import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os


class DQNAgent:
    """
    A "Deep Quality Network" (DQN) agent that is capable of solving the CartPole Problem.
    """

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # The memory of the agent holds up to 2000 previous game states (state, action, reward, next_state, done)
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        """
        Creates and returns the keras neural network model with two hidden layers and output layer.
        :return:
        """
        model = Sequential()
        # 1. Hidden Layer
        model.add(Dense(32, activation="relu", input_dim=self.state_size))
        # 2. Hidden Layer
        model.add(Dense(32, activation="relu"))
        # Output Layer (Determines the action the agent will take)
        model.add(Dense(self.action_size.n, activation="linear"))
        # Use Mean Squared Error as the loss function and the Adam Optimizer
        # See: https://keras.io/api/optimizers/adam/
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """
        Store the given game state as well as the action and its reward into the agents' memory.

        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return:
        """
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward  # if done
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0]))
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act(self, state):
        """

        :param state:
        :return:
        """
        # Epsilon starts out large (1.0) and becomes smaller and smaller.
        # In the beginning we always want random actions!
        if np.random.rand() < self.epsilon:
            return self.action_size.sample()
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def save(self, file_path):
        """
        Store the weights for later re-use.

        :param file_path:
        :return:
        """
        self.model.save_weights(file_path)


if __name__ == '__main__':
    # Create the CartPole environment from the OpenAI gym
    env = gym.make("CartPole-v0")

    # Our model has access to 4 variables (= observation space)
    # 1. Cart Position (-4.8 to 4.8)
    # 2. Cart Velocity (no limit)
    # 3. Pole Angle (-24° to 24°, in radians)
    # 4. Pole Angular Velocity (no limit)
    state_size = env.observation_space.shape[0]
    # We have two possible actions in our action space:
    # 1. Push the cart to the left
    # 2. Push it to the right
    action_size = env.action_space

    # The batch size for the Neural Network
    # TODO: Document what this does (maybe in Obsidian?)
    batch_size = 32
    # The amount of iterations that we train the model on
    n_episodes = 1000

    # Path where we save our model (so that we could use it later on) and create it if necessary:
    output_dir = "model_output/cartpole/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Instantiation of our Neural Network
    agent = DQNAgent(state_size, action_size)

    # Start training the Neural Network
    for episode in range(n_episodes):
        # Reset the environment to (random) start position
        # See: https://www.gymlibrary.dev/environments/classic_control/cart_pole/#starting-state
        # All observations are assigned a uniformly random value in (-0.05, 0.05)
        # TODO: What is the reason for the reshape?
        state = env.reset()  # [-0.01213777 -0.03111566  0.02838673 -0.02328164]
        state = np.reshape(state, [1, state_size])  # [[-0.01213777 -0.03111566  0.02838673 -0.02328164]]

        # The "is_game_over" variable keeps track of whether the game has reached one of its ending requirements:
        # See: https://www.gymlibrary.dev/environments/classic_control/cart_pole/#episode-end
        game_is_over = False
        timestep = 0
        while not game_is_over:
            # Render the current environments current step. This is optional!
            env.render()
            # Determine the action the agent will take
            action = agent.act(state)
            # Then apply that action to the environment
            next_state, reward, game_is_over, _ = env.step(action)

            # If the game is over early (for example because the pole fell down), we want to punish the agent and
            # therefore give it a negative reward.
            if game_is_over:
                reward = -10

            # TODO: What is the reason for the reshape?
            next_state = np.reshape(next_state, [1, state_size])

            agent.remember(state, action, reward, next_state, game_is_over)
            state = next_state
            if game_is_over:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(episode + 1, n_episodes, timestep, agent.epsilon))
            timestep += 1
        if len(agent.memory) > batch_size:
            agent.train(batch_size)

        # Every 50 episodes we want to save the model to a .hdf5 file, so that we could replay it later on.
        if episode % 50 == 0:
            agent.save(output_dir + "weights_" + "{:04d}".format(episode) + ".hdf5")
