"""
This code is based on the article by Satwik Kansal and Brendan Martin:
https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/
"""

import gym
import os
import numpy as np
from time import sleep
import random


class AutomatedCab:
    def __init__(self, env):
        self.q_table = np.zeros([env.observation_space.n, env.action_space.n])

    def train(self, alpha, gamma, epsilon, env, training_duration=100_000):
        for index in range(0, training_duration):
            state = env.reset()

            epochs = 0
            penalties = 0

            done = False

            while not done:
                # If epsilon is 0.1 then we try a random action in 10% of cases.
                # Else we apply what we have learned so far in the Q table.
                if random.uniform(0, 1) < epsilon:
                    action = env.action_space.sample()  # Explore action space
                else:
                    action = np.argmax(self.q_table[state])  # Exploit learned values

                next_state, reward, done, info = env.step(action)

                # TODO: How does this algorithm work exactly?
                old_value = self.q_table[state, action]
                next_max = np.max(self.q_table[next_state])

                # TODO: This is based on a mathematical formula. What does it do exactly?
                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                # Update the Q table: This is the actual learning!
                self.q_table[state, action] = new_value

                if reward == -10:
                    penalties += 1

                state = next_state
                epochs += 1

            if (index + 1) % 100 == 0:
                os.system('clear')
                print(f"Episode: {index + 1}")

        print('Training finished. \n')

    def evaluate(self, env):
        total_epochs, total_penalties = 0, 0
        episodes = 100

        for _ in range(episodes):
            state = env.reset()
            epochs, penalties, reward = 0, 0, 0
            frames = []
            done = False

            while not done:
                action = np.argmax(self.q_table[state])
                state, reward, done, info = env.step(action)

                if reward == -10:
                    penalties += 1

                epochs += 1

            total_penalties += penalties
            total_epochs += epochs

        print(f"Results after {episodes} episodes:")
        print(f"Average timesteps per episode: {total_epochs / episodes}")
        print(f"Average penalties per episode: {total_penalties / episodes}")

    def render_single_run(self, env, delay=0.1):
        state = env.reset()
        done = False
        timestep = 0

        while not done:
            # Clear the console / screen
            os.system('clear')

            # Determine the next action
            action = np.argmax(self.q_table[state])
            # Act on the determined action and retrieve:
            # - The new state
            # - The reward (positive / negative) that we received for this action
            # - Whether the game is over
            # - TODO: What is stored in info?
            state, reward, done, info = env.step(action)

            timestep += 1

            print(env.render(mode='ansi'))
            print(f"Timestep: {timestep + 1}")
            print(f"Action: {action}")
            print(f"Reward: {reward}")

            sleep(delay)


if __name__ == '__main__':
    taxi_enviroment = gym.make("Taxi-v3")
    automated_cab = AutomatedCab(taxi_enviroment)
    automated_cab.train(0.1, 0.6, 0.1, taxi_enviroment, 20_000)
    # automated_cab.evaluate(taxi_enviroment)
    automated_cab.render_single_run(taxi_enviroment, 0.1)
