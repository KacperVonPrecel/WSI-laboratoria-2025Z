import numpy as np
import random
import gymnasium as gym


class QLearningAgent():
    def __init__(self, n_actions, learning_rate=0.1, discount=0.99, epsilon=0.1):
        self.q_table = {}
        self.learning_rate = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.n_actions = n_actions

    def get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        return self.q_table[state]

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            q_values = self.get_q_values(state)
            q_max = np.max(q_values)
            actions_with_q_max = np.where(q_values == q_max)[0]
            return random.choice(actions_with_q_max)

    def learn(self, state, action, reward, next_state):
        current_q = self.get_q_values(state)[action]
        next_q_max = np.max(self.get_q_values(next_state))

        new_q = current_q + self.learning_rate * (reward + self.gamma * next_q_max - current_q)
        self.q_table[state][action] = new_q


def train_generic(env: gym.Env, agent, state_processor_func, episodes=500):
    history = []

    for _ in range(episodes):
        obs, _ = env.reset()
        state = state_processor_func(obs, env)
        total_reward = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = agent.choose_action(state)

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = state_processor_func(next_obs, env)

            agent.learn(state, action, reward, next_state)

            state = next_state
            total_reward += reward

        history.append(total_reward)
    return history
