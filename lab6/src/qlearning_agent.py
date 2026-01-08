import numpy as np
import random


class QLearningAgent():
    def __init__(self, actions, learning_rate=0.1, discount=0.99, epsilon=0.1):
        self.q_table = {}
        self.learning_rate = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.actions = actions

    def get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        return self.q_table[state]

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        else:
            q_values = self.get_q_values(state)
            q_max = np.max(q_values)
            actions_with_q_max = [a for a, q in zip(self.actions, q_values) if q == q_max]
            return random.choice(actions_with_q_max)

    def learn(self, state, action, reward, next_state):
        current_q = self.get_q_values(state)[action]
        next_q_max = np.max(self.get_q_values(next_state))

        new_q = current_q + self.learning_rate * (reward + self.gamma * next_q_max - current_q)
        self.q_table[state][action] = new_q
