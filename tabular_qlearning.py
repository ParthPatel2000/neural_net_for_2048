# q_learning.py

import random
import pickle

class QLearningAgent:
    def __init__(self, actions, \
                alpha=0.1, gamma=0.9, \
                epsilon=0.2, q_table_file="qtable.pkl"):
        self.actions = actions
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # discount factor
        self.epsilon = epsilon  # exploration rate
        self.q_table_file = q_table_file

        # Q-table: {state_tuple: {action: value}}
        self.q_table = {}
        self.load()

    def state_to_key(self, state):
        """Flatten 2D board into hashable tuple"""
        return tuple(tuple(row) for row in state)

    def choose_action(self, state):
        key = self.state_to_key(state)
        if key not in self.q_table:
            self.q_table[key] = {a: 0 for a in self.actions}

        # epsilon-greedy
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return max(self.q_table[key], key=self.q_table[key].get)

    def learn(self, state, action, reward, next_state):
        state_key = self.state_to_key(state)
        next_key = self.state_to_key(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = {a: 0 for a in self.actions}
        if next_key not in self.q_table:
            self.q_table[next_key] = {a: 0 for a in self.actions}

        predict = self.q_table[state_key][action]
        target = reward + self.gamma * max(self.q_table[next_key].values())
        self.q_table[state_key][action] += self.alpha * (target - predict)

    def save(self):
        with open(self.q_table_file, "wb") as f:
            pickle.dump(self.q_table, f)

    def load(self):
        try:
            with open(self.q_table_file, "rb") as f:
                self.q_table = pickle.load(f)
        except FileNotFoundError:
            self.q_table = {}
