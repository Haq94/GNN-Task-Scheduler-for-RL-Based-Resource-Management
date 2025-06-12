import numpy as np
import sys
import os

# Fix imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestEnvironments:
    def __init__(self):
        self.state = np.array([[1.0, 0.0], [0.0, 1.0]])  # 2 nodes, one-hot input
        self.current_step = 0
        self.correct_action = 0  # always correct

    def reset(self):
        self.current_step = 0
        return self.state

    def step(self, action):
        reward = np.array([10 if a == self.correct_action else -5 for a in action])
        self.current_step += 1
        done = self.current_step >= 3  # One-step episode
        return self.state, reward, done

    def get_action_mask(self):
        return np.array([1, 1])  # Both actions always valid

    def get_adjacency_matrix(self, num_nodes):
        return np.eye(num_nodes)  # No edges (identity matrix)
