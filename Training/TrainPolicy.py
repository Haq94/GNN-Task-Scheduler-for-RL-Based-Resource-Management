import numpy as np
# import sys
# import os

# # Fix imports
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Policy_Network.PolicyNetwork import PolicyNetwork
from Policy_Network.LossFunctions import LossFunctions
from Policy_Network.GCNLayer import GCNLayer
from Policy_Network.DenseLayer import DenseLayer
from Policy_Network.Nonlinearity import NonlinearityLayer
from Policy_Network.SoftmaxCrossEntropyLoss import SoftmaxCrossEntropyLoss
from Task_Environment.TaskEnvironment import TaskEnvironment

class TrainPolicy:
    def __init__(self, env, policy_network, num_episodes=1000, lr=0.01, gamma=0.99):
        self.env = env
        self.policy_network = policy_network
        self.num_episodes = num_episodes
        self.lr = lr
        self.gamma = gamma

    def train(self):
        for episode in range(self.num_episodes):
            state = self.env.reset()
            done = False
            log_probs = []
            rewards = []

            while not done:
                X = state  # Node features
                A = self.env.get_adjacency_matrix(X.shape[0])

                # Forward through the policy network
                logits = self.policy_network.forward(X, A)
                logits = logits.T

                # Get action mask
                action_mask = self.env.get_action_mask()

                # Apply masked softmax
                action_probs = self.masked_softmax(logits, action_mask)

                # Sample action
                action = np.random.choice(len(action_probs), p=action_probs)

                # Log prob for policy gradient
                log_prob = np.log(action_probs[action] + 1e-8)

                # Step in environment
                next_state, reward, done, _ = self.env.step(action)

                log_probs.append(log_prob)
                rewards.append(reward)

                state = next_state  # Move to next state

            # End of episode
            returns = self.compute_returns(rewards)

            # Compute policy loss
            policy_loss = self.compute_policy_loss(log_probs, returns)

            # Backward and update
            self.policy_network.backward(policy_loss, self.lr)

            print(f"Episode {episode+1}/{self.num_episodes}, Total Reward: {sum(rewards):.2f}")


    def compute_returns(self, rewards):
        returns = []
        cumulative = 0
        for reward in reversed(rewards):
            cumulative = reward + self.gamma * cumulative
            returns.insert(0, cumulative)
        return np.array(returns)

    def compute_policy_loss(self, log_probs, returns):
        log_probs = np.array(log_probs)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)  # Normalize returns
        loss = -np.sum(log_probs * returns)  # Policy gradient loss
        return loss

    def masked_softmax(self, logits, mask):
        logits = logits.copy()
        logits[np.reshape(mask,logits.shape) == 0] = -1e9  # Set invalid logits to a very large negative number
    
        # Subtract max *per row* for numerical stability
        logits = logits - np.max(logits, axis=1, keepdims=True)
    
        exp_logits = np.exp(logits)
    
        # Sum *per row*
        sum_exp = np.sum(exp_logits, axis=1, keepdims=True)
    
        probs = np.squeeze(exp_logits / sum_exp)
        return probs


if __name__ == "__main__":
    np.random.seed(0)

    # Setup small example
    activation_type = 'relu'
    num_labels = 1
    num_nodes = 5
    input_features = 4  # Memory, CPU, Arrival, Runtime
    dense_feature = 16

    # Build policy network
    gcn_layer = GCNLayer(input_features, dense_feature)
    relu_layer = NonlinearityLayer(activation_type)
    dense_layer = DenseLayer(dense_feature, num_labels)

    network = [gcn_layer, relu_layer, dense_layer]

    policy_net = PolicyNetwork(network)

    # Setup environment
    env = TaskEnvironment(num_tasks=num_nodes)

    # Train
    trainer = TrainPolicy(env, policy_net, num_episodes=500)
    trainer.train()
