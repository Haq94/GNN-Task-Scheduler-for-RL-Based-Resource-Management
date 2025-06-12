import numpy as np
import sys
import os


# Fix imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Policy_Network.PolicyNetwork import PolicyNetwork
from Policy_Network.LossFunctions import LossFunctions
from Policy_Network.GCNLayer import GCNLayer
from Policy_Network.DenseLayer import DenseLayer
from Policy_Network.Nonlinearity import NonlinearityLayer
from Policy_Network.SoftmaxLayer import SoftmaxLayer
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
            Log_probs = np.empty((0, state.shape[0])) 
            Probs = []
            Rewards = np.empty((0, state.shape[0])) 
            Actions = np.empty((0, state.shape[0]), dtype=int) 

            while not done:
                X = state  # Node features
                A = self.env.get_adjacency_matrix(X.shape[0])

                # Forward through the policy network
                logits = self.policy_network.forward(X, A)

                # Get action mask
                action_mask = self.env.get_action_mask()

                # Apply masked softmax
                probs = self.masked_softmax(logits, action_mask)
                # probs = self.softmax(logits)

                # Sample action
                # action = np.random.choice(len(probs), p=probs)
                actions = self.sample_action(probs)

                # Log prob for policy gradient
                log_probs = np.log(probs[np.arange(probs.shape[0]), actions] + 1e-8)

                # Step in environment
                next_state, rewards, done = self.env.step(actions)
                
                Probs.append(probs)
                Log_probs = np.vstack([Log_probs, log_probs])
                Rewards = np.vstack([Rewards, rewards])
                Actions = np.vstack([Actions, actions])
                
                state = next_state  # Move to next state

            # End of episode
            # returns = self.compute_returns(rewards)
            Returns = self.compute_node_returns(Rewards)

            # Compute policy loss
            policy_loss = self.compute_policy_loss(Log_probs, Returns)
            
            # Compute logit gradient of policy loss
            logit_grad = self.reinforce_gradient(Probs, Actions, Returns)

            # Backward and update
            self.policy_network.backward(logit_grad, self.lr)

            print(f"Episode {episode+1}/{self.num_episodes}, Loss: {policy_loss:.4f}, Total Reward: {Rewards.sum():.2f}")
    
    def compute_node_returns(self, rewards):
        """
        Compute discounted returns for each node over time.
    
        Args:
            rewards (np.ndarray): shape (T, N_nodes)
    
        Returns:
            np.ndarray: shape (N_nodes,), the return for each node from t=0
        """
        gamma = self.gamma
        T, N = rewards.shape
        returns = np.zeros_like(rewards)
    
        G = np.zeros(N)
        for t in reversed(range(T)):
            G = rewards[t] + gamma * G
            returns[t] = G
    
        return returns


    def compute_policy_loss(self, log_probs, returns):
        returns = returns.copy()
        returns = (returns - np.mean(returns, axis=0)) / (np.std(returns, axis=0) + 1e-8) 
        loss = -np.sum(log_probs * returns)  # Policy gradient loss
        return loss

    def masked_softmax(self, logits, mask):
        logits = logits.copy()
        logits[np.reshape(mask,logits.shape[0]) == 0, 1] = -1e9  # Set invalid logits to a very large negative number
    
        # Subtract max *per row* for numerical stability
        logits = logits - np.max(logits, axis=1, keepdims=True)
    
        exp_logits = np.exp(logits)
    
        # Sum *per row*
        sum_exp = np.sum(exp_logits, axis=1, keepdims=True)
    
        probs = np.squeeze(exp_logits / sum_exp)
        return probs
    
    def softmax(self, logits):
        logits = logits.copy()
    
        # Subtract max *per row* for numerical stability
        logits = logits - np.max(logits, axis=1, keepdims=True)
    
        exp_logits = np.exp(logits)
    
        # Sum *per row*
        sum_exp = np.sum(exp_logits, axis=1, keepdims=True)
    
        probs = np.squeeze(exp_logits / sum_exp)
        return probs
    
    def sample_action(self, probs):
        num_nodes  = probs.shape[0]
        
        action = np.zeros(num_nodes)
        
        for n in range(num_nodes):
        
            action[n] = np.random.choice(len(probs[n]), p=probs[n])
            
        return action.astype(int)
    
    def reinforce_gradient(self, Probs, Actions, Returns):
        # Normalize returns 
        Returns = Returns.copy()
        Returns = (Returns - np.mean(Returns, axis=0)) / (np.std(Returns, axis=0) + 1e-8)

        # Number of time steps in episode and number of nodes
        num_steps, num_nodes = Actions.shape
        
        # Number of actions
        num_actions = Probs[0].shape[1]
        
        # Build one-hot action matrix
        logit_grad = np.zeros_like(Probs[0])
        
        for node in range(num_nodes):
            for nt in range(num_steps):
                
                # Action probability
                probs = Probs[nt][node]
                
                # Action
                a = Actions[nt, node]
                
                # Discounted return
                G = Returns[nt, node]
                
                # One-hot vector
                one_hot = np.zeros(num_actions)
                one_hot[a] = 1
                
                logit_grad[node] += (probs - one_hot)*G

        # Normalize logit_grad

        logit_grad /= num_steps
                
        return logit_grad



if __name__ == "__main__":
    np.random.seed(0)

    # Setup small example
    activation_type = 'relu'
    num_labels = 2
    num_nodes = 5
    input_features = 4  # Memory, CPU, Arrival, Runtime
    dense_feature = 16

    # Build policy network
    gcn_layer = GCNLayer(input_features, dense_feature)
    relu_layer = NonlinearityLayer(activation_type)
    dense_layer = DenseLayer(dense_feature, num_labels)
    # softmax_layer = SoftmaxLayer()

    network = [gcn_layer, relu_layer, dense_layer]

    policy_net = PolicyNetwork(network)

    # Setup environment
    env = TaskEnvironment(num_tasks=num_nodes)

    # Train
    trainer = TrainPolicy(env, policy_net, num_episodes=500)
    trainer.train()
