# policy_network.py
import numpy as np
from .GCNLayer import GCNLayer
from .DenseLayer import DenseLayer
from .Nonlinearity import NonlinearityLayer
from .SoftmaxCrossEntropyLoss import SoftmaxCrossEntropyLoss

class PolicyNetwork:
    def __init__(self, layers):
        self.layers = layers
    
    def forward(self, X, A):
        for layer in self.layers:
            if isinstance(layer, GCNLayer):
                X = layer.forward(X, A)  # Forward pass for each layer
            else:
                X = layer.forward(X)  # Forward pass for each layer
        return X
    
    def backward(self, grad_output, lr=0.01):
        grad_input = grad_output
        for layer in reversed(self.layers):
            if isinstance(layer, GCNLayer):
                grad_input = layer.backward(grad_input, lr)
            elif isinstance(layer, DenseLayer):
                grad_input = layer.backward(grad_output, lr)
            else:
                grad_input = layer.backward(grad_input)
        return grad_input

if __name__ == "__main__":
    # Example regression test for PolicyNetwork
    np.random.seed(0)
    
    # Parameters
    activation_type = 'relu'
    num_labels = 3
    num_nodes = 5
    batch_size = 7
    gnn_num_input_feat = 6
    gnn_num_output_feat = 3
    
    # Random input data (X) output data y, and adjacency matrix (A)
    X = np.random.rand(num_nodes, gnn_num_input_feat)
    y = np.zeros(num_labels)
    y[np.random.randint(num_labels)] = 1
    A = np.random.randint(2,size=(num_nodes, num_nodes))
    
    # Define newtork
    gcn_layer = GCNLayer(gnn_num_input_feat, gnn_num_output_feat)
    dense_layer = DenseLayer(gnn_num_output_feat, num_labels)
    nonlinearity_layer = NonlinearityLayer(activation_type)
    output_layer = SoftmaxCrossEntropyLoss()
    
    network = [gcn_layer, nonlinearity_layer, dense_layer, nonlinearity_layer]
    
    # Create the policy network
    policy_network = PolicyNetwork(network)
    
    # Forward pass
    y0 = policy_network.forward(X, A)
    
    # Cross Entropy Loss and probability
    loss, probs = output_layer.forward(y, y0)
    print("Output probability:\n", probs)
    print("Loss:\n", loss)
    
    # Backward pass (Gradient computation)
    grad_output = output_layer.grad(y, probs)
    policy_network.backward(grad_output, lr=0.01)


