import numpy as np

class GCNLayer:
    def __init__(self, in_features, out_features):
        """
        Initialize the GCN Layer with input and output features.
        Args:
            in_features (int): Number of input features per node.
            out_features (int): Number of output features per node.
        """
        # Initialize weights and biases for the layer
        self.W = np.random.randn(in_features, out_features)  # Weight matrix
        self.b = np.zeros(out_features)  # Bias vector

    def forward(self, X, A):
        """
        Forward pass through the GCN layer: X is the node feature matrix, A is the adjacency matrix.
        Args:
            X (np.array): Node feature matrix of shape (num_nodes, in_features).
            A (np.array): Adjacency matrix of shape (num_nodes, num_nodes).
        Returns:
            np.array: Output node features after applying the GCN layer, shape (num_nodes, out_features).
        """
        # Store Adjacency Matrix and input
        self.input = X
        self.A = A
        # Compute the graph convolution: A * X * W + b
        H = np.dot(A, X)  # Propagate the features using the adjacency matrix
        H = np.dot(H, self.W)  # Apply the weights
        H += self.b  # Add the bias term
        # DELETE JUST FOR DEBUG
        self.output = H
        return H  # This is the output of the layer

    def backward(self, grad_output, lr=0.01):
        """
        Backward pass through the GCN layer, compute gradients and update parameters.
        Args:
            X (np.array): Input feature matrix of shape (num_nodes, in_features).
            A (np.array): Adjacency matrix of shape (num_nodes, num_nodes).
            grad_output (np.array): Gradient of the loss with respect to the layer's output.
            lr (float): Learning rate.
        Returns:
            np.array: Gradient of the loss with respect to the input features (X).
        """
        # Load Adjacency Matrix and input
        X = self.input
        A = self.A
        # Compute gradients for weights and bias
        grad_W = np.dot(np.dot(A, X).T, grad_output)  # Gradient w.r.t weights
        grad_b = np.sum(grad_output, axis=0)  # Gradient w.r.t bias

        # Update parameters using the gradients
        self.W -= lr * grad_W  # Gradient descent step for weights
        self.b -= lr * grad_b  # Gradient descent step for bias

        # Propagate the gradients back to the input
        grad_input = np.dot(A, np.dot(grad_output, self.W.T))  # Gradient w.r.t input features

        return grad_input

# Example of a regression script for the GNCLayer

if __name__ == "__main__":
    # Random input data: 4 nodes, each with 3 features
    X = np.random.randn(4, 3)  # Node features (4 nodes, 3 features)
    # Random adjacency matrix for a graph with 4 nodes
    A = np.array([[0, 1, 0, 0],
                  [1, 0, 1, 0],
                  [0, 1, 0, 1],
                  [0, 0, 1, 0]])  # Adjacency matrix (4 nodes, 4 nodes)

    # Create the GCN layer with 3 input features and 2 output features
    gcn_layer = GCNLayer(in_features=3, out_features=2)

    # Forward pass through the GCN layer
    output = gcn_layer.forward(X, A)
    print("Forward pass output:")
    print(output)

    # Assuming some random gradient of the loss w.r.t output for backpropagation
    grad_output = np.random.randn(4, 2)  # Gradient of loss w.r.t output (4 nodes, 2 output features)

    # Backward pass through the GCN layer
    grad_input = gcn_layer.backward(grad_output, lr=0.01)
    print("Gradient of loss w.r.t input features (backprop):")
    print(grad_input)





