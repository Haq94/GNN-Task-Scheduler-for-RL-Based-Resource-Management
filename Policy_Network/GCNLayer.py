import numpy as np

class GCNLayer:
    def __init__(self, input_dim, output_dim, lr=0.01):
        """
        Initialize a Graph Convolutional Network (GCN) layer.

        Args:
            input_dim (int): Number of input features per node
            output_dim (int): Number of output features per node
            lr (float): Learning rate for parameter updates
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr

        # Xavier/Glorot initialization for weights
        limit = np.sqrt(6 / (input_dim + output_dim))
        self.weights = np.random.uniform(-limit, limit, (input_dim, output_dim))

        # Bias initialized to zero
        self.bias = np.zeros(output_dim)

    def normalize_adjacency(self, adj):
        """
        Symmetric normalization of the adjacency matrix with added self-loops.

        Args:
            adj (np.array): Adjacency matrix of shape [N x N]

        Returns:
            np.array: Normalized adjacency matrix
        """
        adj = adj + np.eye(adj.shape[0])  # Add self-connections (self-loops)
        degree = np.sum(adj, axis=1)      # Node degree vector
        d_inv_sqrt = np.diag(1.0 / np.sqrt(degree + 1e-8))  # Inverse sqrt degree matrix
        return d_inv_sqrt @ adj @ d_inv_sqrt  # D^(-1/2) * A * D^(-1/2)

    def forward(self, X, adj):
        """
        Forward pass of the GCN layer.

        Args:
            X (np.array): Node features [N x input_dim]
            adj (np.array): Adjacency matrix [N x N]

        Returns:
            np.array: Output node features [N x output_dim]
        """
        self.inputs = X
        self.adj_norm = self.normalize_adjacency(adj)
        return self.adj_norm @ X @ self.weights + self.bias

    def backward(self, grad_output, lr=0.01):
        """
        Backward pass of the GCN layer.

        Args:
            grad_output (np.array): Gradient of loss w.r.t. output [N x output_dim]

        Returns:
            np.array: Gradient w.r.t. input features [N x input_dim]
        """
        # Gradient w.r.t. weights: X^T * A^T * grad_output
        grad_weights = self.inputs.T @ self.adj_norm.T @ grad_output

        # Gradient w.r.t. bias: sum across nodes
        grad_bias = np.sum(grad_output, axis=0)

        # Parameter updates using gradient descent
        self.weights -= lr * grad_weights
        self.bias -= lr * grad_bias

        # Gradient w.r.t. inputs for backpropagation to earlier layers
        grad_input = self.adj_norm.T @ grad_output @ self.weights.T
        return grad_input


# ------------------------------
# Regression Test Script
# ------------------------------
if __name__ == "__main__":
    np.random.seed(1)

    # Dummy node features [4 nodes, 3 features each]
    X = np.random.randn(4, 3)

    # Adjacency matrix of a 4-node graph
    adj = np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0]
    ])

    # Random target output for regression [4 nodes, 2 output features]
    target = np.random.randn(4, 2)

    # Instantiate GCN layer
    gcn = GCNLayer(input_dim=3, output_dim=2)

    # Train for 1000 epochs
    for epoch in range(1000):
        # Forward pass
        out = gcn.forward(X, adj)

        # Mean Squared Error loss
        loss = np.mean((out - target) ** 2)

        # Compute gradient of loss w.r.t. output
        grad_loss = 2 * (out - target) / out.shape[0]

        # Backward pass
        gcn.backward(grad_loss, lr=0.01)

        # Print loss every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")






