import numpy as np

class DenseLayer:
    def __init__(self, input_dim, output_dim):
        # Initialize the weights and bias for the dense layer
        self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros(output_dim)
        self.cache = None

    def forward(self, X):
        """
        Forward pass through the dense layer.
        Args:
            X (numpy array): Input data of shape [batch_size, input_dim].
        Returns:
            Output (numpy array): The result of the linear transformation [batch_size, output_dim].
        """
        self.input = X
        output = np.dot(X, self.W) + self.b
        # DELETE FOR DEBUG
        self.output = output
        return output

    def backward(self, grad_output, lr=0.01):
        """
        Backward pass through the dense layer.
        Args:
            X (numpy array): Input data used in the forward pass [batch_size, input_dim].
            grad_output (numpy array): Gradient of the loss with respect to the output [batch_size, output_dim].
            lr (float): Learning rate for the update step.
        Returns:
            grad_input (numpy array): Gradient of the loss with respect to the input [batch_size, input_dim].
        """
        X = self.input
        grad_W = np.dot(X.T, grad_output)
        grad_b = np.sum(grad_output, axis=0)

        # Update weights and biases
        self.W -= lr * grad_W
        self.b -= lr * grad_b

        # Propagate gradients back to the input
        grad_input = np.dot(grad_output, self.W.T)
        return grad_input

# Regression script for testing
if __name__ == "__main__":
    # Generate random input data (e.g., batch_size=5, input_dim=3)
    X = np.random.randn(5, 3)

    # Define the DenseLayer
    dense_layer = DenseLayer(input_dim=3, output_dim=2)

    # Perform forward pass
    output = dense_layer.forward(X)
    print("Forward pass output:")
    print(output)

    # Generate a random gradient for the backward pass
    grad_output = np.random.randn(5, 2)

    # Perform backward pass and update weights
    grad_input = dense_layer.backward(X, grad_output, lr=0.01)
    print("\nBackward pass gradient wrt input:")
    print(grad_input)

    # Print updated weights and biases
    print("\nUpdated weights:")
    print(dense_layer.W)
    print("\nUpdated biases:")
    print(dense_layer.b)
