import numpy as np

class Nonlinearity:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_backward(input, grad_output):
        return grad_output * (input > 0)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_backward(input, grad_output):
        sig = Nonlinearity.sigmoid(input)
        return grad_output * sig * (1 - sig)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_backward(input, grad_output):
        return grad_output * (1 - np.tanh(input)**2)


class NonlinearityLayer:
    def __init__(self, activation):
        """
        Nonlinearity layer that wraps activation functions.
        
        Args:
            activation (str): 'relu', 'sigmoid', or 'tanh'
        """
        self.activation = activation
        self.input = None  # Will store input for backward pass

    def forward(self, x):
        self.input = x
        if self.activation == 'relu':
            return Nonlinearity.relu(x)
        elif self.activation == 'sigmoid':
            return Nonlinearity.sigmoid(x)
        elif self.activation == 'tanh':
            return Nonlinearity.tanh(x)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

    def backward(self, grad_output):
        if self.activation == 'relu':
            return Nonlinearity.relu_backward(self.input, grad_output)
        elif self.activation == 'sigmoid':
            return Nonlinearity.sigmoid_backward(self.input, grad_output)
        elif self.activation == 'tanh':
            return Nonlinearity.tanh_backward(self.input, grad_output)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")


# Regression script for Nonlinearity
if __name__ == "__main__":
    # Sample input
    X = np.array([[0.5, -0.5, 0.1], [-0.3, 0.8, -0.6]])

    # Test ReLU
    relu_layer = NonlinearityLayer('relu')
    relu_output = relu_layer.forward(X)
    print("ReLU Output:\n", relu_output)

    grad_output = np.array([[1, 1, 1], [1, 1, 1]])
    relu_grad_input = relu_layer.backward(grad_output)
    print("ReLU Backward Output:\n", relu_grad_input)

    # Test Sigmoid
    sigmoid_layer = NonlinearityLayer('sigmoid')
    sigmoid_output = sigmoid_layer.forward(X)
    print("\nSigmoid Output:\n", sigmoid_output)

    sigmoid_grad_input = sigmoid_layer.backward(grad_output)
    print("Sigmoid Backward Output:\n", sigmoid_grad_input)

    # Test Tanh
    tanh_layer = NonlinearityLayer('tanh')
    tanh_output = tanh_layer.forward(X)
    print("\nTanh Output:\n", tanh_output)

    tanh_grad_input = tanh_layer.backward(grad_output)
    print("Tanh Backward Output:\n", tanh_grad_input)

