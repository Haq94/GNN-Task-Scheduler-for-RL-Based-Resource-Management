import numpy as np

class SoftmaxLayer:
    def __init__(self):
        pass  # No parameters (W, b) in softmax

    def forward(self, X):
        """
        Forward pass: apply softmax to logits X.
        Args:
            X (np.array): Input logits (batch_size x num_classes) or (num_classes,)
        Returns:
            np.array: Probabilities after softmax
        """
        # Shift for numerical stability
        X_shifted = X - np.max(X, axis=-1, keepdims=True)
        exp_X = np.exp(X_shifted)
        self.output = exp_X / np.sum(exp_X, axis=-1, keepdims=True)
        return self.output

    def backward(self, grad_output, lr=0.01):
        """
        Backward pass through softmax.
        Args:
            grad_output (np.array): Gradient of loss w.r.t softmax output
        Returns:
            np.array: Gradient of loss w.r.t input logits
        """
        # Shape (num_classes,) or (batch_size, num_classes)
        probs = self.output

        # If 1D (single sample), reshape
        if probs.ndim == 1:
            probs = probs.reshape(1, -1)
            grad_output = grad_output.reshape(1, -1)

        batch_size, num_classes = probs.shape
        grad_input = np.zeros_like(probs)

        for i in range(batch_size):
            p = probs[i].reshape(-1, 1)  # (num_classes, 1)
            jacobian = np.diagflat(p) - np.dot(p, p.T)  # (num_classes, num_classes)
            grad_input[i] = np.dot(jacobian, grad_output[i])

        if grad_input.shape[0] == 1:
            grad_input = grad_input.flatten()

        return grad_input


def grad_loss_layer(probs, action, return_R):
    y = np.zeros_like(probs)
    y[action] = 1
    grad_logits = (probs - y) * return_R
    return grad_logits
