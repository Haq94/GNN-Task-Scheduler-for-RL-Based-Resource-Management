import numpy as np

class SoftmaxCrossEntropyLoss:
    @staticmethod
    def forward(y_true, y_pred):
        """
        Computes the Softmax + Cross-Entropy loss.
        Args:
            y_true: True labels, numpy array of shape [batch_size, num_classes]
            y_pred: Predicted logits, numpy array of shape [batch_size, num_classes]
        Returns:
            loss: Scalar value representing the softmax cross-entropy loss
            probs: Predicted probabilities after softmax, numpy array of shape [batch_size, num_classes]
        """
        # Softmax computation
        exp_preds = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))  # For numerical stability
        probs = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
        
        # Cross-Entropy Loss
        loss = -np.mean(np.sum(y_true * np.log(probs), axis=1))  # Negative log-likelihood
        return loss, probs
    
    @staticmethod
    def grad(y_true, probs):
        """
        Computes the gradient of the loss with respect to the input logits.
        Args:
            y_true: True labels, numpy array of shape [batch_size, num_classes]
            y_pred: Predicted logits, numpy array of shape [batch_size, num_classes]
            probs: Softmax probabilities, numpy array of shape [batch_size, num_classes]
        Returns:
            grad: Gradient of the loss with respect to the input logits
        """
        grad = probs - y_true
        return grad

# Regression script for testing the SoftmaxCrossEntropyLoss class
if __name__ == "__main__":
    # Sample input for regression
    y_true = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])  # One-hot encoded true labels
    y_pred = np.array([[2.0, 1.0, 0.1], [0.5, 1.5, 1.0], [1.2, 0.8, 0.4]])  # Logits (before softmax)
    
    # Forward pass
    loss, probs = SoftmaxCrossEntropyLoss.forward(y_true, y_pred)
    print("Loss:", loss)
    print("Softmax Probabilities:", probs)
    
    # Backward pass (Gradient)
    grad = SoftmaxCrossEntropyLoss.grad(y_true, y_pred, probs)
    print("Gradient w.r.t logits:", grad)
