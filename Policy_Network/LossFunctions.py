# LossFunctions.py
import numpy as np

class LossFunctions:
    @staticmethod
    def loss(y_true, y_pred, loss_type="mse"):
        """
        Compute loss based on loss_type.
        Args:
            y_true: True labels
            y_pred: Predictions
            loss_type: "mse" or "cross_entropy"
        Returns:
            loss: Scalar loss value
        """
        if loss_type == "mse":
            return LossFunctions.mse_loss(y_true, y_pred)
        elif loss_type == "cross_entropy":
            return LossFunctions.cross_entropy_loss(y_true, y_pred)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    @staticmethod
    def grad(y_true, y_pred, loss_type="mse"):
        """
        Compute gradient of the loss based on loss_type.
        Args:
            y_true: True labels
            y_pred: Predictions
            loss_type: "mse" or "cross_entropy"
        Returns:
            grad: Gradient array
        """
        if loss_type == "mse":
            return LossFunctions.mse_grad(y_true, y_pred)
        elif loss_type == "cross_entropy":
            return LossFunctions.cross_entropy_grad(y_true, y_pred)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    @staticmethod
    def mse_loss(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def mse_grad(y_true, y_pred):
        batch_size = y_true.shape[0]
        return (2 / batch_size) * (y_pred - y_true)

    @staticmethod
    def cross_entropy_loss(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    def cross_entropy_grad(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-10, 1 - y_pred)
        grad = -(y_true / y_pred) + ((1 - y_true) / (1 - y_pred))
        return grad / y_true.shape[0]

    @staticmethod
    def policy_loss(log_probs, returns):
        return -np.mean(log_probs * returns)

    @staticmethod
    def policy_grad(log_probs, returns):
        return -returns / log_probs.shape[0]

    @staticmethod
    def value_loss(values, returns):
        return np.mean((values - returns) ** 2)

    @staticmethod
    def value_grad(values, returns):
        batch_size = values.shape[0]
        return (2 / batch_size) * (values - returns)

if __name__ == "__main__":
    # Quick test
    np.random.seed(0)

    y_true = np.random.randn(4, 3)
    y_pred = np.random.randn(4, 3)

    # MSE
    loss = LossFunctions.loss(y_true, y_pred, loss_type="mse")
    grad = LossFunctions.grad(y_true, y_pred, loss_type="mse")
    print("MSE Loss:", loss)
    print("MSE Grad:\n", grad)

    # Cross-Entropy
    y_true_bin = np.random.randint(0, 2, (4, 1))
    y_pred_bin = np.random.rand(4, 1)
    loss = LossFunctions.loss(y_true_bin, y_pred_bin, loss_type="cross_entropy")
    grad = LossFunctions.grad(y_true_bin, y_pred_bin, loss_type="cross_entropy")
    print("\nCross-Entropy Loss:", loss)
    print("Cross-Entropy Grad:\n", grad)
