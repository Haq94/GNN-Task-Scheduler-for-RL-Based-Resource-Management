import numpy as np

class GATLayer:
    def __init__(self, input_dim, output_dim, alpha=0.2):
        limit = np.sqrt(6 / (input_dim + output_dim))
        self.W = np.random.uniform(-limit, limit, (input_dim, output_dim))
        self.a = np.random.uniform(-limit, limit, (2 * output_dim, 1))
        self.alpha = alpha

        self.dW = np.zeros_like(self.W)
        self.da = np.zeros_like(self.a)

        self.input = None
        self.H = None
        self.attention = None
        self.e = None

    def leaky_relu(self, x):
        return np.where(x > 0, x, self.alpha * x)

    def leaky_relu_grad(self, x):
        return np.where(x > 0, 1, self.alpha)

    def forward(self, X, adj):
        self.input = X
        H = X @ self.W
        self.H = H
        N = H.shape[0]

        e = np.full((N, N), -np.inf)
        for i in range(N):
            for j in range(N):
                if adj[i, j] == 1 or i == j:
                    a_input = np.concatenate([H[i], H[j]])
                    e[i, j] = self.leaky_relu(np.dot(a_input, self.a).squeeze())
        self.e = e

        e_exp = np.exp(e - np.max(e, axis=1, keepdims=True))
        e_exp *= (adj + np.eye(N))
        attention = e_exp / (np.sum(e_exp, axis=1, keepdims=True) + 1e-10)
        self.attention = attention

        output = attention @ H
        return output

    def backward(self, grad_output, lr=0.01):
        N, out_dim = self.H.shape
        grad_H = np.zeros_like(self.H)
        self.dW = np.zeros_like(self.W)
        self.da = np.zeros_like(self.a)

        for i in range(N):
            for j in range(N):
                if self.attention[i, j] == 0:
                    continue
                # grad w.r.t attention
                grad_alpha = np.dot(grad_output[i], self.H[j])
                # grad w.r.t e[i, j] via softmax
                sum_exp = np.sum(np.exp(self.e[i] - np.max(self.e[i])))
                grad_eij = grad_alpha * self.attention[i, j] * (1 - self.attention[i, j])

                # leaky relu grad
                a_input = np.concatenate([self.H[i], self.H[j]])
                grad_leaky = self.leaky_relu_grad(np.dot(a_input, self.a).squeeze())

                self.da += grad_eij * grad_leaky * a_input.reshape(-1, 1)

                grad_hi = grad_output[i] * self.attention[i, j]
                grad_hj = grad_output[i] * self.attention[i, j]
                grad_H[i] += grad_hi
                grad_H[j] += grad_hj

        self.dW = self.input.T @ grad_H
        grad_input = grad_H @ self.W.T

        self.W -= lr * self.dW
        self.a -= lr * self.da

        return grad_input
