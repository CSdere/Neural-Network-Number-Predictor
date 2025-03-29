import numpy as np
import matplotlib.pyplot as plt

class NeuralNet:
    def __init__(self, w1, b1, w2, b2, w3, b3, lr=0.01):
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2
        self.w3 = w3
        self.b3 = b3
        self.lr = lr

    def activation(self, z1):
        return np.maximum(0, z1)

    def activation_deriv(self, z1):
        return (z1 > 0).astype(float)  # derivative of ReLU

    def softmax(self, z2):
        z_stable = z2 - np.max(z2, axis=1, keepdims=True)
        exp_z = np.exp(z_stable)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward_prop(self, X):
        self.z1 = X @ self.w1 + self.b1
        self.a1 = self.activation(self.z1)
        self.z2 = self.a1 @ self.w2 + self.b2
        self.a2 = self.activation(self.z2)
        self.z3 = self.a2 @ self.w3 + self.b3
        self.output = self.softmax(self.z3)
        return self.output

    def compute_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def backward_prop(self, X, y_true):
        m = X.shape[0]

        # Output layer gradient
        dZ3 = 2 * (self.output - y_true) / m  # (N, 10)
        dW3 = self.a2.T @ dZ3                 # (64, 10)
        db3 = np.sum(dZ3, axis=0, keepdims=True)  # (1, 10)

        # Second hidden layer
        dA2 = dZ3 @ self.w3.T                 # (N, 64)
        dZ2 = dA2 * self.activation_deriv(self.z2)  # (N, 64)
        dW2 = self.a1.T @ dZ2                 # (128, 64)
        db2 = np.sum(dZ2, axis=0, keepdims=True)  # (1, 64)

        # First hidden layer
        dA1 = dZ2 @ self.w2.T                 # (N, 128)
        dZ1 = dA1 * self.activation_deriv(self.z1)  # (N, 128)
        dW1 = X.T @ dZ1                       # (784, 128)
        db1 = np.sum(dZ1, axis=0, keepdims=True)  # (1, 128)

        # Gradient descent updates
        self.w1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.w2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.w3 -= self.lr * dW3
        self.b3 -= self.lr * db3

    def compute_accuracy(self, y_true, y_pred):
        true_labels = np.argmax(y_true, axis=1)
        pred_labels = np.argmax(y_pred, axis=1)
        accuracy = np.mean(true_labels == pred_labels)
        return accuracy






