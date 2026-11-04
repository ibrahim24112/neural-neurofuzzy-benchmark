from __future__ import annotations

import math

import numpy as np


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    encoded = np.zeros((len(y), n_classes), dtype=float)
    encoded[np.arange(len(y)), y] = 1.0
    return encoded


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


class MLPClassifier:
    def __init__(
        self,
        hidden_layers: tuple[int, ...],
        activation: str,
        learning_rate: float,
        epochs: int,
        batch_size: int,
        l2_penalty: float,
        seed: int,
    ) -> None:
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2_penalty = l2_penalty
        self.seed = seed
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []

    def fit(self, X: np.ndarray, y: np.ndarray, n_classes: int) -> "MLPClassifier":
        self._initialize_parameters(X.shape[1], n_classes)
        rng = np.random.default_rng(self.seed + 7919)
        y_encoded = one_hot(y, n_classes)
        first_moment_w = [np.zeros_like(weight) for weight in self.weights]
        second_moment_w = [np.zeros_like(weight) for weight in self.weights]
        first_moment_b = [np.zeros_like(bias) for bias in self.biases]
        second_moment_b = [np.zeros_like(bias) for bias in self.biases]
        beta_1, beta_2, epsilon = 0.9, 0.999, 1e-8
        step = 0

        for _ in range(self.epochs):
            order = rng.permutation(len(X))
            for start in range(0, len(order), self.batch_size):
                batch_index = order[start : start + self.batch_size]
                gradients_w, gradients_b = self._gradients(X[batch_index], y_encoded[batch_index])
                step += 1
                for index in range(len(self.weights)):
                    first_moment_w[index] = beta_1 * first_moment_w[index] + (1 - beta_1) * gradients_w[index]
                    second_moment_w[index] = beta_2 * second_moment_w[index] + (1 - beta_2) * gradients_w[index] ** 2
                    first_moment_b[index] = beta_1 * first_moment_b[index] + (1 - beta_1) * gradients_b[index]
                    second_moment_b[index] = beta_2 * second_moment_b[index] + (1 - beta_2) * gradients_b[index] ** 2
                    weight_update = first_moment_w[index] / (1 - beta_1 ** step) / (np.sqrt(second_moment_w[index] / (1 - beta_2 ** step)) + epsilon)
                    bias_update = first_moment_b[index] / (1 - beta_1 ** step) / (np.sqrt(second_moment_b[index] / (1 - beta_2 ** step)) + epsilon)
                    self.weights[index] -= self.learning_rate * weight_update
                    self.biases[index] -= self.learning_rate * bias_update
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        activation = X
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            activation = self._activate(activation @ weight + bias)
        return softmax(activation @ self.weights[-1] + self.biases[-1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def _initialize_parameters(self, n_features: int, n_classes: int) -> None:
        rng = np.random.default_rng(self.seed)
        layer_sizes = [n_features, *self.hidden_layers, n_classes]
        self.weights = []
        self.biases = []
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            scale = math.sqrt(2.0 / fan_in) if self.activation == "relu" else math.sqrt(1.0 / fan_in)
            self.weights.append(rng.normal(0.0, scale, size=(fan_in, fan_out)))
            self.biases.append(np.zeros(fan_out, dtype=float))

    def _gradients(self, X: np.ndarray, y_encoded: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        activations = [X]
        pre_activations = []
        activation = X
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            z = activation @ weight + bias
            pre_activations.append(z)
            activation = self._activate(z)
            activations.append(activation)

        probabilities = softmax(activations[-1] @ self.weights[-1] + self.biases[-1])
        delta = (probabilities - y_encoded) / len(X)
        gradients_w: list[np.ndarray] = [np.empty_like(weight) for weight in self.weights]
        gradients_b: list[np.ndarray] = [np.empty_like(bias) for bias in self.biases]
        gradients_w[-1] = activations[-1].T @ delta + self.l2_penalty * self.weights[-1]
        gradients_b[-1] = delta.sum(axis=0)

        for layer in range(len(self.weights) - 2, -1, -1):
            delta = (delta @ self.weights[layer + 1].T) * self._activation_gradient(pre_activations[layer])
            gradients_w[layer] = activations[layer].T @ delta + self.l2_penalty * self.weights[layer]
            gradients_b[layer] = delta.sum(axis=0)
        return gradients_w, gradients_b

    def _activate(self, value: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return np.maximum(0.0, value)
        if self.activation == "tanh":
            return np.tanh(value)
        raise ValueError(f"Unsupported activation: {self.activation}")

    def _activation_gradient(self, value: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return (value > 0.0).astype(float)
        if self.activation == "tanh":
            activated = np.tanh(value)
            return 1.0 - activated * activated
        raise ValueError(f"Unsupported activation: {self.activation}")
