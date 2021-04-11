import streamlit as st
import numpy as np
import plotly.express as px
from typing import Callable


class MSE:
    def __call__(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return ((y_pred - y_true) ** 2).mean()

    def backward(self):
        n = self.y_true.shape[0]
        self.gradient = 2. * (self.y_pred - self.y_true) / n
        # print('MSE backward', self.y_pred.shape, self.y_true.shape, self.gradient.shape)
        return self.gradient


class Linear:
    def __init__(self, input_dim: int, num_hidden: int = 1):
        self.weights = np.random.randn(input_dim, num_hidden) * np.sqrt(2. / input_dim)
        self.bias = np.zeros(num_hidden)

    def __call__(self, x):
        self.x = x
        output = x @ self.weights + self.bias
        return output

    def backward(self, gradient):
        self.weights_gradient = self.x.T @ gradient
        self.bias_gradient = gradient.sum(axis=0)
        self.x_gradient = gradient @ self.weights.T
        return self.x_gradient

    def update(self, lr):
        self.weights = self.weights - lr * self.weights_gradient
        self.bias = self.bias - lr * self.bias_gradient


def plot_3d(x, y_true, y_pred=None, epochs=None):
    x = np.tile(x, (20, 1))
    y_true = np.tile(y_true, (20, 1))
    epochs = np.repeat(np.array(range(20)), 100)

    if y_pred is not None:
        label_underlying = np.tile(np.array(['Target Function']), x.shape[0])
        label_our = np.tile(np.array(['Optimized Function']), x.shape[0])
        label = np.append(label_underlying, label_our)
        y = np.append(y_true, y_pred)
        fig = px.scatter_3d(x=np.tile(x[:, 0], 2),
                            y=np.tile(x[:, 1], 2),
                            z=y, color=label, animation_frame=np.tile(epochs, 2),
                            labels={"color": "label", "animation_frame": "Epoch"})
    else:
        label = np.tile(np.array(['Target Function']), x.shape[0])
        fig = px.scatter_3d(x=x[:, 0], y=x[:, 1], z=y_true, color=label)
    return fig


def fit(x: np.ndarray, y: np.ndarray, model: Callable, loss: Callable, lr: float, num_epochs: int):
  y_pred_array = np.array([])
  for epoch in range(num_epochs):
    y_pred = model(x)
    y_pred_array = np.append(y_pred_array, y_pred)
    loss_value = loss(y_pred, y)
    print(f'Epoch {epoch}, loss {loss_value}')
    gradient_from_loss = loss.backward()
    model.backward(gradient_from_loss)
    model.update(lr)
  return y_pred_array


def main():
    n = 100
    d = 2
    x = np.random.uniform(-2, 2, (n, d))

    st.title("Explore Gradient Descent")
    st.write("Let's use gradient descent to learn the weights and bias that minimizes the loss function."
             "For this, we need the gradient of the loss function and the gradients of the linear function.")
    weights_true = np.array([[4, -4], ]).T
    bias_true = np.array([0.5])

    y_true = x @ weights_true + bias_true

    epochs = 20
    loss = MSE()
    linear = Linear(2)
    y_pred_array = fit(x,
                       y_true,
                       model=linear,
                       loss=loss,
                       lr=0.1,
                       num_epochs=epochs)

    fig = plot_3d(x, y_true, y_pred_array, epochs)
    st.plotly_chart(fig)

if __name__ == '__main__':
    main()