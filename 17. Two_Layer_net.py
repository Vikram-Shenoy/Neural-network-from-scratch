import numpy as np
from numpy.random import randn

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x, y = randn(N, D_in), randn(N, D_out)

# Randomly initialize weights
w1, w2 = randn(D_in, H), randn(H, D_out)

learning_rate = 1e-6
for t in range(10000):
    # Forward pass: compute predicted y
    h = 1 / (1 + np.exp(-x.dot(w1)))
    y_pred = h.dot(w2)

    # Compute and print loss
    loss = np.square(y_pred - y).sum()
    if t % 100 == 0:
        print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h.T.dot(grad_y_pred)
    grad_h = grad_y_pred.dot(w2.T)
    grad_w1 = x.T.dot(grad_h * h * (1 - h))

    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2