# Neural-network-from-scratch

> “What I cannot create, I do not understand” - Richard Feynman.

This repository contains mini Python scripts that build neural networks from scratch using core mathematical principles, without relying on any deep learning libraries.
- This will be done first using arrays and manual dot product.
- Transitioning to using NumPy (because it makes matrix multiplication so much easier).
- Writing loss functions and activation functions using their core math principles and so on.

## Optimization

> Random optimization: Tweaking the weights and biases by a small number, measuring loss and keeping the best weights and biases over many iterations. This technique works for simple vertical data but does not work well when the problem is more complex, i.e fitting spiral data.

## Backpropagation

> Backpropagation: Instead of randomly tweaking weights, this method uses calculus (the chain rule) to compute the gradient of the loss with respect to each weight and bias. The network performs a forward pass to compute predictions, calculates the loss, then runs a backward pass to determine gradients. Finally, weights are updated using gradient descent. This makes training efficient and scalable, allowing neural networks to learn complex non-linear patterns such as spirals, images, or natural language.

