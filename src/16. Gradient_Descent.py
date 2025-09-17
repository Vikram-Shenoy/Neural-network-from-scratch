# Gradient Descent Example

# Loss function: L(w) = (w - 3)^2
# Derivative: dL/dw = 2 * (w - 3)

# Initial values
w = 0.0                # starting weight
learning_rate = 0.1    # step size if step size = 1, 
# we jitter if it is 2 we bounce out of the valley 
# Find the optimal step size for least iterations and max convergence to true minimum
tolerance = 1e-2       # stop when loss is ~0
iteration = 0

while True:
    # Compute loss and gradient
    loss = (w - 3) ** 2
    grad = 2 * (w - 3)
    temp = w
    # Stopping condition
    if loss < tolerance:
        break
    
    # Update rule: w = w - lr * grad
    w = temp - learning_rate * grad
    # Print current status    
    print(f"Iter {iteration}: w = {temp:.6f}, Loss = {loss:.6f}, Grad = {grad:.6f}, Updated w = {w:.6f}")
    iteration += 1
