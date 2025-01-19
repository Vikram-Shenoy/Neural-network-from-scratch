import numpy as np

inputs = [[1, 2, 3, 2.5],
          [2, 5, -1, 2],
          [-1.5, 2.7, 3.3, -0.8]]

weights = [ [0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]
]


biases= [2,3,0.5]

# output = np.dot(weights, inputs) + biases # np.dot(inputs,weights) doesn't work, because shape(4,) x shape (3,4) can't multiply

output = np.dot(inputs, np.array(weights).T) + biases

print(output)
