import math 
layer_output = [4.8, 1.21, 2.385]


# E = 2.71828182846
E = math.e

exp_values = []

for output in layer_output:
    exp_values.append(E**output)

print(exp_values)

norm_base = sum(exp_values)
norm_values = []
for val in exp_values:
    norm_values.append(val/norm_base)
print(norm_values)
print(sum(norm_values))


import nnfs
import numpy as np
nnfs.init()


layer_output = [4.8, 1.21, 2.385]



# E = 2.71828182846
E = math.e

exp_values = np.exp(layer_output)


norm_base = sum(exp_values)
norm_values = exp_values/ sum(exp_values)

print(norm_values)
print(sum(norm_values))