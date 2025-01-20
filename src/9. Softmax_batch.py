import math
import nnfs
import numpy as np
nnfs.init()


layer_output = [[4.8, 1.21, 2.385],
                [8.9, -1.81, 0.2],
                [1.41, 1.051, 0.026]]

exp_values = np.exp(layer_output)
# print(exp_values) 

#print(np.sum(layer_output, axis=1, keepdims=True)) #keepdims is a boolean type parameter, if this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array

norm_values = exp_values/ np.sum(exp_values, axis=1, keepdims=True) 
print(norm_values)
# print(norm_values)
# print(sum(norm_values))