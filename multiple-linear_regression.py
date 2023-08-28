"""
Multiple linear regression
Program which minimize loss function with the precision of designated iteration number.
"""
import numpy as np

target_value = np.array([1, 23, 4, 11])
feature_value_1 = np.array([2, 5, 3, 3])
feature_value_2 = np.array([1, 10, 4, 5])
weight_and_bias = np.array([1, 1, 1], dtype=float)
learning_rate = 0.01
iterations = 1000000


# loss function
def cost_function(weight_and_bias, target_value, feature_value_1, feature_value_2):
    sum = 0
    for x in range(len(feature_value_1)):
        sum += (weight_and_bias[0] * feature_value_1[x] + weight_and_bias[1] * feature_value_2[x] + weight_and_bias[2] -
                target_value[x]) ** 2
    sum /= (2 * len(feature_value_1))
    return sum


# partial derivative in respect of w1
def derivative_w1(weight_and_bias, target_value, feature_value_1, feature_value_2):
    sum = 0
    for x in range(len(feature_value_1)):
        sum += (weight_and_bias[0] * feature_value_1[x] + weight_and_bias[1] * feature_value_2[x] + weight_and_bias[2] -
                target_value[x]) * feature_value_1[x]
    sum /= len(feature_value_1)
    return sum


# partial derivative in respect of w2
def derivative_w2(weight_and_bias, target_value, feature_value_1, feature_value_2):
    sum = 0
    for x in range(len(feature_value_1)):
        sum += (weight_and_bias[0] * feature_value_1[x] + weight_and_bias[1] * feature_value_2[x] + weight_and_bias[2] -
                target_value[x]) * feature_value_2[x]
    sum /= len(feature_value_1)
    return sum


# partial derivative in respect of b
def derivative_b(weight_and_bias, target_value, feature_value_1, feature_value_2):
    sum = 0
    for x in range(len(feature_value_1)):
        sum += (weight_and_bias[0] * feature_value_1[x] + weight_and_bias[1] * feature_value_2[x] + weight_and_bias[2] -
                target_value[x])
    sum /= (len(feature_value_1))
    return sum


# gradient descend
for x in range(iterations):
    weight_and_bias[0] -= learning_rate * derivative_w1(weight_and_bias, target_value, feature_value_1, feature_value_2)
    weight_and_bias[1] -= learning_rate * derivative_w2(weight_and_bias, target_value, feature_value_1, feature_value_2)
    weight_and_bias[2] -= learning_rate * derivative_b(weight_and_bias, target_value, feature_value_1, feature_value_2)
print(weight_and_bias)