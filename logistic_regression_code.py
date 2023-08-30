"""
Code for logistic regression without regularization
"""
import numpy as np

feature = np.array([7, 8, 9, 10, 1, 3, 5.5, 0.5])
target = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=float)
w = 1
b = 1
iterations = 100000
learning_rate = 0.01

def sigmoid(w, b, x):
    return 1 / (1 + np.e ** -(w * x + b))


def derivative_weight(feature, target, w, b):
    sum = 0
    for x in range(len(feature)):
        sum += (sigmoid(w, b, feature[x]) - target[x]) * feature[x]
    sum /= len(target)
    return sum


def derivative_bias(feature, target, w, b):
    sum = 0
    for x in range(len(feature)):
        sum += (sigmoid(w, b, feature[x]) - target[x])
    sum /= len(target)
    return sum

# gradient descent
for x in range(iterations):
    w = w - learning_rate * derivative_weight(feature, target, w, b)
    b = b - learning_rate * derivative_bias(feature, target, w, b)
print(w, b)
