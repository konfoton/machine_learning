'''
MNIST DATA SET NEURAL NETWORK FROM SCRATCH
architecture 784x16x16x10
credits to 3blue1brown for beautiful videos on this subcject
Author: Konrad Burdach
'''
import functionsml as ml
#defining constants
train = "mnist_train.csv"
test = "mnist_test.csv"
architecture = [784, 16, 16, 10]
data_train = ml.read(train)
ml.preprocess(data_train)
data_test = ml.read(test)
ml.preprocess(data_test)
epoch  = 3
number_of_tests = 1000
#defining dynamic variables
learning_rate = 0.01
weights = ml.generate_weights(architecture)
biases = ml.generate_biases(architecture)
delta_weights = ml.generate_delta_weights(architecture)
delta_weights_f = ml.generate_delta_weights(architecture)
delta_biases = ml.generate_delta_biases(architecture)
delta_biases_f = ml.generate_delta_biases(architecture)
#feedforward without answer
def feedforward(activation):
    activations = [activation]
    for w, b in zip(weights, biases):
        activation = ml.add_mat(ml.matmul(activation, ml.transpose(w)), b)
        activations.append(activation)
        ml.sigmoid_matrix(activation)
    max_index = -1
    max_value = -1
    for index, pred in enumerate(activation):
        if pred > max_value:
            max_value = pred
            max_index = index
    return [max_index, activations]
#backprop
print(biases)
def backprop(example):
    information = feedforward(example[1:])
    cost = [0 for x in range(10)]
    cost[example[0]] = -1
    temp_derivative =  ml.mulc(ml.add_mat(ml.sigmoid_matrix_new(information[1][-1]), cost), 2)
    for x in range(1, len(delta_weights) + 1):
        for index, y in enumerate(information[1][-x]):
            temp_derivative[index] *= ml.sigmoid_d(y)
        for index1, a1 in enumerate(temp_derivative):
            delta_biases[-x][index1] = a1
            for index2, a2 in enumerate(information[1][-x-1]):
                delta_weights[-x][index1][index2] = a1 * ml.sigmoid_function(a2)
        temp_derivative = ml.matmul(temp_derivative, weights[-x])
# creating mini-batches
# learning
for x in range(epoch):
    ml.random.shuffle(data_train)
    k = 0
    for example in data_train:
        backprop(example)
        ml.add_matrixw(delta_weights, delta_weights_f)
        ml.add_matrixb(delta_biases, delta_biases_f)
        if k % 500 == 0:    
            ml.matcb(delta_biases_f, -1/100 * learning_rate)
            ml.matcw(delta_weights_f, -1/100 * learning_rate)
            ml.add_matrixw(delta_weights_f, weights)
            ml.add_matrixb(delta_biases_f, biases)
            print("one mini batch")
            print("\n\n")
            print(biases)
            print("\n\n")
            delta_weights_f = ml.generate_delta_weights(architecture)
            delta_biases_f = ml.generate_delta_biases(architecture)
        k += 1
#testing model
correct = 0
for test in data_test[0:number_of_tests]:
    if test[0] == feedforward(test[1:])[0]:
        correct += 1
print(correct / number_of_tests * 100)