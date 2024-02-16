import math
import random
import csv
def transpose(matrix):
        transposed_matrix = [[] for x in range(len(matrix[0]))]
        for x in matrix:
            for index, y in enumerate(x):
                transposed_matrix[index].append(y)
        return transposed_matrix

def dot_product(a, b):
        counter = 0
        for x, y in zip(a, b):
            counter += x * y
        return counter

def matmul(matrixa, matrixb):
        if isinstance(matrixa[0], int) or isinstance(matrixa[0], float):
             matrixa = [matrixa]
        elif isinstance(matrixb[0], int) or isinstance(matrixb[0], float):
             matrixb = [matrixb]
        result = []
        try:
            for x in matrixa:
                temp = []
                for y in transpose(matrixb):
                    temp.append(dot_product(x, y))
                result.append(temp)
            if len(result) == 1:
                 return result[0]
            return result
        except:
            print("something went wrong")

def sigmoid_function(value):
     return 1 / (1 + math.e ** -(value))

def generate_weights(architecture):
     weights = []
     for x in range(len(architecture) - 1):
          weights.append([[random.randint(1, 100) / 1000 for z in range(architecture[x])] for y in range(architecture[x + 1])])
     return weights

def generate_biases(architecture):
     bias = []
     for y in architecture[1:]:
          bias.append([random.randint(1, 100) / 1000 for z in range(y)])
     return bias

def read(adress):
     data = []
     with open(adress, "r") as file:
          csvreader = csv.reader(file)
          next(csvreader)
          for row in csvreader:
               temp = []
               for element in row:
                    temp.append(int(element))
               data.append(temp)
     return data 

def generate_delta_biases(architecture):
     bias = []
     for y in architecture[1:]:
          bias.append([0 for z in range(y)])
     return bias

def generate_delta_weights(architecture):
     weights = []
     for x in range(len(architecture) - 1):
          weights.append([[0 for z in range(architecture[x])] for y in range(architecture[x + 1])])
     return weights   

def add_mat(lista, listb):
     result = []
     for x, y in zip(lista, listb):
          result.append(x + y)
     return result
def sigmoid_matrix(matrix):
     for x in range(len(matrix)):
          matrix[x] = sigmoid_function(matrix[x])

def preprocess(matrix):
     for index, x in enumerate(matrix):
          for indexy, y in enumerate(x[1:]):
               matrix[index][indexy + 1] = y/255

def sigmoid_matrix_new(matrix):
     result = []
     for element in matrix:
          result.append(sigmoid_function(element))
     return result

def mulc(matrix, c):
     result = []
     for element in matrix:
          result.append(element * c)
     return result

def sigmoid_d(value):
    return sigmoid_function(value) * (1 - sigmoid_function(value))

def add_matrixw(matrixa, matrixb):
    for index1, x in enumerate(matrixa):
         for index2, y in enumerate(x):
              for index3, z in enumerate(y):
                   matrixb[index1][index2][index3] += z

def matcw(matrix, c):
     for index1, x in enumerate(matrix):
         for index2, y in enumerate(x):
              for index3, z in enumerate(y):
                   matrix[index1][index2][index3] *= c

def add_matrixb(matrixa, matrixb):
    for index1, x in enumerate(matrixa):
         for index2, y in enumerate(x):
              matrixb[index1][index2] += y

def matcb(matrix, c):
     for index1, x in enumerate(matrix):
         for index2, y in enumerate(x):
              matrix[index1][index2] *= c

if __name__ == "__main__":
    print(transpose([[1, 2]]))
    print(transpose([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    print(dot_product([1, 2, 3], [4, 5, 6]))
    print(matmul([[1, 4], [2, 5], [3, 6]], [[1, 2], [3, 4]]))
    print(generate_weights([5, 10, 3]))
    print(generate_biases([5, 10, 3]))
    print(add_mat([3, 4, 5], [-2, 6, 8]))
    print("\n\n\n")
    print(matmul([1, 2, 3, 4], [[1, 2], [3, 4], [5, 6], [7, 8]]))
    print(mulc([1, 2, 3], 5))
    print("\n\n\n")
    matrix = [[[1, 2], [3, 4]], [[1, 2], [3, 4], [5, 6]]]
    matrix1 = [[[1, 2], [3, 4]], [[1, 2], [3, 4], [5, 20]]]
    bias = [[1, 2, 3], [5, 6, 7]]
    bias1 = [[1, 2, 3], [5, 6, 7]]
    add_matrixw(matrix, matrix1)
    add_matrixb(bias, bias1)
    print(matrix1)
    print(bias1)
    # test = "/Users/konradburdach/Desktop/Mini pw/archive/mnist_test.csv"
    # print(read(test))

