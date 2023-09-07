import pygame
import sys
import ssl
import matplotlib.pyplot as plt
import numpy as np 
ssl._create_default_https_context = ssl._create_unverified_context
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
threshold = 128
x_train = np.where(x_train >= threshold, 255, 0)
x_test = np.where(x_test >= threshold, 255, 0)
x_train = x_train / 255.0
x_test = x_test / 255.0
model = Sequential([tf.keras.layers.Flatten(), Dense(units=25, activation="relu"), 
Dense(units=15, activation="relu"), Dense(units=10, activation="softmax")])
model.compile(loss=SparseCategoricalCrossentropy())
model.fit(x_train, y_train, epochs=5)
pygame.init()
window = pygame.display.set_mode((560, 760))
pygame.display.set_caption('digit recognition')
value = "X"
surface1 = pygame.Surface((20, 20))
surface2 = pygame.Surface((20, 20))
surface3 = pygame.Surface((560, 200))
text = pygame.font.Font(None, 80)
surface_clear = text.render("CLEAR", True, 'white')
surface_predict = text.render("PREDICT", True, 'white')
surface_output = text.render(f"OUTPUT: {value}", True, 'white')
rect_clear = surface_clear.get_rect(center=(120, 600))
rect_predict = surface_clear.get_rect(center=(120, 680))
rect_output = surface_clear.get_rect(center=(350, 600))
surface2.fill('white')
surface3.fill((129, 122, 122))
rectangles = []
array_28x28 = np.zeros((28, 28))
for row in range(28):
    for column in range(28):
        rectangles.append(pygame.Rect(column * 20, row * 20, 20, 20))
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            if rect_clear.collidepoint(event.pos):
                array_28x28 = np.zeros((28, 28))
                value = "X"
            if rect_predict.collidepoint(event.pos):
                # plt.imshow(x_test[9])
                # print(np.argmax(model.predict(x_test[9:10])))
                # plt.show()
                tensor = array_28x28.reshape(1, 28, 28)
                value = np.argmax(model.predict(tensor))
    window.blit(surface3, (0, 560))
    window.blit(surface_clear, rect_clear)
    window.blit(surface_predict, rect_predict)
    window.blit(text.render(f"OUTPUT: {value}", True, 'white'), rect_output)
    mouse_button = pygame.mouse.get_pressed()
    mouse_pos = pygame.mouse.get_pos()
    if mouse_button[0]:
        for rectangle in rectangles:
                if rectangle.collidepoint(mouse_pos):
                    array_28x28[rectangle.y // 20, rectangle.x // 20] = 1
    for rectangle, type in zip(rectangles, array_28x28.flatten()):
        if type == 1:
            window.blit(surface2, rectangle)
        if type == 0:
            window.blit(surface1, rectangle)
    pygame.display.update()