import sys
import matplotlib
from matplotlib import pyplot as plt


import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop
from ga import PixelGA, LineGA
import pickle
from loguru import logger
import json

epochs = 6000
batch_size = 100


input_images = "./data/clock.npy"
data = np.load(input_images)
data = data/255
img_w, img_h = 28, 28
data = np.reshape(data, [data.shape[0], img_w, img_h, 1])

initial_population = np.zeros((batch_size, img_h, img_w, img_h, img_w)).astype(np.bool)
generator = LineGA(initial_population, batch_size, 0.8, 0.1)


def calculate_fitness(population):
    population = generator.in_pixel(population)
    fitness = discriminator.predict(population.astype(np.float32)).flatten()
    # fitness -= 0.5 * np.sum(population, axis=(1, 2, 3)) / population.shape[1] / population.shape[2]
    return fitness

generator.calculate_fitness = calculate_fitness

generator_d = generator.__dict__
generator_d["_parent_generator"] = None

print()
