import matplotlib
from matplotlib import pyplot as plt

matplotlib.interactive(True)

import pickle

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model

from ga import LineGA, PixelGA

input_images = "./data/clock.npy"
data = np.load(input_images)
data = data/255
img_w, img_h = 28, 28
data = np.reshape(data, [data.shape[0], img_w, img_h, 1])


batch_size = 100

discriminator = load_model('discriminator.h5')

initial_population = np.zeros((batch_size, img_h, img_w, img_h, img_w)).astype(np.bool)
generator = LineGA(initial_population, batch_size, 0.8, 0.1)

generator_d = {}
with open('generator.pickle', 'rb') as handle:
    generator_d = pickle.load(handle)

generator.population = generator_d["population"]
generator.fitness = generator_d["fitness"]
generator.offspring_num = generator_d["offspring_num"]
generator.mutation_probability = generator_d["mutation_probability"]
generator.crossover_probability = generator_d["crossover_probability"]

real_data = np.array(data[np.random.choice(len(data), batch_size * 2, replace=False)])
gen_imgs = generator.in_pixel(generator.select_best(9))

print(discriminator.predict(gen_imgs)[:3])
print(discriminator.predict(real_data)[:3])


r, c = 3, 3

# Rescale images 0 - 1
gen_imgs = 0.5 * gen_imgs + 0.5

fig, axs = plt.subplots(r, c)
cnt = 0
for i in range(r):
    for j in range(c):
        axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
        axs[i,j].axis('off')
        cnt += 1
fig.savefig("test.png")
plt.close()
