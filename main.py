import json
import pickle
import os
import sys

import matplotlib
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

from loguru import logger
from matplotlib import pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop

from ga import LineGA, PixelGA

logger.remove()

logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS!UTC}</green> | <level>{level: <8}</level> |"
    " <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)

logger.add(
    "train.log",
    backtrace=True,
    diagnose=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS!UTC}</green> | <level>{level: <8}</level> |"
    " <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)


def make_discriminator_model():
    depth = 64
    p = 0.4
    model = tf.keras.Sequential()

    model.add(
        layers.Conv2D(
            depth * 1,
            5,
            strides=2,
            padding="same",
            activation="relu",
            input_shape=(28, 28, 1),
        )
    )
    model.add(layers.Dropout(p))

    model.add(layers.Conv2D(depth * 2, 5, strides=2, padding="same", activation="relu"))
    model.add(layers.Dropout(p))

    model.add(layers.Conv2D(depth * 4, 5, strides=2, padding="same", activation="relu"))
    model.add(layers.Dropout(p))

    model.add(layers.Conv2D(depth * 8, 5, strides=1, padding="same", activation="relu"))
    model.add(layers.Dropout(p))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="sigmoid"))
    model.summary()
    return model


epochs = 6000
batch_size = 100


input_images = "./data/sheep.npy"
data = np.load(input_images)
data = data / 255
img_w, img_h = 28, 28
data = np.reshape(data, [data.shape[0], img_w, img_h, 1])

# initial_population = np.random.randint(2, size=(batch_size, img_w, img_h, 1))
# initial_population = np.random.rand(batch_size, img_w, img_h, 1)
# initial_population = np.zeros((batch_size, img_w, img_h, 1))

initial_population = np.zeros((batch_size, img_h, img_w, img_h, img_w)).astype(np.bool)
generator = LineGA(initial_population, batch_size, 0.8, 0.1)


discriminator = make_discriminator_model()
discriminator.compile(
    optimizer=RMSprop(lr=0.0008), loss="binary_crossentropy", metrics=["accuracy"]
)


def calculate_fitness(population):
    population = generator.in_pixel(population)
    fitness = discriminator.predict(population.astype(np.float32)).flatten()
    # fitness -= 0.5 * np.sum(population, axis=(1, 2, 3)) / population.shape[1] / population.shape[2]
    return fitness


generator.calculate_fitness = calculate_fitness
es = EarlyStopping(monitor="loss", mode="min", verbose=1, patience=5)
min_generation = 10

for epoch in range(epochs):

    real_data = np.array(
        data[np.random.choice(len(data), batch_size * 2, replace=False)]
    )
    fake_data = generator.in_pixel(generator.select_best(batch_size)).astype(np.float32)
    fake_data = np.concatenate(
        (fake_data, np.random.randint(2, size=(batch_size, img_w, img_h, 1)))
    )
    x = np.concatenate((real_data, fake_data))
    y = np.concatenate((np.ones(len(real_data)), 0 * np.ones(len(fake_data))))

    result = discriminator.evaluate(x, y, verbose=0)
    # train only when accuracy is less than X
    if result[1] < 0.95:
        result = discriminator.train_on_batch(x, y)

    if (epoch + 1) % 1 == 0:
        logger.info("#", epoch)
        logger.info(result)
        discriminator.save("discriminator.h5")
        logger.info("Epoch #{}".format(epoch + 1))
        logger.info(result)

        generator_d = generator.__dict__.copy()
        generator_d["_parent_generator"] = None
        generator_d["calculate_fitness"] = None

        with open("generator.pickle", "wb") as handle:
            pickle.dump(generator_d, handle, protocol=pickle.HIGHEST_PROTOCOL)

        gen_imgs = generator.in_pixel(generator.select_best(9))

        logger.info(discriminator.predict(gen_imgs)[:3])
        logger.info(discriminator.predict(real_data)[:3])

        r, c = 3, 3

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap="gray")
                axs[i, j].axis("off")
                cnt += 1
        fig.savefig("./images/%d.png" % (epoch + 1))
        plt.close()

    generator.breed(min_generation)
