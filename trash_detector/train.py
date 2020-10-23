import os
import random
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, transform

print("Using Tensorflow Version:", tf.__version__)

RESIZE_RES = (600, 600)

class_names = ["clean", "trash"]

training_data = []
training_labels = []

# load training data
for class_name in os.listdir("trash_detector/training_data"):
    if class_name != ".DS_Store":
        for filename in os.listdir(f"trash_detector/training_data/{class_name}"):
            if filename != ".DS_Store":
                image = io.imread(f"trash_detector/training_data/{class_name}/{filename}", as_gray=True) / 255
                resized = transform.resize(image, RESIZE_RES)
                training_data.append(resized)
                training_labels.append(class_names.index(class_name))

# shuffle around the data
shuffle_seed = random.randint(1, 10)

random.seed(shuffle_seed)
random.shuffle(training_data)

random.seed(shuffle_seed)
random.shuffle(training_labels)

# view training images
# for i in range(len(training_data)):
#     plt.title(class_names[training_labels[i]])
#     plt.imshow(training_data[i])
#     plt.show()

# create model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=RESIZE_RES),
    keras.layers.Dense(265, activation="relu"),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(2, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# train and save model
model.fit(np.array(training_data), np.array(training_labels), epochs=5, batch_size=3)
model.save("trash_detector/trash_detector_model.h5")
