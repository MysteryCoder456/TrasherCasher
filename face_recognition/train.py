import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt

root_folder = "face_recognition"

training_data = []
training_labels = []

class_names = []
class_index = 0

for class_name in os.listdir(f"{root_folder}/training_data"):
    if class_name != ".DS_Store":
        class_names.append(class_name)
        for filename in os.listdir(f"{root_folder}/training_data/{class_name}"):
            if filename != ".DS_Store":
                image = io.imread(f"{root_folder}/training_data/{class_name}/{filename}", as_gray=True)
                resized = transform.resize(image, (200, 300))
                training_data.append(resized)
                training_labels.append(class_index)
        class_index += 1

class_names_fname = f"{root_folder}/class_names.txt"

# clear the file
with open(class_names_fname, "w") as _:
    print("Clearing class names file...")

with open(class_names_fname, "a") as class_names_file:
    for name in class_names:
        class_names_file.write(name + "\n")

# shows all training images
# for i in range(len(training_data)):
#     plt.xlabel(training_labels[i])
#     plt.imshow(training_data[i])
#     plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(200, 300)),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(512, activation="relu"),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(512, activation="relu"),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(len(class_names), activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# model.fit(training_data, training_labels, epochs=5)
model.fit(np.array(training_data), np.array(training_labels), epochs=50)
model.save(f"{root_folder}/face_recog_model.h5")
