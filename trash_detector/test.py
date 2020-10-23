import os
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform

class_names = ["clean", "trash"]
RESIZE_RES = (600, 600)
model = keras.models.load_model("trash_detector/trash_detector_model.h5")

# load testing data
for filename in os.listdir(f"trash_detector/testing_data"):
    if filename != ".DS_Store":
        image = io.imread(f"trash_detector/testing_data/{filename}", as_gray=True) / 255
        resized = transform.resize(image, RESIZE_RES)
        predictions = model.predict(np.expand_dims(resized, 0))

        # display image with prediction
        plt.subplot(1, 2, 1)
        plt.title(f"Prediciction Label: {class_names[np.argmax(predictions)]}")
        plt.xlabel(f"Confidence: {max(predictions)}")
        plt.imshow(image, cmap=plt.cm.binary)
        plt.subplot(1, 2, 2)
        plt.imshow(resized, cmap=plt.cm.binary)
        plt.show()
