import os
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

class_names = ["clean", "trash"]
RESIZE_RES = (224, 224)
model = keras.models.load_model("trash_detector/trash_detector_model.h5")
folder_size = len(os.listdir("trash_detector/testing_data"))
data = np.ndarray(shape=(folder_size - 1, 224, 224, 3), dtype=np.float32)

# load testing data
index = 0
for filename in os.listdir("trash_detector/testing_data"):
    if filename != ".DS_Store":
        image = image = Image.open(f'trash_detector/testing_data/{filename}')
        resized = ImageOps.fit(image, RESIZE_RES, Image.ANTIALIAS)
        image_array = np.asarray(resized)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[index] = normalized_image_array
        index += 1


predictions = model.predict(data)

for i in range(len(predictions)):
    # display image with prediction
    plt.title(f"Prediciction Label: {class_names[np.argmax(predictions[i])]}")
    plt.xlabel(f"Confidence: {max(predictions[i])}")
    plt.imshow(data[i])
    plt.show()
