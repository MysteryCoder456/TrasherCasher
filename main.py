from tensorflow import keras
import numpy as np
import cv2
from skimage import transform

class_names = ["Rehatbir Singh", "Keanu Reeves", "Navneet Kaur"]
face_recog = keras.models.load_model("face_recognition/face_recog_model.h5")

cap = cv2.VideoCapture(0)
face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_classifier_alt = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

while True:
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(frame, minSize=(120, 120), minNeighbors=4)

    for (x, y, w, h) in faces:
        cropped = frame[x:x + w, y:y + h]
        cropped = transform.resize(frame, (200, 300))
        prediction = face_recog.predict(np.expand_dims(cropped, 0))
        print(class_names[np.argmax(prediction)])
        cv2.rectangle(frame, (x, y), (x + w, y + h), 3, 3)

    cv2.imshow("Camera Footage", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
