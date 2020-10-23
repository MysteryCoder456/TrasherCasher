import os
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import face_recognition
from PIL import Image, ImageOps
from skimage import transform

trash_detector = load_model("trash_detector/trash_detector_model.h5")
class_names = ["clean", "trash"]

TOLERANCE = 0.5
FRAME_THICKNESS = 2
FONT_THICKNESS = 2
MODEL = "hog"
INVERT_PIC = True
RESIZE_RES = (224, 224)

face_recog_folder = "face_recognition"
known_faces = []
known_names = []

print("loading known faces...")
for face_name in os.listdir(f"{face_recog_folder}/known_faces"):
    if face_name != ".DS_Store":
        for filename in os.listdir(f"{face_recog_folder}/known_faces/{face_name}"):
            if filename != ".DS_Store":
                image = face_recognition.load_image_file(f"{face_recog_folder}/known_faces/{face_name}/{filename}")
                encoding = face_recognition.face_encodings(image)[0]
                known_faces.append(encoding)
                known_names.append(face_name)
print("finished loading known faces!")


def main():
    print("starting camera...")
    cap = cv2.VideoCapture(0)

    while True:
        _, cam_image = cap.read()

        # if INVERT_PIC:
        #     cam_image = transform.rotate(cam_image, 180)

        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        resized = transform.resize(cam_image, RESIZE_RES)  # ImageOps.fit(cam_image, RESIZE_RES, Image.ANTIALIAS)
        image_array = np.asarray(resized)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array

        trash_detected = class_names[np.argmax(trash_detector.predict(data))]
        print(trash_detected)

        if trash_detected == "trash":
            face_locations = face_recognition.face_locations(cam_image, model=MODEL)
            face_encodings = face_recognition.face_encodings(cam_image, face_locations)

            for f_enc, f_loc in zip(face_encodings, face_locations):
                results = face_recognition.compare_faces(known_faces, f_enc, TOLERANCE)

                if True in results:
                    match = known_names[results.index(True)]

                    top_left = (f_loc[3], f_loc[0])
                    bottom_right = (f_loc[1], f_loc[2])
                    cv2.rectangle(cam_image, top_left, bottom_right, (0, 255, 0), FRAME_THICKNESS)

                    print(f"Match found: {match}")

        cv2.imshow("Camera Footage", cam_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
