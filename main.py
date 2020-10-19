import os
from tensorflow import keras
import numpy as np
import cv2
import face_recognition
from skimage import transform

TOLERANCE = 0.5
FRAME_THICKNESS = 2
FONT_THICKNESS = 2
MODEL = "hog"

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
    cap = cv2.VideoCapture(1)

    while True:
        _, cam_image = cap.read()

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
