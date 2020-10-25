import os
import json
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import face_recognition
from PIL import Image, ImageOps
from skimage import transform

from emid import EmiratesID

trash_detector = load_model("trash_detector/trash_detector_model.h5")
class_names = ["clean", "trash"]

TOLERANCE = 0.5
FRAME_THICKNESS = 2
FONT_THICKNESS = 2
MODEL = "hog"
INVERT_PIC = False
RESIZE_RES = (224, 224)
DISPLAY_IMAGE = True

face_recog_folder = "face_recognition"
known_faces = []
known_names = []
emirates_id_list = []

with open("fines.json", "r") as fines_file:
    fines = json.load(fines_file)

print("Loading known faces...")

for face_name in os.listdir(f"{face_recog_folder}/known_faces"):
    if face_name != ".DS_Store":
        for filename in os.listdir(f"{face_recog_folder}/known_faces/{face_name}"):
            if filename != ".DS_Store":
                if filename == "emid.txt":
                    with open(f"{face_recog_folder}/known_faces/{face_name}/emid.txt", "r") as emid_file:
                        id_number = int(emid_file.readlines()[0])

                        emirates_id = EmiratesID(id_number, face_name)
                        if str(id_number) in fines.keys():
                            emirates_id.fine_amount = fines.get(str(id_number))

                        emirates_id_list.append(emirates_id)
                        print(f"Created Emirates ID entry for {face_name} with ID number {id_number}!")
                else:
                    image = face_recognition.load_image_file(f"{face_recog_folder}/known_faces/{face_name}/{filename}")
                    encoding = face_recognition.face_encodings(image)[0]
                    known_faces.append(encoding)
                    known_names.append(face_name)

print("Finished loading known faces!")


def exit_sequence(video_capture):
    video_capture.release()
    cv2.destroyAllWindows()

    fines_dict = {}
    with open("fines.json", "w") as fines_file:
        for e in emirates_id_list:
            fines_dict[e.id] = e.fine_amount

        json.dump(fines_dict, fines_file)


def main():
    print("Starting camera...")
    cap = cv2.VideoCapture(0)
    previous_trash_label = "trash"

    try:
        while True:
            _, cam_image = cap.read()

            if INVERT_PIC:
                cam_image = transform.rotate(cam_image, 180)

            cv2.imwrite("temp.jpg", cam_image)

            pred_image = Image.open("temp.jpg")
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            resized = ImageOps.fit(pred_image, RESIZE_RES, Image.ANTIALIAS)
            image_array = np.asarray(resized)
            normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
            data[0] = normalized_image_array

            trash_detected = trash_detector.predict(data)
            trash_label = class_names[np.argmax(trash_detected)]

            if DISPLAY_IMAGE:
                cv2.putText(cam_image, trash_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, FONT_THICKNESS)
            else:
                print(trash_label)

            if trash_label == "trash" and previous_trash_label == "clean":
                face_locations = face_recognition.face_locations(cam_image, model=MODEL)
                face_encodings = face_recognition.face_encodings(cam_image, face_locations)

                for f_enc, f_loc in zip(face_encodings, face_locations):
                    results = face_recognition.compare_faces(known_faces, f_enc, TOLERANCE)

                    if True in results:
                        match = known_names[results.index(True)]

                        if DISPLAY_IMAGE:
                            top_left = (f_loc[3], f_loc[0])
                            bottom_right = (f_loc[1], f_loc[2])
                            cv2.rectangle(cam_image, top_left, bottom_right, (0, 255, 0), FRAME_THICKNESS)
                            cv2.putText(cam_image, match, (top_left[0], bottom_right[1] + 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), FONT_THICKNESS)

                        print(f"Match found: {match}")

                        for e in emirates_id_list:
                            if e.name == match:
                                e.fine_amount += 3000  # Apply fine

            if DISPLAY_IMAGE:
                cv2.imshow("Camera Footage", cam_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            previous_trash_label = trash_label

    except KeyboardInterrupt:
        exit_sequence(cap)

    exit_sequence(cap)


if __name__ == "__main__":
    main()
