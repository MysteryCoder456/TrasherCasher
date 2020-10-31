import time
import datetime
import os
import json
import numpy as np
import cv2
import face_recognition
from skimage import transform

from RPi import GPIO
from twilio.rest import Client

from emid import EmiratesID

# trash_detector = load_model("trash_detector/trash_detector_model.h5")
# class_names = ["clean", "trash"]

# Load Yolo
net = cv2.dnn.readNet("trash_detector/yolov3_training_last.weights", "trash_detector/yolov3_testing.cfg")
class_names = ["trash"]

TOLERANCE = 0.5
FRAME_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_THICKNESS = 2
MODEL = "hog"
INVERT_PIC = False
RESIZE_RES = (224, 224)
DISPLAY_IMAGE = True
CLIENT = Client()
FINE_AMOUNT = 1000
MSG_SENDER = "whatsapp:+14155238886"

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
face_recog_folder = "face_recognition"
known_faces = []
known_names = []
emirates_id_list = []

with open("fines.json", "r") as fines_file:
    fines = json.load(fines_file)

print("Loading person information...")

for face_name in os.listdir(f"{face_recog_folder}/known_faces"):
    if face_name != ".DS_Store":
        for filename in os.listdir(f"{face_recog_folder}/known_faces/{face_name}"):
            if filename != ".DS_Store":
                if filename == "emid.txt":
                    with open(f"{face_recog_folder}/known_faces/{face_name}/emid.txt", "r") as emid_file:
                        emid_info = emid_file.readlines()
                        id_number = int(emid_info[0])
                        phone_number = str(emid_info[1])
                        emirates_id = EmiratesID(id_number, face_name, phone_number)

                        if str(id_number) in fines.keys():
                            emirates_id.fine_amount = fines.get(str(id_number))

                        emirates_id_list.append(emirates_id)
                        print(f"Created Emirates ID entry for {face_name} with ID number {id_number}!")
                else:
                    image = face_recognition.load_image_file(f"{face_recog_folder}/known_faces/{face_name}/{filename}")
                    encoding = face_recognition.face_encodings(image)[0]
                    known_faces.append(encoding)
                    known_names.append(face_name)


def exit_sequence(video_capture):
    GPIO.cleanup()
    video_capture.release()
    cv2.destroyAllWindows()

    fines_dict = {}
    with open("fines.json", "w") as fines_file:
        for e in emirates_id_list:
            fines_dict[e.id] = e.fine_amount

        json.dump(fines_dict, fines_file)


def send_msg(phone_number, message):
    msg_receiver = "whatsapp:" + phone_number
    CLIENT.messages.create(
        body=message,
        from_=MSG_SENDER,
        to=msg_receiver
    )


def get_distance(trigger, echo):
    GPIO.output(trigger, GPIO.HIGH)
    time.sleep(0.0001)
    GPIO.output(trigger, GPIO.LOW)

    while not GPIO.input(echo):
        start = time.time()

    while GPIO.input(echo):
        end = time.time()

    sig_time = end - start
    dist = sig_time / 0.000058
    return dist


def main():
    print("Setting up GPIO pins...")
    GPIO.setmode(GPIO.BCM)
    GPIO.cleanup()

    echo_pin = 23
    trig_pin = 24
    led_pin = 25

    GPIO.setup(echo_pin, GPIO.IN)
    GPIO.setup(trig_pin, GPIO.OUT)
    GPIO.setup(led_pin, GPIO.OUT)
    previous_distance = 0
    previous_trash_present = False

    print("Starting camera...")
    cap = cv2.VideoCapture(0)

    try:
        while True:
            trash_present = False
            GPIO.output(led_pin, GPIO.LOW)
            _, cam_image = cap.read()

            if INVERT_PIC:
                cam_image = transform.rotate(cam_image, 180)

            resized_img = cv2.resize(cam_image, None, fx=0.4, fy=0.4)
            height, width, channels = cam_image.shape

            # Detecting trash objects
            blob = cv2.dnn.blobFromImage(cam_image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.3:
                        trash_present = True
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            if DISPLAY_IMAGE:
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                for i in range(len(boxes)):
                    if i in indexes:
                        x, y, w, h = boxes[i]
                        label = str(class_names[class_ids[i]])
                        color = (0, 0, 255)
                        cv2.rectangle(cam_image, (x, y), (x + w, y + h), color, FRAME_THICKNESS)
                        cv2.putText(cam_image, label, (x, y + h + 30), FONT, 1, color, FONT_THICKNESS)

            distance = get_distance(trig_pin, echo_pin)
            dist_diff = previous_distance - distance

            if trash_present or (dist_diff > distance / 10 and previous_distance != 0):
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
                            cv2.putText(cam_image, match, (top_left[0], bottom_right[1] + 30), FONT, 1, (0, 255, 0), FONT_THICKNESS)
                        else:
                            print(f"Match found: {match}")

                        match_phone = None
                        for e in emirates_id_list:
                            if e.name == match:
                                match_phone = e.phone_number
                                if trash_present and not previous_trash_present:  # if the person has thrown some trash
                                    # Save snapshot
                                    snap_datetime = datetime.datetime.now().strftime("%d.%m.%Y %H:%M")
                                    cv2.imwrite(f"snapshots/{snap_filename}.jpg", cam_image)

                                    # Apply fine and send message
                                    e.fine_amount += FINE_AMOUNT
                                    send_msg(e.phone_number, f"Dear {e.name}, you have been fined {FINE_AMOUNT} for littering in public.")
                                    print(f"Applied fine to {e.name}")
                                break

                        if dist_diff > distance / 10:
                            send_msg(match_phone, f"Dear {match}, thank you for throwing the trash in the proper place :)")
                            print("Trash entered the bin!")
                            GPIO.output(led_pin, GPIO.HIGH)

                        if DISPLAY_IMAGE:
                            cv2.putText(cam_image, str(distance), (0, 50), FONT, 1, (255, 255, 255), FONT_THICKNESS)

            # update "previous" variables
            previous_distance = distance
            previous_trash_present

            if DISPLAY_IMAGE:
                cv2.imshow("Camera Footage", cam_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        exit_sequence(cap)

    exit_sequence(cap)


if __name__ == "__main__":
    main()
