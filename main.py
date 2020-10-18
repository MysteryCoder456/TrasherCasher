import cv2

cap = cv2.VideoCapture(0)
face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_classifier_alt = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

while True:
    _, frame = cap.read()
    faces = face_classifier.detectMultiScale(
        frame, minSize=(120, 120), minNeighbors=4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), 3, 3)

    cv2.imshow("Camera Footage", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
