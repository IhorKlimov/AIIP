import cv2
import numpy
import imutils


def find_single_face():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    scaling_factor = 0.5
    frame = cv2.imread("person.jpeg")
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    face_rects = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in face_rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

    cv2.imshow("person.jpeg", frame)
    cv2.waitKey(0)
    print(f"Found {len(face_rects)} faces!")


def find_multiple_faces():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    scaling_factor = 0.5
    frame = cv2.imread("people.jpeg")
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    face_rects = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in face_rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

    cv2.imshow("Many people", frame)
    cv2.waitKey(0)
    print(f"Found {len(face_rects)} faces!")


def find_face_smile_eyes(file):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    nose_cascade = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")
    left_ears_cascade = cv2.CascadeClassifier('haarcascade_mcs_left_ear.xml')
    right_ears_cascade = cv2.CascadeClassifier('haarcascade_mcs_right_ear.xml')

    image = cv2.imread(file)

    gray_filter = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(gray_filter, 7, 4)
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.7, minNeighbors=5)

    print(len(faces))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray_filter[y:y + h, x:x + w]
        roi_color = image[y:y + h, x:x + w]
        smile = smile_cascade.detectMultiScale(roi_gray)
        eye = eye_cascade.detectMultiScale(roi_gray)
        noses = nose_cascade.detectMultiScale(roi_gray)
        for (sx, sy, sw, sh) in smile:
            print("smile")
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 1)
        for (ex, ey, ew, eh) in eye:
            print("eye")
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 1)
        for (ex, ey, ew, eh) in noses:
            print("nose")
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 1)

    cv2.imshow("Image", image)
    cv2.waitKey(0)


def find_number_of_people():
    scaling_factor = 0.5
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    image = cv2.imread("people_walking.jpeg")
    image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    people_rects = hog.detectMultiScale(image, winStride=(8, 8), padding=(30, 30), scale=1.06)

    for (x, y, w, h) in people_rects[0]:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("People", image)
    cv2.waitKey(0)
    print(f"Found {len(people_rects[0])} people!")


def find_people_in_video():
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    cv2.startWindowThread()
    cap = cv2.VideoCapture("person.mp4")

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (800, 560))
        # gray_filter = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))
        boxes = numpy.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

        for (xa, ya, xb, yb) in boxes:
            cv2.rectangle(frame, (xa, ya), (xb, yb), (0, 255, 0), 1)
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break

    cap.release()
    cv2.destoryAllWindows()


if __name__ == '__main__':
    # find_single_face()
    # find_multiple_faces()
    find_face_smile_eyes("students.jpeg")
    # find_number_of_people()
    # find_people_in_video()
