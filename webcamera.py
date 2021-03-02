import cv2

face_cascade = cv2.CascadeClassifier('face-localization.xml')

def face_localization(frame):
    faces = face_cascade.detectMultiScale(frame, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x - 25, y - 25), (x + w + 25, y + h + 25), (0, 0, 255), 2)
    return frame


if __name__ == '__main__':
    vc = cv2.VideoCapture(0)
    frame = None
    if vc.isOpened():
        rval, frame = vc.read()
        frame = face_localization(frame)
    else:
        rval = False

    while rval and frame is not None:
        cv2.imshow("preview", frame)

        rval, frame = vc.read()
        frame = face_localization(frame)
        key = cv2.waitKey(20)
        if key == 27:
            break

    vc.release()
