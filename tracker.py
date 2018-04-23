import cv2
import math
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


def detectPupils(color_frame_full, preprocessed_frame_full, face, eye):
    (x, y, width, height) = face
    (ex, ey, ew, eh) = eye

    just_the_face_frame = preprocessed_frame_full[y: y + height, x: x + width]
    just_the_eye_frame = just_the_face_frame[ey: ey + eh, ex: ex + ew]


    just_the_eye_frame = preprocess_just_the_eye_frame(just_the_eye_frame)

    return preprocessed_frame_full


def preprocess_just_the_eye_frame(just_the_eye_frame):
    cv2.imshow('Video eyes', just_the_eye_frame)

    return just_the_eye_frame


def detectEyes(color_frame_full, preprocessed_frame_full, face):
    (x, y, width, height) = face

    just_the_face_frame = preprocessed_frame_full[y: y + height, x: x + width]

    eyes = eye_cascade.detectMultiScale(just_the_face_frame, 1.1, 7)

    for eye in eyes:
        (ex, ey, ew, eh) = eye

        cv2.rectangle(
            just_the_face_frame,
            (ex, ey),
            (ex + ew, ey + eh),
            (0, 255, 0),
            2
            )

        print('eye : ', (ex, ey, ew, eh))

        preprocessed_frame_full = detectPupils(color_frame_full, preprocessed_frame_full, face, eye)

    return preprocessed_frame_full


def detectFaces(color_frame_full, preprocessed_frame_full):
    faces = face_cascade.detectMultiScale(
        preprocessed_frame_full,
        1.3, # scale factor
        5 # minimum neighbors each candidate rectangle should have to retain it.
        )

    for face in faces:
        (x, y, width, height) = face
        # draw a rectangle in the color frame
        cv2.rectangle(
            preprocessed_frame_full,
            (x, y),
            (x + width, y + height),
            (255, 0, 0), # color of rectangle
            2 # width of rectangle
            )

        print('face : ', (x, y, width, height))

        preprocessed_frame_full = detectEyes(color_frame_full, preprocessed_frame_full, face)

    return preprocessed_frame_full


def preprocess_color_frame_full(color_frame_full):
    preprocessed_frame_full = cv2.cvtColor(color_frame_full, cv2.COLOR_BGR2GRAY)

    return preprocessed_frame_full


def start(color_frame_full):
    preprocessed_frame_full = preprocess_color_frame_full(color_frame_full)

    preprocessed_frame_full = detectFaces(color_frame_full, preprocessed_frame_full)

    return preprocessed_frame_full


def main():
    # We turn the webcam on
    video_capture = cv2.VideoCapture(0)

    while True:
        # get a frame
        _, color_frame_full = video_capture.read()
        # perform detection
        canvas = start(color_frame_full)
        # display frame with the rectangles
        cv2.imshow('Video', canvas)
        # check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # We turn the webcam off.
    video_capture.release()
    # close all windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()