import cv2
import math
#import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

params = cv2.SimpleBlobDetector_Params()

def detectPupils(gray, frame, eye):
    (ex, ey, ew, eh) = eye

    roi_gray = gray[ey: ey + eh, ex: ex + ew]
    roi_color = frame[ey: ey + eh, ex: ex + ew]

    canny_edge = cv2.Canny(gray, 0, 0)

    # Change thresholds
    params.minThreshold = 1
    
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 50
    params.minArea = 800
    
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.7
    
    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.9
    
    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.001

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
    
    # Detect blobs.
    keypoints = detector.detect(canny_edge)

    for keypoint in keypoints:
       x = int(keypoint.pt[0])
       y = int(keypoint.pt[1])
       s = keypoint.size
       r = int(math.floor(s/2))
       
       cv2.circle(frame, (x, y), r, (255, 255, 0), 2)

    return frame


def detectEyes(gray, frame, face):
    (x, y, width, height) = face

    roi_gray = gray[y: y + height, x: x + width]
    roi_color = frame[y: y + height, x: x + width]

    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)

    for eye in eyes:
        (ex, ey, ew, eh) = eye

        cv2.rectangle(
            roi_color,
            (ex, ey),
            (ex + ew, ey + eh),
            (0, 255, 0),
            2
            )

    for eye in eyes:
        frame = detectPupils(gray, frame, eye)

    return frame


def detect(gray, frame):
    gray = cv2.blur(gray, (7,7))
    # provides a list of coordinates of faces from the gray frame
    faces = face_cascade.detectMultiScale(
        gray,
        1.3, # scale factor
        6 # minimum neighbors each candidate rectangle should have to retain it.
        )

    for face in faces:
        (x, y, width, height) = face
        # draw a rectangle in the color frame
        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            (255, 0, 0), # color of rectangle
            2 # width of rectangle
            )

    for face in faces:
        frame = detectEyes(gray, frame, face)

    return frame


# We turn the webcam on
video_capture = cv2.VideoCapture(0)

while True:
    # get a frame
    _, frame = video_capture.read()
    # convert frame to black and white
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # get a color frame with the rectangles around the faces with help of gray frame
    canvas = detect(gray, frame)
    # display frame with the rectangles
    cv2.imshow('Video', canvas)
    # check for quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# We turn the webcam off.
video_capture.release()
# close all windows
cv2.destroyAllWindows()
