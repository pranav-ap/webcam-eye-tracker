import cv2
import math
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def detect(gray, frame):
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 30
#    params.maxThreshold = 200

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 50

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.6

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.4

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.4

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)


    # Detect blobs.
    keypoints = detector.detect(gray)

    for keypoint in keypoints:
        x = int(keypoint.pt[0])
        y = int(keypoint.pt[1])
        s = keypoint.size
        r = int(math.floor(s/2))

        print(x, y)

        cv2.circle(gray, (x, y), r, (0, 0, 255), -1)

    return gray


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
