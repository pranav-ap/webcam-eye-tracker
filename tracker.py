import cv2
cv2.__version__
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def detect(gray, frame):
    # provides a list of coordinates of faces from the gray frame
    faces = face_cascade.detectMultiScale(
        gray,
        1.3, # scale factor
        5 # minimum neighbors each candidate rectangle should have to retain it.
        )

    for (x, y, width, height) in faces:
        # draw a rectangle in the color frame
        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            (255, 0, 0), # color of rectangle
            2 # width of rectangle
            )

        # region of interest where we try to find eyes
        # [rows : cols]
        roi_gray = gray[y: y + height, x: x + width]
        roi_color = frame[y: y + height, x: x + width]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(
                roi_color,
                (ex, ey),
                (ex + ew, ey + eh),
                (0, 255, 0),
                2
                )

            roi_eyes_gray = gray[ey: ey + height, ex: ex + width]
            roi_eyes_color = frame[ey: ey + height, ex: ex + width]
#            roi_eyes_gray = cv2.medianBlur(roi_eyes_gray, 2)
        
            circles = cv2.HoughCircles(
                    roi_eyes_gray, 
                    cv2.HOUGH_GRADIENT, 
                    1, 
                    2, 
#                    param1 = 10,
                    param2 = 12,
                    minRadius = 2,
                    maxRadius = 13
                    )
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    center = (i[0], i[1])
                    # circle center
                    cv2.circle(roi_eyes_color, center, 1, (0, 100, 100), 3)
                    # circle outline
                    radius = i[2]
                    cv2.circle(roi_eyes_color, center, radius, (255, 0, 255), 3)
            
            print (circles)

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
