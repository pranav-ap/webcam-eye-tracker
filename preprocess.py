import cv2
import numpy as np


def preprocessFrame(gray, frame):
    # Create kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
#
#    # Sharpen image
    gray = cv2.filter2D(gray, -1, kernel)

#    # Apply adaptive thresholding
#    max_output_value = 255
#    neighorhood_size = 99
#    subtract_from_mean = 40
#    gray = cv2.adaptiveThreshold(gray,
#                                max_output_value,
#                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                cv2.THRESH_BINARY,
#                                neighorhood_size,
#                                subtract_from_mean)

#    gray = cv2.GaussianBlur(gray,(5,5),0)

#    kernel = np.ones((3,3),np.uint8)
#    gray = cv2.erode(gray,kernel,iterations = 1)
#    gray = cv2.dilate(gray,kernel,iterations = 1)

#    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)


    return gray


# We turn the webcam on
video_capture = cv2.VideoCapture(0)

while True:
    # get a frame
    _, frame = video_capture.read()
    # convert frame to black and white
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # get a color frame with the rectangles around the faces with help of gray frame
    canvas = preprocessFrame(gray, frame)
    # display frame with the rectangles
    cv2.imshow('Video', canvas)
    # check for quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# We turn the webcam off.
video_capture.release()
# close all windows
cv2.destroyAllWindows()

