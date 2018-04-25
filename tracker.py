import cv2
import logging


face_cascade = cv2.CascadeClassifier(r'.\haarcascades\haarcascade_frontalface_default.xml')
right_eye_cascade = cv2.CascadeClassifier(r'.\haarcascades\haarcascade_right_eye.xml')

pupil_detector = cv2.SimpleBlobDetector_create()

previous_position = (1, 1)

def detectPupils(color_frame_full, preprocessed_frame_full, face, eye):
    (x, y, width, height) = face
    (ex, ey, ew, eh) = eye

    just_the_face_frame = preprocessed_frame_full[y: y + height, x: x + width]
    just_the_eye_frame = just_the_face_frame[ey: ey + eh, ex: ex + ew]

    preprocesses_just_the_eye_frame = preprocess_just_the_eye_frame(just_the_eye_frame)

    # Detect blobs
    keypoints = pupil_detector.detect(preprocesses_just_the_eye_frame)

    print(just_the_eye_frame.shape)

    for keypoint in keypoints:
        x = int(keypoint.pt[0])
        y = int(keypoint.pt[1])

        just_the_eye_frame = cv2.drawMarker(
           just_the_eye_frame,
           (x, y),
           (255, 255, 0),
           markerSize = 10
           )

        global previous_position

        cue = ''

        if (previous_position[0] > x):
            cue += ' right'
        if (previous_position[0] < x):
            cue += ' left'

        if (previous_position[1] > y):
            cue += ' down'
        if (previous_position[1] < y):
            cue += ' up'

        previous_position = (x, y)

        logging.info('Pupil coordinates : ' + str(x) + ' ,' + str(y) + ' Eye Cue : ' + cue)

    cv2.imshow('preprocessed eyes', preprocesses_just_the_eye_frame)
    cv2.imshow('detection eyes', just_the_eye_frame)

    return preprocessed_frame_full


def preprocess_just_the_eye_frame(just_the_eye_frame):
    # Apply adaptive thresholding
    max_output_value = 100
    neighorhood_size = 99
    subtract_from_mean = 8

    just_the_eye_frame = cv2.adaptiveThreshold(
        just_the_eye_frame,
        max_output_value,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        neighorhood_size,
        subtract_from_mean
        )

    return just_the_eye_frame


def detectEyes(color_frame_full, preprocessed_frame_full, face):
    (x, y, width, height) = face

    just_the_face_frame = preprocessed_frame_full[y: y + height, x: x + width]

    right_eyes = right_eye_cascade.detectMultiScale(just_the_face_frame, 1.3, 12)

    for eye in right_eyes:
        (ex, ey, ew, eh) = eye

        cv2.rectangle(
            just_the_face_frame,
            (ex, ey),
            (ex + ew, ey + eh),
            (0, 255, 0),
            2
            )

        cv2.putText(
            just_the_face_frame,
            "Eye",
            (ex, ey - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2
            )

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

        cv2.putText(
            preprocessed_frame_full,
            "Face",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2
            )

        preprocessed_frame_full = detectEyes(color_frame_full, preprocessed_frame_full, face)

    return preprocessed_frame_full


def preprocess_color_frame_full(color_frame_full):
    preprocessed_frame_full = cv2.cvtColor(color_frame_full, cv2.COLOR_BGR2GRAY)

    return preprocessed_frame_full


def start(color_frame_full):
    preprocessed_frame_full = preprocess_color_frame_full(color_frame_full)

    preprocessed_frame_full = detectFaces(color_frame_full, preprocessed_frame_full)

    return preprocessed_frame_full


def setup():
    # -- Setup logger --
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
    	level=logging.INFO,
		datefmt='%Y-%m-%d %H:%M:%S',
        filename = 'tracker.log',
        filemode = 'w')


    # -- Setup pupil detector --
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 30

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 35
    params.minArea = 100

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.4

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.5

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.3

    # Create a detector with the parameters
    global pupil_detector
    pupil_detector = cv2.SimpleBlobDetector_create(params)


def main():
    # perform setup
    setup()

    # We turn the webcam on
    video_capture = cv2.VideoCapture(0)

    while True:
        # get a frame
        _, color_frame_full = video_capture.read()
        # perform detection
        canvas = start(color_frame_full)
        # display frame with the rectangles
        cv2.imshow('Main Frame', canvas)
        # check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # We turn the webcam off.
    video_capture.release()
    # close all windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
