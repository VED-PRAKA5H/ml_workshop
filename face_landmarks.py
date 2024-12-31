import cv2
import imutils
from imutils import face_utils
import dlib

# Load the face detector and shape predictor model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../model/full_face.dat")

# Initialize video capture from the default camera (0)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error in opening the camera.")
    exit()  # Exit if the camera cannot be opened

# Start a loop that runs as long as the camera is opened successfully
while cap.isOpened():
    ret, frame = cap.read()  # Capture frame-by-frame

    if not ret:
        print("Failed to grab frame")  # Print an error message if frame capture fails
        break  # Exit the loop if there was an error

    # Resize the frame for better processing speed
    image = imutils.resize(frame, width=950)

    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image; up-sample by 1 time
    rects = detector(gray, 1)

    # Loop over detected faces
    for rect in rects:
        shape = predictor(gray, rect)  # Get facial landmarks
        shape = face_utils.shape_to_np(shape)  # Convert to NumPy array

        (x, y, w, h) = face_utils.rect_to_bb(rect)  # Convert rectangle to bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)  # Draw bounding box

        # Draw circles at each landmark point
        for (x_landmark, y_landmark) in shape:
            cv2.circle(image, (x_landmark, y_landmark), 1, (0, 255, 0), thickness=1)

    # Show the output image with detected faces and landmarks
    cv2.imshow("Your Face", image)

    # Check for key presses; wait for 11 ms for a key event
    if cv2.waitKey(11) & 0xFF == ord('q'):
        break  # Break the loop if 'q' is pressed

# Release the video capture object and close all OpenCV windows
cap.release()  # Release the camera resource
cv2.destroyAllWindows()  # Close all OpenCV windows
