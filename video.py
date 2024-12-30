import cv2  # Import the OpenCV library

# Initialize video capture object to access the default camera (0)
cap = cv2.VideoCapture(0)

# Start a loop that runs as long as the camera is opened successfully
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Check if the frame was captured successfully
    if not ret:
        print("Failed to grab frame")  # Print an error message if frame capture fails
        break  # Exit the loop if there was an error

    # Display the captured frame in a window titled 'Your face'
    cv2.imshow('Your face', frame)

    # Check for key presses; wait for 1 ms for a key event
    # If 'q' is pressed, exit the loop
    if cv2.waitKey(11) & 0xFF == ord('q'):
        break  # Break the loop if 'q' is pressed

# Release the video capture object and close all OpenCV windows
cap.release()  # Release the camera resource
cv2.destroyAllWindows()  # Close all OpenCV windows
