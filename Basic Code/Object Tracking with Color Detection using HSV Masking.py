import cv2
import numpy as np

# Initialize the webcam. This will use the default camera (0).
cap = cv2.VideoCapture(0)

# Set the video capture resolution to 1280x720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Define the HSV color range for the color you want to track (e.g., blue).
# You may need to adjust these values based on your lighting conditions.
# H (Hue), S (Saturation), V (Value)
lower_blue = np.array([100, 150, 0])
upper_blue = np.array([140, 255, 255])

while True:
    # Read a frame from the video capture.
    ret, frame = cap.read()

    # If the frame was not captured correctly, break the loop.
    if not ret:
        break

    # Convert the frame from BGR (Blue, Green, Red) to HSV (Hue, Saturation, Value).
    # This color space is better for color-based segmentation.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a binary mask that shows only the pixels within the defined blue range.
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Use the mask to perform a bitwise AND operation on the original frame.
    # This will result in an image that only contains the tracked color.
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Display the original, the mask, and the tracked object frames in separate windows.
    cv2.imshow('Original', frame)
    cv2.imshow('Mask', mask)
    cv2.imshow('Tracked Object', result)

    # Break the loop if the 'q' key is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows.
cap.release()
cv2.destroyAllWindows()
