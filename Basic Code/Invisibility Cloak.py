import cv2
import numpy as np
import time

# Capture the background first
cap = cv2.VideoCapture(0)
time.sleep(2)

# Set the video capture resolution to 1280x720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Read a few frames to allow camera to adjust
for i in range(30):
    ret, background = cap.read()

# Flip background for consistency
background = np.flip(background, axis=1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = np.flip(frame, axis=1)

    # Convert to HSV for color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range for the color to be cloaked (e.g., red)
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for red color
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    # Refine the mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    # Inverse mask to segment out the cloak
    mask_inv = cv2.bitwise_not(mask)

    # Segment the person without cloak
    part1 = cv2.bitwise_and(frame, frame, mask=mask_inv)

    # Segment the background where the cloak is
    part2 = cv2.bitwise_and(background, background, mask=mask)

    # Combine the two to make the cloak area show the background
    final_output = cv2.addWeighted(part1, 1, part2, 1, 0)

    # Show the output
    cv2.imshow('Invisibility Cloak', final_output)

    # Quit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
