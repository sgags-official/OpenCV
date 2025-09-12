import cv2
import numpy as np

# Create a white canvas with a fixed size
canvas_width = 1280
canvas_height = 720
canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

# Dark blue HSV range
lower_color = np.array([100, 150, 50])
upper_color = np.array([130, 255, 255])

# Variables for drawing
prev_x, prev_y = None, None
radius = 5
color = (255, 0, 0)  # Blue drawing color (BGR)

# Start capturing video from the default webcam
cap = cv2.VideoCapture(0)

# Set the video capture resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, canvas_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, canvas_height)

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for intuitive experience
    frame = cv2.flip(frame, 1)

    # ✅ Resize frame to match canvas size
    frame = cv2.resize(frame, (canvas_width, canvas_height))

    # Convert the frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create mask for dark blue
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Apply a Gaussian blur to remove noise
    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours and len(contours) > 0:
        # Get largest contour (assumed fingertip/marker)
        contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(contour) > 500:
            # Find center
            (x, y), radius_detect = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))

            # Draw detection circle
            cv2.circle(frame, center, int(radius_detect), (0, 255, 0), 2)

            # Draw on canvas
            if prev_x is not None and prev_y is not None:
                cv2.line(canvas, (prev_x, prev_y), center, color, radius * 2)

            prev_x, prev_y = center
        else:
            prev_x, prev_y = None, None
    else:
        prev_x, prev_y = None, None

    # ✅ Combine video and canvas (now same size)
    combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Show results
    cv2.imshow('Finger Paint (Dark Blue)', combined)
    cv2.imshow('Mask', mask)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        # Clear canvas
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
