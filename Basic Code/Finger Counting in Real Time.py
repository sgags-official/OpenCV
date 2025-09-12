import cv2
import numpy as np
import math

# Initialize webcam. This will use the default camera (0).
cap = cv2.VideoCapture(0)

# Set video capture resolution to 1280x720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    # Read a frame from the video capture.
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for natural user interaction.
    frame = cv2.flip(frame, 1)

    # Define a Region of Interest (ROI) for hand detection.
    # The new coordinates are centered for the 1280x720 resolution.
    roi_top = 260
    roi_bottom = 620
    roi_left = 440
    roi_right = 800

    roi = frame[roi_top:roi_bottom, roi_left:roi_right]
    cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 0), 2)

    # Convert ROI to grayscale.
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise.
    blur = cv2.GaussianBlur(gray, (35, 35), 0)

    # Threshold the image to create a binary mask.
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours in the thresholded image.
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # Get the largest contour, which is assumed to be the hand.
        contour = max(contours, key=cv2.contourArea)

        # Create a bounding rectangle around the hand contour.
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Get the convex hull of the hand contour.
        hull = cv2.convexHull(contour)

        # Draw the hand contour and the convex hull.
        cv2.drawContours(roi, [contour], -1, (255, 0, 0), 2)
        cv2.drawContours(roi, [hull], -1, (0, 255, 0), 2)

        # Find convexity defects for finger counting.
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        if hull_indices is not None and len(hull_indices) > 3:
            defects = cv2.convexityDefects(contour, hull_indices)

            if defects is not None:
                count_defects = 0

                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(contour[s][0])
                    end = tuple(contour[e][0])
                    far = tuple(contour[f][0])

                    # Calculate lengths of the triangle sides using distance formula.
                    a = math.dist(start, end)
                    b = math.dist(start, far)
                    c = math.dist(end, far)

                    # Use the cosine rule to find the angle at the defect point.
                    angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

                    # If the angle is less than 90 degrees, it is considered a defect (valley between fingers).
                    if angle <= 90:
                        count_defects += 1
                        cv2.circle(roi, far, 5, (0, 0, 255), -1)

                    # Draw lines around the defects for visualization.
                    cv2.line(roi, start, end, (0, 255, 0), 2)

                # Count the number of fingers based on the defects.
                fingers = count_defects + 1

                # Display the number of fingers on the main frame.
                cv2.putText(frame, f"Fingers: {fingers}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the frames.
    cv2.imshow("Frame", frame)
    cv2.imshow("ROI", roi)
    cv2.imshow("Thresh", thresh)

    # Exit on pressing 'q'.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows.
cap.release()
cv2.destroyAllWindows()
