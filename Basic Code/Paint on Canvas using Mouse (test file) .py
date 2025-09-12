import cv2
import numpy as np

# Initialize a blank canvas
canvas = np.ones((600, 800, 3), dtype=np.uint8) * 255

# Default values
drawing = False
ix, iy = -1, -1
color = (0, 0, 255)  # Red color
radius = 5


# Mouse callback function
def draw(event, x, y, flags, param):
    global ix, iy, drawing, color, canvas

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(canvas, (x, y), radius, color, -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(canvas, (x, y), radius, color, -1)

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Erase with white color when right-clicked
        cv2.circle(canvas, (x, y), radius * 3, (255, 255, 255), -1)


# Create window and set callback
cv2.namedWindow('Paint Canvas')
cv2.setMouseCallback('Paint Canvas', draw)

while True:
    cv2.imshow('Paint Canvas', canvas)
    key = cv2.waitKey(1) & 0xFF

    # Press 'c' to clear the canvas
    if key == ord('c'):
        canvas = np.ones((600, 800, 3), dtype=np.uint8) * 255

    # Press 'r' for red
    if key == ord('r'):
        color = (0, 0, 255)

    # Press 'g' for green
    if key == ord('g'):
        color = (0, 255, 0)

    # Press 'b' for blue
    if key == ord('b'):
        color = (255, 0, 0)

    # Press 'q' to quit
    if key == ord('q'):
        break

cv2.destroyAllWindows()
