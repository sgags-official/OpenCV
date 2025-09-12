import cv2
import numpy as np

image = cv2.imread('shapes.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
    cv2.drawContours(image, [approx], 0, (0, 255, 0), 3)
    x = approx.ravel()[0]
    y = approx.ravel()[1] - 10
    cv2.putText(image, f'{len(approx)} sides', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

cv2.imshow('Shapes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
