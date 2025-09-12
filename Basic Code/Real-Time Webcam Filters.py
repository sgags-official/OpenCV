import cv2

cap = cv2.VideoCapture(0)

# Set the video capture resolution to 1280x720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(frame, (15, 15), 0)
    invert = cv2.bitwise_not(frame)

    cv2.imshow('Original', frame)
    cv2.imshow('Grayscale', gray)
    cv2.imshow('Blurred', blur)
    cv2.imshow('Inverted', invert)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
q