import cv2
import pytesseract
import numpy as np

# If needed, specify the tesseract path here
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# External feed (video file, USB cam, or IP cam)
# Example: video file
cap = cv2.VideoCapture("License Plate Detection Test.mp4")

# Example: external USB webcam -> cap = cv2.VideoCapture(1)
# Example: IP camera -> cap = cv2.VideoCapture("http://192.168.1.100:8080/video")

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame received or end of video.")
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply preprocessing
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(blur, 30, 200)

    # Find contours
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    plate = None

    for cnt in contours:
        # Approximate contour to polygon
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.018 * peri, True)

        if len(approx) == 4:  # Select contour with 4 corners
            x, y, w, h = cv2.boundingRect(approx)
            plate = frame[y:y + h, x:x + w]
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
            break

    if plate is not None:
        # OCR on detected plate
        plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(plate_gray, config='--psm 8')
        cv2.putText(frame, f'Plate: {text.strip()}', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("License Plate Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
