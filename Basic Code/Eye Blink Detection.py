import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Indices for eye landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]


def eye_aspect_ratio(landmarks, eye_indices):
    # Extract the 2D points
    eye = np.array([(landmarks[i].x, landmarks[i].y) for i in eye_indices])

    # Compute vertical distances
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    # Compute horizontal distance
    C = np.linalg.norm(eye[0] - eye[3])

    ear = (A + B) / (2.0 * C)
    return ear


# Thresholds
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 3
blink_count = 0
frame_counter = 0

# Start video capture
cap = cv2.VideoCapture(0)

# Set the video capture resolution to 1280x720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            # Calculate EAR for both eyes
            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
            ear = (left_ear + right_ear) / 2.0

            # Draw eye contours
            for index in LEFT_EYE + RIGHT_EYE:
                x = int(landmarks[index].x * w)
                y = int(landmarks[index].y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Check for blink
            if ear < EAR_THRESHOLD:
                frame_counter += 1
            else:
                if frame_counter >= CONSEC_FRAMES:
                    blink_count += 1
                frame_counter = 0

            # Display EAR and blink count
            cv2.putText(frame, f'Blink Count: {blink_count}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f'EAR: {ear:.2f}', (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Eye Blink Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
