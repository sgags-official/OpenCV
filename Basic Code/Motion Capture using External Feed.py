import cv2
import numpy as np

# ====== Setup Video Source ======
# For video file: replace 'video.mp4' with your file path
# For IP camera: replace with stream URL, e.g., 'http://192.168.0.101:8080/video'
video_source = ('Motion Detection.mp4')  # or 'http://your_ip_camera_stream'

cap = cv2.VideoCapture(video_source)

# Read the first frame to initialize background
ret, frame1 = cap.read()
if not ret:
    print("Failed to load video")
    exit()

frame1 = cv2.resize(frame1, (640, 480))
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv_mask = np.zeros_like(frame1)
hsv_mask[..., 1] = 255  # Set saturation to maximum

while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    frame2 = cv2.resize(frame2, (640, 480))
    next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate dense optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Convert flow to magnitude and angle
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv_mask[..., 0] = angle * 180 / np.pi / 2
    hsv_mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Convert HSV to BGR
    motion_visual = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)

    # Show both original and motion tracking
    cv2.imshow('Original Feed', frame2)
    cv2.imshow('Motion Tracking', motion_visual)

    # Prepare for next iteration
    prvs = next_frame.copy()

    # Exit on pressing 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
