import cv2
import numpy as np
import os

# Create output directory
os.makedirs("synthetic_videos", exist_ok=True)

# Video settings
frame_width = 256
frame_height = 256
fps = 10
duration_seconds = 3
total_frames = fps * duration_seconds

# Define output video writer
video_name = "synthetic_tbut_eye.avi"
out = cv2.VideoWriter(f"synthetic_videos/{video_name}", cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

# Colors
eye_white = (255, 255, 255)
eye_black = (0, 0, 0)
eye_gray = (180, 180, 180)
tear_break_color = (255, 200, 200)

# Draw a synthetic eye shape
def draw_eye(frame, open_eye=True, tbut=False):
    eye_center = (frame_width // 2, frame_height // 2)
    radius = 80

    if open_eye:
        # Draw open eye
        cv2.circle(frame, eye_center, radius, eye_gray, -1)
        cv2.circle(frame, eye_center, 20, eye_black, -1)  # pupil

        if tbut:
            # Add a tear break-up effect (red cracks or spots)
            for i in range(10):
                pt = (
                    np.random.randint(eye_center[0] - radius//2, eye_center[0] + radius//2),
                    np.random.randint(eye_center[1] - radius//2, eye_center[1] + radius//2)
                )
                cv2.circle(frame, pt, 5, tear_break_color, -1)
    else:
        # Draw closed eye (line)
        cv2.line(frame, (60, eye_center[1]), (frame_width - 60, eye_center[1]), eye_black, 12)

    return frame

# Generate video frames
for i in range(total_frames):
    frame = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255

    # Blink between frame 10-15
    if 10 <= i <= 15:
        frame = draw_eye(frame, open_eye=False)
    # After frame 20, simulate TBUT
    elif i >= 20:
        frame = draw_eye(frame, open_eye=True, tbut=True)
    else:
        frame = draw_eye(frame, open_eye=True)

    out.write(frame)

out.release()
print(f"✅ Synthetic video saved at: synthetic_videos/{video_name}")
