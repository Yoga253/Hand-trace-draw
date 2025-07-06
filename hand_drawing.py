import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# --- Initialize MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# --- Drawing Variables ---
points = deque(maxlen=512)
drawing = False  # Toggle to start/stop drawing

# --- Shape Detection Function ---
def detect_shape(points):
    if len(points) < 10:
        return "Too few points"

    contour = np.array(points).reshape((-1, 1, 2)).astype(np.int32)
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    num_vertices = len(approx)

    if num_vertices == 3:
        return "Triangle"
    elif num_vertices == 4:
        return "Rectangle"
    elif num_vertices > 6:
        return "Circle"
    else:
        return f"Unknown ({num_vertices} vertices)"

# --- Start Webcam ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert to RGB for MediaPipe
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Detect hand
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            tip = hand_landmarks.landmark[8]  # Index finger tip
            x, y = int(tip.x * w), int(tip.y * h)

            if drawing:
                points.append((x, y))

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Draw the traced path
    for i in range(1, len(points)):
        if points[i - 1] and points[i]:
            cv2.line(frame, points[i - 1], points[i], (0, 255, 0), 3)

    # Display UI Instructions
    cv2.putText(frame, "'D'=Draw | 'C'=Clear | 'S'=Shape Detect | 'Q'=Quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Hand Drawing", frame)

    # --- Handle Key Presses ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d'):
        drawing = not drawing
    elif key == ord('c'):
        points.clear()
    elif key == ord('s'):
        if len(points) > 10:
            shape = detect_shape(points)
            print(f"🟢 Detected shape: {shape}")
            points.clear()

cap.release()
cv2.destroyAllWindows()
