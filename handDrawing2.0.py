import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import math

# --- Initialize MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# --- Drawing Variables ---
points = deque(maxlen=1024)
drawing = False

# --- Utility Functions ---
def angle(p1, p2, p3):
    """Return the angle at p2 formed by p1, p2, p3"""
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def detect_shape(points):
    if len(points) < 20:
        return "Too few points"

    contour = np.array(points).reshape((-1, 1, 2)).astype(np.int32)
    area = cv2.contourArea(contour)
    if area < 100:
        return "Too small to detect"

    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertices = len(approx)

    if vertices == 3:
        return "Triangle"
    elif vertices == 4:
        # Check aspect ratio
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if 0.9 <= aspect_ratio <= 1.1:
            return "Square"
        else:
            return "Rectangle"
    elif vertices > 6:
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        if circularity > 0.7:
            return "Circle"
        else:
            return "Ellipse"
    else:
        return f"Polygon ({vertices} sides)"

# --- Start Webcam ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Detect hand and track fingertip
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            tip = hand_landmarks.landmark[8]
            x, y = int(tip.x * w), int(tip.y * h)

            if drawing:
                points.append((x, y))

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Draw the traced path
    for i in range(1, len(points)):
        if points[i - 1] and points[i]:
            cv2.line(frame, points[i - 1], points[i], (0, 255, 0), 3)

    # --- Drawing Mode Indicator ---
    if drawing:
        cv2.circle(frame, (20, 20), 8, (0, 0, 255), -1)
        cv2.putText(frame, "Drawing ON", (35, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # --- Display UI ---
    cv2.putText(frame, "'D'=Draw | 'C'=Clear | 'S'=Shape Detect | 'Q'=Quit",
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("Hand Drawing", frame)

    # --- Key Controls ---
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
