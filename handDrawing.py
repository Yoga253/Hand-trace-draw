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
pen_down = True  # Toggle this with 'e'

# --- Shape Detection ---
def detect_shape(points):
    contour = np.array([pt for pt in points if pt is not None], dtype=np.int32)
    if len(contour) < 3:
        return "Too few points"

    contour = contour.reshape((-1, 1, 2))
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    sides = len(approx)

    if sides == 3:
        return "Triangle"
    elif sides == 4:
        x, y, w, h = cv2.boundingRect(approx)
        return "Square" if 0.95 < float(w)/h < 1.05 else "Rectangle"
    elif sides == 5:
        return "Pentagon"
    elif sides == 6:
        return "Hexagon"
    elif 7 <= sides <= 10:
        return "Circle"
    else:
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            major, minor = ellipse[1]
            return "Circle (fitted)" if abs(major - minor) < 10 else "Ellipse"
        return f"Polygon ({sides} sides)"

# --- Start Webcam ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            tip = hand_landmarks.landmark[8]
            x, y = int(tip.x * w), int(tip.y * h)

            if drawing and pen_down:
                points.append((x, y))

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Draw path
    for i in range(1, len(points)):
        if points[i - 1] and points[i]:
            cv2.line(frame, points[i - 1], points[i], (0, 255, 0), 3)

    # Drawing mode status
    if drawing:
        color = (0, 0, 255)
        text = "Drawing ON"
        if not pen_down:
            text += " (Pen Lifted)"
        cv2.circle(frame, (20, 20), 8, color, -1)
        cv2.putText(frame, text, (35, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # UI Footer
    cv2.putText(frame, "'D'=Draw | 'E'=Lift/Resume | 'C'=Clear | 'S'=Shape | 'Q'=Quit",
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Hand Drawing", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d'):
        drawing = not drawing
    elif key == ord('e'):
        pen_down = not pen_down
        if not pen_down:
            points.append(None)  # Insert gap
    elif key == ord('c'):
        points.clear()
    elif key == ord('s'):
        if len(points) > 10:
            shape = detect_shape(points)
            print(f"🟢 Detected shape: {shape}")
            points.clear()

cap.release()
cv2.destroyAllWindows()
