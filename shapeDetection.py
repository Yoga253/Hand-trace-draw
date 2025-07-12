import cv2
import mediapipe as mp
import numpy as np

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Drawing state
points = []
drawing = False
pen_down = True
eraser_mode = False

# Create a persistent canvas
canvas = None

# Shape detection
def detect_shape(points):
    contour = np.array([p for p in points if p is not None], dtype=np.int32)
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
        aspect_ratio = float(w) / h
        if 0.95 < aspect_ratio < 1.05:
            return "Square"
        else:
            return "Rectangle"
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
            if abs(major - minor) < 10:
                return "Circle (fitted)"
            else:
                return "Ellipse"
        return f"Polygon ({sides} sides)"

# Erase nearby points - optimized version
def erase_near(x, y, radius=30):
    global points
    # More efficient eraser - modify in place
    for i in range(len(points)):
        if points[i] is not None:
            if ((points[i][0]-x)**2 + (points[i][1]-y)**2)**0.5 <= radius:
                points[i] = None

def undo_last_stroke():
    global points
    if not points:
        return
    
    # Find the last stroke by looking for the last None separator
    last_none_index = -1
    for i in range(len(points) - 1, -1, -1):
        if points[i] is None:
            last_none_index = i
            break
    
    if last_none_index == -1:
        # No None found, remove all points (first stroke)
        points.clear()
    else:
        # Remove everything after the last None
        points = points[:last_none_index]

def redraw_canvas(h, w):
    global canvas, points
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Draw all strokes on canvas
    for i in range(1, len(points)):
        if points[i - 1] and points[i]:
            cv2.line(canvas, points[i - 1], points[i], (0, 255, 0), 3)

# Open webcam
cap = cv2.VideoCapture(0)

# Set camera properties for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

frame_count = 0
redraw_needed = True

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_count += 1
    
    # Initialize canvas on first frame
    if canvas is None:
        h, w, _ = frame.shape
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Process hand tracking every frame
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    current_drawing = False
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            tip = hand_landmarks.landmark[8]
            x, y = int(tip.x * w), int(tip.y * h)

            if drawing and pen_down and not eraser_mode:
                points.append((x, y))
                current_drawing = True
                redraw_needed = True
            elif eraser_mode:
                erase_near(x, y)
                redraw_needed = True

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Only redraw canvas when needed
    if redraw_needed:
        redraw_needed = False
    
    # Draw lines directly on frame (original method)
    for i in range(1, len(points)):
        if points[i - 1] and points[i]:
            cv2.line(frame, points[i - 1], points[i], (0, 255, 0), 3)

    # Drawing indicator
    if drawing:
        color = (0, 0, 255)
        text = "Drawing ON"
        if not pen_down:
            text += " (Pen Lifted)"
        if eraser_mode:
            text = "Eraser Mode"
        cv2.circle(frame, (20, 20), 8, color, -1)
        cv2.putText(frame, text, (35, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # UI instructions
    cv2.putText(frame, "'D'=Draw | 'W'=Pen Up/Down | 'E'=Eraser | 'S'=Shape | 'C'=Clear|",
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)

    cv2.imshow("Hand Drawing + Shape + Erase", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d'):
        drawing = not drawing
    elif key == ord('w'):
        pen_down = not pen_down
        if not pen_down:
            points.append(None)
    elif key == ord('e'):
        eraser_mode = not eraser_mode
    elif key == ord('c'):
        points.clear()
        redraw_needed = True
    elif key == ord('s'):
        if len(points) > 10:
            shape = detect_shape(points)
            print(f"🟢 Detected shape: {shape}")
            points.clear()
            redraw_needed = True
    elif key == 26:  # Ctrl+Z
        undo_last_stroke()
        redraw_needed = True

cap.release()
cv2.destroyAllWindows()