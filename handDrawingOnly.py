import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

points = []
drawing = False
pen_down = True

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

    for i in range(1, len(points)):
        if points[i - 1] and points[i]:
            cv2.line(frame, points[i - 1], points[i], (0, 255, 0), 3)

    if drawing:
        color = (0, 0, 255)
        text = "Drawing ON"
        if not pen_down:
            text += " (Pen Lifted)"
        cv2.circle(frame, (20, 20), 8, color, -1)
        cv2.putText(frame, text, (35, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(frame, "'D'=Draw | 'E'=Lift/Resume | 'C'=Clear | 'Q'=Quit",
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
            points.append(None)
    elif key == ord('c'):
        points.clear()

cap.release()
cv2.destroyAllWindows()
