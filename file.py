import cv2
import mediapipe as mp
import numpy as np
import math
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)

calc_x, calc_y = 180, 150

buttons = [
    ('7', (calc_x + 0, calc_y + 30)), ('8', (calc_x + 60, calc_y + 30)), ('9', (calc_x + 120, calc_y + 30)), ('/', (calc_x + 180, calc_y + 30)),
    ('4', (calc_x + 0, calc_y + 90)), ('5', (calc_x + 60, calc_y + 90)), ('6', (calc_x + 120, calc_y + 90)), ('*', (calc_x + 180, calc_y + 90)),
    ('1', (calc_x + 0, calc_y + 150)), ('2', (calc_x + 60, calc_y + 150)), ('3', (calc_x + 120, calc_y + 150)), ('-', (calc_x + 180, calc_y + 150)),
    ('0', (calc_x + 0, calc_y + 210)), ('.', (calc_x + 60, calc_y + 210)), ('=', (calc_x + 120, calc_y + 210)), ('+', (calc_x + 180, calc_y + 210)),
    ('<-', (calc_x + 180, calc_y + 270))
]

expression = ""
last_tap_time = 0
tap_delay = 0.5

def detect_button(x, y):
    for text, pos in buttons:
        bx, by = pos
        if bx - 25 < x < bx + 25 and by - 25 < y < by + 25:
            return text
    return None

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (calc_x - 20, calc_y - 40), (calc_x + 240, calc_y + 320), (0, 0, 0), -1)
    alpha = 0.6
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    for text, pos in buttons:
        bx, by = pos
        cv2.rectangle(frame, (bx - 30, by - 30), (bx + 30, by + 30), (0, 255, 0), 2)
        cv2.putText(frame, text, (bx - 10, by + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(frame, expression, (calc_x, calc_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark
            index_finger = landmarks[8]
            thumb = landmarks[4]

            ix, iy = int(index_finger.x * w), int(index_finger.y * h)
            tx, ty = int(thumb.x * w), int(thumb.y * h)

            cv2.circle(frame, (ix, iy), 8, (255, 0, 0), -1)
            cv2.circle(frame, (tx, ty), 8, (0, 255, 0), -1)

            pinch_distance = math.hypot(tx - ix, ty - iy)

            if pinch_distance < 25 and (time.time() - last_tap_time) > tap_delay:
                key = detect_button(ix, iy)
                if key:
                    last_tap_time = time.time()
                    if key == "=":
                        try:
                            expression = str(eval(expression))
                        except:
                            expression = "Error"
                    elif key == "<-":
                        expression = expression[:-1]
                    else:
                        expression += key

    cv2.imshow("AirMath", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
