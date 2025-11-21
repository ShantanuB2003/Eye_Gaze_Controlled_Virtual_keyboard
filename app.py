import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import threading
import time

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

keyboard = [
    ['Q','W','E','R','T','Y','U','I','O','P'],
    ['A','S','D','F','G','H','J','K','L'],
    ['Z','X','C','V','B','N','M'],
    ['1','2','3','4','5','6','7','8','9','0'],
    ['SPACE', 'BACKSPACE']
]

current_index = [0]

def get_letter_from_index(index):
    flat_keys = [key for row in keyboard for key in row]
    return flat_keys[index] if index < len(flat_keys) else None

def get_row_col_from_index(index):
    flat_keys = [(i, j) for i, row in enumerate(keyboard) for j in range(len(row))]
    return flat_keys[index] if index < len(flat_keys) else (None, None)

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def blink_detection():
    cap = cv2.VideoCapture(0)
    blink_counter = 0
    BLINK_THRESHOLD = 0.21
    CONSEC_FRAMES = 2
    last_blink_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            shape = predictor(gray, face)
            shape_np = np.array([[p.x, p.y] for p in shape.parts()])
            left_eye = shape_np[36:42]
            right_eye = shape_np[42:48]

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)

            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            cv2.putText(frame, f'Left EAR: {left_ear:.2f}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f'Right EAR: {right_ear:.2f}', (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if left_ear < BLINK_THRESHOLD or right_ear < BLINK_THRESHOLD:
                blink_counter += 1
            else:
                if blink_counter >= CONSEC_FRAMES:
                    now = time.time()
                    if now - last_blink_time > 1.0:
                        letter = get_letter_from_index(current_index[0])
                        row, col = get_row_col_from_index(current_index[0])
                        print(f"[BLINK] Typed: {letter} (Row: {row}, Col: {col})")
                        socketio.emit('type_letter', {'letter': letter, 'row': row, 'col': col})
                        last_blink_time = now
                blink_counter = 0

        cv2.imshow("Blink Detection", frame)
        if cv2.waitKey(1) == 27: 
            break

        time.sleep(0.05)

    cap.release()
    cv2.destroyAllWindows()

def highlight_loop():
    total_letters = sum(len(row) for row in keyboard)
    while True:
        row, col = get_row_col_from_index(current_index[0])
        socketio.emit('highlight', {'row': row, 'col': col})
        current_index[0] = (current_index[0] + 1) % total_letters
        time.sleep(1.5)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    threading.Thread(target=highlight_loop, daemon=True).start()
    threading.Thread(target=blink_detection, daemon=True).start()
    socketio.run(app, host='0.0.0.0', port=5001)