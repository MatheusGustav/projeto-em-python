"""
🖐️ Hand Mouse Controller
Controla o cursor do mouse com a mão usando webcam.

INSTALAÇÃO:
    pip install mediapipe opencv-python pyautogui

GESTOS:
    - Mão aberta movendo             → Move o cursor
    - Indicador empurra pra frente   → Clique esquerdo
    - Dedo médio empurra pra frente  → Clique direito
"""

import cv2
import pyautogui
import time
import math

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions

# ── Configurações ─────────────────────────────────────────
CAMERA_INDEX      = 0
SMOOTHING         = 6
MOVE_MARGIN       = 0.15
CLICK_COOLDOWN    = 0.4
PINCH_THRESH      = 0.05   # distância normalizada para considerar toque
# ──────────────────────────────────────────────────────────

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

SCREEN_W, SCREEN_H = pyautogui.size()

# Índices dos landmarks (mesmos do MediaPipe antigo)
THUMB_TIP  = 4
INDEX_TIP  = 8
INDEX_PIP  = 6
MIDDLE_TIP = 12
MIDDLE_PIP = 10
RING_TIP   = 16
RING_PIP   = 14
PINKY_TIP  = 20
PINKY_PIP  = 18


def map_range(val, in_min, in_max, out_min, out_max):
    val = max(in_min, min(in_max, val))
    return (val - in_min) / (in_max - in_min) * (out_max - out_min) + out_min


def is_hand_open(lm):
    fingers = [
        lm[INDEX_TIP].y  < lm[INDEX_PIP].y,
        lm[MIDDLE_TIP].y < lm[MIDDLE_PIP].y,
        lm[RING_TIP].y   < lm[RING_PIP].y,
        lm[PINKY_TIP].y  < lm[PINKY_PIP].y,
    ]
    return sum(fingers) >= 3


class PinchDetector:
    def __init__(self):
        self.pinching = False

    def update(self, lm):
        dx = lm[INDEX_TIP].x - lm[THUMB_TIP].x
        dy = lm[INDEX_TIP].y - lm[THUMB_TIP].y
        dist = math.sqrt(dx * dx + dy * dy)
        if not self.pinching and dist < PINCH_THRESH:
            self.pinching = True
            return True
        if self.pinching and dist >= PINCH_THRESH:
            self.pinching = False
        return False


class SmoothCursor:
    def __init__(self, factor):
        self.factor = factor
        self.x, self.y = pyautogui.position()

    def update(self, tx, ty):
        self.x += (tx - self.x) / self.factor
        self.y += (ty - self.y) / self.factor
        pyautogui.moveTo(int(self.x), int(self.y))


def draw_landmarks_on_frame(frame, landmarks):
    """Desenha os pontos da mão no frame manualmente."""
    h, w = frame.shape[:2]
    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17)
    ]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in connections:
        cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)
    for i, (x, y) in enumerate(pts):
        color = (0, 255, 200) if i in [8, 12] else (0, 200, 255)
        cv2.circle(frame, (x, y), 5, color, -1)


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cursor          = SmoothCursor(SMOOTHING)
    right_detector = PinchDetector()
    left_detector  = PinchDetector()
    last_click_time = 0

    # Nova API do MediaPipe
    base_options = python.BaseOptions(model_asset_path=None)
    options = HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_buffer=_load_hand_model()),
        num_hands=2,
        min_hand_detection_confidence=0.75,
        min_hand_presence_confidence=0.75,
        min_tracking_confidence=0.75,
        running_mode=vision.RunningMode.VIDEO
    )

    print("🖐️  Hand Mouse iniciado! Pressione 'Q' na janela para sair.")

    with HandLandmarker.create_from_options(options) as landmarker:
        frame_ts = 0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]
            frame_ts += 1

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result    = landmarker.detect_for_video(mp_image, frame_ts)

            cv2.rectangle(frame, (0, 0), (w, 40), (20, 20, 20), -1)
            cv2.putText(frame, "Hand Mouse  |  Q = sair", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            status_text  = "sem mao detectada"
            status_color = (50, 50, 255)
            right_lm = None
            left_lm  = None

            # imagem espelhada: MediaPipe "Left" = mão direita do usuário
            if result.hand_landmarks:
                for lm, handedness in zip(result.hand_landmarks, result.handedness):
                    if handedness[0].display_name == "Left":
                        right_lm = lm
                    else:
                        left_lm = lm

            now = time.time()

            if right_lm is not None:
                raw_x = map_range(right_lm[INDEX_TIP].x, MOVE_MARGIN, 1 - MOVE_MARGIN, 0, SCREEN_W)
                raw_y = map_range(right_lm[INDEX_TIP].y, MOVE_MARGIN, 1 - MOVE_MARGIN, 0, SCREEN_H)
                cursor.update(raw_x, raw_y)

                if right_detector.update(right_lm) and now - last_click_time > CLICK_COOLDOWN:
                    pyautogui.click()
                    last_click_time = now
                    status_text, status_color = "CLIQUE ESQUERDO", (0, 255, 100)
                elif right_detector.pinching:
                    status_text, status_color = "mao direita pincando...", (0, 220, 80)
                else:
                    status_text, status_color = "mao direita - movendo", (180, 180, 180)

                draw_landmarks_on_frame(frame, right_lm)

            if left_lm is not None:
                if left_detector.update(left_lm) and now - last_click_time > CLICK_COOLDOWN:
                    pyautogui.rightClick()
                    last_click_time = now
                    status_text, status_color = "CLIQUE DIREITO", (0, 150, 255)
                elif left_detector.pinching:
                    status_text, status_color = "mao esquerda pincando...", (0, 120, 220)
                else:
                    status_text, status_color = "mao esquerda detectada", (150, 150, 200)

                draw_landmarks_on_frame(frame, left_lm)

            cv2.putText(frame, status_text, (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.imshow("Hand Mouse", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Hand Mouse encerrado.")


def _load_hand_model():
    """Baixa o modelo de detecção de mãos do MediaPipe."""
    import urllib.request
    import os
    model_path = "/tmp/hand_landmarker.task"
    if not os.path.exists(model_path):
        print("📥 Baixando modelo de detecção de mãos...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
        print("✅ Modelo baixado!")
    with open(model_path, "rb") as f:
        return f.read()


if __name__ == "__main__":
    main()
