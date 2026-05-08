"""
Hand Mouse Controller
Controla o cursor do mouse com a mão usando webcam (roda em segundo plano).

INSTALAÇÃO:
    pip install mediapipe opencv-python pyautogui pynput

GESTOS:
    - Dedo indicador direito apontando → Move o cursor (direção do dedo, não posição)
    - Dedo médio + polegar direito → Clique esquerdo
    - Dedo médio + polegar esquerdo → Clique direito

PARAR: Ctrl+Shift+Q
"""

import threading
import cv2
import pyautogui
import time
import math

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from pynput import keyboard

# ── Configurações ─────────────────────────────────────────
CAMERA_INDEX      = 0
SMOOTHING         = 12
DEAD_ZONE         = 12     # pixels — movimentos menores que isso são ignorados (anti-tremor)
POINT_RANGE       = 0.08   # alcance angular do dedo (PIP→TIP) para cobrir a tela inteira
DIR_ALPHA         = 0.25   # suavização da direção: 0=parado, 1=sem filtro
DIR_MAX_JUMP      = 0.04   # salto máximo de direção por frame — acima disso é glitch, ignora
CLICK_COOLDOWN    = 0.4
PINCH_THRESH      = 0.05   # distância normalizada para considerar toque
# ──────────────────────────────────────────────────────────

STOP_HOTKEY = {keyboard.Key.ctrl_l, keyboard.Key.shift, keyboard.KeyCode(char='q')}

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
    def __init__(self, tip=MIDDLE_TIP):
        self.tip = tip
        self.pinching = False

    def update(self, lm):
        dx = lm[self.tip].x - lm[THUMB_TIP].x
        dy = lm[self.tip].y - lm[THUMB_TIP].y
        dist = math.sqrt(dx * dx + dy * dy)
        if not self.pinching and dist < PINCH_THRESH:
            self.pinching = True
            return True
        if self.pinching and dist >= PINCH_THRESH:
            self.pinching = False
        return False


class DirFilter:
    def __init__(self, alpha, max_jump):
        self.alpha    = alpha
        self.max_jump = max_jump
        self.x = None
        self.y = None

    def update(self, dx, dy):
        if self.x is None:
            self.x, self.y = dx, dy
            return dx, dy
        jump = math.sqrt((dx - self.x) ** 2 + (dy - self.y) ** 2)
        if jump > self.max_jump:
            return None  # salto grande = glitch de tracking, ignora frame
        self.x += self.alpha * (dx - self.x)
        self.y += self.alpha * (dy - self.y)
        return self.x, self.y


class SmoothCursor:
    def __init__(self, factor):
        self.factor = factor
        self.x, self.y = pyautogui.position()

    def update(self, tx, ty):
        dx = tx - self.x
        dy = ty - self.y
        if abs(dx) < DEAD_ZONE and abs(dy) < DEAD_ZONE:
            return
        self.x += dx / self.factor
        self.y += dy / self.factor
        pyautogui.moveTo(int(self.x), int(self.y))


def main():
    stop_event = threading.Event()
    pressed_keys = set()

    def on_press(key):
        pressed_keys.add(key)
        if all(k in pressed_keys for k in STOP_HOTKEY):
            print("Hand Mouse: encerrando...")
            stop_event.set()

    def on_release(key):
        pressed_keys.discard(key)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cursor         = SmoothCursor(SMOOTHING)
    dir_filter     = DirFilter(DIR_ALPHA, DIR_MAX_JUMP)
    right_detector = PinchDetector()
    left_detector  = PinchDetector()
    last_click_time = 0

    options = HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_buffer=_load_hand_model()),
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        running_mode=vision.RunningMode.VIDEO
    )

    print("Hand Mouse iniciado! Pressione Ctrl+Shift+Q para encerrar.")

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener, \
         HandLandmarker.create_from_options(options) as landmarker:
        frame_ts = 0
        while cap.isOpened() and not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                break

            frame    = cv2.flip(frame, 1)
            frame_ts += 1

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result    = landmarker.detect_for_video(mp_image, frame_ts)

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
                raw_dir_x = right_lm[INDEX_TIP].x - right_lm[INDEX_PIP].x
                raw_dir_y = right_lm[INDEX_TIP].y - right_lm[INDEX_PIP].y
                filtered = dir_filter.update(raw_dir_x, raw_dir_y)
                if filtered is not None:
                    dir_x, dir_y = filtered
                    raw_x = map_range(dir_x, -POINT_RANGE, POINT_RANGE, 0, SCREEN_W)
                    raw_y = map_range(dir_y, -POINT_RANGE, POINT_RANGE, 0, SCREEN_H)
                    cursor.update(raw_x, raw_y)

                if right_detector.update(right_lm) and now - last_click_time > CLICK_COOLDOWN:
                    pyautogui.click()
                    last_click_time = now

            if left_lm is not None:
                if left_detector.update(left_lm) and now - last_click_time > CLICK_COOLDOWN:
                    pyautogui.rightClick()
                    last_click_time = now

        listener.stop()

    cap.release()
    print("Hand Mouse encerrado.")


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
