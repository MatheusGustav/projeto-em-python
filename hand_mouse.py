"""
Hand Mouse Controller v2
Controla o cursor com a mão via webcam. Roda em segundo plano.

GESTOS (mão direita):
    - Dedo indicador estendido → Move o cursor (posição do dedo = posição na tela)
    - Levanta o polegar        → Clique esquerdo
    - Levanta o mindinho       → Clique direito
    - Sinal de paz (dedos V)   → Scroll vertical

PARAR: Ctrl+Shift+Q
"""

import collections
import threading
import math
import time

import cv2
import pyautogui
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from pynput import keyboard

# ── Configurações ──────────────────────────────────────────────────────────────
CAMERA_INDEX   = 0
CLICK_COOLDOWN = 0.5    # segundos mínimos entre cliques
PINCH_THRESH   = 0.065  # distância normalizada para detectar pinça
PINCH_FRAMES   = 4      # frames consecutivos para confirmar pinça (anti-clique-acidental)

# One Euro Filter — ajuste fino da suavização:
#   min_cutoff: menor = mais suave em repouso (remove tremor); maior = mais responsivo
#   beta:       maior = menos lag em movimentos rápidos intencionais
OEF_MIN_CUTOFF = 0.05   # muito suave em repouso — elimina tremor de braço
OEF_BETA       = 0.6    # responde rápido a movimentos intencionais

# Filtro de mediana antes do One Euro: remove picos de glitch do tracking
MEDIAN_SIZE = 7         # número de amostras (ímpar)

# Zona morta em pixels: cursor só move se a mudança for maior que isso
CURSOR_DEAD_ZONE = 5    # pixels — movimentos menores são tremor, ignorar

# Margem do frame da câmera mapeada para a borda da tela.
# 0.15 = os 15% das bordas do frame cobrem os extremos da tela.
MARGIN = 0.15

SCROLL_DEADZONE = 0.003  # movimento mínimo normalizado para acionar scroll
SCROLL_SCALE    = 200    # multiplicador: delta_norm * SCROLL_SCALE = clicks de scroll
# ───────────────────────────────────────────────────────────────────────────────

STOP_HOTKEY = {keyboard.Key.ctrl_l, keyboard.Key.shift, keyboard.KeyCode(char='q')}
pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0
SCREEN_W, SCREEN_H = pyautogui.size()

# Índices dos landmarks do MediaPipe
THUMB_TIP  = 4;  THUMB_IP   = 3
INDEX_TIP  = 8;  INDEX_PIP  = 6
MIDDLE_TIP = 12; MIDDLE_PIP = 10
RING_TIP   = 16; RING_PIP   = 14
PINKY_TIP  = 20; PINKY_PIP  = 18


# ── Filtros ────────────────────────────────────────────────────────────────────

class MedianFilter:
    """Remove picos de glitch do tracking mantendo um buffer circular de amostras."""

    def __init__(self, size=MEDIAN_SIZE):
        self.buf = collections.deque(maxlen=size)

    def __call__(self, x):
        self.buf.append(x)
        return sorted(self.buf)[len(self.buf) // 2]

    def reset(self):
        self.buf.clear()


class _LowPass:
    def __init__(self, alpha):
        self.alpha = alpha
        self.s = None

    def __call__(self, x):
        self.s = x if self.s is None else self.alpha * x + (1.0 - self.alpha) * self.s
        return self.s


class OneEuroFilter:
    """Filtro adaptativo: remove jitter em repouso, mantém responsividade em movimento."""

    def __init__(self, freq, min_cutoff=OEF_MIN_CUTOFF, beta=OEF_BETA, d_cutoff=1.0):
        self.freq       = freq
        self.min_cutoff = min_cutoff
        self.beta       = beta
        self.x_filt     = _LowPass(self._alpha(min_cutoff))
        self.dx_filt    = _LowPass(self._alpha(d_cutoff))

    def _alpha(self, cutoff):
        te  = 1.0 / self.freq
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def reset(self):
        self.x_filt.s  = None
        self.dx_filt.s = None

    def __call__(self, val):
        prev  = self.x_filt.s
        deriv = (val - prev) * self.freq if prev is not None else 0.0
        edx   = self.dx_filt(deriv)
        self.x_filt.alpha = self._alpha(self.min_cutoff + self.beta * abs(edx))
        return self.x_filt(val)


# ── Detecção de gestos ─────────────────────────────────────────────────────────

class GestureDetector:
    """Detecta um gesto booleano com debounce: exige N frames consecutivos para confirmar."""

    def __init__(self, frames=PINCH_FRAMES):
        self.frames = frames
        self._count = 0
        self.active = False

    def update(self, triggered: bool):
        if triggered:
            self._count = min(self._count + 1, self.frames)
        else:
            self._count = 0
            self.active = False

        if self._count >= self.frames and not self.active:
            self.active = True
            return True  # evento: gesto confirmado agora
        return False

    def reset(self):
        self._count = 0
        self.active = False


def _extended(lm, tip, pip):
    return lm[tip].y < lm[pip].y


def _map_to_screen(nx, ny):
    """Mapeia coordenada normalizada [0,1] do frame para coordenada de tela."""
    sx = (nx - MARGIN) / (1.0 - 2 * MARGIN)
    sy = (ny - MARGIN) / (1.0 - 2 * MARGIN)
    return max(0.0, min(1.0, sx)) * SCREEN_W, max(0.0, min(1.0, sy)) * SCREEN_H


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    stop_event   = threading.Event()
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
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    median_x           = MedianFilter()
    median_y           = MedianFilter()
    filt_x             = OneEuroFilter(fps)
    filt_y             = OneEuroFilter(fps)
    lclick_det         = GestureDetector()  # polegar levantado → clique esquerdo
    rclick_det         = GestureDetector()  # mindinho levantado → clique direito
    last_click_t       = 0.0
    scroll_prev_y      = None
    prev_right_visible = False
    cursor_x, cursor_y = pyautogui.position()

    options = HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_buffer=_load_hand_model()),
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        running_mode=vision.RunningMode.VIDEO,
    )

    print("Hand Mouse v2 iniciado! Pressione Ctrl+Shift+Q para encerrar.")

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener, \
         HandLandmarker.create_from_options(options) as landmarker:

        frame_ts = 0
        while cap.isOpened() and not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                break

            frame     = cv2.flip(frame, 1)
            frame_ts += 1
            rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result    = landmarker.detect_for_video(mp_img, frame_ts)

            right_lm = None
            if result.hand_landmarks:
                for lm, hand in zip(result.hand_landmarks, result.handedness):
                    # Frame espelhado: MediaPipe "Left" = mão direita do usuário
                    if hand[0].display_name == "Left":
                        right_lm = lm

            now = time.time()

            # ── Mão direita: cursor + clique esquerdo + scroll ─────────────────
            if right_lm is not None:
                if not prev_right_visible:
                    # Reseta filtros para não arrastar de posição antiga
                    median_x.reset(); median_y.reset()
                    filt_x.reset();   filt_y.reset()
                prev_right_visible = True

                idx_ext = _extended(right_lm, INDEX_TIP, INDEX_PIP)
                mid_ext = _extended(right_lm, MIDDLE_TIP, MIDDLE_PIP)

                if idx_ext and mid_ext:
                    # Sinal de paz (V) → scroll vertical
                    ref_y = (right_lm[INDEX_TIP].y + right_lm[MIDDLE_TIP].y) / 2
                    if scroll_prev_y is not None:
                        delta = scroll_prev_y - ref_y  # mão subindo → delta + → scroll up
                        if abs(delta) > SCROLL_DEADZONE:
                            clicks = round(delta * SCROLL_SCALE)
                            if clicks:
                                pyautogui.scroll(clicks)
                    scroll_prev_y = ref_y

                elif idx_ext:
                    # Indicador estendido → mover cursor pela posição do dedo
                    scroll_prev_y = None
                    raw_x, raw_y  = _map_to_screen(right_lm[INDEX_TIP].x,
                                                   right_lm[INDEX_TIP].y)
                    # Pipeline: mediana → One Euro Filter → zona morta
                    sx = int(filt_x(median_x(raw_x)))
                    sy = int(filt_y(median_y(raw_y)))
                    if math.hypot(sx - cursor_x, sy - cursor_y) > CURSOR_DEAD_ZONE:
                        pyautogui.moveTo(sx, sy)
                        cursor_x, cursor_y = sx, sy

                else:
                    scroll_prev_y = None

                # Gestos de clique — só ativos fora do modo scroll
                if not (idx_ext and mid_ext):
                    thumb_up  = right_lm[THUMB_TIP].y < right_lm[THUMB_IP].y
                    pinky_ext = _extended(right_lm, PINKY_TIP, PINKY_PIP)

                    if lclick_det.update(thumb_up) and now - last_click_t > CLICK_COOLDOWN:
                        pyautogui.click()
                        last_click_t = now

                    if rclick_det.update(pinky_ext) and now - last_click_t > CLICK_COOLDOWN:
                        pyautogui.rightClick()
                        last_click_t = now
                else:
                    lclick_det.reset()
                    rclick_det.reset()

            else:
                prev_right_visible = False
                scroll_prev_y      = None

        listener.stop()

    cap.release()
    print("Hand Mouse encerrado.")


def _load_hand_model():
    import urllib.request
    import os
    path = "/tmp/hand_landmarker.task"
    if not os.path.exists(path):
        print("Baixando modelo de detecção de mãos...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            path,
        )
    with open(path, "rb") as f:
        return f.read()


if __name__ == "__main__":
    main()
