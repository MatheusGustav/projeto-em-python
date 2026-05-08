#!/usr/bin/env python3
"""Daemon de hotkey global para Wayland/KDE usando evdev. Super+M abre/fecha hand_mouse.py."""

import subprocess
import select
import glob
import sys
import os
import signal

import evdev
from evdev import ecodes

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_mouse.py")
HOTKEY_META = {ecodes.KEY_LEFTMETA, ecodes.KEY_RIGHTMETA}
HOTKEY_KEY = ecodes.KEY_M


def find_keyboards():
    keyboards = []
    for path in sorted(glob.glob("/dev/input/event*")):
        try:
            dev = evdev.InputDevice(path)
            caps = dev.capabilities().get(ecodes.EV_KEY, [])
            if ecodes.KEY_A in caps and ecodes.KEY_LEFTMETA in caps:
                keyboards.append(dev)
        except Exception:
            pass
    return keyboards


def main():
    keyboards = find_keyboards()
    if not keyboards:
        print("Nenhum teclado encontrado em /dev/input/", file=sys.stderr)
        sys.exit(1)

    print(f"Monitorando {len(keyboards)} teclado(s). Pressione Super+M para alternar hand_mouse.")

    meta_pressed = False
    process = None

    fds = {dev.fd: dev for dev in keyboards}

    while True:
        r, _, _ = select.select(fds.keys(), [], [], 1.0)
        for fd in r:
            dev = fds[fd]
            try:
                for event in dev.read():
                    if event.type != ecodes.EV_KEY:
                        continue
                    key = evdev.categorize(event)
                    code = key.scancode
                    state = key.keystate

                    if code in HOTKEY_META:
                        meta_pressed = state in (key.key_down, key.key_hold)

                    elif code == HOTKEY_KEY and state == key.key_down and meta_pressed:
                        if process is None or process.poll() is not None:
                            process = subprocess.Popen(
                                [sys.executable, SCRIPT],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            )
                            print(f"hand_mouse iniciado (pid {process.pid})")
                        else:
                            os.kill(process.pid, signal.SIGTERM)
                            process = None
                            print("hand_mouse encerrado")
            except OSError:
                pass


if __name__ == "__main__":
    main()
