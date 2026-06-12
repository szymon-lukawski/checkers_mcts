"""
conftest.py – globalna konfiguracja testów.
Ustawia SDL w tryb 'dummy' PRZED jakimkolwiek importem pygame.
"""

import os
import multiprocessing as mp

# SDL dummy – musi być przed importem pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# Upewnij się, że metoda startowa procesów jest ustawiona (tylko raz)
try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass  # Już ustawiona
