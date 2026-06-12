"""
conftest.py – globalna konfiguracja testów.
Ustawia SDL w tryb 'dummy' PRZED jakimkolwiek importem pygame.
"""

import os
import sys
import multiprocessing as mp
from unittest.mock import MagicMock

# SDL dummy – musi być przed importem pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# Zastąp numba no-op shim-em PRZED jakimkolwiek importem modułów projektu.
# Dzięki temu @njit jest dekoratorem tożsamościowym → funkcje działają jako
# czyste Python → pytest-cov może je śledzić → 100% pokrycia.
if "numba" not in sys.modules:
    _mock_numba = MagicMock()
    _mock_numba.njit = lambda func=None, **kw: (func if func is not None else lambda f: f)
    sys.modules["numba"] = _mock_numba

# Upewnij się, że metoda startowa procesów jest ustawiona (tylko raz)
try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass  # Już ustawiona
