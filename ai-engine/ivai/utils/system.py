"""System helpers."""

from __future__ import annotations

import contextlib
import sys


@contextlib.contextmanager
def prevent_sleep():
    """Keep the computer awake while the block runs (Windows only; no-op elsewhere).

    A training run is not "user activity": without this, Windows may enter sleep in
    the middle of an epoch, which corrupts the measured training time (observed in
    Phase 2). The normal power policy is restored when the block exits.
    """
    if sys.platform != "win32":
        yield
        return
    import ctypes

    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    kernel32 = ctypes.windll.kernel32
    kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    try:
        yield
    finally:
        kernel32.SetThreadExecutionState(ES_CONTINUOUS)
