"""Tests for wyoming-faster-whisper."""

from pathlib import Path

_DIR = Path(__file__).parent
_PROGRAM_DIR = _DIR.parent
_LOCAL_DIR = _PROGRAM_DIR / "local"
_SAMPLES_PER_CHUNK = 1024

# Generous: on a fresh machine the server downloads the 785 MB model before it
# answers anything
_START_TIMEOUT = 900
_TRANSCRIBE_TIMEOUT = 60
