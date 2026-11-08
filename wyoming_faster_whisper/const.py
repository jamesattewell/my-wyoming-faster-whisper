"""Constants and the transcriber interface."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

AUTO_LANGUAGE = "auto"

# Merged-decoder export: the biasing prompt's KV cache is reusable across
# utterances. The older split-decoder export is not supported.
DEFAULT_MODEL = "rhasspy/qwen3-asr-0.6b-onnx-int4-merged"

# Where to look for Home Assistant when biasing toward its names.
HASS_API_URL = "http://homeassistant.local:8123/api"


class Transcriber(ABC):
    """What the event handler and name cache need from a model.

    Qwen3AsrTranscriber is the only implementation. The interface exists so that
    the server logic can be tested against a fake instead of a multi-GB model.
    """

    @abstractmethod
    def transcribe(
        self,
        wav_path: Union[str, Path],
        language: Optional[str],
        initial_prompt: Optional[str] = None,
    ) -> str:
        """Transcribe a 16 kHz 16-bit mono WAV file."""

    @abstractmethod
    def count_prompt_tokens(self, text: str) -> int:
        """Count the tokens ``text`` would use as a biasing prompt.

        Used to fit as many entity names as possible into the prompt budget.
        """
