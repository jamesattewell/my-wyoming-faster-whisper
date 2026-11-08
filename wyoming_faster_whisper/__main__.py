#!/usr/bin/env python3
import argparse
import asyncio
import logging
from functools import partial
from typing import TYPE_CHECKING, Optional

from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer, AsyncTcpServer

from . import __version__
from .const import AUTO_LANGUAGE, DEFAULT_MODEL, HASS_API_URL, Transcriber
from .env_args import ENV_PREFIX
from .env_args import parse_args as parse_args_with_env
from .handler import TranscriptionHandler
from .languages import SUPPORTED_LANGUAGES, qwen3_asr_language
from .qwen3_asr_handler import Qwen3AsrTranscriber
from .vocabulary import DEFAULT_PROMPT_MAX_TOKENS

if TYPE_CHECKING:
    # Imported for typing only: aiohttp is an optional extra.
    from .name_cache import HassNameCache

_LOGGER = logging.getLogger(__name__)


def positive_float(value: str) -> float:
    """Parse a strictly positive floating-point command-line value."""
    number = float(value)
    if number <= 0:
        raise ValueError("must be greater than zero")
    return number


def get_parser() -> argparse.ArgumentParser:
    """Return the command-line parser.

    Every option here also has an environment variable, so an argument added
    below needs nothing extra to be settable from a container's environment.
    """
    parser = argparse.ArgumentParser(
        epilog=(
            "Every option can also be set from the environment: --some-option "
            f"reads {ENV_PREFIX}SOME_OPTION, or {ENV_PREFIX}SOME_OPTION_FILE "
            "to read the value out of a file (docker secrets). "
            "Command-line arguments win over both."
        )
    )
    parser.add_argument("--uri", required=True, help="unix:// or tcp://")
    #
    parser.add_argument(
        "--zeroconf",
        nargs="?",
        const="faster-whisper",
        help="Enable discovery over zeroconf with optional name (default: faster-whisper)",
    )
    #
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face repo id or local directory of a Qwen3-ASR ONNX export "
        f"(default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        action="append",
        help="Data directory to check for downloaded models",
    )
    parser.add_argument(
        "--download-dir",
        help="Directory to download models into (default: first data dir)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to use for inference: cpu, cuda, or cuda:N (default: cpu). "
        "CUDA requires onnxruntime-gpu; see README.",
    )
    parser.add_argument(
        "--language",
        default=AUTO_LANGUAGE,
        help=f"Default language to set for transcription (default: {AUTO_LANGUAGE})",
    )
    parser.add_argument(
        "--cpu-threads",
        default=4,
        type=int,
        help="Number of CPU threads to use for inference (default: 4)",
    )
    parser.add_argument(
        "--initial-prompt",
        help="Biasing context for the model, e.g. a list of names to spell right",
    )
    parser.add_argument(
        "--vad-endpointing",
        type=positive_float,
        help="End a command and send its transcript after this many seconds of "
        "VAD-detected silence (default: wait for audio-stop)",
    )
    parser.add_argument(
        "--vad-clip",
        action="store_true",
        help="Use pysilero-vad to clip leading/trailing silence before "
        "transcription (default: disabled)",
    )
    parser.add_argument(
        "--vad-clip-threshold",
        type=float,
        default=0.5,
        help="Speech probability threshold for --vad-clip (default: 0.5)",
    )
    parser.add_argument(
        "--vad-clip-pad-ms",
        type=int,
        default=400,
        help="Context kept around speech for --vad-clip in ms (default: 400)",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Never download a model: fail if it isn't already in --download-dir "
        "(cached models are loaded without a download regardless)",
    )
    # Home Assistant name biasing (extra: hass)
    parser.add_argument(
        "--hass-token",
        help="Long-lived access token for Home Assistant. Enables biasing toward "
        "the names of exposed entities, areas, and floors (extra: hass). Keep it "
        f"off the command line with {ENV_PREFIX}HASS_TOKEN_FILE",
    )
    parser.add_argument(
        "--hass-api",
        default=HASS_API_URL,
        help=f"URL of the Home Assistant API (default: {HASS_API_URL})",
    )
    parser.add_argument(
        "--hass-refresh-seconds",
        type=float,
        default=0.0,
        help="Minimum seconds between name refreshes (default: 0, refresh on "
        "every utterance)",
    )
    parser.add_argument(
        "--hass-prompt-max-tokens",
        type=int,
        default=DEFAULT_PROMPT_MAX_TOKENS,
        help=f"Token budget for names in the prompt (default: {DEFAULT_PROMPT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--hass-prompt-timeout",
        type=float,
        default=1.0,
        help="Seconds to wait for an unfinished name refresh before transcribing "
        "with the previous names (default: 1.0)",
    )
    #
    parser.add_argument("--debug", action="store_true", help="Log DEBUG messages")
    parser.add_argument(
        "--log-format", default=logging.BASIC_FORMAT, help="Format for log messages"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="Print version and exit",
    )

    return parser


async def main() -> None:
    """Main entry point."""
    args = parse_args_with_env(get_parser())

    if not args.download_dir:
        # Download to first data dir by default
        args.download_dir = args.data_dir[0]

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO, format=args.log_format
    )
    _LOGGER.debug(args)

    language = None if args.language == AUTO_LANGUAGE else args.language
    _warn_if_language_unsupported(language)

    _LOGGER.debug("Loading model '%s' (device=%s)", args.model, args.device)
    transcriber = Qwen3AsrTranscriber(
        args.model,
        args.download_dir,
        local_files_only=args.local_files_only,
        cpu_threads=args.cpu_threads,
        device=args.device,
    )

    names = await _load_names(args, transcriber)

    wyoming_info = build_info(
        args.model, requires_external_vad=args.vad_endpointing is None
    )

    server = AsyncServer.from_uri(args.uri)

    if args.zeroconf:
        if not isinstance(server, AsyncTcpServer):
            raise ValueError("Zeroconf requires tcp:// uri")

        from wyoming.zeroconf import HomeAssistantZeroconf

        tcp_server: AsyncTcpServer = server
        hass_zeroconf = HomeAssistantZeroconf(
            name=args.zeroconf, port=tcp_server.port, host=tcp_server.host
        )
        await hass_zeroconf.register_server()
        _LOGGER.debug("Zeroconf discovery enabled")

    _LOGGER.info("Ready")
    await server.run(
        partial(
            TranscriptionHandler,
            wyoming_info,
            transcriber,
            names,
            language=language,
            initial_prompt=args.initial_prompt,
            vad_clip=args.vad_clip,
            vad_clip_threshold=args.vad_clip_threshold,
            vad_clip_pad_ms=args.vad_clip_pad_ms,
            vad_endpointing=args.vad_endpointing,
        )
    )


def build_info(model_name: str, *, requires_external_vad: bool = True) -> Info:
    """Build Wyoming service metadata."""
    return Info(
        asr=[
            AsrProgram(
                name="qwen3-asr",
                description="Qwen3-ASR speech recognition with onnxruntime",
                attribution=Attribution(
                    name="Qwen",
                    url="https://huggingface.co/Qwen",
                ),
                installed=True,
                version=__version__,
                requires_external_vad=requires_external_vad,
                models=[
                    AsrModel(
                        name=model_name,
                        description=model_name,
                        attribution=Attribution(
                            name="Qwen",
                            url="https://huggingface.co/Qwen",
                        ),
                        installed=True,
                        languages=sorted(SUPPORTED_LANGUAGES),
                        version=None,
                    )
                ],
            )
        ],
    )


# -----------------------------------------------------------------------------


def _warn_if_language_unsupported(language: Optional[str]) -> None:
    """Warn at startup when --language names something the model cannot do.

    Unvalidated, a typo only shows up as every transcription being
    auto-detected. A region-qualified tag is accepted.
    """
    if (not language) or (qwen3_asr_language(language) is not None):
        return

    _LOGGER.warning(
        "Language '%s' is not supported by Qwen3-ASR (supported: %s). "
        "Transcription will fall back to auto-detection.",
        language,
        ", ".join(sorted(SUPPORTED_LANGUAGES)),
    )


async def _load_names(
    args: argparse.Namespace, transcriber: Transcriber
) -> Optional["HassNameCache"]:
    """Set up biasing toward Home Assistant's names, if a token was given."""
    if not args.hass_token:
        return None

    try:
        from .hass_api import HomeAssistant
        from .name_cache import HassNameCache
    except ImportError as exc:
        raise ImportError(
            "--hass-token requires the 'hass' extra: "
            "pip install 'wyoming-faster-whisper[hass]'"
        ) from exc

    names = HassNameCache(
        HomeAssistant(args.hass_token, api_url=args.hass_api),
        transcriber.count_prompt_tokens,
        prefix=args.initial_prompt,
        max_tokens=args.hass_prompt_max_tokens,
        refresh_seconds=args.hass_refresh_seconds,
        wait_timeout=args.hass_prompt_timeout,
    )

    # Fetch once now so the first utterance is biased too. Best effort: Home
    # Assistant may still be starting up, and that must not stop this server
    # from serving. Every utterance retries.
    if await names.refresh():
        _LOGGER.info("Biasing toward names from %s", args.hass_api)
    else:
        _LOGGER.warning(
            "No names loaded from %s yet; will retry on each utterance",
            args.hass_api,
        )

    return names


# -----------------------------------------------------------------------------


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
