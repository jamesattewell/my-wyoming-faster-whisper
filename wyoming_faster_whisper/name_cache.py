"""Keep Home Assistant's names fresh without making the user wait for them.

The refresh is timed to hide inside the utterance. When ``AudioStart`` arrives
the speaker has not finished talking -- a voice command runs a couple of seconds
-- so a fetch kicked off there has ample time to land before ``AudioStop``, when
the prompt is actually needed. By then the names are current, and nothing was
added to the latency the user perceives.

Everything degrades to the last good snapshot. A fetch that fails, or is still
in flight when the audio stops, costs freshness and nothing else: the previous
names are used, or the configured ``--initial-prompt`` alone if there has never
been a successful fetch. Home Assistant being unreachable must never turn into a
failed transcription.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from .hass_api import HomeAssistant, HomeAssistantError
from .vocabulary import DEFAULT_PROMPT_MAX_TOKENS, CountTokens, RecognitionContext

_LOGGER = logging.getLogger(__name__)


class HassNameCache:
    """A snapshot of Home Assistant's names, refreshed in the background."""

    def __init__(
        self,
        hass: HomeAssistant,
        count_tokens: CountTokens,
        prefix: Optional[str] = None,
        max_tokens: int = DEFAULT_PROMPT_MAX_TOKENS,
        refresh_seconds: float = 0.0,
        wait_timeout: float = 1.0,
    ) -> None:
        self._hass = hass
        # The model's own tokenizer, so the budget is exact.
        self._count_tokens = count_tokens
        self._prefix = prefix
        self._max_tokens = max_tokens

        # 0 refreshes on every utterance; a positive value throttles.
        self._refresh_seconds = refresh_seconds

        # How long AudioStop will wait on a fetch that has not landed yet.
        self._wait_timeout = wait_timeout

        self._context: Optional[RecognitionContext] = None
        self._updated: Optional[float] = None
        self._task: Optional[asyncio.Task] = None

        # Whether the previous attempt succeeded, so a broken Home Assistant is
        # reported once instead of on every utterance.
        self._was_ok = True

    @property
    def has_names(self) -> bool:
        """Whether a fetch has ever succeeded."""
        return self._context is not None

    def schedule_refresh(self) -> None:
        """Start a refresh in the background, if one is due and none is running.

        Safe to call on every AudioStart from every connection: concurrent calls
        collapse onto the one in-flight fetch.
        """
        if (self._task is not None) and (not self._task.done()):
            # Already in flight.
            return

        if not self._is_stale():
            return

        self._task = asyncio.create_task(self._refresh())

    async def refresh(self) -> bool:
        """Refresh now and report whether names are available afterward.

        Used at start-up so the very first utterance is biased too.
        """
        await self._refresh()
        return self.has_names

    async def wait_for_refresh(self) -> None:
        """Give an in-flight refresh a bounded chance to land."""
        task = self._task
        if (task is None) or task.done():
            return

        try:
            # Shielded: a timeout here must not cancel the fetch. It keeps
            # running and its names are used by the next utterance.
            await asyncio.wait_for(asyncio.shield(task), timeout=self._wait_timeout)
        except asyncio.TimeoutError:
            _LOGGER.debug(
                "Names not refreshed within %ss; using the previous snapshot",
                self._wait_timeout,
            )

    async def initial_prompt(self) -> Optional[str]:
        """The prompt for this utterance: configured prefix plus current names."""
        await self.wait_for_refresh()

        context = self._context
        if context is None:
            return self._prefix

        prompt = context.biasing_prompt(
            self._count_tokens,
            max_tokens=self._max_tokens,
            prefix=self._prefix,
        )
        return prompt or self._prefix

    def _is_stale(self) -> bool:
        if self._updated is None:
            return True

        if self._refresh_seconds <= 0:
            return True

        return (time.monotonic() - self._updated) >= self._refresh_seconds

    async def _refresh(self) -> None:
        """Fetch the names, keeping the previous snapshot on failure."""
        try:
            context = await self._hass.get_context()
        except HomeAssistantError as exc:
            # Warn on the first failure of a run, then stay quiet: this runs once
            # per utterance and an unreachable Home Assistant would flood the log.
            if self._was_ok:
                _LOGGER.warning("Failed to load names from Home Assistant: %s", exc)
            else:
                _LOGGER.debug(
                    "Still failing to load names from Home Assistant: %s", exc
                )

            self._was_ok = False
            return

        self._was_ok = True
        self._updated = time.monotonic()

        if (self._context is not None) and (context == self._context):
            # Nothing was renamed. Keep the existing instance so the prompt it
            # already built stays cached and no tokenizing is repeated -- which
            # is what makes refreshing on every utterance cheap.
            return

        self._context = context
        _LOGGER.debug("Names updated (%s total)", len(context.all_names()))
