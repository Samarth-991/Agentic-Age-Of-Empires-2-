"""Perception Loop — Article III of the Constitution.

Screenshot → VLM (gemma4:latest) → GameStateSnapshot

This loop runs on its own asyncio task, completely decoupled from the
Strategist Loop. A slow or failing VLM call never blocks the next capture.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections import deque
from typing import Callable, Deque

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from config import (
    OLLAMA_BASE_URL,
    VLM_MODEL,
    PERCEPTION_INTERVAL_SECS,
    SCREENSHOT_OUT_DIR,
    HASH_THRESHOLD,
)
from schemas import GameStateSnapshot
from .capture import CaptureLoop

log = logging.getLogger(__name__)

VISION_SYSTEM_PROMPT = """\
You are the vision component of a real-time Age of Empires II: Definitive Edition \
coaching assistant. You will be shown one screenshot of the game in progress. \
Describe what you see by filling in the GameStateSnapshot JSON schema.
Do NOT invent numbers you cannot actually read.

What to look for:

- Top-left resource bar: food, wood, gold, stone (in that order), then population \
  (current / cap), then the Age icon. If a value is obscured or you are not confident, \
  set it to null rather than guessing.
- An animated "idle villager" icon near the population counter: set idle_villagers_visible \
  accordingly. If unclear, leave it false.
- Main viewport: count visible units by type and owner. Units in the player's own color \
  = "self"; differently colored = "enemy"; unclear = "unknown". Group identical units.
- Visible buildings in the viewport (Town Center, houses, production buildings, etc.).
- Minimap: enemy dots near the player's economy = threat_level "high"; elsewhere = "low"; \
  no enemy activity = "none".
- Anything notable that doesn't fit a field above goes in notes as one short sentence.

Respond with ONLY a valid JSON object matching the GameStateSnapshot schema. \
No markdown fences, no explanation — just the JSON.

Schema:
{
  "age": "Dark|Feudal|Castle|Imperial",
  "population": int or null,
  "pop_cap": int or null,
  "resources": {"food": int|null, "wood": int|null, "gold": int|null, "stone": int|null},
  "visible_units": [{"label": str, "owner": "self|enemy|unknown", "approx_count": int}],
  "visible_buildings": [str, ...],
  "idle_villagers_visible": bool,
  "threat_level": "none|low|high",
  "notes": str
}
"""


def _img_to_b64(img) -> str:
    """Convert PIL Image to base64 PNG string."""
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _parse_snapshot(raw: str) -> GameStateSnapshot | None:
    """Parse VLM text output into GameStateSnapshot; return None on failure."""
    text = raw.strip()
    # Strip accidental markdown fences
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        data = json.loads(text)
        return GameStateSnapshot.model_validate(data)
    except Exception as exc:
        log.warning("Failed to parse VLM output: %s\nRaw: %.200s", exc, raw)
        return None


class PerceptionLoop:
    """Async perception loop.

    Args:
        on_snapshot: Callback invoked with each successfully parsed snapshot.
        on_alarm: Callback invoked with a list of alarm strings each tick.
        interval: Seconds between captures (overrides config default).
    """

    def __init__(
        self,
        on_snapshot: Callable[[GameStateSnapshot], None],
        on_alarm: Callable[[list[str]], None] | None = None,
        interval: float = PERCEPTION_INTERVAL_SECS,
    ) -> None:
        self._on_snapshot = on_snapshot
        self._on_alarm = on_alarm
        self._interval = interval
        self._capture = CaptureLoop(
            out_dir=SCREENSHOT_OUT_DIR,
            interval=interval,
            hash_threshold=HASH_THRESHOLD,
        )
        self._vlm = ChatOllama(
            model=VLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0,
        )
        self._history: Deque[GameStateSnapshot] = deque(maxlen=20)
        self._last_snapshot: GameStateSnapshot | None = None

    @property
    def snapshot_history(self) -> Deque[GameStateSnapshot]:
        return self._history

    @property
    def latest_snapshot(self) -> GameStateSnapshot | None:
        return self._last_snapshot

    async def _call_vlm(self, img) -> GameStateSnapshot | None:
        """Send screenshot to VLM and parse the structured output."""
        b64 = _img_to_b64(img)
        messages = [
            SystemMessage(content=VISION_SYSTEM_PROMPT),
            HumanMessage(
                content=[
                    {"type": "text", "text": "Analyze this Age of Empires II screenshot."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ]
            ),
        ]
        try:
            response = await self._vlm.ainvoke(messages)
            return _parse_snapshot(response.content)
        except Exception as exc:
            log.error("VLM call failed: %s", exc)
            return None

    async def run(self, stop_event: asyncio.Event | None = None) -> None:
        """Run until stop_event is set (or forever if None)."""
        log.info("Perception loop starting (interval=%.1fs)", self._interval)
        while stop_event is None or not stop_event.is_set():
            try:
                _path, img, needs_vlm = self._capture.run_once()
                if needs_vlm:
                    snap = await self._call_vlm(img)
                    if snap is not None:
                        self._history.append(snap)
                        self._last_snapshot = snap
                        self._on_snapshot(snap)
                        if self._on_alarm:
                            from ..alarms.rules import check_alarms
                            alarms = check_alarms(snap, self._history)
                            if alarms:
                                self._on_alarm(alarms)
                else:
                    log.debug("Frame skipped (no significant change)")
            except Exception as exc:
                log.error("Perception loop error: %s", exc)

            await asyncio.sleep(self._interval)
