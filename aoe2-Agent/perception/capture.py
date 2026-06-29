"""Screenshot capture with perceptual-hash deduplication.

Adapted from old/src/aoe2_coach/capture/screenshot.py — refactored to be
importable without old/ path hacks and to accept a Config object.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional, Tuple

import imagehash
from PIL import Image
import mss


def get_aoe2_window_bbox() -> Optional[Tuple[int, int, int, int]]:
    """Best-effort: locate the AoE2 window and return (left, top, width, height).

    Returns None when the window isn't found or pygetwindow isn't installed.
    Falls back to full-screen capture in the loop.
    """
    try:
        import pygetwindow as gw
    except Exception:
        return None

    for title in ("Age of Empires", "AoE2"):
        matches = [w for w in gw.getWindowsWithTitle(title) if w.width > 0]
        if matches:
            w = matches[0]
            return (w.left, w.top, w.width, w.height)
    return None


def capture_screenshot(bbox: Optional[Tuple[int, int, int, int]] = None) -> Image.Image:
    """Capture screen (or a region) and return a PIL Image in RGB mode."""
    with mss.mss() as s:
        if bbox is None:
            monitor = s.monitors[1]
        else:
            left, top, width, height = bbox
            monitor = {"left": left, "top": top, "width": width, "height": height}
        raw = s.grab(monitor)
        return Image.frombytes("RGB", raw.size, raw.rgb)


def save_screenshot(img: Image.Image, out_dir: str = "logs/screenshots") -> str:
    """Save image to out_dir with millisecond timestamp; return the file path."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(out_dir, f"screenshot_{int(time.time() * 1000)}.png")
    img.save(path)
    return path


class CaptureLoop:
    """Manages one-shot and continuous screenshot capture with dedup.

    Attributes:
        out_dir: Directory where screenshots are written.
        interval: Seconds between captures.
        hash_threshold: Perceptual-hash diff below which VLM call is skipped.
    """

    def __init__(
        self,
        out_dir: str = "logs/screenshots",
        interval: float = 3.0,
        hash_threshold: int = 8,
    ) -> None:
        self.out_dir = out_dir
        self.interval = interval
        self.hash_threshold = hash_threshold
        self._last_hash: Optional[imagehash.ImageHash] = None
        self._last_mini_hash: Optional[imagehash.ImageHash] = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _hash(self, img: Image.Image, size: Tuple[int, int] = (64, 64)) -> imagehash.ImageHash:
        return imagehash.average_hash(img.resize(size).convert("L"))

    def _crop_minimap(self, img: Image.Image) -> Image.Image:
        """Heuristically crop the minimap region (bottom-right ~18% of screen)."""
        w, h = img.size
        mw, mh = int(w * 0.18), int(h * 0.18)
        return img.crop((w - mw, h - mh, w, h))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_once(self) -> Tuple[str, Image.Image, bool]:
        """Capture one frame.

        Returns:
            (path, img, needs_vlm) where needs_vlm is False when the frame
            is visually too similar to the previous one to warrant a VLM call.
        """
        bbox = get_aoe2_window_bbox()
        img = capture_screenshot(bbox=bbox)
        path = save_screenshot(img, out_dir=self.out_dir)

        current_hash = self._hash(img)
        mini_hash = self._hash(self._crop_minimap(img))

        needs_vlm = True
        if self._last_hash is not None:
            diff = current_hash - self._last_hash
            mini_diff = (mini_hash - self._last_mini_hash) if self._last_mini_hash is not None else 99
            if diff <= self.hash_threshold and mini_diff <= (self.hash_threshold // 2):
                needs_vlm = False

        self._last_hash = current_hash
        self._last_mini_hash = mini_hash
        return path, img, needs_vlm
