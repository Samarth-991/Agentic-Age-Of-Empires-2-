from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

VLM_MODEL = "gemma4:latest"
OLLAMA_BASE_URL = "http://localhost:11434"

STRATEGY_MODEL = 'llama3.1:latest'

# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------

KB_PATH = Path("/mnt/e/Personal/Samarth/repository/AOE-Agent/knowledge_base")
KB_CIVS = KB_PATH / "civilizations" / "civs.json"
KB_UNITS = KB_PATH / "units" / "units.json"
KB_COUNTERS = KB_PATH / "counters" / "counters.json"
KB_BUILDINGS = KB_PATH / "buildings" / "building.json"

# ---------------------------------------------------------------------------
# Loop timing (seconds)
# ---------------------------------------------------------------------------

PERCEPTION_INTERVAL_SECS: float = 3.0
STRATEGIST_INTERVAL_SECS: float = 15.0

# ---------------------------------------------------------------------------
# Deterministic alarm thresholds (Article V — externally configurable)
# ---------------------------------------------------------------------------

ALARM_FOOD_FLOOR: int = 200
ALARM_WOOD_FLOOR: int = 100
ALARM_GOLD_FLOOR: int = 50
ALARM_IDLE_VILLAGER_SECS: int = 30
ALARM_POP_HEADROOM: int = 5      # alert when pop_cap - population <= this value

# ---------------------------------------------------------------------------
# Screenshot capture
# ---------------------------------------------------------------------------

SCREENSHOT_OUT_DIR: str = "logs/screenshots"
HASH_THRESHOLD: int = 8          # perceptual-hash diff below which VLM is skipped
