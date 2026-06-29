"""Knowledge base query tools for the Strategist Deep Agent.

All four functions are plain Python callables with docstrings — Deep Agents
registers them as tools automatically. Data is loaded once at import time to
avoid repeated disk I/O during a session.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..config import KB_BUILDINGS, KB_CIVS, KB_COUNTERS, KB_UNITS


# ---------------------------------------------------------------------------
# Internal loaders — cached after first load
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_civs() -> list[dict]:
    with open(KB_CIVS, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_units() -> list[dict]:
    with open(KB_UNITS, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_counters() -> dict:
    with open(KB_COUNTERS, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_buildings() -> list[dict]:
    with open(KB_BUILDINGS, encoding="utf-8") as f:
        return json.load(f)


def _fuzzy_find(records: list[dict], name: str, key: str = "name") -> dict | None:
    """Case-insensitive substring search across a list of records."""
    needle = name.strip().lower()
    # Exact match first
    for r in records:
        if r.get(key, "").lower() == needle:
            return r
    # Substring match
    for r in records:
        if needle in r.get(key, "").lower():
            return r
    return None


def _fmt(obj: Any) -> str:
    return json.dumps(obj, indent=2)


# ---------------------------------------------------------------------------
# Tool functions (registered by Deep Agents)
# ---------------------------------------------------------------------------


def get_civilization_info(civ_name: str) -> str:
    """Return bonuses and help text for a named Age of Empires II civilization.

    Args:
        civ_name: The civilization name, e.g. 'Britons', 'Mongols', 'Franks'.

    Returns:
        JSON string with the civilization record, or an error message if not found.
    """
    civs = _load_civs()
    needle = civ_name.strip().lower()
    match = next(
        (c for c in civs if c.get("name", "").lower() == needle or c.get("key", "").lower() == needle),
        None,
    )
    if not match:
        # Partial match fallback
        match = next(
            (c for c in civs if needle in c.get("name", "").lower()),
            None,
        )
    if match:
        return _fmt(match)
    available = sorted(c["name"] for c in civs)
    return f"Civilization '{civ_name}' not found. Available: {available}"


def get_unit_counters(unit_type: str) -> str:
    """Return what a unit type is strong against and weak against.

    Args:
        unit_type: Archetype such as 'archer', 'cavalry', 'infantry', 'knight',
                   'siege', 'monk', 'camel', 'spearman', 'skirmisher', 'eagle_warrior'.

    Returns:
        JSON string with strong_vs, weak_vs, and strategic notes.
    """
    counters = _load_counters()
    needle = unit_type.strip().lower()
    if needle in counters:
        return _fmt({needle: counters[needle]})
    # Partial match
    matches = {k: v for k, v in counters.items() if needle in k}
    if matches:
        return _fmt(matches)
    return (
        f"Unit type '{unit_type}' not found in counter matrix. "
        f"Available types: {sorted(counters.keys())}"
    )


def get_unit_stats(unit_name: str) -> str:
    """Return stats (HP, attack, armor, cost) for a specific unit by name.

    Args:
        unit_name: The unit's name, e.g. 'Archer', 'Knight', 'Mangudai', 'Hussar'.

    Returns:
        JSON string with the unit record (id, name, hit_points, attack, melee_armor,
        pierce_armor, cost, class), or an error message if not found.
    """
    units = _load_units()
    record = _fuzzy_find(units, unit_name)
    if record:
        # Return only the most useful fields to keep context compact
        summary = {
            "id": record.get("id"),
            "name": record.get("name"),
            "class": record.get("class"),
            "hit_points": record.get("hit_points"),
            "attack": record.get("attack"),
            "melee_armor": record.get("melee_armor"),
            "pierce_armor": record.get("pierce_armor"),
            "line_of_sight": record.get("line_of_sight"),
            "cost": record.get("cost"),
            "total_cost": record.get("total_cost"),
        }
        return _fmt(summary)
    return f"Unit '{unit_name}' not found. Try a partial name or check spelling."


def get_building_info(building_name: str) -> str:
    """Return stats and cost for a specific building by name.

    Args:
        building_name: The building's name, e.g. 'Town Center', 'Archery Range', 'Castle'.

    Returns:
        JSON string with the building record (id, name, category, hit_points,
        melee_armor, pierce_armor, cost), or an error message if not found.
    """
    buildings = _load_buildings()
    record = _fuzzy_find(buildings, building_name)
    if record:
        summary = {
            "id": record.get("id"),
            "name": record.get("name"),
            "category": record.get("category"),
            "hit_points": record.get("hit_points"),
            "melee_armor": record.get("melee_armor"),
            "pierce_armor": record.get("pierce_armor"),
            "garrison_capacity": record.get("garrison_capacity"),
            "cost": record.get("cost"),
        }
        return _fmt(summary)
    return f"Building '{building_name}' not found. Try a partial name or check spelling."


# Expose all tools as a list for easy import
KB_TOOLS = [get_civilization_info, get_unit_counters, get_unit_stats, get_building_info]
