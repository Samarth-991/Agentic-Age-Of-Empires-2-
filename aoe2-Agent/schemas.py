from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DetectedUnit(BaseModel):
    label: str = Field(description="Unit type name, e.g. 'knight', 'villager', 'skirmisher'")
    owner: Literal["self", "enemy", "unknown"] = Field(
        description="'self' for player units, 'enemy' for opponent, 'unknown' when unclear"
    )
    approx_count: int = Field(description="Approximate number of this unit visible on screen")


class GameStateSnapshot(BaseModel):
    age: Literal["Dark", "Feudal", "Castle", "Imperial"] = Field(
        description="Current in-game age"
    )
    population: int | None = Field(
        default=None,
        description="Current population count; null if unreadable",
    )
    pop_cap: int | None = Field(
        default=None,
        description="Current population cap; null if unreadable",
    )
    resources: dict[str, int | None] = Field(
        default_factory=dict,
        description="Resource counts keyed by name: food, wood, gold, stone. Null if unreadable.",
    )
    visible_units: list[DetectedUnit] = Field(
        default_factory=list,
        description="All visible units grouped by label and owner",
    )
    visible_buildings: list[str] = Field(
        default_factory=list,
        description="Names of visible buildings in the viewport",
    )
    idle_villagers_visible: bool = Field(
        default=False,
        description="True when the idle-villager icon is visible near the population counter",
    )
    threat_level: Literal["none", "low", "high"] = Field(
        default="none",
        description="Threat assessment based on enemy unit positions on minimap",
    )
    notes: str = Field(
        default="",
        description="Any notable events not captured by other fields",
    )


class MatchContext(BaseModel):
    """Metadata provided at session start — never changes mid-game."""

    player_civ: str
    opponent_civ: str | None = None
