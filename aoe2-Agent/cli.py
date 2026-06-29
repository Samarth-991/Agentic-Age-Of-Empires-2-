"""Typer CLI entry point for the AoE2 Coaching Agent.

Usage:
    python -m aoe2_agent start                # Launches interactive session
    python -m aoe2_agent start --civ Britons  # Skip the civ selection prompt
    python -m aoe2_agent list-civs            # Print all available civilizations
    python -m aoe2_agent test-kb              # Smoke-test the knowledge-base tools
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import IntPrompt, Prompt
from rich.table import Table
from rich import print as rprint

from .config import KB_CIVS, PERCEPTION_INTERVAL_SECS, STRATEGIST_INTERVAL_SECS
from .display.overlay import CoachOverlay
from .perception.loop import PerceptionLoop
from .schemas import GameStateSnapshot
from .strategy.agent import StrategistLoop

app = typer.Typer(
    name="aoe2-agent",
    help="AoE2 Coaching Agent — real-time read-only strategy advisor",
    add_completion=False,
)
console = Console()
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_civs() -> list[dict]:
    with open(KB_CIVS, encoding="utf-8") as f:
        return json.load(f)


def _select_civ(civs: list[dict]) -> str:
    """Interactive civilization selection — returns the chosen civ name."""
    console.print(
        Panel(
            "[bold cyan]Age of Empires II — Coaching Agent[/bold cyan]\n"
            "[dim]Read-only coaching assistant. It watches; you play.[/dim]",
            border_style="cyan",
        )
    )

    table = Table(title="Available Civilizations", show_lines=False, box=None)
    table.add_column("#", style="dim", width=4)
    table.add_column("Civilization", style="bold white")
    table.add_column("Bonus preview", style="dim", no_wrap=False, max_width=60)

    for i, civ in enumerate(civs, 1):
        first_bonus = civ.get("bonuses", ["—"])[0] if civ.get("bonuses") else "—"
        table.add_row(str(i), civ["name"], first_bonus)

    console.print(table)
    console.print()

    while True:
        choice = IntPrompt.ask(
            f"[bold]Select your civilization[/bold] [dim](1–{len(civs)})[/dim]"
        )
        if 1 <= choice <= len(civs):
            selected = civs[choice - 1]
            break
        console.print(f"[red]Please enter a number between 1 and {len(civs)}.[/red]")

    # Show the selected civ's bonuses
    console.print()
    bonuses = selected.get("bonuses", [])
    bonus_text = "\n".join(f"  • {b}" for b in bonuses) or "  [dim]No bonuses listed[/dim]"
    console.print(
        Panel(
            f"[bold green]{selected['name']}[/bold green]\n\n{bonus_text}",
            title="Selected Civilization",
            border_style="green",
        )
    )

    confirmed = Prompt.ask(
        "[bold]Start coaching session with this civilization?[/bold]",
        choices=["y", "n"],
        default="y",
    )
    if confirmed.lower() != "y":
        return _select_civ(civs)  # Let them pick again

    return selected["name"]


def _select_opponent_civ(civs: list[dict], player_civ: str) -> str:
    """Ask the player which civilization their opponent is playing.

    Returns the civ name, or 'Unknown' if the player skips.
    """
    console.print()
    console.print(
        Panel(
            "[bold yellow]Who is your opponent playing?[/bold yellow]\n"
            "[dim]Knowing the opponent's civilization lets the agent give matchup-aware advice.\n"
            "Press Enter to skip if you don't know yet.[/dim]",
            border_style="yellow",
        )
    )

    # Compact numbered list (exclude the player's own civ)
    others = [c for c in civs if c["name"] != player_civ]
    table = Table(show_lines=False, box=None, show_header=False)
    table.add_column("#", style="dim", width=4)
    table.add_column("Civilization", style="white")

    for i, civ in enumerate(others, 1):
        table.add_row(str(i), civ["name"])

    console.print(table)
    console.print()

    raw = Prompt.ask(
        f"[bold]Opponent civilization[/bold] [dim](1–{len(others)}, or Enter to skip)[/dim]",
        default="",
    )

    if not raw.strip():
        console.print("[dim]Opponent unknown — matchup analysis will be skipped.[/dim]")
        return "Unknown"

    try:
        idx = int(raw.strip()) - 1
        if 0 <= idx < len(others):
            chosen = others[idx]["name"]
        else:
            raise ValueError
    except (ValueError, IndexError):
        # Try treating input as a name
        needle = raw.strip().lower()
        match = next((c["name"] for c in civs if needle in c["name"].lower()), None)
        if match:
            chosen = match
        else:
            console.print("[yellow]Not recognized — treating opponent as Unknown.[/yellow]")
            return "Unknown"

    console.print(
        Panel(
            f"[bold red]Opponent:[/bold red] [bold]{chosen}[/bold]",
            border_style="red",
            padding=(0, 1),
        )
    )
    return chosen


async def _main_loop(player_civ: str, opponent_civ: str, overlay: CoachOverlay) -> None:
    """Async dual-loop orchestration (Article III)."""
    stop_event = asyncio.Event()

    def on_snapshot(snap: GameStateSnapshot) -> None:
        overlay.set_snapshot(snap)

    def on_alarm(alarms: list[str]) -> None:
        overlay.set_alarms(alarms)

    def on_strategy(text: str) -> None:
        overlay.set_strategy(text)
        overlay.clear_alarms()  # Reset alarms after each strategist cycle

    perception = PerceptionLoop(
        on_snapshot=on_snapshot,
        on_alarm=on_alarm,
        interval=PERCEPTION_INTERVAL_SECS,
    )
    strategist = StrategistLoop(
        player_civ=player_civ,
        opponent_civ=opponent_civ,
        on_strategy=on_strategy,
        interval=STRATEGIST_INTERVAL_SECS,
        perception_loop=perception,
    )

    try:
        await asyncio.gather(
            perception.run(stop_event),
            strategist.run(stop_event),
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        stop_event.set()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def start(
    civ: Optional[str] = typer.Option(
        None, "--civ", "-c", help="Your civilization name (skip interactive picker)."
    ),
    opponent: Optional[str] = typer.Option(
        None, "--opponent", "-o", help="Opponent's civilization name (skip interactive picker)."
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging."),
) -> None:
    """Start a coaching session. Asks for both civilizations if not provided."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    civs = _load_civs()
    civ_names_lower = [c["name"].lower() for c in civs]

    # ── Player civilization ───────────────────────────────────────────────────
    if civ:
        if civ.lower() not in civ_names_lower:
            console.print(f"[red]Unknown civilization '{civ}'. Run `aoe2-agent list-civs` for options.[/red]")
            raise typer.Exit(1)
        player_civ = next(c["name"] for c in civs if c["name"].lower() == civ.lower())
    else:
        player_civ = _select_civ(civs)

    # ── Opponent civilization ─────────────────────────────────────────────────
    if opponent:
        if opponent.lower() not in civ_names_lower:
            console.print(f"[yellow]Unknown opponent civ '{opponent}' — treating as Unknown.[/yellow]")
            opponent_civ = "Unknown"
        else:
            opponent_civ = next(c["name"] for c in civs if c["name"].lower() == opponent.lower())
    else:
        opponent_civ = _select_opponent_civ(civs, player_civ)

    # ── Session summary ───────────────────────────────────────────────────────
    console.print()
    console.print(
        Panel(
            f"[bold cyan]{player_civ}[/bold cyan]  vs  [bold red]{opponent_civ}[/bold red]\n\n"
            "[dim]Matchup-aware strategy will be generated each cycle.\n"
            "Initializing Ollama models and knowledge base — first analysis may take a moment.[/dim]",
            title="[bold]Coaching Session[/bold]",
            border_style="cyan",
        )
    )

    overlay = CoachOverlay(player_civ=f"{player_civ} vs {opponent_civ}")

    # Run the async loops in a background thread so the Rich Live display
    # can own the main thread (it uses terminal raw mode).
    loop = asyncio.new_event_loop()

    def run_loops() -> None:
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_main_loop(player_civ, opponent_civ, overlay))
        finally:
            loop.close()

    bg = threading.Thread(target=run_loops, daemon=True)
    bg.start()

    try:
        overlay.start()  # Blocks on main thread; Ctrl+C to exit
    except KeyboardInterrupt:
        overlay.stop()
        console.print("\n[bold]Session ended. Good game![/bold]")


@app.command("list-civs")
def list_civs() -> None:
    """Print all available civilizations with their bonuses."""
    civs = _load_civs()
    table = Table(title=f"AoE2 Civilizations ({len(civs)} total)", show_lines=True)
    table.add_column("Civilization", style="bold cyan", width=20)
    table.add_column("Bonuses")

    for civ in sorted(civs, key=lambda c: c["name"]):
        bonuses = civ.get("bonuses", [])
        bonus_text = "\n".join(f"• {b}" for b in bonuses[:4])
        if len(bonuses) > 4:
            bonus_text += f"\n[dim]+{len(bonuses) - 4} more[/dim]"
        table.add_row(civ["name"], bonus_text or "[dim]—[/dim]")

    console.print(table)


@app.command("test-kb")
def test_kb(
    civ: str = typer.Argument("Britons", help="Civilization to look up"),
    unit: str = typer.Argument("Archer", help="Unit to look up"),
) -> None:
    """Smoke-test the knowledge base tools — useful for verifying setup."""
    from .knowledge.tools import (
        get_building_info,
        get_civilization_info,
        get_unit_counters,
        get_unit_stats,
    )

    console.print(f"\n[bold cyan]KB test[/bold cyan]\n")

    console.print(f"[bold]Civilization: {civ}[/bold]")
    console.print(get_civilization_info(civ))

    console.print(f"\n[bold]Unit stats: {unit}[/bold]")
    console.print(get_unit_stats(unit))

    console.print(f"\n[bold]Unit counters: archer[/bold]")
    console.print(get_unit_counters("archer"))

    console.print(f"\n[bold]Building: Town Center[/bold]")
    console.print(get_building_info("Town Center"))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
