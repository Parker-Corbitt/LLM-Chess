#!/usr/bin/env python3
"""Prompt loading and rendering for Chess AI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Prompt file paths relative to project root
ROUTER_PROMPT_FILE = "prompts/router.txt"
OPENING_PROMPT_FILE = "prompts/opening.txt"
MIDDLEGAME_PROMPT_FILE = "prompts/middlegame.txt"
ENDGAME_PROMPT_FILE = "prompts/endgame.txt"

ROUTER_FALLBACK_PROMPT = """You are a chess game-state router.
Inputs:
- FEN: {fen}
- PGN so far: {pgn}
- Side to move: {side_to_move}
- Move number: {move_number}
- Retrieved opening line context: {opening_db_entries}
- Optional clock/time control: {time_info}
- Optional repetition/50-move info: {draw_state}

Output JSON only:
{"phase":"OPENING|MIDDLEGAME|ENDGAME","advantage":"WHITE|BLACK|EQUAL","confidence":0.0}
"""

PHASE_FALLBACK_PROMPT = """You are a chess move selector.
Inputs:
- FEN: {fen}
- PGN so far: {pgn}
- Side to move: {side_to_move}
- Legal moves: {legal_moves}
- Retrieved opening line context: {opening_db_entries}

Output JSON only:
{"chosen_move_uci":"uci"}
"""

def load_prompt(path_str: str, fallback: str) -> str:
    """Load prompt text from disk, or return fallback text if file is missing."""
    path = Path(path_str)
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return fallback

def to_json_text(value: Any) -> str:
    """Serialize values for text-template placeholders."""
    return json.dumps(value, ensure_ascii=False)

def render_prompt_template(template: str, payload: dict) -> str:
    """Render `{placeholder}` fields used by prompt files from request payload."""
    legal_moves = payload.get("legal_moves", [])
    replacements = {
        "fen": payload.get("fen", ""),
        "pgn": payload.get("pgn", ""),
        "side_to_move": payload.get("side_to_move", ""),
        "move_number": payload.get("move_number", payload.get("fullmove_number", "")),
        "legal_moves": " ".join(legal_moves),
        "opening_db_entries": to_json_text(payload.get("opening_db_entries", [])),
        "draw_state": to_json_text(payload.get("draw_state", {})),
        "time_info": to_json_text(payload.get("time_info", None)),
    }

    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace("{" + key + "}", str(value))
    return rendered

def build_router_system_prompt(master_prompt: str) -> str:
    """Compose system instructions for the routing call."""
    base = (
        "You are a chess phase router. "
        "Return strictly JSON and classify one phase: OPENING, MIDDLEGAME, or ENDGAME."
    )
    if master_prompt:
        return f"{base}\n\nGlobal strategy policy:\n{master_prompt}"
    return base

def build_phase_system_prompt(master_prompt: str, phase: str) -> str:
    """Compose system instructions for a phase selection call."""
    base = (
        f"You are a chess {phase.lower()} move selector. "
        "Return strictly JSON as requested and choose exactly one legal move from legal_moves."
    )
    if master_prompt:
        return f"{base}\n\nGlobal strategy policy:\n{master_prompt}"
    return base
