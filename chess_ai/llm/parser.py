#!/usr/bin/env python3
"""Move normalization and regex-based parsing for Chess AI."""

from __future__ import annotations

import json
import re
from typing import Any
import chess

UCI_MOVE_RE = re.compile(r"\b([a-h][1-8])([a-h][1-8])([qrbnQBRN])?\b")
LAN_MOVE_RE = re.compile(
    r"\b(?:[KQRBN])?([a-h][1-8])\s*[-x]\s*([a-h][1-8])(?:\s*=\s*([qrbnQBRN]))?[+#]?\b",
    re.IGNORECASE,
)
SAN_FRAGMENT_RE = re.compile(
    r"\b(?:O-O-O|O-O|0-0-0|0-0|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QBRNqbrn])?[+#]?|"
    r"[KQRBN][a-h]?[1-8]?[+#]?)\b",
    re.IGNORECASE,
)

def clean_move_text(text: str) -> str:
    """Normalize lightweight notation noise around a candidate move string."""
    cleaned = text.strip().strip("`\"'")
    cleaned = cleaned.replace("–", "-").replace("—", "-").replace("−", "-")
    cleaned = cleaned.replace("0-0-0", "O-O-O").replace("0-0", "O-O")
    cleaned = re.sub(r"\s+", "", cleaned)
    cleaned = re.sub(r"[!?]+", "", cleaned)
    cleaned = cleaned.rstrip(".,;:")
    return cleaned

def standardize_san_text(text: str) -> str:
    """Normalize SAN/LAN casing for piece letters, promotions, and castling."""
    cleaned = clean_move_text(text)
    cleaned = re.sub(r"^(o-o-o|0-0-0)$", "O-O-O", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(o-o|0-0)$", "O-O", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^([kqrbn])", lambda match: match.group(1).upper(), cleaned)
    cleaned = re.sub(r"=([kqrbn])", lambda match: "=" + match.group(1).upper(), cleaned)
    return cleaned

def move_from_uci_text(text: str, board: chess.Board) -> chess.Move | None:
    """Parse UCI-like coordinate notation into a legal move when possible."""
    cleaned = clean_move_text(text).lower()
    match = UCI_MOVE_RE.fullmatch(cleaned)
    if match is None:
        return None

    promotion = (match.group(3) or "").lower()
    candidate = f"{match.group(1)}{match.group(2)}{promotion}"
    try:
        move = chess.Move.from_uci(candidate)
    except ValueError:
        return None
    return move if move in board.legal_moves else None

def move_from_lan_text(text: str, board: chess.Board) -> chess.Move | None:
    """Parse long algebraic / coordinate-hyphen notation into a legal move."""
    cleaned = standardize_san_text(text)
    match = LAN_MOVE_RE.fullmatch(cleaned)
    if match is None:
        return None

    promotion = (match.group(3) or "").lower()
    candidate = f"{match.group(1)}{match.group(2)}{promotion}"
    try:
        move = chess.Move.from_uci(candidate)
    except ValueError:
        return None
    return move if move in board.legal_moves else None

def move_from_san_text(text: str, board: chess.Board) -> chess.Move | None:
    """Parse SAN or standard castling notation into a legal move."""
    cleaned = standardize_san_text(text)
    if not cleaned:
        return None
    try:
        return board.parse_san(cleaned)
    except ValueError:
        return None

def move_from_castling_phrase(text: str, board: chess.Board) -> chess.Move | None:
    """Parse common natural-language castling phrases."""
    lowered = text.lower()
    castle_targets: list[str] = []
    if "queenside castle" in lowered or "castle queenside" in lowered or "long castle" in lowered:
        castle_targets.append("O-O-O")
    if "kingside castle" in lowered or "castle kingside" in lowered or "short castle" in lowered:
        castle_targets.append("O-O")

    for notation in castle_targets:
        move = move_from_san_text(notation, board)
        if move is not None:
            return move
    return None

def candidate_fragments(text: str) -> list[str]:
    """Extract move-like fragments from noisy model text."""
    fragments: list[str] = []

    def add(fragment: str) -> None:
        cleaned = fragment.strip()
        if cleaned and cleaned not in fragments:
            fragments.append(cleaned)

    add(text.strip())
    for pattern in (UCI_MOVE_RE, LAN_MOVE_RE, SAN_FRAGMENT_RE):
        for match in pattern.finditer(text):
            add(match.group(0))
    return fragments

def parse_move_candidate(text: str, board: chess.Board) -> chess.Move | None:
    """Best-effort parsing for common LLM move output formats."""
    for fragment in candidate_fragments(text):
        for parser in (
            move_from_uci_text,
            move_from_lan_text,
            move_from_san_text,
            move_from_castling_phrase,
        ):
            move = parser(fragment, board)
            if move is not None:
                return move
    return None

def move_to_uci(move: chess.Move) -> str:
    """Return the canonical UCI string for a parsed move."""
    return move.uci()

def collect_candidates(obj: Any, out: list[str]) -> None:
    """Recursively gather move-like string fields from nested JSON."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in {"chosen_move_uci", "best_move", "chosen_move", "move_uci", "move", "uci"} and isinstance(value, str):
                out.append(value)
            collect_candidates(value, out)
    elif isinstance(obj, list):
        for item in obj:
            collect_candidates(item, out)

def extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of a JSON object from model output text."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        snippet = cleaned[start : end + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None

    return None

def extract_phase(router_text: str) -> str | None:
    """Extract routed phase from router output JSON or fallback regex scan."""
    parsed = extract_json_object(router_text)
    if parsed is not None:
        phase_value = parsed.get("phase")
        if isinstance(phase_value, str):
            token = phase_value.strip().upper()
            for phase in ("OPENING", "MIDDLEGAME", "ENDGAME"):
                if phase == token or phase in token:
                    return phase

    regex_match = re.search(r"\b(OPENING|MIDDLEGAME|ENDGAME)\b", router_text, re.IGNORECASE)
    if regex_match:
        token = regex_match.group(1).strip().upper()
        for phase in ("OPENING", "MIDDLEGAME", "ENDGAME"):
            if phase == token or phase in token:
                return phase
    return None

def extract_move_from_phase_output(text: str, board: chess.Board) -> str:
    """Extract a legal move from phase output, scanning nested JSON first."""
    parsed = extract_json_object(text)
    if parsed is not None:
        candidates: list[str] = []
        collect_candidates(parsed, candidates)
        for candidate in candidates:
            try:
                move = parse_move_candidate(candidate, board)
                if move is not None:
                    return move_to_uci(move)
            except (ValueError, json.JSONDecodeError):
                continue

    move = parse_move_candidate(text, board)
    if move is not None:
        return move_to_uci(move)

    raise ValueError(f"No legal move found in model output: {text!r}")

def is_out_of_book_opening_response(text: str) -> bool:
    """Detect whether an OPENING response indicates out-of-book status."""
    parsed = extract_json_object(text)
    if parsed is not None:
        status = parsed.get("status")
        if isinstance(status, str) and status.strip().upper() == "OUT_OF_BOOK":
            return True
        out_of_book = parsed.get("out_of_book")
        if isinstance(out_of_book, bool):
            return out_of_book
        if isinstance(out_of_book, str):
            return out_of_book.strip().lower() in {"1", "true", "yes", "on"}
    return "OUT_OF_BOOK" in text.upper()
