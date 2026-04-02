#!/usr/bin/env python3
"""Ollama adapter for `play_chess.py`.

Execution model per turn:
1. Read payload JSON from stdin.
2. Route the position into OPENING/MIDDLEGAME/ENDGAME.
3. Run the selected phase prompt to choose a move.
4. Validate/extract a legal UCI move and print it to stdout.

On model/parsing/network failures, falls back to the first legal move so the
game loop can continue.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

try:
    import chess
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: python-chess. Install with: pip install python-chess"
    ) from exc

SCRIPT_DIR = Path(__file__).resolve().parent

MODEL = os.getenv("OLLAMA_MODEL", "llama4:latest")
OLLAMA_CHAT_URL = os.getenv("OLLAMA_CHAT_URL", "http://127.0.0.1:11434/api/chat")
TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))
DEFAULT_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.5"))

ROUTER_PROMPT_FILE = os.getenv("ROUTER_PROMPT_FILE", "router.txt")
OPENING_PROMPT_FILE = os.getenv("OPENING_PROMPT_FILE", "opening.txt")
MIDDLEGAME_PROMPT_FILE = os.getenv("MIDDLEGAME_PROMPT_FILE", "middlegame.txt")
ENDGAME_PROMPT_FILE = os.getenv("ENDGAME_PROMPT_FILE", "endgame.txt")

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
{"best_move":"uci"}
"""

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


def resolve_prompt_path(path_str: str) -> Path:
    """Resolve prompt path relative to this script unless already absolute."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path


def load_prompt(path_str: str, fallback: str) -> str:
    """Load prompt text from disk, or return fallback text if file is missing."""
    path = resolve_prompt_path(path_str)
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return fallback


def parse_debug_level(value: object) -> int:
    """Coerce common debug representations into an integer debug level."""
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "0", "false", "no", "off"}:
            return 0
        if normalized in {"true", "yes", "on"}:
            return 1
        try:
            return max(0, int(normalized))
        except ValueError:
            return 1
    return 0


def parse_temperature(value: object) -> float:
    """Parse per-request temperature value with env-backed default fallback."""
    if value is None:
        return DEFAULT_TEMPERATURE
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return DEFAULT_TEMPERATURE
    return DEFAULT_TEMPERATURE


def normalize_phase(text: str) -> str | None:
    """Normalize noisy phase text into OPENING/MIDDLEGAME/ENDGAME when possible."""
    token = text.strip().upper()
    for phase in ("OPENING", "MIDDLEGAME", "ENDGAME"):
        if phase == token:
            return phase
    for phase in ("OPENING", "MIDDLEGAME", "ENDGAME"):
        if phase in token:
            return phase
    return None


def to_json_text(value: object) -> str:
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


def call_ollama(system_prompt: str, user_prompt: str, temperature: float) -> tuple[str, str]:
    """Call Ollama `/api/chat` and return `(message.content, raw_json_response)`."""
    body = {
        "model": MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {"temperature": temperature},
    }

    req = urllib.request.Request(
        OLLAMA_CHAT_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as response:
        raw = response.read().decode("utf-8")
    data = json.loads(raw)
    content = data.get("message", {}).get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Ollama response missing `message.content`.")
    return content.strip(), raw


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of a JSON object from model output text."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\\s*```$", "", cleaned)

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
            if key in {"best_move", "chosen_move", "move", "uci"} and isinstance(value, str):
                out.append(value)
            collect_candidates(value, out)
    elif isinstance(obj, list):
        for item in obj:
            collect_candidates(item, out)


def extract_phase(router_text: str) -> str | None:
    """Extract routed phase from router output JSON or fallback regex scan."""
    parsed = extract_json_object(router_text)
    if parsed is not None:
        phase_value = parsed.get("phase")
        if isinstance(phase_value, str):
            normalized = normalize_phase(phase_value)
            if normalized:
                return normalized

    regex_match = re.search(r"\b(OPENING|MIDDLEGAME|ENDGAME)\b", router_text, re.IGNORECASE)
    if regex_match:
        return normalize_phase(regex_match.group(1))
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


def extract_reasoning_summary(text: str) -> str | None:
    """Extract a concise reasoning summary from phase output JSON when available."""
    parsed = extract_json_object(text)
    if parsed is None:
        return None

    for key in (
        "explanation",
        "justification",
        "retrieval_summary",
        "why_advantage",
        "plan",
    ):
        value = parsed.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    candidate_moves = parsed.get("candidate_moves")
    if isinstance(candidate_moves, list):
        snippets: list[str] = []
        for item in candidate_moves[:2]:
            if not isinstance(item, dict):
                continue
            move = item.get("move")
            why = item.get("why")
            if isinstance(move, str) and isinstance(why, str) and why.strip():
                snippets.append(f"{move}: {why.strip()}")
        if snippets:
            return " | ".join(snippets)

    top3 = parsed.get("top3")
    if isinstance(top3, list):
        snippets = []
        for item in top3[:2]:
            if not isinstance(item, dict):
                continue
            move = item.get("move")
            idea = item.get("strategic_idea")
            if isinstance(move, str) and isinstance(idea, str) and idea.strip():
                snippets.append(f"{move}: {idea.strip()}")
        if snippets:
            return " | ".join(snippets)

    return None


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


def main() -> int:
    """Adapter entrypoint used by `play_chess.py`.

    Input: JSON payload on stdin.
    Output: one UCI move on stdout.
    """
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON input: {exc}", file=sys.stderr)
        return 2

    legal_moves = payload.get("legal_moves", [])
    if not legal_moves:
        print("No legal moves in payload.", file=sys.stderr)
        return 1

    fen = payload.get("fen")
    if not isinstance(fen, str) or not fen.strip():
        print("Missing valid FEN in payload.", file=sys.stderr)
        return 2

    try:
        board = chess.Board(fen)
    except ValueError as exc:
        print(f"Invalid FEN in payload: {exc}", file=sys.stderr)
        return 2

    debug = parse_debug_level(payload.get("debug", 0))
    temperature = parse_temperature(payload.get("temperature"))
    master_prompt = payload.get("system_prompt", "")

    router_template = load_prompt(ROUTER_PROMPT_FILE, ROUTER_FALLBACK_PROMPT)
    opening_template = load_prompt(OPENING_PROMPT_FILE, PHASE_FALLBACK_PROMPT)
    middlegame_template = load_prompt(MIDDLEGAME_PROMPT_FILE, PHASE_FALLBACK_PROMPT)
    endgame_template = load_prompt(ENDGAME_PROMPT_FILE, PHASE_FALLBACK_PROMPT)

    try:
        router_prompt = render_prompt_template(router_template, payload)
        router_system = build_router_system_prompt(master_prompt)
        router_text, router_raw = call_ollama(router_system, router_prompt, temperature)

        phase = extract_phase(router_text) or "MIDDLEGAME"
        routed_phase = phase

        phase_templates = {
            "OPENING": opening_template,
            "MIDDLEGAME": middlegame_template,
            "ENDGAME": endgame_template,
        }
        phase_template = phase_templates[phase]
        phase_prompt = render_prompt_template(phase_template, payload)
        phase_system = build_phase_system_prompt(master_prompt, phase)
        phase_text, phase_raw = call_ollama(phase_system, phase_prompt, temperature)

        phase_call_logs: list[tuple[str, str, str]] = [(phase, phase_raw, phase_text)]

        if phase == "OPENING":
            try:
                opening_move = extract_move_from_phase_output(phase_text, board)
                if is_out_of_book_opening_response(phase_text):
                    raise ValueError("Opening response marked OUT_OF_BOOK.")
                move = opening_move
            except ValueError:
                phase = "MIDDLEGAME"
                phase_template = phase_templates[phase]
                phase_prompt = render_prompt_template(phase_template, payload)
                phase_system = build_phase_system_prompt(master_prompt, phase)
                phase_text, phase_raw = call_ollama(phase_system, phase_prompt, temperature)
                phase_call_logs.append((phase, phase_raw, phase_text))
                move = extract_move_from_phase_output(phase_text, board)
        else:
            move = extract_move_from_phase_output(phase_text, board)

        if debug >= 1:
            print(
                (
                    f"[ollama-debug] model={MODEL} temperature={temperature} "
                    f"timeout={TIMEOUT_SECONDS}s"
                ),
                file=sys.stderr,
            )
            print(f"[ollama-debug] routed_phase: {routed_phase}", file=sys.stderr)
            print(f"[ollama-debug] final_phase: {phase}", file=sys.stderr)
            print(f"[ollama-debug] router_extracted_content: {router_text!r}", file=sys.stderr)
            reasoning_summary = extract_reasoning_summary(phase_text)
            if reasoning_summary:
                print(f"[ollama-debug] reasoning: {reasoning_summary}", file=sys.stderr)
            for idx, (phase_name, raw_phase, extracted_phase) in enumerate(phase_call_logs, start=1):
                print(f"[ollama-debug] phase_call_{idx}: {phase_name}", file=sys.stderr)
                print(
                    f"[ollama-debug] phase_extracted_content: {extracted_phase!r}",
                    file=sys.stderr,
                )
                if debug >= 2:
                    print("[ollama-debug] phase_raw_response:", file=sys.stderr)
                    print(raw_phase.rstrip(), file=sys.stderr)
            if debug >= 2:
                print("[ollama-debug] router_raw_response:", file=sys.stderr)
                print(router_raw.rstrip(), file=sys.stderr)
            print(f"[ollama-debug] selected_move: {move}", file=sys.stderr)

    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        fallback = legal_moves[0]
        print(f"Ollama adapter warning: {exc}. Falling back to {fallback}.", file=sys.stderr)
        print(fallback)
        return 0

    print(move)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
