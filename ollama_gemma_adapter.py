#!/usr/bin/env python3
"""Ollama adapter for play_chess.py.

Reads JSON payload on stdin and prints one legal UCI move to stdout.
Uses two LLM calls per turn: router -> phase-specific prompt.
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

SCRIPT_DIR = Path(__file__).resolve().parent

MODEL = os.getenv("OLLAMA_MODEL", "gemma3:12b")
OLLAMA_CHAT_URL = os.getenv("OLLAMA_CHAT_URL", "http://127.0.0.1:11434/api/chat")
TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "90"))
DEFAULT_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.4"))

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

Output JSON only:
{"best_move":"uci"}
"""


def resolve_prompt_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path


def load_prompt(path_str: str, fallback: str) -> str:
    path = resolve_prompt_path(path_str)
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return fallback


def parse_debug_flag(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def parse_temperature(value: object) -> float:
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
    token = text.strip().upper()
    for phase in ("OPENING", "MIDDLEGAME", "ENDGAME"):
        if phase == token:
            return phase
    for phase in ("OPENING", "MIDDLEGAME", "ENDGAME"):
        if phase in token:
            return phase
    return None


def to_json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def render_prompt_template(template: str, payload: dict) -> str:
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


def extract_uci(text: str, legal_moves: set[str]) -> str:
    candidate = text.strip()
    if candidate.startswith("{"):
        data = json.loads(candidate)
        for key in ("move", "best_move", "chosen_move", "uci"):
            value = data.get(key)
            if isinstance(value, str):
                candidate = value.strip()
                break

    candidate = candidate.split()[0]
    candidate = re.sub(r"[^a-h1-8qrbn]", "", candidate.lower())
    if candidate in legal_moves:
        return candidate

    for match in re.findall(r"[a-h][1-8][a-h][1-8][qrbn]?", text.lower()):
        if match in legal_moves:
            return match

    raise ValueError(f"No legal UCI move found in model output: {text!r}")


def collect_candidates(obj: Any, out: list[str]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in {"best_move", "chosen_move", "move", "uci"} and isinstance(value, str):
                out.append(value)
            collect_candidates(value, out)
    elif isinstance(obj, list):
        for item in obj:
            collect_candidates(item, out)


def extract_phase(router_text: str) -> str | None:
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


def extract_move_from_phase_output(text: str, legal_moves: set[str]) -> str:
    parsed = extract_json_object(text)
    if parsed is not None:
        candidates: list[str] = []
        collect_candidates(parsed, candidates)
        for candidate in candidates:
            try:
                return extract_uci(candidate, legal_moves)
            except (ValueError, json.JSONDecodeError):
                continue
    return extract_uci(text, legal_moves)


def is_out_of_book_opening_response(text: str) -> bool:
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
    base = (
        "You are a chess phase router. "
        "Return strictly JSON and classify one phase: OPENING, MIDDLEGAME, or ENDGAME."
    )
    if master_prompt:
        return f"{base}\n\nGlobal strategy policy:\n{master_prompt}"
    return base


def build_phase_system_prompt(master_prompt: str, phase: str) -> str:
    base = (
        f"You are a chess {phase.lower()} move selector. "
        "Return strictly JSON as requested and choose exactly one legal move from legal_moves."
    )
    if master_prompt:
        return f"{base}\n\nGlobal strategy policy:\n{master_prompt}"
    return base


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON input: {exc}", file=sys.stderr)
        return 2

    legal_moves = payload.get("legal_moves", [])
    if not legal_moves:
        print("No legal moves in payload.", file=sys.stderr)
        return 1

    legal_set = set(legal_moves)
    debug = parse_debug_flag(payload.get("debug", False))
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
                opening_move = extract_move_from_phase_output(phase_text, legal_set)
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
                move = extract_move_from_phase_output(phase_text, legal_set)
        else:
            move = extract_move_from_phase_output(phase_text, legal_set)

        if debug:
            print(
                (
                    f"[ollama-debug] model={MODEL} temperature={temperature} "
                    f"timeout={TIMEOUT_SECONDS}s"
                ),
                file=sys.stderr,
            )
            print(f"[ollama-debug] routed_phase: {routed_phase}", file=sys.stderr)
            print(f"[ollama-debug] final_phase: {phase}", file=sys.stderr)
            print("[ollama-debug] router_raw_response:", file=sys.stderr)
            print(router_raw.rstrip(), file=sys.stderr)
            print(f"[ollama-debug] router_extracted_content: {router_text!r}", file=sys.stderr)
            for idx, (phase_name, raw_phase, extracted_phase) in enumerate(phase_call_logs, start=1):
                print(f"[ollama-debug] phase_call_{idx}: {phase_name}", file=sys.stderr)
                print("[ollama-debug] phase_raw_response:", file=sys.stderr)
                print(raw_phase.rstrip(), file=sys.stderr)
                print(
                    f"[ollama-debug] phase_extracted_content: {extracted_phase!r}",
                    file=sys.stderr,
                )
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
