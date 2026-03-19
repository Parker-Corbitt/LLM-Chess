#!/usr/bin/env python3
"""Minimal adapter for play_chess.py.

Reads JSON payload on stdin and prints one UCI move to stdout.
Replace the selection logic with an LLM call when ready.
"""

from __future__ import annotations

import json
import sys


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

    # Placeholder policy: choose first legal move.
    # Swap this with your LLM/tool call and return a UCI move string.
    print(legal_moves[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
