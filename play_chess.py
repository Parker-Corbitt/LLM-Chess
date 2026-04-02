#!/usr/bin/env python3
"""Terminal chess runner with optional LLM-backed players.

This module owns the game loop and move validation. For LLM turns, it builds
position JSON and calls an external adapter command on stdin/stdout.
"""

from __future__ import annotations

import argparse
from typing import Any
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional

try:
    import chess
    import chess.pgn
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: python-chess. Install with: pip install python-chess"
    ) from exc

from opening import DEFAULT_BOOK_PATH, OpeningBook, load_opening_book


def board_to_pgn(board: chess.Board) -> str:
    """Return the current game as compact PGN (no headers/variations/comments)."""
    game = chess.pgn.Game.from_board(board)
    exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
    return game.accept(exporter).strip()


def parse_uci_or_none(text: str, board: chess.Board) -> Optional[chess.Move]:
    """Parse a UCI move and ensure it is legal in `board`; otherwise return `None`."""
    move_text = text.strip()
    if not move_text:
        return None
    try:
        move = chess.Move.from_uci(move_text)
    except ValueError:
        return None
    return move if move in board.legal_moves else None


def render_position(board: chess.Board) -> None:
    """Print an ASCII board plus machine-readable context (FEN/PGN/side to move)."""
    print("\n" + str(board))
    print(f"FEN: {board.fen()}")
    if board.move_stack:
        print(f"PGN: {board_to_pgn(board)}")
    print(f"Turn: {'white' if board.turn == chess.WHITE else 'black'}")


def show_legal_moves(board: chess.Board) -> None:
    """Print current legal moves in UCI form on one line."""
    legal = [move.uci() for move in board.legal_moves]
    print("Legal moves (UCI):")
    print(" ".join(legal))


def get_human_move(board: chess.Board, show_legal: bool) -> Optional[chess.Move]:
    """Prompt until a valid action is provided.

    Returns:
    - `None` if user quits
    - `chess.Move.null()` if user resigns
    - a legal move otherwise
    """
    if show_legal:
        show_legal_moves(board)
    while True:
        raw = input("Enter UCI move (`legal`, `resign`, `quit`): ").strip()
        key = raw.lower()
        if key in {"quit", "exit"}:
            return None
        if key == "legal":
            show_legal_moves(board)
            continue
        if key == "resign":
            return chess.Move.null()
        move = parse_uci_or_none(raw, board)
        if move is None:
            print("Invalid or illegal UCI move. Example format: e2e4, g1f3, e7e8q")
            continue
        return move


def extract_move_from_llm_stdout(stdout_text: str) -> str:
    """Extract a candidate move from adapter stdout.

    Supported adapter outputs:
    - raw UCI text: `e2e4`
    - JSON object with one of: `move`, `best_move`, `uci`
    """
    text = stdout_text.strip()
    if not text:
        raise ValueError("LLM command returned empty output.")
    if text.startswith("{"):
        data = json.loads(text)
        for key in ("move", "best_move", "uci"):
            if key in data and isinstance(data[key], str):
                return data[key].strip()
        raise ValueError("LLM JSON output missing one of keys: move, best_move, uci")
    return text.split()[0]


def get_llm_move(
    board: chess.Board,
    llm_command: str,
    system_prompt: str,
    opening_db_entries: Optional[list[dict[str, Any]]] = None,
    debug: bool = False,
) -> chess.Move:
    """Request a move from the configured adapter command.

    The adapter receives JSON on stdin (position, legal moves, prompt context)
    and must print a move on stdout. This function validates legality before
    returning.
    """
    payload = {
        "fen": board.fen(),
        "pgn": board_to_pgn(board),
        "side_to_move": "white" if board.turn == chess.WHITE else "black",
        "move_number": board.fullmove_number,
        "fullmove_number": board.fullmove_number,
        "halfmove_clock": board.halfmove_clock,
        "legal_moves": [move.uci() for move in board.legal_moves],
        "last_move": board.move_stack[-1].uci() if board.move_stack else None,
        "opening_db_entries": opening_db_entries or [],
        "time_info": None,
        "draw_state": {
            "halfmove_clock": board.halfmove_clock,
            "can_claim_fifty_moves": board.can_claim_fifty_moves(),
            "can_claim_threefold_repetition": board.can_claim_threefold_repetition(),
            "is_repetition_3": board.is_repetition(3),
        },
        "system_prompt": system_prompt,
        "debug": debug,
    }

    cmd = shlex.split(llm_command)
    if not cmd:
        raise ValueError("LLM command is empty.")

    result = subprocess.run(
        cmd,
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
    )
    if debug and result.stderr.strip():
        print("[llm-debug] Adapter stderr:", file=sys.stderr)
        print(result.stderr.rstrip(), file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            "LLM command failed "
            f"(exit={result.returncode}). stderr:\n{result.stderr.strip()}"
        )
    if debug:
        print(f"[llm-debug] Adapter stdout: {result.stdout.strip()!r}", file=sys.stderr)

    move_text = extract_move_from_llm_stdout(result.stdout)
    move = parse_uci_or_none(move_text, board)
    if move is None:
        raise ValueError(f"LLM returned invalid/illegal move: {move_text!r}")
    return move


def parse_args() -> argparse.Namespace:
    """Define and parse CLI arguments for terminal play."""
    parser = argparse.ArgumentParser(
        description="Play chess with UCI moves in terminal (human and/or LLM players)."
    )
    parser.add_argument(
        "--white-player",
        choices=("human", "llm"),
        default="human",
        help="Who plays White.",
    )
    parser.add_argument(
        "--black-player",
        choices=("human", "llm"),
        default="human",
        help="Who plays Black.",
    )
    parser.add_argument(
        "--llm-command",
        default="python3 ollama_adapter.py",
        help=(
            "Command used for each LLM move. It receives JSON on stdin and must output "
            "a UCI move or JSON with move/best_move/uci. "
            "Default: `python3 ollama_adapter.py`"
        ),
    )
    parser.add_argument(
        "--prompt-file",
        default="Master.txt",
        help="Optional system prompt file passed through to the adapter payload.",
    )
    parser.add_argument(
        "--opening-book-file",
        default=str(DEFAULT_BOOK_PATH),
        help=(
            "Optional opening book JSON file. If the current position matches, "
            "the game uses the book move immediately instead of calling the LLM."
        ),
    )
    parser.add_argument(
        "--start-fen",
        default="",
        help="Optional custom FEN to start from.",
    )
    parser.add_argument(
        "--show-legal",
        action="store_true",
        help="Show legal UCI moves before each human move prompt.",
    )
    parser.add_argument(
        "--max-plies",
        type=int,
        default=0,
        help="Optional max half-moves to play before stopping (0 means no limit).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print adapter/Ollama debug output to terminal stderr during LLM turns.",
    )
    return parser.parse_args()


def load_system_prompt(prompt_file: str) -> str:
    """Load optional strategy text from disk; returns empty string if missing."""
    path = Path(prompt_file)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def load_opening_book_or_none(book_path: str, debug: bool = False) -> Optional[OpeningBook]:
    """Load an opening book if available; otherwise continue without it."""
    path = Path(book_path)
    if not path.exists():
        if debug:
            print(f"[opening-debug] Opening book not found at {path}", file=sys.stderr)
        return None

    try:
        return load_opening_book(path)
    except Exception as exc:  # noqa: BLE001
        if debug:
            print(
                f"[opening-debug] Failed to load opening book {path}: {exc}",
                file=sys.stderr,
            )
        return None


def main() -> int:
    """Run one full game until termination condition is reached."""
    args = parse_args()

    if not args.llm_command:
        print("Error: --llm-command cannot be empty.")
        return 2

    try:
        board = chess.Board(args.start_fen) if args.start_fen else chess.Board()
    except ValueError as exc:
        print(f"Error: invalid --start-fen value: {exc}")
        return 2
    system_prompt = load_system_prompt(args.prompt_file)
    opening_book = load_opening_book_or_none(args.opening_book_file, args.debug)

    plies = 0
    while not board.is_game_over(claim_draw=True):
        if args.max_plies and plies >= args.max_plies:
            print(f"Stopping after max plies: {args.max_plies}")
            break

        render_position(board)
        side_name = "white" if board.turn == chess.WHITE else "black"
        player = args.white_player if board.turn == chess.WHITE else args.black_player

        if player == "human":
            move = get_human_move(board, args.show_legal)
            if move is None:
                print("Game terminated by user.")
                return 0
            if move == chess.Move.null():
                winner = "black" if board.turn == chess.WHITE else "white"
                print(f"{side_name} resigned. Winner: {winner}.")
                return 0
        else:
            book_move_text = opening_book.best_move(board) if opening_book is not None else None
            if book_move_text:
                move = chess.Move.from_uci(book_move_text)
                if args.debug:
                    print(
                        f"[opening-debug] exact opening hit for {side_name}: {book_move_text}",
                        file=sys.stderr,
                    )
                print(f"Book ({side_name}) plays: {move.uci()} ({board.san(move)})")
            else:
                opening_db_entries = (
                    opening_book.retrieve_line_context(board) if opening_book is not None else []
                )
                if args.debug and opening_db_entries:
                    print(
                        f"[opening-debug] passing {len(opening_db_entries)} retrieved opening "
                        f"context entries to adapter for {side_name}",
                        file=sys.stderr,
                    )
                try:
                    move = get_llm_move(
                        board,
                        args.llm_command,
                        system_prompt,
                        opening_db_entries=opening_db_entries,
                        debug=args.debug,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"LLM move error: {exc}")
                    return 1
                print(f"LLM ({side_name}) plays: {move.uci()} ({board.san(move)})")

        board.push(move)
        plies += 1

    render_position(board)
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        print("Game ended without an outcome.")
        return 0

    winner = "draw" if outcome.winner is None else ("white" if outcome.winner else "black")
    print(f"Game over: {winner}. Termination: {outcome.termination.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
