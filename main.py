#!/usr/bin/env python3
"""Main entry point for Chess AI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import chess
from chess_ai.core.engine import ChessGame
from chess_ai.core.opening_book import OpeningBook, load_opening_book, DEFAULT_BOOK_PATH
from chess_ai.llm.orchestrator import LLMOrchestrator

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
        "--prompt-file",
        default="prompts/Master.txt",
        help="Optional system prompt file passed to the AI.",
    )
    parser.add_argument(
        "--opening-book-file",
        default=str(DEFAULT_BOOK_PATH),
        help="Optional opening book JSON file.",
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
        action="count",
        default=0,
        help="Increase debug verbosity.",
    )
    return parser.parse_args()

def load_system_prompt(prompt_file: str) -> str:
    """Load optional strategy text from disk; returns empty string if missing."""
    path = Path(prompt_file)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()

def load_opening_book_or_none(book_path: str, debug: int = 0) -> Optional[OpeningBook]:
    """Load an opening book if available; otherwise continue without it."""
    path = Path(book_path)
    if not path.exists():
        if debug >= 1:
            print(f"[opening-debug] Opening book not found at {path}")
        return None
    try:
        return load_opening_book(path)
    except Exception as exc:
        if debug >= 1:
            print(f"[opening-debug] Failed to load opening book {path}: {exc}")
        return None

def main() -> int:
    """Run one full game until termination condition is reached."""
    args = parse_args()

    try:
        game = ChessGame(args.start_fen)
    except ValueError as exc:
        print(f"Error: invalid --start-fen value: {exc}")
        return 2

    system_prompt = load_system_prompt(args.prompt_file)
    opening_book = load_opening_book_or_none(args.opening_book_file, args.debug)

    orchestrator = LLMOrchestrator(
        master_prompt=system_prompt,
        temperature=0.5, # Can be moved to args
        debug=args.debug
    )

    plies = 0
    while not game.is_game_over():
        if args.max_plies and plies >= args.max_plies:
            print(f"Stopping after max plies: {args.max_plies}")
            break

        game.render_position()
        side_name = "white" if game.board.turn == chess.WHITE else "black"
        player = args.white_player if game.board.turn == chess.WHITE else args.black_player

        if player == "human":
            move = game.get_human_move(args.show_legal)
            if move is None:
                print("Game terminated by user.")
                return 0
            if move == chess.Move.null():
                winner = "black" if game.board.turn == chess.WHITE else "white"
                print(f"{side_name} resigned. Winner: {winner}.")
                return 0
        else:
            # Check Opening Book
            book_move_text = opening_book.best_move(game.board) if opening_book is not None else None
            if book_move_text:
                move = chess.Move.from_uci(book_move_text)
                if args.debug >= 1:
                    print(f"[opening-debug] exact opening hit for {side_name}: {book_move_text}")
                print(f"Book ({side_name}) plays: {move.uci()} ({game.board.san(move)})")
            else:
                # Prepare payload for LLM
                opening_db_entries = (
                    opening_book.retrieve_line_context(game.board) if opening_book is not None else []
                )

                payload = {
                    "fen": game.board.fen(),
                    "pgn": game.board_to_pgn(),
                    "side_to_move": side_name,
                    "move_number": game.board.fullmove_number,
                    "fullmove_number": game.board.fullmove_number,
                    "halfmove_clock": game.board.halfmove_clock,
                    "legal_moves": [move.uci() for move in game.board.legal_moves],
                    "last_move": game.board.move_stack[-1].uci() if game.board.move_stack else None,
                    "opening_db_entries": opening_db_entries,
                    "time_info": None,
                    "draw_state": {
                        "halfmove_clock": game.board.halfmove_clock,
                        "can_claim_fifty_moves": game.board.can_claim_fifty_moves(),
                        "can_claim_threefold_repetition": game.board.can_claim_threefold_repetition(),
                        "is_repetition_3": game.board.is_repetition(3),
                    },
                }

                try:
                    move = orchestrator.get_ai_move(payload, game.board)
                except Exception as exc:
                    print(f"LLM move error: {exc}")
                    return 1
                print(f"LLM ({side_name}) plays: {move.uci()} ({game.board.san(move)})")

        game.push_move(move)
        plies += 1

    game.render_position()
    outcome = game.get_outcome()
    if outcome is None:
        print("Game ended without an outcome.")
        return 0

    winner = "draw" if outcome.winner is None else ("white" if outcome.winner else "black")
    print(f"Game over: {winner}. Termination: {outcome.termination.name}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
