#!/usr/bin/env python3
"""Game loop and board management for the Chess AI."""

from __future__ import annotations

from typing import Optional
import chess
import chess.pgn

class ChessGame:
    """Encapsulates the chess board state and game-related utility functions."""

    def __init__(self, start_fen: str = ""):
        self.board = chess.Board(start_fen) if start_fen else chess.Board()

    def board_to_pgn(self) -> str:
        """Return the current game as compact PGN (no headers/variations/comments)."""
        game = chess.pgn.Game.from_board(self.board)
        exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
        return game.accept(exporter).strip()

    def parse_uci_or_none(self, text: str) -> Optional[chess.Move]:
        """Parse a UCI move and ensure it is legal in the current board; otherwise return None."""
        move_text = text.strip()
        if not move_text:
            return None
        try:
            move = chess.Move.from_uci(move_text)
        except ValueError:
            return None
        return move if move in self.board.legal_moves else None

    def render_position(self) -> None:
        """Print an ASCII board plus machine-readable context (FEN/PGN/side to move)."""
        print("\n" + str(self.board))
        print(f"FEN: {self.board.fen()}")
        if self.board.move_stack:
            print(f"PGN: {self.board_to_pgn()}")
        print(f"Turn: {'white' if self.board.turn == chess.WHITE else 'black'}")

    def show_legal_moves(self) -> None:
        """Print current legal moves in UCI form on one line."""
        legal = [move.uci() for move in self.board.legal_moves]
        print("Legal moves (UCI):")
        print(" ".join(legal))

    def get_human_move(self, show_legal: bool) -> Optional[chess.Move]:
        """Prompt until a valid action is provided.

        Returns:
        - None if user quits
        - chess.Move.null() if user resigns
        - a legal move otherwise
        """
        if show_legal:
            self.show_legal_moves()
        while True:
            raw = input("Enter UCI move (`legal`, `resign`, `quit`): ").strip()
            key = raw.lower()
            if key in {"quit", "exit"}:
                return None
            if key == "legal":
                self.show_legal_moves()
                continue
            if key == "resign":
                return chess.Move.null()
            move = self.parse_uci_or_none(raw)
            if move is None:
                print("Invalid or illegal UCI move. Example format: e2e4, g1f3, e7e8q")
                continue
            return move

    def push_move(self, move: chess.Move) -> None:
        """Push a move onto the board."""
        self.board.push(move)

    def is_game_over(self) -> bool:
        """Check if the game is over."""
        return self.board.is_game_over(claim_draw=True)

    def get_outcome(self) -> Optional[chess.Outcome]:
        """Get the outcome of the game."""
        return self.board.outcome(claim_draw=True)
