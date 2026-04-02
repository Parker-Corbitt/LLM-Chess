#!/usr/bin/env python3
"""Opening-book loader and exact-position lookup helpers.

This module implements the first step of the opening-book plan:
1. Load a small curated opening database from disk.
2. Index each reachable position to one or more known book moves.
3. Return structured metadata for later integration with `play_chess.py`
   and the LLM adapter payload.

The lookup is deterministic and position-based. It does not use embeddings or
semantic retrieval; for chess openings, exact board-state matching is the right
primitive.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import chess

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BOOK_PATH = SCRIPT_DIR / "opening_book.json"


def position_key(board: chess.Board) -> str:
    """Return a stable key for exact position matching.

    The key intentionally excludes halfmove/fullmove counters because opening
    lookup should match the same board state regardless of move count.
    """

    castling = board.castling_xfen() or "-"
    ep_square = chess.square_name(board.ep_square) if board.ep_square is not None else "-"
    turn = "w" if board.turn == chess.WHITE else "b"
    return f"{board.board_fen()} {turn} {castling} {ep_square}"


@dataclass(frozen=True)
class OpeningLine:
    """One named opening line loaded from the JSON database."""

    eco: str
    name: str
    moves: tuple[str, ...]
    source: str = "builtin"
    start_fen: str = chess.STARTING_FEN

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OpeningLine":
        """Validate and normalize one JSON opening line record."""

        eco = data.get("eco")
        name = data.get("name")
        moves = data.get("moves")
        source = data.get("source", "builtin")
        start_fen = data.get("start_fen", chess.STARTING_FEN)

        if not isinstance(eco, str) or not eco.strip():
            raise ValueError(f"Opening line missing valid eco code: {data!r}")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Opening line missing valid name: {data!r}")
        if not isinstance(moves, list) or not moves or not all(isinstance(m, str) for m in moves):
            raise ValueError(f"Opening line missing valid moves list: {data!r}")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"Opening line missing valid source: {data!r}")
        if not isinstance(start_fen, str) or not start_fen.strip():
            raise ValueError(f"Opening line missing valid start_fen: {data!r}")

        return cls(
            eco=eco.strip(),
            name=name.strip(),
            moves=tuple(move.strip() for move in moves),
            source=source.strip(),
            start_fen=start_fen.strip(),
        )


@dataclass(frozen=True)
class OpeningChoice:
    """Aggregated book move metadata for one exact position."""

    move: str
    weight: int
    eco_codes: tuple[str, ...]
    names: tuple[str, ...]
    sources: tuple[str, ...]
    max_depth: int

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable form suitable for adapter payloads."""

        return {
            "move": self.move,
            "weight": self.weight,
            "eco_codes": list(self.eco_codes),
            "opening_names": list(self.names),
            "sources": list(self.sources),
            "max_depth": self.max_depth,
        }


class OpeningBook:
    """Indexed opening book keyed by exact board state."""

    def __init__(self, index: dict[str, list[OpeningChoice]], line_count: int) -> None:
        self._index = index
        self.line_count = line_count

    @classmethod
    def from_lines(cls, lines: Iterable[OpeningLine]) -> "OpeningBook":
        """Build an opening book from validated opening lines."""

        index: dict[str, dict[str, dict[str, Any]]] = {}
        line_count = 0

        for line in lines:
            board = chess.Board(line.start_fen)
            line_count += 1

            for depth, move_text in enumerate(line.moves, start=1):
                try:
                    move = chess.Move.from_uci(move_text)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid UCI move {move_text!r} in line {line.eco} {line.name}"
                    ) from exc

                if move not in board.legal_moves:
                    raise ValueError(
                        "Illegal move in opening line "
                        f"{line.eco} {line.name}: {move_text!r} at position {board.fen()}"
                    )

                key = position_key(board)
                move_bucket = index.setdefault(key, {})
                aggregate = move_bucket.setdefault(
                    move_text,
                    {
                        "weight": 0,
                        "eco_codes": set(),
                        "names": set(),
                        "sources": set(),
                        "max_depth": 0,
                    },
                )
                aggregate["weight"] += 1
                aggregate["eco_codes"].add(line.eco)
                aggregate["names"].add(line.name)
                aggregate["sources"].add(line.source)
                aggregate["max_depth"] = max(aggregate["max_depth"], depth - 1)

                board.push(move)

        normalized_index: dict[str, list[OpeningChoice]] = {}
        for key, move_map in index.items():
            choices = [
                OpeningChoice(
                    move=move,
                    weight=data["weight"],
                    eco_codes=tuple(sorted(data["eco_codes"])),
                    names=tuple(sorted(data["names"])),
                    sources=tuple(sorted(data["sources"])),
                    max_depth=data["max_depth"],
                )
                for move, data in move_map.items()
            ]
            choices.sort(key=lambda choice: (-choice.weight, choice.move))
            normalized_index[key] = choices

        return cls(index=normalized_index, line_count=line_count)

    def lookup(self, board: chess.Board) -> list[OpeningChoice]:
        """Return all known book continuations for the current position."""

        return list(self._index.get(position_key(board), []))

    def lookup_payload(self, board: chess.Board) -> list[dict[str, Any]]:
        """Return opening choices in payload-ready JSON form."""

        return [choice.to_payload() for choice in self.lookup(board)]

    def contextual_payload(self, board: chess.Board) -> list[dict[str, Any]]:
        """Return opening context for the current board, falling back to recent history.

        Priority:
        1. Exact current-position book continuations.
        2. Most recent earlier position in the current game that matched the book.
        """

        current_matches = self.lookup(board)
        if current_matches:
            return [
                {
                    "match_type": "current_position",
                    "plies_from_start": len(board.move_stack),
                    **choice.to_payload(),
                }
                for choice in current_matches
            ]

        history_board = board.copy(stack=True)
        plies_from_start = len(history_board.move_stack)

        while history_board.move_stack:
            history_board.pop()
            plies_from_start -= 1
            historical_matches = self.lookup(history_board)
            if historical_matches:
                return [
                    {
                        "match_type": "recent_book_position",
                        "plies_from_start": plies_from_start,
                        **choice.to_payload(),
                    }
                    for choice in historical_matches
                ]

        return []

    def best_move(self, board: chess.Board) -> str | None:
        """Return the highest-weight book move for the position, if any."""

        matches = self.lookup(board)
        return matches[0].move if matches else None

    def __len__(self) -> int:
        return len(self._index)


def load_opening_lines(book_path: Path | str = DEFAULT_BOOK_PATH) -> list[OpeningLine]:
    """Load and validate opening lines from a JSON file."""

    path = Path(book_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    lines_data = raw.get("lines")
    if not isinstance(lines_data, list):
        raise ValueError(f"Opening book must contain a top-level 'lines' list: {path}")
    return [OpeningLine.from_dict(item) for item in lines_data]


def load_opening_book(book_path: Path | str = DEFAULT_BOOK_PATH) -> OpeningBook:
    """Convenience helper to build an indexed opening book from disk."""

    return OpeningBook.from_lines(load_opening_lines(book_path))


def board_from_args(fen: str, moves: Sequence[str]) -> chess.Board:
    """Build a board from CLI args for quick manual lookup testing."""

    board = chess.Board(fen) if fen else chess.Board()
    for move_text in moves:
        move = chess.Move.from_uci(move_text)
        if move not in board.legal_moves:
            raise ValueError(f"Illegal move {move_text!r} for position {board.fen()}")
        board.push(move)
    return board


def parse_args() -> argparse.Namespace:
    """Parse CLI args for ad hoc opening-book inspection."""

    parser = argparse.ArgumentParser(description="Inspect the local opening book.")
    parser.add_argument(
        "--book-path",
        default=str(DEFAULT_BOOK_PATH),
        help="Path to the opening book JSON file.",
    )
    parser.add_argument(
        "--fen",
        default="",
        help="Optional starting FEN. Defaults to the standard initial position.",
    )
    parser.add_argument(
        "--moves",
        nargs="*",
        default=[],
        help="Optional sequence of UCI moves to apply before lookup.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint for inspecting opening matches."""

    args = parse_args()
    book = load_opening_book(args.book_path)
    board = board_from_args(args.fen, args.moves)
    print(
        json.dumps(
            {
                "lines_loaded": book.line_count,
                "indexed_positions": len(book),
                "position": position_key(board),
                "matches": book.lookup_payload(board),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
