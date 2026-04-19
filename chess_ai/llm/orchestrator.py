#!/usr/bin/env python3
"""High-level AI routing and move selection logic."""

from __future__ import annotations

from typing import Any, Optional
import chess

from .client import call_ollama
from .prompts import (
    load_prompt,
    render_prompt_template,
    build_router_system_prompt,
    build_phase_system_prompt,
    ROUTER_FALLBACK_PROMPT,
    PHASE_FALLBACK_PROMPT,
)
from .parser import extract_phase, extract_move_from_phase_output, is_out_of_book_opening_response

class LLMOrchestrator:
    """Coordinates the flow from routing to phase-specific move selection."""

    def __init__(self, master_prompt: str = "", temperature: float = 0.5, debug: int = 0):
        self.master_prompt = master_prompt
        self.temperature = temperature
        self.debug = debug

    def get_ai_move(self, payload: dict, board: chess.Board) -> chess.Move:
        """Determine a move by routing through phases.

        Args:
            payload: Context containing FEN, PGN, etc.
            board: The current chess.Board object.

        Returns:
            A legal chess.Move.

        Raises:
            RuntimeError: If all AI attempts fail, returns first legal move as absolute fallback.
        """
        legal_moves = [move.uci() for move in board.legal_moves]
        if not legal_moves:
            raise RuntimeError("No legal moves available.")

        try:
            # 1. Routing Phase
            router_template = load_prompt("prompts/router.txt", ROUTER_FALLBACK_PROMPT)
            router_prompt = render_prompt_template(router_template, payload)
            router_system = build_router_system_prompt(self.master_prompt)

            router_text, router_raw = call_ollama(router_system, router_prompt, self.temperature)
            phase = extract_phase(router_text) or "MIDDLEGAME"
            routed_phase = phase

            # 2. Phase-Specific Selection
            phase_templates = {
                "OPENING": load_prompt("prompts/opening.txt", PHASE_FALLBACK_PROMPT),
                "MIDDLEGAME": load_prompt("prompts/middlegame.txt", PHASE_FALLBACK_PROMPT),
                "ENDGAME": load_prompt("prompts/endgame.txt", PHASE_FALLBACK_PROMPT),
            }

            phase_template = phase_templates[phase]
            phase_prompt = render_prompt_template(phase_template, payload)
            phase_system = build_phase_system_prompt(self.master_prompt, phase)

            phase_text, phase_raw = call_ollama(phase_system, phase_prompt, self.temperature)

            # Handle Opening -> Middlegame fallback
            if phase == "OPENING":
                try:
                    move_uci = extract_move_from_phase_output(phase_text, board)
                    if is_out_of_book_opening_response(phase_text):
                        raise ValueError("Opening response marked OUT_OF_BOOK.")
                    move = chess.Move.from_uci(move_uci)
                except (ValueError, RuntimeError):
                    # Fallback to Middlegame
                    phase = "MIDDLEGAME"
                    phase_template = phase_templates[phase]
                    phase_prompt = render_prompt_template(phase_template, payload)
                    phase_system = build_phase_system_prompt(self.master_prompt, phase)
                    phase_text, phase_raw = call_ollama(phase_system, phase_prompt, self.temperature)
                    move_uci = extract_move_from_phase_output(phase_text, board)
                    move = chess.Move.from_uci(move_uci)
            else:
                move_uci = extract_move_from_phase_output(phase_text, board)
                move = chess.Move.from_uci(move_uci)

            if self.debug >= 1:
                print(f"[llm-debug] routed_phase: {routed_phase}, final_phase: {phase}", file=sys.stderr)
                print(f"[llm-debug] selected_move: {move.uci()}", file=sys.stderr)

            return move

        except Exception as exc:
            if self.debug >= 1:
                print(f"AI Orchestrator warning: {exc}. Falling back to first legal move.", file=sys.stderr)
            # Absolute fallback: first legal move
            return list(board.legal_moves)[0]
