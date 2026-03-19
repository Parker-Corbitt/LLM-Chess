# UCI Chess Runner (Terminal + LLM Adapter)

This project includes a Python chess runner that accepts moves in **UCI format**.

- `play_chess.py`: main game loop
- `ollama_gemma_adapter.py`: Ollama adapter with router -> phase pipeline (defaults to `gemma3:12b`)
- `llm_move_stub.py`: minimal example adapter (non-Ollama)

## 1) Install dependency

```bash
python3 -m pip install -r requirements.txt
```

## 2) Terminal play (human vs human)

```bash
python3 play_chess.py --white-player human --black-player human --show-legal
```

Enter moves like:

- `e2e4`
- `g1f3`
- `e7e8q` (promotion)

Other commands:

- `legal` to print legal moves
- `resign` to resign
- `quit` to exit

## 3) Play against an LLM adapter

### Ollama setup (Gemma 3 12B)

```bash
ollama serve
```

In another terminal:

```bash
ollama pull gemma3:12b
```

### White human, Black LLM (default: `gemma3:12b`)

```bash
python3 play_chess.py \
  --white-player human \
  --black-player llm \
  --show-legal \
  --debug
```

### Both sides LLM

```bash
python3 play_chess.py \
  --white-player llm \
  --black-player llm
```

`play_chess.py` now defaults to:

```bash
--llm-command "python3 ollama_gemma_adapter.py"
```

Override model (optional):

```bash
OLLAMA_MODEL=gemma3:12b python3 play_chess.py --white-player human --black-player llm
```

You can also override adapter command completely:

```bash
python3 play_chess.py --white-player human --black-player llm --llm-command "python3 llm_move_stub.py"
```

## 4) Adapter contract (for custom adapters)

`play_chess.py` sends JSON to your adapter on stdin:

```json
{
  "fen": "...",
  "pgn": "...",
  "side_to_move": "white|black",
  "move_number": 1,
  "fullmove_number": 1,
  "halfmove_clock": 0,
  "legal_moves": ["e2e4", "d2d4", "..."],
  "last_move": "e7e5",
  "opening_db_entries": [],
  "time_info": null,
  "draw_state": {"halfmove_clock": 0, "can_claim_threefold_repetition": false},
  "system_prompt": "...",
  "debug": false
}
```

Your adapter must print either:

- a raw UCI move (example: `e2e4`)
- or JSON with one of `move`, `best_move`, `uci`

Examples:

```text
e2e4
```

or

```json
{"best_move":"e2e4"}
```

## 5) Phase routing prompts

`ollama_gemma_adapter.py` now runs two model calls per move:

1. `router.txt` chooses `OPENING`, `MIDDLEGAME`, or `ENDGAME`.
2. The chosen phase prompt file (`opening.txt`, `middlegame.txt`, `endgame.txt`) selects a move.
3. If `OPENING` returns out-of-book/no legal move, adapter falls through to `middlegame.txt` in the same turn.

`Master.txt` is still loaded by `play_chess.py` and passed as global strategy policy (`system_prompt`) for both calls.

You can still override the global policy file:

```bash
python3 play_chess.py --black-player llm --prompt-file Master.txt
```

## 6) Optional flags

- `--start-fen "<fen>"` to start from a custom position
- `--max-plies N` to stop after N half-moves
- `--debug` to print raw adapter/Ollama diagnostics to terminal stderr

## 7) Debugging model output and variation

Show raw Ollama response each LLM turn:

```bash
python3 play_chess.py --white-player human --black-player llm --debug
```

Tune sampling (default is now `0.4`):

```bash
OLLAMA_TEMPERATURE=0.7 python3 play_chess.py --white-player human --black-player llm --debug
```
