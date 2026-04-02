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
--llm-command "python3 ollama_adapter.py"
```

When an LLM side is on move, `play_chess.py` now checks the local opening book first.
If the current position matches `opening_book.json`, it plays the top book move immediately and skips the adapter call for that turn.

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
  "opening_db_entries": [
    {
      "match_type": "current_position|recent_book_position",
      "plies_from_start": 4,
      "move": "f1c4",
      "weight": 3,
      "eco_codes": ["C50"],
      "opening_names": ["Italian Game"],
      "sources": ["builtin"],
      "max_depth": 4
    }
  ],
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

`opening_db_entries` now carries either:
- exact `current_position` book continuations, or
- `recent_book_position` lineage from the latest earlier position still recognized in the local opening book.

Only `current_position` entries should be treated as legal book moves for the current board.

`Master.txt` is still loaded by `play_chess.py` and passed as global strategy policy (`system_prompt`) for both calls.

You can still override the global policy file:

```bash
python3 play_chess.py --black-player llm --prompt-file Master.txt
```

## 6) Optional flags

- `--start-fen "<fen>"` to start from a custom position
- `--opening-book-file opening_book.json` to override or disable the local opening book source
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

## 8) Code reference

### `play_chess.py`

- Purpose: terminal game loop, move legality checks, and adapter subprocess integration.
- Key functions:
  - `get_llm_move(...)`: builds payload and calls adapter command on stdin/stdout.
  - `extract_move_from_llm_stdout(...)`: accepts raw UCI or JSON (`move`/`best_move`/`uci`).
  - `get_human_move(...)`: interactive human input handling (`legal`, `resign`, `quit`).

### `ollama_gemma_adapter.py`

- Purpose: Ollama-backed adapter that performs `router -> phase` selection each turn.
- Flow:
  1. Render `router.txt`, call model, extract phase.
  2. Render selected phase prompt (`opening.txt`/`middlegame.txt`/`endgame.txt`), call model.
  3. Extract legal UCI move; if opening is out-of-book, fall through to middlegame.
- Safety:
  - On parsing/network failures, returns first legal move as fallback.
  - Debug mode prints raw router/phase responses to stderr.

### `llm_move_stub.py`

- Purpose: minimal contract example for custom adapters.
- Behavior: reads payload JSON and prints first legal move.
