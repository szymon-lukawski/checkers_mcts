# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv run python main.py                        # launch the game (menu → play)
uv run pytest                                # run all 400 tests + coverage report
uv run pytest tests/test_move_generator.py   # run a single test file
uv run pytest -k "test_apply_move"           # run tests matching a name pattern
uv run pytest --cov-report=html              # HTML coverage report → htmlcov/
uv run python benchmark_agents.py            # time all agents, write agents_benchmark_report.md
uv sync --dev                                # install/sync all dependencies
```

## Architecture

The codebase is split into four layers:

**`engine/`** — pure bitboard logic, no classes, no Pydantic. All functions take `(wp, bp, kings, current_player)` integers and return plain Python values. `move_generator.py` is the core: it exports `get_legal_moves`, `apply_move`, and `is_game_over`. The `NEIGHBORS` lookup table (`list[tuple[UL,UR,DL,DR]]`) is precomputed at import time from `_build_neighbors()` — this eliminates all manual bit-shift arithmetic and is the reason edge-case capture bugs were eliminated.

**`models/`** — Pydantic v2 DTOs used only at layer boundaries: `BoardState` / `Move` for AI↔UI communication, `AgentConfig` / `GameConfig` for menu→game config. Never used inside tight engine loops.

**`ai/`** — Each agent implements `BaseAgent.get_best_move(state: BoardState) -> Move | None`. `AIProcess` wraps any agent in a `multiprocessing.Process` (spawn mode, required on macOS) and exposes a non-blocking `request_move()` / `poll_move()` API for the Pygame loop. Queue messages are plain `dict` (`model_dump` / `model_validate`) — never raw Pydantic objects — to avoid pickle issues.

**`ui/`** — `Renderer` handles all drawing; `Renderer.pixel_to_sq()` converts mouse clicks to board indices. `menu.py` runs its own event loop and returns a `GameConfig`, then `PygameGame` in `main.py` takes over.

## Brazilian Checkers Rules (critical for engine correctness)

- **Mandatory capture**: if any capture exists, only captures are legal.
- **Majority rule**: among captures, only paths that take the maximum number of pieces are legal.
- **Pawns capture backward**: `ALL_DIRS = (0,1,2,3)` is used in `_find_captures_from` for both pawns and kings.
- **Flying kings**: kings travel the full diagonal in simple moves and in captures (land anywhere beyond the jumped piece).
- **Captured pieces stay on board** during a sequence — `captured_mask` tracks them; `effective_occupied = ((wp|bp) & ~captured_mask) | (1<<sq)` makes them transparent to subsequent jumps.

## Bitboard Layout

32 dark squares, numbered 0–31 top-left to bottom-right:
- Even rows (0,2,4,6): squares occupy columns 1,3,5,7
- Odd rows (1,3,5,7): squares occupy columns 0,2,4,6
- `WHITE_PROMO = 0x0000_000F` (bits 0–3, top rows) — white promotes here
- `BLACK_PROMO = 0xF000_0000` (bits 28–31, bottom rows) — black promotes here
- `current_player`: `1` = white, `0` = black

## MCTS Perspective Convention

`MCTSNode.wins` counts wins for the player who **made the move to reach** that node (not the current player at the node). `_backpropagate` therefore starts with `current_result = 1.0 - result` (a pre-flip) before walking up the tree. Getting this wrong causes the agent to choose the worst move — it has happened before.

## Testing

Tests require `SDL_VIDEODRIVER=dummy` for pygame (set in `tests/conftest.py`). The `if not moves:` branch in `_simulate` (mcts_agent.py) and the light-square guard in `_build_neighbors` are marked `# pragma: no cover` — they are structurally unreachable.
