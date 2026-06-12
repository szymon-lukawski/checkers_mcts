# Checkers MCTS

[![CI](https://github.com/szymon-lukawski/checkers_mcts/actions/workflows/ci.yml/badge.svg)](https://github.com/szymon-lukawski/checkers_mcts/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/szymon-lukawski/checkers_mcts/graph/badge.svg)](https://codecov.io/gh/szymon-lukawski/checkers_mcts)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

**Brazilian Checkers** (Warcaby Brazylijskie) with three AI opponents — Random, Minimax + Alpha-Beta, and MCTS — featuring a full Pygame GUI, animated piece movement, and a pre-game menu.

> 🎬 **Demo** — _add your screen recording here_

```
  .  ●  .  ●  .  ●  .  ●      ← Black (top)
  ●  .  ●  .  ●  .  ●  .
  .  ●  .  ●  .  ●  .  ●
  .  .  .  .  .  .  .  .
  .  .  .  .  .  .  .  .
  ○  .  ○  .  ○  .  ○  .
  .  ○  .  ○  .  ○  .  ○
  ○  .  ○  .  ○  .  ○  .      ← White (bottom)
```

---

## Features

- **Full Brazilian rules** — mandatory capture, majority-capture rule, pawns capture backward, flying kings
- **Three AI agents** selectable per side in the menu:
  - **Random** — baseline, near-instant
  - **Minimax + Alpha-Beta** — configurable depth (1–12), with material + positional evaluation
  - **MCTS** — configurable time budget (500–5 000 ms), UCT selection
- **Animated movement** — pieces glide smoothly through each jump of a multi-capture; speed configurable in the menu (50 ms – 2 000 ms per segment)
- **Bitboard engine** — compact 32-field representation, precomputed neighbour table, no manual bit-shift errors
- **Non-blocking AI** — AI runs in a separate `multiprocessing` process; the UI stays responsive
- **Pygame GUI** — board, piece highlights, legal-move dots, last-move indicator, status bar
- **485 tests, 100 % line coverage** — `pytest` + `pytest-cov`

---

## Installation

Requires [uv](https://github.com/astral-sh/uv) and Python 3.11+.

```bash
git clone https://github.com/szymon-lukawski/checkers_mcts.git
cd checkers_mcts
uv sync
```

---

## Usage

```bash
uv run python main.py        # launch the game (menu → play)
```

Pick the agent for each side in the menu, adjust depth / time budget, then press **Start** or **Enter**.

---

## AI Agents

### Minimax + Alpha-Beta

Negamax search with iterative move ordering (captures first, then by centre bonus).  
Evaluation function weights:

| Component | Value |
|-----------|-------|
| Pawn | 100 |
| King | 300 |
| Centre bonus (4 inner squares) | +15 |
| Advancement bonus (row-based) | +0–3 |

### MCTS (Monte Carlo Tree Search)

Standard UCT with `c = √2`.  
Each iteration: **select → expand → simulate (random rollout) → backpropagate**.  
The agent picks the child with the highest visit count after the time budget expires.

### Performance (Apple M-class, single core)

| Agent | Position | Mean (ms) |
|-------|----------|-----------|
| Random | initial | < 0.1 |
| Minimax d=2 | initial | 1.7 |
| Minimax d=4 | initial | 13.7 |
| Minimax d=6 | initial | 69.1 |
| MCTS 500 ms | initial | 500.8 |
| MCTS 1 000 ms | initial | 1 001.3 |
| MCTS 1 500 ms | initial | 1 500.8 |

Full benchmark: [`agents_benchmark_report.md`](agents_benchmark_report.md)

---

## Project Structure

```
checkers_mcts/
├── engine/
│   ├── bitboard.py          # constants & bit utilities
│   ├── move_generator.py    # legal moves, captures, apply_move, is_game_over
│   └── game_logic.py        # Board OOP wrapper
├── ai/
│   ├── base_agent.py        # BaseAgent ABC
│   ├── random_agent.py
│   ├── minimax_agent.py     # Minimax + Alpha-Beta + evaluation
│   ├── mcts_agent.py        # MCTS / UCT
│   └── ai_process.py        # multiprocessing worker
├── models/
│   ├── board_state.py       # Pydantic DTOs (BoardState, Move)
│   └── config.py            # AgentConfig, GameConfig, AgentType
├── ui/
│   ├── renderer.py          # Pygame drawing
│   └── menu.py              # pre-game menu screen
├── tests/                   # 400 pytest tests, 100 % coverage
├── main.py                  # entry point
├── benchmark_agents.py      # timing script → agents_benchmark_report.md
└── pyproject.toml
```

---

## Development

### Run tests

```bash
uv run pytest                        # tests + coverage report in terminal
uv run pytest --cov-report=html      # HTML report → htmlcov/index.html
```

### Run benchmark

```bash
uv run python benchmark_agents.py    # writes agents_benchmark_report.md
```

### Headless mode (no GUI, stress-test)

```python
# inside a Python session or script
from main import run_headless
run_headless(num_games=1000)
```

---

## Technical Notes

**Bitboard layout** — 32 dark squares numbered 0–31 (top-left to bottom-right).  
`NEIGHBORS[sq]` is a precomputed `(UL, UR, DL, DR)` tuple; `-1` means board edge.  
This eliminates all manual shift arithmetic and edge-case bugs.

**Multiprocessing on macOS** — `spawn` start method is set explicitly to avoid fork-related pygame/OpenGL issues.

**Brazilian capture rules** implemented exactly:
- Pawns capture in all 4 diagonal directions (not only forward)
- Captured pieces remain physically on the board until the full sequence is complete (their squares are transparent to subsequent jumps via `captured_mask`)
- Only paths with the maximum number of captures are legal (majority rule)

---

## License

[MIT](LICENSE) © 2026 Szymon Łukawski
