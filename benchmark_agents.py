"""
Benchmark agentów AI – mierzy czas myślenia (get_best_move) dla każdego agenta.
Wyniki zapisuje do agents_benchmark_report.md
"""

import statistics
import time
from datetime import datetime

from models.board_state import BoardState
from ai.random_agent import RandomAgent
from ai.minimax_agent import MinimaxAgent
from ai.mcts_agent import MCTSAgent


def time_agent(agent, state: BoardState, n_trials: int) -> dict:
    times_ms = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        agent.get_best_move(state)
        times_ms.append((time.perf_counter() - t0) * 1000)
    return {
        "n_trials": n_trials,
        "mean_ms": statistics.mean(times_ms),
        "median_ms": statistics.median(times_ms),
        "stdev_ms": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
    }


# Pozycje testowe
INITIAL = BoardState.initial()

# Środek gry: białe mają 4 pionki (sq20-23), czarne mają 4 pionki (sq8-11)
_w = (1 << 20) | (1 << 21) | (1 << 22) | (1 << 23)
_b = (1 << 8) | (1 << 9) | (1 << 10) | (1 << 11)
MIDGAME = BoardState(white_pieces=_w, black_pieces=_b, kings=0, current_player=1)

AGENTS = [
    ("Random", RandomAgent(), 200),
    ("Minimax d=2", MinimaxAgent(depth=2), 100),
    ("Minimax d=4", MinimaxAgent(depth=4), 30),
    ("Minimax d=6", MinimaxAgent(depth=6), 5),
    ("MCTS 500ms", MCTSAgent(time_limit_ms=500), 5),
    ("MCTS 1000ms", MCTSAgent(time_limit_ms=1000), 3),
    ("MCTS 1500ms", MCTSAgent(time_limit_ms=1500), 3),
]


def run_benchmarks() -> dict:
    results = {}
    for name, agent, n in AGENTS:
        print(f"Benchmarking {name} (initial, n={n})...", flush=True)
        results[name] = {
            "initial": time_agent(agent, INITIAL, n),
            "midgame": time_agent(agent, MIDGAME, n),
        }
    return results


def write_report(results: dict) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Raport benchmarku agentów AI",
        "",
        f"Data: {now}",
        "",
        "## Pozycja startowa (12 pionków/strona)",
        "",
        "| Agent | N prób | Śr. (ms) | Mediana (ms) | Min (ms) | Max (ms) | Odch. std (ms) |",
        "|-------|--------|----------|--------------|----------|----------|----------------|",
    ]
    for name, res in results.items():
        r = res["initial"]
        lines.append(
            f"| {name} | {r['n_trials']} | {r['mean_ms']:.1f} | {r['median_ms']:.1f} | "
            f"{r['min_ms']:.1f} | {r['max_ms']:.1f} | {r['stdev_ms']:.1f} |"
        )
    lines += [
        "",
        "## Pozycja środkowej gry (4 pionki/strona)",
        "",
        "| Agent | N prób | Śr. (ms) | Mediana (ms) | Min (ms) | Max (ms) | Odch. std (ms) |",
        "|-------|--------|----------|--------------|----------|----------|----------------|",
    ]
    for name, res in results.items():
        r = res["midgame"]
        lines.append(
            f"| {name} | {r['n_trials']} | {r['mean_ms']:.1f} | {r['median_ms']:.1f} | "
            f"{r['min_ms']:.1f} | {r['max_ms']:.1f} | {r['stdev_ms']:.1f} |"
        )
    lines += [
        "",
        "## Uwagi",
        "",
        "- **Random**: wybiera losowy ruch z listy legalnych, czas proporcjonalny do generowania ruchów",
        "- **Minimax**: czas rośnie wykładniczo z głębokością; alpha-beta znacznie przyspiesza",
        "- **MCTS**: czas = budżet czasowy; liczba symulacji zależy od szybkości rolloutów",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    results = run_benchmarks()
    report = write_report(results)
    print(report)
    path = "agents_benchmark_report.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nRaport zapisany do {path}")
