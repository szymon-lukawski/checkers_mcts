# Raport benchmarku agentów AI

Data: 2026-06-12 23:32:51

## Pozycja startowa (12 pionków/strona)

| Agent | N prób | Śr. (ms) | Mediana (ms) | Min (ms) | Max (ms) | Odch. std (ms) |
|-------|--------|----------|--------------|----------|----------|----------------|
| Random | 200 | 0.0 | 0.0 | 0.0 | 0.1 | 0.0 |
| Minimax d=2 | 100 | 1.7 | 1.2 | 1.0 | 37.8 | 3.7 |
| Minimax d=4 | 30 | 13.7 | 13.2 | 11.1 | 17.5 | 1.6 |
| Minimax d=6 | 5 | 69.1 | 68.5 | 64.5 | 74.6 | 3.7 |
| MCTS 500ms | 5 | 500.8 | 500.6 | 500.1 | 502.0 | 0.8 |
| MCTS 1000ms | 3 | 1001.3 | 1001.3 | 1000.6 | 1002.0 | 0.7 |
| MCTS 1500ms | 3 | 1500.8 | 1501.1 | 1500.1 | 1501.1 | 0.6 |

## Pozycja środkowej gry (4 pionki/strona)

| Agent | N prób | Śr. (ms) | Mediana (ms) | Min (ms) | Max (ms) | Odch. std (ms) |
|-------|--------|----------|--------------|----------|----------|----------------|
| Random | 200 | 0.0 | 0.0 | 0.0 | 0.1 | 0.0 |
| Minimax d=2 | 100 | 0.6 | 0.5 | 0.5 | 1.1 | 0.1 |
| Minimax d=4 | 30 | 9.0 | 8.9 | 7.2 | 11.2 | 1.3 |
| Minimax d=6 | 5 | 61.9 | 47.2 | 46.6 | 119.7 | 32.3 |
| MCTS 500ms | 5 | 500.2 | 500.2 | 500.1 | 500.4 | 0.1 |
| MCTS 1000ms | 3 | 1000.2 | 1000.1 | 1000.1 | 1000.2 | 0.0 |
| MCTS 1500ms | 3 | 1500.2 | 1500.2 | 1500.0 | 1500.4 | 0.2 |

## Uwagi

- **Random**: wybiera losowy ruch z listy legalnych, czas proporcjonalny do generowania ruchów
- **Minimax**: czas rośnie wykładniczo z głębokością; alpha-beta znacznie przyspiesza
- **MCTS**: czas = budżet czasowy; liczba symulacji zależy od szybkości rolloutów
