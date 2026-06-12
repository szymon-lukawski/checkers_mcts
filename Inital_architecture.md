## 1. Architektura Systemu (Komponenty)

System zostanie podzielony na trzy niezależne warstwy (MVC) oraz odizolowany podsystem AI działający w osobnym procesie.

### A. Warstwa Danych i Logiki (Model) – Bitboard Engine

To serce aplikacji. Operuje wyłącznie na liczbach 32-bitowych. Zero obiektów typu `Pionek`.

* **Stan planszy (Board State):** Reprezentowany przez cztery liczby 32-bitowe:
* `white_pieces` – bity ustawione na 1 tam, gdzie stoją białe pionki/damki.
* `black_pieces` – bity ustawione na 1 tam, gdzie stoją czarne pionki/damki.
* `kings` – bity ustawione na 1 tam, gdzie stoi dowolna damka (biała lub czarna).
* `current_player` – znacznik, czyj jest ruch (np. 1 dla białych, 0 dla czarnych).


* **Generator Ruchów (Move Generator):** Funkcje przesuwające bity (bit-shifts) w celu wyznaczenia dozwolonych ruchów i bić. W warcabach przesunięcie o "skos" na 32-bitowej planszy zależy od tego, czy rząd jest parzysty, czy nieparzysty (zazwyczaj są to przesunięcia o 3, 4 lub 5 bitów).

### B. Warstwa Kontroli i Współbieżności (Controller)

Zarządza przebiegiem gry i pośredniczy między UI a procesem AI.

* **Game Manager:** Pętla gry. Sprawdza, czy gra się skończyła, liczy punkty (w warcabach: liczba zbitych pionków/stan materiału na planszy).
* **Process Wrapper:** Odpowiada za wysyłanie stanu planszy (cztery inty) do procesu AI przez `multiprocessing.Queue` i odbieranie ruchu (dwie liczby: skąd -> dokąd).

### C. Warstwa Prezentacji (View) – Pygame

Odpowiada wyłącznie za rysowanie i kliknięcia.

* **Renderer:** Rysuje planszę 8x8. Mapuje indeksy bitowe (0-31) na współrzędne pikseli ekranu $(x, y)$.
* **Input Handler:** Przechwytuje kliknięcie myszką na kwadrat, mapuje piksele na indeks bitowy (0-31) i przekazuje do Controllera.

### D. Podsystem AI (Niezależny Proces)

Klasa-robot, która żyje we własnym procesie i czeka na zadania. Posiada wspólny interfejs dla każdego agenta: `get_best_move(board_state)`.

---

## 2. Architektura Modułów (Struktura plików)

```text
checkers_project/
│
├── engine/                  # Czysta logika (Model)
│   ├── __init__.py
│   ├── bitboard.py          # Operacje bitowe, definicje masek
│   └── game_logic.py        # Klasa Board, generator ruchów, zasady brazylijskie
│
├── ai/                      # Agenci AI (Osobny proces)
│   ├── __init__.py
│   ├── ai_process.py        # Współbieżność (multiprocessing worker)
│   ├── random_agent.py      # Agent losowy (do testów)
│   ├── minimax_agent.py     # Minimax + Alpha-Beta
│   └── mcts_agent.py        # Monte Carlo Tree Search
│
├── ui/                      # Interfejs (View)
│   ├── __init__.py
│   └── pygame_app.py        # Okno Pygame, pętla rysowania
│
└── main.py                  # Punkt startowy (Controller), konfiguracja trybów

```

---

## 3. Priorytetyzowany Plan Wdrożenia (Roadmap)

Projekt podzieliłem na 4 fazy według priorytetów. Nie przechodź do kolejnej, dopóki poprzednia nie działa w 100%.

### Priorytet 1: Fundament Bitboardów i Silnik w Konsoli (MVP)

**Cel:** Stworzenie bezbłędnej logiki gry działającej w terminalu.

1. Zmapowanie pól planszy 8x8 na 32 bity (stworzenie maski i indeksacji).
2. Implementacja ruchów i bić dla **zwykłych pionków** za pomocą operacji bitowych.
3. Zasada obowiązkowego bicia (jeśli generator znajdzie bicie, ruchy zwykłe są ignorowane).
4. Stworzenie `RandomAgent` (wybiera losowy bit-ruch).
5. **Testy:** Pętla w konsoli, gdzie Random Agent gra przeciwko drugiemu Random Agentowi przez 1000 partii. Jeśli program się nie zawiesi i poprawnie wykryje koniec gry – fundament jest stabilny.

### Priorytet 2: UI (Pygame) i Multiprocessing

**Cel:** Wizualizacja gry i płynne sterowanie człowiek vs komputer.

1. Napisanie kodu Pygame mapującego kliki na bity. Wizualizacja stanu bitboardu na ekranie.
2. Wdrożenie `multiprocessing.Process` dla AI. Uruchomienie `RandomAgent` w osobnym procesie. Pygame ma pozostać responsywny (płynne 60 FPS) podczas "myślenia" bota.
3. Dodanie obsługi **Damek** do Bitboardu (Damka w wersjach brazylijskich "lata" po całych przekątnych – dla bitboardów wymaga to pętli bitowych/masek linii, to najtrudniejsza część logiki).

### Priorytet 3: Konfigurowalny Minimax i Zasady Zaawansowane

**Cel:** Pierwsze inteligentne AI i pełne zasady turniejowe.

1. Wdrożenie **zasady większości** bić (Brazylijska: musisz bić tam, gdzie zbĳesz najwięcej pionków).
2. Napisanie funkcji oceniającej stan planszy (Heurystyka: waga pionka = 100, waga damki = 300, bonus za pozycje centralne).
3. Implementacja **Minimax z odcinaniem Alpha-Beta**.
4. Dodanie konfiguratora: suwak głębokości (depth) w Pygame oraz flaga włączająca prostą bazę końcówek (np. szybki warunek: jeśli zostały 2 damki na 1, użyj gotowego algorytmu pogoni).

### Priorytet 4: Monte Carlo Tree Search (MCTS) i Szlify

**Cel:** Potężne AI i finalny produkt.

1. Implementacja MCTS (Kroki: Selection, Expansion, Simulation, Backpropagation).
2. Dzięki bitboardom faza Simulation (losowa rozgrywka do końca) powinna wykonywać się błyskawicznie.
3. Dodanie ekranu menu w Pygame: Wybór trybu (Człowiek vs Człowiek, Człowiek vs AI, AI vs AI) oraz menu wyboru bota z listy (Random, Minimax, MCTS) z parametrami.

---

## 4. Krytyczne Zagadnienie Techniczne na Start: Indeksowanie

Zanim zaczniesz, musisz wybrać schemat indeksowania 32 bitów. Najpopularniejszy i najwygodniejszy dla warcabów to indeksowanie rzędami:

```text
   Czarny Gracz (Góra planszy)
   .  0  .  1  .  2  .  3
   4  .  5  .  6  .  7  .
   .  8  .  9  . 10  . 11
  12  . 13  . 14  . 15  .
   . 16  . 17  . 18  . 19
  20  . 21  . 22  . 23  .
   . 24  . 25  . 26  . 27
  28  . 29  . 30  . 31  .
   Biały Gracz (Dół planszy)

```

Przy takim układzie, ruch zwykłego białego pionka "w lewo-skos" z rzędu nieparzystego (np. z pola 20) to przesunięcie bitowe o 4 (pole 16). Z rzędu parzystego (np. z pola 16) "w lewo-skos" to pole 12 (też o 4), ale "w prawo-skos" z pola 17 to pole 13 (o 4) lub z pola 21 to pole 17 (o 4). Ruchy i bicia zamieniają się w stałe przesunięcia o wartości `3, 4, 5`.
