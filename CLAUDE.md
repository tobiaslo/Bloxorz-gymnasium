# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the game

```bash
python3 bloxorz.py [--cols N] [--rows N] [--no-gate]
```

Requires `pygame`. Install with `pip install pygame`.

| Flag | Default | Effect |
|------|---------|--------|
| `--cols` | 18 | Grid width in tiles |
| `--rows` | 14 | Grid height in tiles |
| `--no-gate` | off | Disable bridge/gate mechanic |

## Architecture

Everything lives in `bloxorz.py`. Key concepts:

**Block state** is `(col, row, ori)` where `ori` is one of `UPRIGHT`, `FLAT_H`, or `FLAT_V`. `FLAT_H` occupies `(col, row)` and `(col+1, row)`; `FLAT_V` occupies `(col, row)` and `(col, row+1)`.

**Board** is a 2D list of ints: `0` = void, `1` = tile, `2` = goal, `3` = gate tile (closed/open), `4` = switch. The goal requires the block to arrive `UPRIGHT`.

**Move logic** (`move()`) encodes the full roll table mapping `(ori, direction)` → `(new_col, new_row, new_ori)`. Position delta depends on orientation — e.g. rolling right from `UPRIGHT` moves +1 col and becomes `FLAT_H`, but rolling right from `FLAT_H` moves +2 col and becomes `UPRIGHT`.

**Board generation** (`generate_board()`) has two modes controlled by `gate=True/False`:

- `gate=True` (default): two-island split at `mid = grid_cols // 2`. Right island (goal side) stays in `col > mid`; left island (player+switch side) stays in `col < mid-1`. A 2-tile bridge at columns `mid-1` and `mid` is the only connection. Generation: reverse-walk each island separately (20 steps each), then scatter branch tiles per island (20 attempts each, step-by-2 offset, staying on correct side).
- `gate=False`: single unconstrained reverse-walk (40 steps) from the goal across the full board, then 40 branch tile attempts (step-by-2) with no side constraint.

**Gate mechanic:** `gate_open` bool in the game loop. Standing `UPRIGHT` on the switch (cell `4`) toggles `gate_open`. `is_valid()` treats cell `3` as impassable when `gate_open=False`. Switch colour changes from yellow (closed) to green (open).

**Controls:** Arrow keys move, `R` resets, `N` generates a new board, `Q` quits.
