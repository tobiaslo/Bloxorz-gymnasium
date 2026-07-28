# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A [Gymnasium](https://gymnasium.farama.org/) reinforcement-learning environment for a 2D Bloxorz puzzle, plus a standalone pygame game for playing it by hand. Levels are randomly generated. The observation is an RGB image rendered in pure NumPy (no display needed for training).

## Install & run

```bash
pip install -e .          # installs numpy, gymnasium, pygame
bloxorz [--cols N] [--rows N] [--no-gate]   # console-script -> bloxorz.core:main
python3 test_human.py [--record [DIR]]   # human-playable harness driving the Gym env
```

| Flag | Default | Effect |
|------|---------|--------|
| `--cols` | 14 | Playable grid width in tiles |
| `--rows` | 10 | Playable grid height in tiles |
| `--no-gate` | off | Disable the bridge/gate mechanic |

**Controls (both `bloxorz` and `test_human.py`):** Arrow keys move, `R` resets the same board, `N` generates a new board, `Q` quits.

## Package layout

`pyproject.toml` (setuptools) defines package `bloxorz` and the `bloxorz` console script. Source:

- `bloxorz/core.py` — pure game logic + the standalone pygame game (`main`). Holds all constants/colors, the orientation/move table, board generation, validity/win checks, and pygame draw helpers.
- `bloxorz/env.py` — `BloxorzEnv(gym.Env)`. Wraps `core` logic, renders observations, and computes rewards.
- `bloxorz/registration.py` — `gym.register(...)` calls for the two env IDs (imported for side effects).
- `bloxorz/__init__.py` — re-exports `BloxorzEnv`, constants, and core functions; imports `registration` so `import bloxorz` registers the envs.
- `test_human.py` — pygame harness that lets a human drive the registered env via `env.step()`. `--record [DIR]` (default `frames/`) writes every observation to `DIR/<timestamp>/episode_NNN/frame_NNNN.png`; `S` still saves a single one-off PNG to the cwd.

## Core concepts (`core.py`)

**Block state** is `(col, row, ori)` where `ori` is `UPRIGHT` (0), `FLAT_H` (1), or `FLAT_V` (2). `FLAT_H` occupies `(col, row)` and `(col+1, row)`; `FLAT_V` occupies `(col, row)` and `(col, row+1)`. `occupied_cells()` is the single source of truth for this.

**Board** is a 2D list indexed `board[row][col]` of ints: `0` = void, `1` = tile, `2` = goal, `3` = gate/bridge tile, `4` = switch. `is_win()` requires the block to arrive `UPRIGHT` on a goal cell.

**Padding:** the playable `grid_cols × grid_rows` area sits inside a `BOARD_PAD` (=2) void border, so the actual board array is `(grid_cols + 2*BOARD_PAD) × (grid_rows + 2*BOARD_PAD)`. The pad gives the block room to roll off the edge and "fall."

**Move logic** (`move()`) encodes the full roll table mapping `(ori, direction)` → `(new_col, new_row, new_ori)`. Position delta depends on orientation — e.g. rolling right from `UPRIGHT` moves +1 col and becomes `FLAT_H`, but rolling right from `FLAT_H` moves +2 col and becomes `UPRIGHT`. Directions are `(1,0)=right, (-1,0)=left, (0,1)=down, (0,-1)=up`.

**`is_valid()`** rejects out-of-bounds, any void cell under the block, and gate cells (`3`) when `gate_open=False`.

**Board generation** (`generate_board(grid_cols, grid_rows, path_length, branch_tiles, rng, gate)`) takes a NumPy `rng` (the env passes `self.np_random` for reproducible seeding) and returns `(board, start_col, start_row, start_ori, switch_cell, bridge_cell)`. Every level is solvable because the path is a **reverse walk from the goal** — start at the goal and take random valid rolls; retracing reaches the goal.

- `gate=False`: single unconstrained reverse-walk of `path_length` steps from a goal placed in the right third, then `branch_tiles` scattered extra tiles (step-by-2 offsets). `switch_cell`/`bridge_cell` are `None`.
- `gate=True`: two-island split at `mid = board_cols // 2`. Right island (goal) stays `col > mid`; left island (player + switch) stays `col < mid-1`. Each island gets a `path_length//2` reverse walk; the right walk is then nudged toward column `mid+1` to anchor a 2-tile bridge (cells `(mid-1, row)` and `(mid, row)`, value `3`). The switch (value `4`) is placed at the first `UPRIGHT` cell roughly a third of the way along the left walk. Branch tiles are split per island and kept on the correct side, avoiding the bridge cells.

## The environment (`env.py`)

- **Action space:** `Discrete(4)` — `0=right, 1=left, 2=down, 3=up` (`ACTIONS` list).
- **Observation space:** `Box(0,255, (H, W, 3), uint8)` RGB image, drawn in `_obs()` with NumPy slicing (no pygame surface needed). Image size derives from grid + `BOARD_PAD` + `MARGIN`.
- **Rewards** (`REWARDS` dict): win `+1.0`, fall `-1.0`, open gate `+0.1`, close gate `-0.1`, per-step `-0.01` (step reward and gate reward stack).
- **Termination:** on win or fall. **Truncation:** after `max_steps` (constructor default `500`; the registered envs override via Gym's `max_episode_steps=100` wrapper). `info["final_info"]["solved"]` is set when the episode ends.
- **`fall_terminates`** (constructor, default `True`): when `False`, an invalid move (off-board / void / closed gate) no longer terminates — the block stays in place and the `fall` reward (`-1.0`) is applied as a penalty. The episode then only ends on win or truncation.
- **Gate state:** `_gate_open` toggles when the block stands `UPRIGHT` on `_switch_cell`.
- **Render modes:** `metadata["render_modes"] = ["human", "ansi", "rgb_array"]`. `human` opens a pygame window via `_render_human`; `ansi` prints a text grid via `_render_ansi`; `rgb_array` returns the observation.

### Registered envs (`registration.py`)

| ID | gate | grid | path_length | branch_tiles | notes |
|----|------|------|-------------|--------------|-------|
| `Bloxorz-v0` | True | 14×10 | 40 | 0 | |
| `Bloxorz-NoGate-v0` | False | 10×10 | 40 | 20 | |
| `Bloxorz-NoFall-v0` | False | 10×10 | 40 | 20 | `fall_terminates=False` — invalid moves penalize but don't end the episode |

Both register with `max_episode_steps=100`. Note the **registration kwargs differ from `BloxorzEnv.__init__` defaults** (18×14, `branch_tiles=40`, `max_steps=500`) — when constructing the env directly, pass explicit kwargs.

```python
import gymnasium as gym
import bloxorz  # registers the envs
env = gym.make("Bloxorz-v0")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## Gotchas

- Keep `core.py` the single source of truth: `env.py`'s `_obs`/`_render_*` and `core.py`'s `draw_board`/`draw_block` are parallel renderers of the same board — changes to cell colors or block geometry must stay in sync between them.
- Board indexing is `board[row][col]`, but block state and most function signatures are `(col, row)` order. Don't transpose by accident.
- `core.main()` (the `bloxorz` CLI) is the standalone pygame game and does **not** go through `BloxorzEnv`; it has its own game loop.
