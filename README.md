# Bloxorz

A Gymnasium environment for the Bloxorz puzzle game, where a rectangular block must be rolled onto a goal tile.

This version is a 2d version of the original game, with only randomly generated levels. There is also a lot of the more advanced mechanics that are not implemented yet.
## Installation

```bash
pip install -e .
```

Requires Python 3.10+. Dependencies (`numpy`, `gymnasium`, `pygame`) are installed automatically.

## Playing the game

```bash
bloxorz [--cols N] [--rows N] [--no-gate]
```

| Flag | Default | Effect |
|------|---------|--------|
| `--cols` | 14 | Playable grid width in tiles |
| `--rows` | 10 | Playable grid height in tiles |
| `--no-gate` | off | Disable the bridge/gate mechanic |

**Controls:** Arrow keys move the block, `R` resets, `N` generates a new board, `Q` quits.

## Gymnasium environments

Two environments are registered:

| ID | Gate | Grid | Path length | Branch tiles |
|----|------|------|-------------|--------------|
| `Bloxorz-v0` | Yes | 14×10 | 40 | 0 |
| `Bloxorz-NoGate-v0` | No | 12×8 | 40 | 20 |

```python
import gymnasium as gym
import bloxorz  # registers the environments

env = gym.make("Bloxorz-v0")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

### Actions

| Index | Direction |
|-------|-----------|
| 0 | Right |
| 1 | Left |
| 2 | Down |
| 3 | Up |

### Observations

RGB image of shape `(H, W, 3)` rendered in pure NumPy (no display required).

### Rewards

| Event | Reward |
|-------|--------|
| Win (block upright on goal) | +1.0 |
| Fall (invalid move) | −1.0 |
| Gate opened (step on switch) | +0.1 |
| Gate closed (step on switch again) | −0.1 |
| Step | −0.01 |

### Episode termination

Episodes end on win or fall, or after `max_episode_steps=100` steps (truncation).

## Game mechanics

**Block state** is `(col, row, ori)` where `ori` is one of `UPRIGHT`, `FLAT_H`, or `FLAT_V`. `FLAT_H` occupies `(col, row)` and `(col+1, row)`; `FLAT_V` occupies `(col, row)` and `(col, row+1)`. The goal requires the block to arrive `UPRIGHT`.

**Board cells:** `0` = void, `1` = tile, `2` = goal, `3` = gate tile, `4` = switch.

**Gate mechanic:** Standing `UPRIGHT` on the switch (cell `4`) toggles the gate. Gate tiles (cell `3`) are impassable while the gate is closed. The gate must be opened to cross the bridge to the goal island.

**Board generation** uses a reverse walk from the goal — the generator starts at the goal tile and takes random steps, producing a path the player can always solve by retracing it. `path_length` controls the number of walk steps; `branch_tiles` adds extra scattered tiles around the path.

## Environment arguments

All arguments can be passed to `gym.make` or `BloxorzEnv` directly:

| Argument | Default | Description |
|----------|---------|-------------|
| `grid_cols` | 14 | Playable grid width (excluding padding border) |
| `grid_rows` | 10 | Playable grid height (excluding padding border) |
| `gate` | `True` | Enable two-island bridge/gate mechanic |
| `path_length` | 40 | Reverse-walk steps during board generation |
| `branch_tiles` | 0 | Extra scattered tiles added around the path |
| `render_mode` | `None` | `None`, `"human"`, or `"ansi"` |

```python
env = gym.make("Bloxorz-v0", grid_cols=20, grid_rows=14, branch_tiles=10)
```
