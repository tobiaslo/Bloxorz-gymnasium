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

You can also drive the registered Gym environment by hand:

```bash
python3 test_human.py [--record [DIR]]
```

`--record` (default `DIR=frames/`) writes every observation to `DIR/<timestamp>/episode_NNN/frame_NNNN.png`; `S` saves a single one-off PNG to the current directory.

## Gymnasium environments

Four environments are registered:

| ID | Gate | Grid | Path length | Branch tiles | Notes |
|----|------|------|-------------|--------------|-------|
| `Bloxorz-v0` | Yes | 14×10 | 40 | 0 | |
| `Bloxorz-NoGate-v0` | No | 10×10 | 30 | 20 | |
| `Bloxorz-NoGate-v1` | No | 10×10 | 30 | 20 | Larger terminal rewards: +10 win, −2 fall |
| `Bloxorz-NoFall-v0` | No | 10×10 | 30 | 50 | Invalid moves penalize but don't end the episode |

All four truncate at `max_episode_steps=100`.

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

Any subset of these can be overridden per-env with the `rewards` argument (e.g. `Bloxorz-NoGate-v1` uses `{"win": 10.0, "fall": -2.0}`):

```python
env = gym.make("Bloxorz-NoGate-v0", rewards={"win": 5.0, "fall": -5.0})
```

### Episode termination

Episodes end on win or fall, or after `max_episode_steps=100` steps (truncation).

With `fall_terminates=False` (the `Bloxorz-NoFall-v0` env) an invalid move no longer ends the episode: the block stays in place, the fall reward (−1.0) is applied as a penalty, and the episode only ends on a win or truncation.

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
| `fall_terminates` | `True` | If `False`, invalid moves penalize instead of ending the episode |
| `rewards` | `None` | Dict overriding any of the default reward values (`win`, `fall`, `gate_open`, `gate_close`, `step`) |
| `render_mode` | `None` | `None`, `"human"`, `"ansi"`, or `"rgb_array"` |

```python
env = gym.make("Bloxorz-v0", grid_cols=20, grid_rows=14, branch_tiles=10)
```

## JAX version (`bloxorz.jx`)

A JAX port of the no-gate environments: level generation and dynamics are pure, jittable, vmappable functions, so a whole batch of environments (including generating fresh levels on reset) runs in one jitted call, or inside `jax.lax.scan`. Install with `pip install -e ".[jax]"`. JAX is only imported when you use `bloxorz.jx`.

```python
import gymnasium as gym
import bloxorz  # registers the JAX ids too

envs = gym.make_vec("JaxBloxorz-NoGate-v0", num_envs=1024)   # native batched env
obs, info = envs.reset(seed=0)                                # (1024, 96, 96, 3) uint8
obs, reward, terminated, truncated, info = envs.step(envs.action_space.sample())
```

| ID | Matches | Notes |
|----|---------|-------|
| `JaxBloxorz-NoGate-v0` | `Bloxorz-NoGate-v0` | |
| `JaxBloxorz-NoGate-v1` | `Bloxorz-NoGate-v1` | +10 win, −2 fall |
| `JaxBloxorz-NoFall-v0` | `Bloxorz-NoFall-v0` | `fall_terminates=False` |

`gym.make(id)` gives the single-env `BloxorzGymEnv`. Episodes truncate at `max_steps=100` inside the env (there is no `TimeLimit` wrapper). The vector env uses Gymnasium's `NEXT_STEP` autoreset by default; pass `autoreset_mode=` for `SAME_STEP` or `DISABLED`, and `jax_output=True` to keep outputs on the device.

For a fully on-device training loop, use the functional env directly:

```python
import jax
from bloxorz import jx

env = jx.Bloxorz(reward_win=10.0)               # any GenParams/EnvParams field by name
obs, state = jax.vmap(env.reset)(jax.random.split(jax.random.key(0), 1024))
obs, state, reward, terminated, truncated, info = jax.vmap(env.step_autoreset)(
    jax.random.split(jax.random.key(1), 1024), state, actions)
```

`step_autoreset` returns the next episode's first observation when an episode ends. The episode that ended is in `info["final_obs"]`, `info["final_info"]` (`episode_return`, `time`, `solved`) and `info["final_state"]`.

**Same game, same levels distribution.** Rules, rewards and the RGB observation are identical to `BloxorzEnv(gate=False)`. The tests replay random play on the same boards through both envs and compare every reward, done flag and pixel. The generator is the same reverse walk plus branch fill, so every level is solvable (the tests BFS-check them). The same seed does not give the same level as the NumPy env, because the RNGs differ. `obs_mode="planes"` gives a compact float32 `(12, 12, 3)` observation (tile / goal / block planes) instead of the image.

On a laptop CPU, the batched env runs at roughly 200k steps/s with RGB observations in `lax.scan` and 360k with `planes` (the NumPy env runs at about 9k). Most of that time is spent generating levels for autoreset.

Run the tests with `pip install -e ".[dev]" && pytest`.
