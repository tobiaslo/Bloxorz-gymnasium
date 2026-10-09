"""Pure-JAX Bloxorz dynamics (no gate) over procedurally generated levels.

Everything here is a jittable, vmappable function over pytrees. The class
:class:`Bloxorz` binds those functions to the generation and reward parameters,
the way ``jaxoban.Sokoban`` does.

The rules are ``BloxorzEnv``'s with ``gate=False``: an action rolls the block;
a roll that leaves the board or lands on void is a fall, which leaves the block
where it was and either ends the episode or, with ``fall_terminates=False``,
only costs the fall penalty. Standing upright on the goal wins.
"""

from __future__ import annotations

import functools
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

from ..core import TILE_SIZE, UPRIGHT
from .generate import GenParams, Level, generate_level
from .render import image_shape, render_rgb, to_ansi
from .rules import GOAL, VOID, fits, roll, second_cell


class EnvState(NamedTuple):
    """A single environment's state. Every field is a device array."""

    board: jnp.ndarray  # uint8 [H, W]  cell codes, static for the episode
    block: jnp.ndarray  # int32 [3]     (col, row, ori)
    time: jnp.ndarray  # int32 []       steps taken this episode
    episode_return: jnp.ndarray  # float32 []


class EnvParams(NamedTuple):
    """Episode length and rewards, defaulting to ``Bloxorz-NoGate-v0``'s.

    ``max_steps`` plays the part of the registered envs' ``max_episode_steps``.
    """

    max_steps: int = 100
    reward_win: float = 1.0
    reward_fall: float = -1.0
    reward_step: float = -0.01
    fall_terminates: bool = True


# --- state ------------------------------------------------------------------
def state_from_level(level: Level) -> EnvState:
    return EnvState(
        board=jnp.asarray(level.board, jnp.uint8),
        block=jnp.asarray(level.start, jnp.int32),
        time=jnp.zeros((), jnp.int32),
        episode_return=jnp.zeros((), jnp.float32),
    )


def reset(key, gen_params: GenParams = GenParams()) -> EnvState:
    """Generate a fresh level and return its initial state."""
    return state_from_level(generate_level(key, gen_params))


# --- dynamics ---------------------------------------------------------------
def step(state: EnvState, action, params: EnvParams = EnvParams()):
    """Advance one step. Returns ``(state, reward, terminated, truncated, info)``.

    No auto-reset: the returned state is terminal when ``terminated | truncated``.
    """
    rolled = roll(state.block, jnp.asarray(action, jnp.int32))
    valid = fits(state.board != VOID, rolled)
    # `fits` already bounds-checked `rolled` when `valid`; clamping only keeps
    # the gather legal when it is not.
    h, w = state.board.shape
    on_goal = state.board[jnp.clip(rolled[1], 0, h - 1), jnp.clip(rolled[0], 0, w - 1)] == GOAL
    solved = valid & (rolled[2] == UPRIGHT) & on_goal
    fell = ~valid

    reward = jnp.where(
        fell, params.reward_fall, jnp.where(solved, params.reward_win, params.reward_step)
    ).astype(jnp.float32)
    terminated = solved | (fell & params.fall_terminates)
    time = state.time + 1
    truncated = (time >= params.max_steps) & ~terminated
    episode_return = state.episode_return + reward

    new_state = state._replace(
        block=jnp.where(valid, rolled, state.block),
        time=time,
        episode_return=episode_return,
    )
    info = {"solved": solved, "fell": fell, "time": time, "episode_return": episode_return}
    return new_state, reward, terminated, truncated, info


def reset_aware_info(info: dict, state: EnvState, was_reset) -> dict:
    """Make ``info`` describe the episode whose observation is being returned.

    On an auto-reset step the observation belongs to the fresh episode, so the
    counters come from that state and the step's events read False.
    """
    return dict(
        info,
        time=state.time,
        episode_return=state.episode_return,
        solved=info["solved"] & ~was_reset,
        fell=info["fell"] & ~was_reset,
    )


def episode_info(state: EnvState, solved) -> dict:
    """The stats of the episode that just ended, for ``info["final_info"]``."""
    return {"episode_return": state.episode_return, "time": state.time, "solved": solved}


# --- observations -----------------------------------------------------------
#: ``"rgb"`` is ``BloxorzEnv``'s image observation. ``"planes"`` is a compact
#: float32 ``[H, W, 3]`` of (tile, goal, block) indicator planes, one cell per
#: pixel -- the same information at a fraction of the size.
OBS_MODES = ("rgb", "planes")


def observe(state: EnvState, mode: str = "rgb", tile_size: int = TILE_SIZE) -> jnp.ndarray:
    if mode == "rgb":
        return render_rgb(state.board, state.block, tile_size)
    if mode == "planes":
        col2, row2 = second_cell(state.block)
        block = jnp.zeros(state.board.shape, bool)
        block = block.at[state.block[1], state.block[0]].set(True).at[row2, col2].set(True)
        return jnp.stack(
            [state.board != VOID, state.board == GOAL, block], axis=-1
        ).astype(jnp.float32)
    raise ValueError(f"unknown observation mode {mode!r}; expected one of {OBS_MODES}")


def obs_shape(mode: str, board_shape, tile_size: int = TILE_SIZE):
    if mode == "rgb":
        return image_shape(board_shape, tile_size)
    if mode == "planes":
        return (*board_shape, 3)
    raise ValueError(f"unknown observation mode {mode!r}; expected one of {OBS_MODES}")


# --- functional environment -------------------------------------------------
class Bloxorz:
    """Functional Bloxorz environment: state in, state out, jit/vmap friendly.

    ``reset``/``step`` are single-environment functions; batch them with
    ``jax.vmap``, or use :class:`bloxorz.jx.BloxorzVectorEnv` for the
    Gymnasium vector API.

    Args:
        gen_params: board size, walk length and branch count.
        params: episode length and rewards.
        obs_mode: ``"rgb"`` (default, as ``BloxorzEnv``) or ``"planes"``.
        tile_size: pixels per cell for ``"rgb"``.
        **overrides: any :class:`GenParams` or :class:`EnvParams` field by
            name, e.g. ``grid_cols=14`` or ``reward_win=10.0``.

    Example::

        env = Bloxorz()
        obs, state = env.reset(jax.random.key(0))
        obs, state, reward, terminated, truncated, info = env.step(state, 0)
    """

    def __init__(
        self,
        gen_params: GenParams = GenParams(),
        params: EnvParams = EnvParams(),
        obs_mode: str = "rgb",
        tile_size: int = TILE_SIZE,
        **overrides,
    ):
        if obs_mode not in OBS_MODES:
            raise ValueError(f"unknown observation mode {obs_mode!r}; expected one of {OBS_MODES}")
        unknown = set(overrides) - set(GenParams._fields) - set(EnvParams._fields)
        if unknown:
            raise TypeError(f"unknown parameters: {sorted(unknown)}")
        self.gen_params = gen_params._replace(
            **{k: int(v) for k, v in overrides.items() if k in GenParams._fields}
        )
        self.params = params._replace(
            **{k: v for k, v in overrides.items() if k in EnvParams._fields}
        )
        self.obs_mode = obs_mode
        self.tile_size = int(tile_size)

    @property
    def board_shape(self):
        return self.gen_params.board_shape

    @property
    def obs_shape(self):
        return obs_shape(self.obs_mode, self.board_shape, self.tile_size)

    def observe(self, state: EnvState) -> jnp.ndarray:
        return observe(state, self.obs_mode, self.tile_size)

    # -- core API -----------------------------------------------------------
    def reset_state(self, key) -> EnvState:
        """A fresh episode's initial state. Un-jitted, for use inside other jits."""
        return reset(key, self.gen_params)

    @functools.partial(jax.jit, static_argnums=0)
    def reset(self, key):
        """Start a fresh episode and return ``(obs, state)``."""
        state = self.reset_state(key)
        return self.observe(state), state

    @functools.partial(jax.jit, static_argnums=0)
    def step(self, state: EnvState, action):
        """Advance one step without auto-reset."""
        state, reward, terminated, truncated, info = step(state, action, self.params)
        return self.observe(state), state, reward, terminated, truncated, info

    @functools.partial(jax.jit, static_argnums=0)
    def step_autoreset(self, key, state: EnvState, action):
        """Advance one step, starting a fresh episode whenever this one ends.

        Same-step auto-reset: the returned observation and ``info`` belong to
        the *next* episode, and ``info["final_obs"]``, ``info["final_info"]``
        and ``info["final_state"]`` describe the one that ended. This is the
        form you want inside ``lax.scan``.
        """
        new_state, reward, terminated, truncated, info = step(state, action, self.params)
        done = terminated | truncated
        fresh = self.reset_state(key)
        state_out = jax.tree.map(lambda a, b: jnp.where(done, a, b), fresh, new_state)
        info = dict(
            reset_aware_info(info, state_out, done),
            final_obs=self.observe(new_state),
            final_info=episode_info(new_state, info["solved"]),
            final_state=new_state,
        )
        return self.observe(state_out), state_out, reward, terminated, truncated, info

    # -- misc ---------------------------------------------------------------
    def render(self, state: EnvState, tile_size: Optional[int] = None) -> jnp.ndarray:
        return render_rgb(state.board, state.block, tile_size or self.tile_size)

    def to_ansi(self, state: EnvState) -> str:
        return to_ansi(np.asarray(state.board), np.asarray(state.block))

    def __repr__(self):
        g, p = self.gen_params, self.params
        return (
            f"Bloxorz(grid={g.grid_cols}x{g.grid_rows}, path_length={g.path_length}, "
            f"branch_tiles={g.branch_tiles}, obs={self.obs_mode}{self.obs_shape}, "
            f"fall_terminates={p.fall_terminates})"
        )

