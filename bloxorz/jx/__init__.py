"""Bloxorz in JAX: jittable, vmappable generation and dynamics (no gate yet).

Three layers, from fastest to most convenient:

* :mod:`bloxorz.jx.generate` -- level generation as pure functions.
* :mod:`bloxorz.jx.env` -- dynamics as pure functions over :class:`EnvState`,
  plus :class:`Bloxorz`, which binds them to generation and reward parameters.
* :class:`BloxorzVectorEnv` / :class:`BloxorzGymEnv` -- the Gymnasium APIs,
  also reachable as ``gym.make_vec("JaxBloxorz-NoGate-v0", num_envs=n)``.
"""

from .env import (
    OBS_MODES,
    Bloxorz,
    EnvParams,
    EnvState,
    episode_info,
    obs_shape,
    observe,
    reset,
    state_from_level,
    step,
)
from .generate import GenParams, Level, generate_level, generate_levels
from .gym_api import BloxorzGymEnv, BloxorzVectorEnv
from .render import render_rgb, save_gif, to_ansi
from .rules import ACTION_NAMES, ACTIONS, MOVES

__all__ = [
    "ACTIONS", "ACTION_NAMES", "MOVES", "OBS_MODES",
    "Bloxorz", "EnvParams", "EnvState", "GenParams", "Level",
    "BloxorzGymEnv", "BloxorzVectorEnv",
    "episode_info", "generate_level", "generate_levels", "obs_shape", "observe",
    "render_rgb", "reset", "save_gif", "state_from_level", "step", "to_ansi",
]
