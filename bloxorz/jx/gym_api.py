"""Gymnasium wrappers around the functional JAX environment.

:class:`BloxorzVectorEnv` implements ``gymnasium.vector.VectorEnv``: the whole
batch -- level generation included -- is stepped by one jitted call.
:class:`BloxorzGymEnv` is the single-environment ``gymnasium.Env``, a drop-in
for ``BloxorzEnv(gate=False)``, for debugging and rendering.
"""

from __future__ import annotations

import functools
from typing import Optional

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space

from . import env as core
from .env import Bloxorz, EnvState, episode_info, reset_aware_info
from .rules import ACTION_NAMES


def _observation_space(obs_mode: str, shape) -> gym.Space:
    if obs_mode == "rgb":
        return gym.spaces.Box(0, 255, shape=shape, dtype=np.uint8)
    return gym.spaces.Box(0.0, 1.0, shape=shape, dtype=np.float32)


class BloxorzVectorEnv(VectorEnv):
    """Vectorised Bloxorz on top of JAX.

    Args:
        num_envs: how many environments to step in lockstep.
        obs_mode, tile_size, gen_params, params, **overrides: as for
            :class:`bloxorz.jx.Bloxorz`.
        autoreset_mode: Gymnasium's ``NEXT_STEP`` (default), ``SAME_STEP``,
            or ``DISABLED``.
        jax_output: return device arrays instead of NumPy, which saves a
            device-to-host copy per step when the policy is JAX too.

    Example::

        envs = BloxorzVectorEnv(num_envs=1024)
        obs, info = envs.reset(seed=0)
        obs, rew, term, trunc, info = envs.step(envs.action_space.sample())
    """

    metadata = {"render_modes": ["rgb_array"], "autoreset_mode": AutoresetMode.NEXT_STEP, "jax": True}

    def __init__(
        self,
        num_envs: int = 1,
        obs_mode: str = "rgb",
        autoreset_mode: AutoresetMode = AutoresetMode.NEXT_STEP,
        jax_output: bool = False,
        render_mode: Optional[str] = "rgb_array",
        seed: Optional[int] = None,
        **env_kwargs,
    ):
        self.env = Bloxorz(obs_mode=obs_mode, **env_kwargs)
        self._key = jax.random.key(0 if seed is None else int(seed))
        self.num_envs = int(num_envs)
        self.autoreset_mode = AutoresetMode(autoreset_mode)
        self.jax_output = bool(jax_output)
        self.render_mode = render_mode
        self.metadata = dict(self.metadata, autoreset_mode=self.autoreset_mode)

        self.single_action_space = gym.spaces.Discrete(4)
        self.single_observation_space = _observation_space(obs_mode, self.env.obs_shape)
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)

        self._state: Optional[EnvState] = None
        self._prev_done: Optional[jnp.ndarray] = None
        self._reset_batch = jax.jit(jax.vmap(self.env.reset_state))
        self._step_batch = jax.jit(functools.partial(_vector_step, self.env, self.autoreset_mode))

    def _split(self):
        self._key, sub = jax.random.split(self._key)
        return jax.random.split(sub, self.num_envs)

    def _out(self, tree):
        return tree if self.jax_output else jax.tree.map(np.asarray, tree)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._key = jax.random.key(int(seed))
        self._state = self._reset_batch(self._split())
        self._prev_done = jnp.zeros((self.num_envs,), bool)
        return self._out(jax.vmap(self.env.observe)(self._state)), {}

    def step(self, actions):
        if self._state is None:
            raise RuntimeError("call reset() before step()")
        actions = jnp.asarray(actions, jnp.int32).reshape(self.num_envs)
        state, obs, reward, terminated, truncated, info = self._step_batch(
            self._state, actions, self._prev_done, self._split()
        )
        self._state = state
        self._prev_done = terminated | truncated
        return self._out((obs, reward, terminated, truncated, info))

    def render(self):
        """Every environment as ``uint8[num_envs, H, W, 3]``."""
        if self._state is None:
            raise RuntimeError("call reset() before render()")
        return np.asarray(jax.vmap(self.env.render)(self._state))

    @property
    def state(self) -> Optional[EnvState]:
        """The underlying batched JAX state."""
        return self._state

    def __repr__(self):
        return f"BloxorzVectorEnv(num_envs={self.num_envs}, {self.env!r})"


def _vector_step(env: Bloxorz, autoreset_mode: AutoresetMode, state, actions, prev_done, keys):
    """One jitted batched step, including whichever auto-reset mode is in force."""
    step_one = functools.partial(core.step, params=env.params)
    new_state, reward, terminated, truncated, info = jax.vmap(step_one)(state, actions)
    done = terminated | truncated

    if autoreset_mode is AutoresetMode.DISABLED:
        out_state = new_state
    else:
        fresh = jax.vmap(env.reset_state)(keys)
        if autoreset_mode is AutoresetMode.SAME_STEP:
            # Reward and done flags describe the transition that ended; the
            # observation and info already belong to the next episode.
            out_state = _where_tree(done, fresh, new_state)
            final_info = episode_info(new_state, info["solved"])
            info = reset_aware_info(info, out_state, done)
            info["final_obs"] = jax.vmap(env.observe)(new_state)
            info["final_info"] = final_info
        else:  # NEXT_STEP: the step after a terminal one resets and ignores its action.
            out_state = _where_tree(prev_done, fresh, new_state)
            reward = jnp.where(prev_done, 0.0, reward)
            terminated = terminated & ~prev_done
            truncated = truncated & ~prev_done
            info = reset_aware_info(info, out_state, prev_done)
            # As BloxorzEnv reports it, on the step an episode ends.
            info["final_info"] = {"solved": info["solved"] & (terminated | truncated)}

    obs = jax.vmap(env.observe)(out_state)
    return out_state, obs, reward, terminated, truncated, info


def _where_tree(mask, a, b):
    return jax.tree.map(
        lambda x, y: jnp.where(mask.reshape(mask.shape + (1,) * (x.ndim - 1)), x, y), a, b
    )


class BloxorzGymEnv(gym.Env):
    """Single-environment ``gymnasium.Env``: the slow path, one host sync per step.

    Render modes: ``"rgb_array"`` and ``"ansi"``. Use :class:`BloxorzVectorEnv`
    for training.
    """

    metadata = {"render_modes": ["rgb_array", "ansi"], "render_fps": 8}

    def __init__(
        self,
        obs_mode: str = "rgb",
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        **env_kwargs,
    ):
        self.env = Bloxorz(obs_mode=obs_mode, **env_kwargs)
        self._key = jax.random.key(0 if seed is None else int(seed))
        self.render_mode = render_mode
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = _observation_space(obs_mode, self.env.obs_shape)
        self._state: Optional[EnvState] = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._key = jax.random.key(int(seed))
        self._key, sub = jax.random.split(self._key)
        obs, self._state = self.env.reset(sub)
        return np.asarray(obs), {}

    def step(self, action):
        if self._state is None:
            raise RuntimeError("call reset() before step()")
        obs, self._state, reward, terminated, truncated, info = self.env.step(
            self._state, jnp.asarray(action, jnp.int32)
        )
        terminated, truncated = bool(terminated), bool(truncated)
        info = {k: np.asarray(v).item() for k, v in info.items()}
        info["action.name"] = ACTION_NAMES[int(action)]
        if terminated or truncated:
            info["final_info"] = {"solved": info["solved"]}
        return np.asarray(obs), float(reward), terminated, truncated, info

    def render(self):
        if self._state is None:
            raise RuntimeError("call reset() before render()")
        if self.render_mode == "ansi":
            return self.env.to_ansi(self._state)
        return np.asarray(self.env.render(self._state))

    @property
    def state(self) -> Optional[EnvState]:
        return self._state
