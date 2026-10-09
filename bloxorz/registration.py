import gymnasium as gym

gym.register(
    id="Bloxorz-v0",
    entry_point="bloxorz.env:BloxorzEnv",
    max_episode_steps=100,
    kwargs={
        "grid_cols": 14,
        "grid_rows": 10,
        "gate": True,
        "path_length": 40,
        "branch_tiles": 0,
        "render_mode": None,
    },
)

gym.register(
    id="Bloxorz-NoGate-v0",
    entry_point="bloxorz.env:BloxorzEnv",
    max_episode_steps=100,
    kwargs={
        "grid_cols": 10,
        "grid_rows": 10,
        "gate": False,
        "path_length": 30,
        "branch_tiles": 20,
        "render_mode": None,
    },
)

# Same as Bloxorz-NoGate-v0, but with larger terminal rewards: +5 for solving
# and -5 for falling off (step/gate rewards keep their defaults).
gym.register(
    id="Bloxorz-NoGate-v1",
    entry_point="bloxorz.env:BloxorzEnv",
    max_episode_steps=100,
    kwargs={
        "grid_cols": 10,
        "grid_rows": 10,
        "gate": False,
        "path_length": 30,
        "branch_tiles": 20,
        "rewards": {"win": 10.0, "fall": -2.0},
        "render_mode": None,
    },
)

# Same as Bloxorz-NoGate-v0, but invalid moves (off-board / void) keep the
# block in place and apply the fall penalty instead of ending the episode.
gym.register(
    id="Bloxorz-NoFall-v0",
    entry_point="bloxorz.env:BloxorzEnv",
    max_episode_steps=100,
    kwargs={
        "grid_cols": 10,
        "grid_rows": 10,
        "gate": False,
        "path_length": 30,
        "branch_tiles": 50,
        "fall_terminates": False,
        "render_mode": None,
    },
)

# --- JAX versions (bloxorz.jx) ---------------------------------------------
# The no-gate envs above, ported to JAX. `gym.make(id)` gives the single
# BloxorzGymEnv; `gym.make_vec(id, num_envs=n)` gives the native batched
# BloxorzVectorEnv, stepped by one jitted call. The episode limit lives in the
# env (`max_steps`) rather than a TimeLimit wrapper, so it applies to both.
# Entry points are strings, so registering here does not import JAX.
_JAX_NO_GATE = {"grid_cols": 10, "grid_rows": 10, "path_length": 30, "max_steps": 100}

for _id, _kwargs in {
    "JaxBloxorz-NoGate-v0": {"branch_tiles": 20},
    "JaxBloxorz-NoGate-v1": {"branch_tiles": 20, "reward_win": 10.0, "reward_fall": -2.0},
    "JaxBloxorz-NoFall-v0": {"branch_tiles": 50, "fall_terminates": False},
}.items():
    gym.register(
        id=_id,
        entry_point="bloxorz.jx.gym_api:BloxorzGymEnv",
        vector_entry_point="bloxorz.jx.gym_api:BloxorzVectorEnv",
        max_episode_steps=None,
        kwargs={**_JAX_NO_GATE, **_kwargs},
    )
