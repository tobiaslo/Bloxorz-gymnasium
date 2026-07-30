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
