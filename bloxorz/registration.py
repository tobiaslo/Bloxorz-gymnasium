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
        "path_length": 40,
        "branch_tiles": 20,
        "render_mode": None,
    },
)
