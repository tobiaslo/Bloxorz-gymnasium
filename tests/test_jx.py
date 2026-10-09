"""The JAX port (bloxorz.jx) checked against the NumPy original (bloxorz.env/core)."""

from collections import deque

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gymnasium.vector import AutoresetMode

import bloxorz  # noqa: F401  (registers the envs)
from bloxorz import jx
from bloxorz.core import UPRIGHT, generate_board, is_valid, is_win, move
from bloxorz.env import ACTIONS, BloxorzEnv

GEN = jx.GenParams()  # Bloxorz-NoGate-v0's geometry


def numpy_env(board, block, **kwargs):
    """A BloxorzEnv put into the given board and block, bypassing its generator."""
    env = BloxorzEnv(grid_cols=GEN.grid_cols, grid_rows=GEN.grid_rows, gate=False, **kwargs)
    env.reset(seed=0)
    env._board = [list(map(int, row)) for row in np.asarray(board)]
    env._col, env._row, env._ori = (int(x) for x in block)
    return env


def solvable(board, start):
    """BFS over block states with core.py's own rules."""
    board = [list(map(int, row)) for row in board]
    seen, queue = {tuple(start)}, deque([tuple(start)])
    while queue:
        state = queue.popleft()
        if is_win(*state, board):
            return True
        for d in ACTIONS:
            nxt = move(*state, d)
            if nxt not in seen and is_valid(*nxt, board):
                seen.add(nxt)
                queue.append(nxt)
    return False


@pytest.fixture(scope="module")
def levels():
    return jax.device_get(jx.generate_levels(jax.random.key(0), 300, GEN))


def test_roll_table_matches_core():
    for ori in range(3):
        for a, d in enumerate(ACTIONS):
            block = jnp.array([5, 6, ori])
            assert tuple(int(x) for x in jx.rules.roll(block, a)) == move(5, 6, ori, d)


def test_generated_levels_are_solvable(levels):
    for board, start in zip(levels.board, levels.start):
        assert solvable(board, start)
        assert is_valid(*map(int, start), [list(map(int, r)) for r in board])
        assert (board == jx.rules.GOAL).sum() == 1


def test_levels_stay_inside_the_border(levels):
    assert levels.board.shape[1:] == GEN.board_shape
    border = np.ones(GEN.board_shape, bool)
    border[1:-1, 1:-1] = False
    assert (levels.board[:, border] == 0).all()


def test_generation_matches_numpy_distribution(levels):
    """Same generator, different RNG: the level statistics should agree."""
    rng = np.random.default_rng(0)
    np_boards = [
        np.array(generate_board(GEN.grid_cols, GEN.grid_rows, GEN.path_length,
                                GEN.branch_tiles, rng=rng, gate=False)[0])
        for _ in range(len(levels.board))
    ]
    np_tiles = np.array([(b > 0).sum() for b in np_boards])
    jx_tiles = (levels.board > 0).sum(axis=(1, 2))
    # Standard error of each mean is ~0.5 tiles over 300 levels.
    assert abs(np_tiles.mean() - jx_tiles.mean()) < 2.5, (np_tiles.mean(), jx_tiles.mean())
    np_goal_cols = np.array([np.argwhere(b == 2)[0][1] for b in np_boards])
    jx_goal_cols = np.array([np.argwhere(b == 2)[0][1] for b in levels.board])
    assert set(np_goal_cols) == set(jx_goal_cols)


@pytest.mark.parametrize("fall_terminates", [True, False])
def test_dynamics_match_numpy(levels, fall_terminates):
    """Random play on the same boards gives the same transitions and pixels."""
    env = jx.Bloxorz(fall_terminates=fall_terminates)
    rng = np.random.default_rng(1)
    for board, start in list(zip(levels.board, levels.start))[:60]:
        ref = numpy_env(board, start, fall_terminates=fall_terminates, max_steps=100)
        state = jx.state_from_level(jx.Level(board, start))
        assert (np.asarray(env.observe(state)) == ref._obs()).all()
        for _ in range(100):
            action = int(rng.integers(4))
            want = ref.step(action)
            obs, state, reward, terminated, truncated, info = env.step(state, action)
            assert float(reward) == pytest.approx(want[1])
            assert bool(terminated) == want[2]
            assert bool(truncated) == want[3]
            assert (np.asarray(obs) == want[0]).all()
            if want[2] or want[3]:
                assert bool(info["solved"]) == want[4]["final_info"]["solved"]
                break


def test_numpy_boards_play_the_same(levels):
    """And boards from core.generate_board run unchanged in the JAX env."""
    rng = np.random.default_rng(2)
    env = jx.Bloxorz()
    for _ in range(30):
        board, c, r, o, _, _ = generate_board(GEN.grid_cols, GEN.grid_rows, GEN.path_length,
                                              GEN.branch_tiles, rng=rng, gate=False)
        state = jx.state_from_level(jx.Level(np.array(board, np.uint8), np.array([c, r, o])))
        ref = numpy_env(board, (c, r, o))
        for _ in range(30):
            action = int(rng.integers(4))
            want = ref.step(action)
            obs, state, reward, terminated, *_ = env.step(state, action)
            assert float(reward) == pytest.approx(want[1])
            assert (np.asarray(obs) == want[0]).all()
            if want[2]:
                break


def test_win_reward_and_override():
    board = np.zeros(GEN.board_shape, np.uint8)
    board[5, 3:6] = 1
    board[5, 6] = 2
    level = jx.Level(board, np.array([4, 5, 1]))  # FLAT_H on cols 4-5, rolls right onto 6
    for env, win in ((jx.Bloxorz(), 1.0), (jx.Bloxorz(reward_win=10.0), 10.0)):
        _, _, reward, terminated, _, info = env.step(jx.state_from_level(level), 0)
        assert float(reward) == win and bool(terminated) and bool(info["solved"])


def test_no_fall_keeps_the_block():
    board = np.zeros(GEN.board_shape, np.uint8)
    board[5, 4:6] = 1
    board[5, 8] = 2
    state = jx.state_from_level(jx.Level(board, np.array([4, 5, 1])))
    _, nxt, reward, terminated, _, info = jx.Bloxorz(fall_terminates=False).step(state, 2)
    assert bool(info["fell"]) and not bool(terminated) and float(reward) == -1.0
    assert (nxt.block == state.block).all() and int(nxt.time) == 1


def test_scan_rollout_with_autoreset():
    env = jx.Bloxorz()
    _, state = env.reset(jax.random.key(0))

    def body(carry, key):
        state = carry
        k_act, k_reset = jax.random.split(key)
        action = jax.random.randint(k_act, (), 0, 4)
        _, state, reward, terminated, truncated, info = env.step_autoreset(k_reset, state, action)
        return state, (reward, terminated | truncated, info["final_info"]["time"])

    _, (rewards, dones, times) = jax.lax.scan(body, state, jax.random.split(jax.random.key(1), 500))
    assert dones.sum() > 10
    assert (times[dones] >= 1).all() and (times[dones] <= 100).all()


@pytest.mark.parametrize("mode", list(AutoresetMode))
def test_vector_env_autoreset(mode):
    envs = jx.BloxorzVectorEnv(num_envs=16, autoreset_mode=mode, seed=0)
    obs, _ = envs.reset()
    assert obs.shape == (16, 96, 96, 3) and obs.dtype == np.uint8
    assert envs.observation_space.contains(obs)
    prev_done = np.zeros(16, bool)
    for _ in range(50):
        actions = envs.action_space.sample()
        obs, reward, terminated, truncated, info = envs.step(actions)
        done = terminated | truncated
        if mode is AutoresetMode.NEXT_STEP:
            assert (reward[prev_done] == 0).all() and not done[prev_done].any()
            assert (info["time"][prev_done] == 0).all()
        if mode is AutoresetMode.SAME_STEP:
            assert (info["time"][done] == 0).all()
            assert (info["final_info"]["time"][done] >= 1).all()
        prev_done = done
        if mode is AutoresetMode.DISABLED and done.any():
            break


def test_registered_ids():
    envs = gym.make_vec("JaxBloxorz-NoGate-v1", num_envs=4)
    assert isinstance(envs, jx.BloxorzVectorEnv)
    assert envs.env.params.reward_win == 10.0 and envs.env.params.reward_fall == -2.0
    nofall = gym.make("JaxBloxorz-NoFall-v0").unwrapped
    assert nofall.env.params.fall_terminates is False
    assert nofall.env.gen_params.branch_tiles == 50


def test_gym_env_passes_checker():
    from gymnasium.utils.env_checker import check_env

    check_env(jx.BloxorzGymEnv(), skip_render_check=True)


def test_planes_observation():
    env = jx.Bloxorz(obs_mode="planes")
    obs, state = env.reset(jax.random.key(0))
    assert obs.shape == (*GEN.board_shape, 3)
    assert obs[..., 2].sum() == (1 if int(state.block[2]) == UPRIGHT else 2)
    assert obs[..., 1].sum() == 1
