"""Level generation as pure, jittable functions.

A port of the ``gate=False`` branch of :func:`bloxorz.core.generate_board`:

1. **Goal** -- a random cell in the right third of the board.
2. **Reverse walk** -- ``path_length`` random rolls from upright on the goal,
   each one uniform over the rolls that stay inside the playable area, stamping
   every cell the block covers. Wherever the walk ends is the start, so every
   level is solvable: retracing the walk reaches the goal.
3. **Branch fill** -- ``branch_tiles`` times, pick a tile (the walk's tiles plus
   every branch tile added so far, duplicates and all) and add the cell two steps
   away in a random direction, if it is inside the playable area.

Both loops have a fixed trip count, so they are ``lax.scan``s and the whole
generator is jittable and vmappable with no rejection sampling. The NumPy
version's "shuffle the four directions, take the first valid one" is the same
distribution as "uniform over the valid ones", which is what stage 2 samples.
Levels come from the same distribution as ``core.generate_board``, but not the
same levels for the same seed, since JAX's RNG differs from NumPy's.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..core import BOARD_PAD, UPRIGHT
from .rules import DIRS, GOAL, MOVES, TILE, VOID, fits, stamp


class GenParams(NamedTuple):
    """Generation knobs, defaulting to ``Bloxorz-NoGate-v0``'s."""

    grid_cols: int = 10  # playable width, BOARD_PAD border excluded
    grid_rows: int = 10  # playable height, BOARD_PAD border excluded
    path_length: int = 30
    branch_tiles: int = 20

    @property
    def board_shape(self):
        """``(rows, cols)`` of the board array, void border included."""
        return (self.grid_rows + 2 * BOARD_PAD, self.grid_cols + 2 * BOARD_PAD)


class Level(NamedTuple):
    board: jnp.ndarray  # uint8 [H, W]  cell codes, indexed board[row, col]
    start: jnp.ndarray  # int32 [3]     (col, row, ori)


def playable_mask(params: GenParams) -> jnp.ndarray:
    """Bool ``[H, W]``: the cells inside the ``BOARD_PAD`` border."""
    h, w = params.board_shape
    rows = jnp.arange(h)[:, None]
    cols = jnp.arange(w)[None, :]
    return (
        (rows >= BOARD_PAD) & (rows < h - BOARD_PAD)
        & (cols >= BOARD_PAD) & (cols < w - BOARD_PAD)
    )


def reverse_walk(key, playable: jnp.ndarray, goal: jnp.ndarray, path_length: int):
    """Random-walk ``path_length`` rolls from upright on ``goal``.

    Returns ``(tiles, end)``: bool ``[H, W]`` of every cell the block covered,
    and the ``(col, row, ori)`` it ended in.
    """
    start = jnp.stack([goal[0], goal[1], jnp.int32(UPRIGHT)])

    def body(carry, u):
        tiles, block = carry
        d = MOVES[block[2]]  # [4, 3]
        cands = jnp.stack([block[0] + d[:, 0], block[1] + d[:, 1], d[:, 2]], axis=1)
        ok = jax.vmap(fits, in_axes=(None, 0))(playable, cands)
        # The k-th open roll, k uniform: uniform over the open rolls.
        k = jnp.floor(u * ok.sum()).astype(jnp.int32)
        pick = jnp.argmax(ok & (jnp.cumsum(ok) - 1 == k))
        # A playable area of at least 2x2 always leaves some roll open; the
        # guard only keeps an impossible dead end from reading garbage.
        block = jnp.where(ok.any(), cands[pick], block)
        return (stamp(tiles, block), block), None

    # Drawn up front: one RNG call instead of one per roll, which on CPU is
    # most of the cost of generating.
    uniforms = jax.random.uniform(key, (path_length,))
    tiles = stamp(jnp.zeros_like(playable), start)
    (tiles, end), _ = jax.lax.scan(body, (tiles, start), uniforms)
    return tiles, end


def branch_fill(key, playable: jnp.ndarray, tiles: jnp.ndarray, branch_tiles: int):
    """Add up to ``branch_tiles`` cells, each two steps from an existing tile.

    Mirrors the NumPy version's growing ``tile_list``: it starts as the distinct
    walk tiles and every accepted branch tile is appended, even one that was
    already a tile, so cells are picked with that same weighting.
    """
    if branch_tiles == 0:
        return tiles
    h, w = tiles.shape
    capacity = h * w + branch_tiles
    pool = jnp.nonzero(tiles.reshape(-1), size=capacity, fill_value=0)[0].astype(jnp.int32)
    count = jnp.sum(tiles).astype(jnp.int32)

    def body(carry, draw):
        tiles, pool, count = carry
        u, d = draw
        cell = pool[jnp.minimum(jnp.floor(u * count).astype(jnp.int32), count - 1)]
        col = cell % w + 2 * DIRS[d, 0]
        row = cell // w + 2 * DIRS[d, 1]
        inside = (col >= 0) & (col < w) & (row >= 0) & (row < h)
        ok = inside & playable[row, col]  # clamped gather, gated by `inside`
        tiles = tiles.at[row, col].set(tiles[row, col] | ok, mode="drop")
        pool = pool.at[count].set(jnp.where(ok, row * w + col, pool[count]))
        return (tiles, pool, count + ok), None

    k_cell, k_dir = jax.random.split(key)
    draws = (
        jax.random.uniform(k_cell, (branch_tiles,)),
        jax.random.randint(k_dir, (branch_tiles,), 0, 4),
    )
    (tiles, _, _), _ = jax.lax.scan(body, (tiles, pool, count), draws)
    return tiles


def generate_level(key, params: GenParams = GenParams()) -> Level:
    """Generate one solvable level. Jit with ``params`` static."""
    h, w = params.board_shape
    k_col, k_row, k_walk, k_branch = jax.random.split(key, 4)
    # Inclusive ranges, as core.py's `_randint`.
    goal = jnp.stack([
        jax.random.randint(k_col, (), w * 2 // 3, w - BOARD_PAD),
        jax.random.randint(k_row, (), BOARD_PAD, h - BOARD_PAD),
    ]).astype(jnp.int32)

    playable = playable_mask(params)
    tiles, start = reverse_walk(k_walk, playable, goal, params.path_length)
    tiles = branch_fill(k_branch, playable, tiles, params.branch_tiles)

    board = jnp.where(tiles, TILE, VOID).astype(jnp.uint8)
    board = board.at[goal[1], goal[0]].set(GOAL)
    return Level(board=board, start=start.astype(jnp.int32))


def generate_levels(key, n: int, params: GenParams = GenParams()) -> Level:
    """``n`` levels at once, as a :class:`Level` with a leading batch axis."""
    gen = jax.jit(jax.vmap(generate_level, in_axes=(0, None)), static_argnums=1)
    return gen(jax.random.split(key, int(n)), params)
