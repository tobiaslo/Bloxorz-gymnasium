"""Block geometry and the roll table, as array lookups.

The table is built from :func:`bloxorz.core.move` rather than restated, so
``core.py`` stays the single source of truth for how the block rolls.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..core import FLAT_H, FLAT_V, UPRIGHT, move

# Same order as bloxorz.env.ACTIONS: 0=right, 1=left, 2=down, 3=up. Restated
# rather than imported, because bloxorz.env pulls in Gymnasium.
ACTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))
ACTION_NAMES = ("right", "left", "down", "up")

# Board cell codes, as in core.py.
VOID, TILE, GOAL, GATE, SWITCH = 0, 1, 2, 3, 4

#: ``MOVES[ori, action] = (dcol, drow, new_ori)``: rolling from the origin
#: gives the position delta directly.
MOVES = jnp.asarray(
    np.array([[move(0, 0, ori, d) for d in ACTIONS] for ori in (UPRIGHT, FLAT_H, FLAT_V)],
             dtype=np.int32)
)

#: Unit steps in the order of ``ACTIONS``, as ``(dcol, drow)``.
DIRS = jnp.asarray(ACTIONS, jnp.int32)


def roll(block: jnp.ndarray, action) -> jnp.ndarray:
    """The ``(col, row, ori)`` reached by rolling ``block`` in ``action``'s direction."""
    d = MOVES[block[2], action]
    return jnp.stack([block[0] + d[0], block[1] + d[1], d[2]])


def second_cell(block: jnp.ndarray):
    """The ``(col, row)`` of the block's second cell (its only cell if upright)."""
    col, row, ori = block[0], block[1], block[2]
    return col + (ori == FLAT_H), row + (ori == FLAT_V)


def fits(mask: jnp.ndarray, block: jnp.ndarray) -> jnp.ndarray:
    """Whether every cell under ``block`` is inside ``mask`` (bool ``[H, W]``)."""
    h, w = mask.shape
    col, row = block[0], block[1]
    col2, row2 = second_cell(block)
    # The second cell is never above or left of the first, so these four
    # comparisons bound both cells.
    inside = (col >= 0) & (row >= 0) & (col2 < w) & (row2 < h)
    # Out-of-bounds gathers clamp, which is harmless once `inside` gates them.
    return inside & mask[row, col] & mask[row2, col2]


def stamp(mask: jnp.ndarray, block: jnp.ndarray) -> jnp.ndarray:
    """``mask`` with the cells under ``block`` set. ``block`` must be in bounds."""
    col2, row2 = second_cell(block)
    return mask.at[block[1], block[0]].set(True).at[row2, col2].set(True)
