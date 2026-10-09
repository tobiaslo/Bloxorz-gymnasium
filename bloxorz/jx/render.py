"""RGB rendering in JAX, matching ``BloxorzEnv._obs`` pixel for pixel.

Board tiles are one gather from a per-code tileset; the block is a rectangle
mask over pixel coordinates, so rendering stays inside jit/vmap. Colors and
geometry come from ``core.py`` -- keep this in sync with ``BloxorzEnv._obs``
and ``core.draw_board``/``core.draw_block``.
"""

from __future__ import annotations

import functools

import jax.numpy as jnp
import numpy as np

from ..core import (
    COLOR_BG, COLOR_BLOCK, COLOR_BLOCK_BORDER, COLOR_GATE_CLOSED, COLOR_GOAL,
    COLOR_SWITCH, COLOR_TILE, FLAT_H, FLAT_V, MARGIN, TILE_SIZE,
)
from .rules import GATE, GOAL, SWITCH, TILE, VOID

BLOCK_PAD = 1  # pixels between the block and its cells' edges
BLOCK_BORDER = 3  # border width in pixels

# The gate is always closed and the switch never pressed, since the gate
# mechanic is not ported yet; the codes are rendered for completeness.
_FILL = {TILE: COLOR_TILE, GOAL: COLOR_GOAL, GATE: COLOR_GATE_CLOSED, SWITCH: COLOR_SWITCH}


@functools.lru_cache(maxsize=8)
def make_tileset(tile_size: int = TILE_SIZE) -> np.ndarray:
    """``uint8[5, T, T, 3]``: each cell code's tile, a 1px background rim around a fill."""
    t = tile_size
    tiles = np.broadcast_to(np.array(COLOR_BG, np.uint8), (5, t, t, 3)).copy()
    for code, fill in _FILL.items():
        tiles[code, 1:t - 1, 1:t - 1] = fill
    assert (tiles[VOID] == COLOR_BG).all()
    return tiles


def image_shape(board_shape, tile_size: int = TILE_SIZE):
    h, w = board_shape
    return (MARGIN * 2 + h * tile_size, MARGIN * 2 + w * tile_size, 3)


# The NumPy tileset is cached, never its device array: a jnp array built while
# tracing is a tracer, and caching one would leak it into later traces.
def render_rgb(board: jnp.ndarray, block: jnp.ndarray, tile_size: int = TILE_SIZE) -> jnp.ndarray:
    """Render one board and block to ``uint8[H_px, W_px, 3]``."""
    t = tile_size
    h, w = board.shape
    img = jnp.asarray(make_tileset(t))[board]  # [H, W, T, T, 3]
    img = img.transpose(0, 2, 1, 3, 4).reshape(h * t, w * t, 3)
    if MARGIN:
        canvas = jnp.broadcast_to(jnp.asarray(COLOR_BG, jnp.uint8), image_shape(board.shape, t))
        img = canvas.at[MARGIN:MARGIN + h * t, MARGIN:MARGIN + w * t].set(img)

    col, row, ori = block[0], block[1], block[2]
    x0 = MARGIN + col * t + BLOCK_PAD
    y0 = MARGIN + row * t + BLOCK_PAD
    x1 = MARGIN + (col + 1 + (ori == FLAT_H)) * t - BLOCK_PAD
    y1 = MARGIN + (row + 1 + (ori == FLAT_V)) * t - BLOCK_PAD

    ys = jnp.arange(img.shape[0])[:, None]
    xs = jnp.arange(img.shape[1])[None, :]
    body = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
    b = BLOCK_BORDER
    inner = (xs >= x0 + b) & (xs < x1 - b) & (ys >= y0 + b) & (ys < y1 - b)
    color = jnp.where(
        inner[..., None],
        jnp.asarray(COLOR_BLOCK, jnp.uint8),
        jnp.asarray(COLOR_BLOCK_BORDER, jnp.uint8),
    )
    return jnp.where(body[..., None], color, img)


def to_ansi(board: np.ndarray, block) -> str:
    """Text rendering, matching ``BloxorzEnv._render_ansi``."""
    cell_mark = {VOID: "  ", TILE: "░░", GOAL: "GG", GATE: "##", SWITCH: "SW"}
    ori_mark = {0: "[]", FLAT_H: "==", FLAT_V: "||"}
    col, row, ori = (int(x) for x in block)
    covered = {(col, row), (col + (ori == FLAT_H), row + (ori == FLAT_V))}
    return "\n".join(
        "".join(ori_mark[ori] if (c, r) in covered else cell_mark.get(int(v), "??")
                for c, v in enumerate(line))
        for r, line in enumerate(np.asarray(board))
    )


def save_gif(frames, path, duration: int = 150):
    """Write a sequence of RGB frames to an animated GIF (needs Pillow)."""
    from PIL import Image

    images = [Image.fromarray(np.asarray(f, np.uint8)) for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=duration, loop=0)
    return path
