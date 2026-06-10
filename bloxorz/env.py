import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .core import (
    generate_board, move, is_valid, is_win, occupied_cells,
    UPRIGHT, FLAT_H, FLAT_V,
    TILE_SIZE, MARGIN, BOARD_PAD,
    COLOR_BG, COLOR_TEXT,
    COLOR_TILE, COLOR_TILE_BORDER,
    COLOR_GOAL, COLOR_GOAL_BORDER,
    COLOR_GATE_OPEN, COLOR_GATE_OPEN_BORDER,
    COLOR_GATE_CLOSED, COLOR_GATE_CLOSED_BORDER,
    COLOR_SWITCH, COLOR_SWITCH_BORDER,
    COLOR_BLOCK, COLOR_BLOCK_BORDER,
)

ACTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # right, left, down, up

REWARDS = {
    "win":          1.0,
    "fall":        -1.0,
    "gate_open":    0.1,
    "gate_close":  -0.1,
    "step":        -0.01,
}


class BloxorzEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi", "rgb_array"], "render_fps": 60}

    def __init__(self, grid_cols=18, grid_rows=14, gate=True, render_mode=None,
                 path_length=40, branch_tiles=40, max_steps=500):
        super().__init__()
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows
        self.gate = gate
        self.render_mode = render_mode
        self.path_length = path_length
        self.branch_tiles = branch_tiles
        self.max_steps = max_steps

        self._img_w = MARGIN * 2 + (grid_cols + 2 * BOARD_PAD) * TILE_SIZE
        self._img_h = MARGIN * 2 + (grid_rows + 2 * BOARD_PAD) * TILE_SIZE

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(self._img_h, self._img_w, 3),
            dtype=np.uint8,
        )

        self._board: list[list[int]] = [[0] * grid_cols for _ in range(grid_rows)]
        self._col: int = 0
        self._row: int = 0
        self._ori: int = UPRIGHT
        self._switch_cell: tuple[int, int] | None = None
        self._gate_open: bool = False
        self._done: bool = False
        self._steps: int = 0

        self._display_surface = None
        self._display_clock = None

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        board, sc, sr, so, switch_cell, _ = generate_board(
            self.grid_cols, self.grid_rows,
            path_length=self.path_length, branch_tiles=self.branch_tiles,
            rng=self.np_random, gate=self.gate,
        )
        self._board = board
        self._col, self._row, self._ori = sc, sr, so
        self._switch_cell = switch_cell
        self._gate_open = False
        self._done = False
        self._steps = 0

        obs = self._obs()
        if self.render_mode == "human":
            self._render_human(obs)
        return obs, {}

    def step(self, action):
        assert not self._done, "Episode is done; call reset()."

        direction = ACTIONS[action]
        nc, nr, no = move(self._col, self._row, self._ori, direction)

        if not is_valid(nc, nr, no, self._board, self._gate_open):
            reward = REWARDS["fall"]
            terminated = True
        else:
            self._col, self._row, self._ori = nc, nr, no

            gate_reward = 0.0
            if (self._switch_cell and self._ori == UPRIGHT
                    and (self._col, self._row) == self._switch_cell):
                self._gate_open = not self._gate_open
                gate_reward = REWARDS["gate_open"] if self._gate_open else REWARDS["gate_close"]

            if is_win(self._col, self._row, self._ori, self._board):
                reward = REWARDS["win"]
                terminated = True
            else:
                reward = REWARDS["step"] + gate_reward
                terminated = False

        self._steps += 1
        truncated = (not terminated) and (self._steps >= self.max_steps)
        self._done = terminated or truncated

        info = {}
        if self._done:
            info["final_info"] = {"solved": terminated and reward == REWARDS["win"]}

        obs = self._obs()
        if self.render_mode == "human":
            self._render_human(obs)
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self._render_human(self._obs())
        elif self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "rgb_array":
            return self._obs()

    def close(self):
        if self._display_surface is not None:
            import pygame
            pygame.quit()
            self._display_surface = None

    # ------------------------------------------------------------------
    def _init_pygame(self):
        import pygame
        if not pygame.get_init():
            pygame.init()

    _CELL_COLORS = {
        1: (COLOR_TILE, COLOR_TILE_BORDER),
        2: (COLOR_GOAL, COLOR_GOAL_BORDER),
    }

    def _obs(self):
        img = np.full((self._img_h, self._img_w, 3), COLOR_BG, dtype=np.uint8)
        T = TILE_SIZE

        for r, row in enumerate(self._board):
            for c, cell in enumerate(row):
                if cell == 0:
                    continue
                if cell == 3:
                    fill, border = (
                        (COLOR_GATE_OPEN, COLOR_GATE_OPEN_BORDER)
                        if self._gate_open
                        else (COLOR_GATE_CLOSED, COLOR_GATE_CLOSED_BORDER)
                    )
                elif cell == 4:
                    fill, border = (
                        ((80, 220, 80), (40, 160, 40))
                        if self._gate_open
                        else (COLOR_SWITCH, COLOR_SWITCH_BORDER)
                    )
                else:
                    fill, border = self._CELL_COLORS[cell]
                x = MARGIN + c * T
                y = MARGIN + r * T
                img[y+2:y+T-2, x+2:x+T-2] = fill
                img[y+2:y+4,   x+2:x+T-2] = border
                img[y+T-4:y+T-2, x+2:x+T-2] = border
                img[y+2:y+T-2, x+2:x+4]   = border
                img[y+2:y+T-2, x+T-4:x+T-2] = border

        cells = occupied_cells(self._col, self._row, self._ori)
        min_c = min(c for c, _ in cells)
        min_r = min(r for _, r in cells)
        max_c = max(c for c, _ in cells)
        max_r = max(r for _, r in cells)
        pad = 6
        bx = MARGIN + min_c * T + pad
        by = MARGIN + min_r * T + pad
        bw = (max_c - min_c + 1) * T - pad * 2
        bh = (max_r - min_r + 1) * T - pad * 2
        if bw > 0 and bh > 0:
            img[by:by+bh, bx:bx+bw] = COLOR_BLOCK
            img[by:by+3,      bx:bx+bw] = COLOR_BLOCK_BORDER
            img[by+bh-3:by+bh, bx:bx+bw] = COLOR_BLOCK_BORDER
            img[by:by+bh, bx:bx+3]      = COLOR_BLOCK_BORDER
            img[by:by+bh, bx+bw-3:bx+bw] = COLOR_BLOCK_BORDER

        return img

    def _render_human(self, obs: np.ndarray):
        import pygame
        self._init_pygame()

        if self._display_surface is None:
            win_h = self._img_h + 40  # extra row for status text
            self._display_surface = pygame.display.set_mode((self._img_w, win_h))
            pygame.display.set_caption("Bloxorz — RL env")
            self._display_clock = pygame.time.Clock()
            self._font = pygame.font.SysFont(None, 28)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self._display_surface.fill(COLOR_BG)
        obs_surface = pygame.surfarray.make_surface(
            np.transpose(obs, (1, 0, 2))
        )
        self._display_surface.blit(obs_surface, (0, 40))
        hint = self._font.render(
            f"ori={self._ori}  gate={'open' if self._gate_open else 'closed'}",
            True, COLOR_TEXT,
        )
        self._display_surface.blit(hint, (MARGIN, 10))
        pygame.display.flip()
        if self._display_clock is not None:
            self._display_clock.tick(self.metadata["render_fps"])

    def _render_ansi(self):
        CELL = {0: "  ", 1: "░░", 2: "GG", 3: "##", 4: "SW"}
        ORI_MARK = {UPRIGHT: "[]", FLAT_H: "==", FLAT_V: "||"}
        cells = occupied_cells(self._col, self._row, self._ori)
        mark = ORI_MARK[self._ori]
        lines = []
        for r, row in enumerate(self._board):
            line = ""
            for c, cell in enumerate(row):
                if (c, r) in cells:
                    line += mark
                else:
                    line += CELL.get(cell, "??")
            lines.append(line)
        result = "\n".join(lines)
        print(result)
        return result


