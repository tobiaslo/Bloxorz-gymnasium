import pygame
import sys
import argparse
import numpy as np

# --- Constants ---
TILE_SIZE = 8
MARGIN = 0
FPS = 60

COLOR_BG = (30, 30, 30)
COLOR_TILE = (150, 150, 160)
COLOR_TILE_BORDER = (115, 115, 125)
COLOR_GOAL = (60, 160, 80)
COLOR_GOAL_BORDER = (40, 120, 60)
COLOR_BLOCK = (220, 100, 40)
COLOR_BLOCK_BORDER = (160, 60, 20)
COLOR_TEXT = (220, 220, 220)
COLOR_WIN = (80, 220, 120)
COLOR_FAIL = (220, 80, 80)
COLOR_BLOCK_FALL = (180, 60, 60)
COLOR_SWITCH = (220, 200, 40)
COLOR_SWITCH_BORDER = (160, 140, 20)
COLOR_GATE_CLOSED = (140, 40, 40)
COLOR_GATE_CLOSED_BORDER = (100, 20, 20)
COLOR_GATE_OPEN = (80, 160, 200)
COLOR_GATE_OPEN_BORDER = (50, 110, 150)

BOARD_PAD = 1  # void-tile border around the playable area on each side

# Orientations
UPRIGHT = 0
FLAT_H = 1   # occupies (col, row) and (col+1, row)
FLAT_V = 2   # occupies (col, row) and (col, row+1)

def generate_board(grid_cols=14, grid_rows=10, path_length=40, branch_tiles=40,
                   rng=None, gate=True):
    """
    grid_cols/grid_rows are the playable dimensions (excluding the BOARD_PAD border).
    gate=True:  Two-island board with a 2-tile bridge gate.
                Right island (goal) col > mid, left island (player+switch) col < mid-1.
    gate=False: Single-island board — full unconstrained reverse-walk with branch fill.
    Returns (board, start_col, start_row, start_ori, switch_cell, bridge_cell).
    """
    if rng is None:
        rng = np.random.default_rng()

    def _randint(a, b):
        return int(rng.integers(a, b + 1))

    def _choice(lst):
        return lst[int(rng.integers(len(lst)))]

    def _shuffle(lst):
        rng.shuffle(lst)
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    board_cols = grid_cols + 2 * BOARD_PAD
    board_rows = grid_rows + 2 * BOARD_PAD

    def ok(col, row):
        return BOARD_PAD <= col < board_cols - BOARD_PAD and BOARD_PAD <= row < board_rows - BOARD_PAD

    def in_bounds(col, row, o):
        return all(ok(cc, rr) for cc, rr in occupied_cells(col, row, o))

    if not gate:
        # Single unconstrained reverse-walk from goal + branch fill
        gc = _randint(board_cols * 2 // 3, board_cols - BOARD_PAD - 1)
        gr = _randint(BOARD_PAD, board_rows - BOARD_PAD - 1)
        c, r, ori = gc, gr, UPRIGHT
        tiles = set(occupied_cells(c, r, ori))
        for _ in range(path_length):
            _shuffle(dirs)
            for d in dirs:
                nc, nr, no = move(c, r, ori, d)
                if in_bounds(nc, nr, no):
                    tiles.update(occupied_cells(nc, nr, no))
                    c, r, ori = nc, nr, no
                    break
        tile_list = list(tiles)
        for _ in range(branch_tiles):
            bc, br = _choice(tile_list)
            dc, dr = _choice(dirs)
            nc, nr = bc + dc * 2, br + dr * 2
            if ok(nc, nr):
                tiles.add((nc, nr))
                tile_list.append((nc, nr))
        board = [[0] * board_cols for _ in range(board_rows)]
        for tc, tr in tiles:
            board[tr][tc] = 1
        board[gr][gc] = 2
        return board, c, r, ori, None, None

    # --- Gate mode: two-island spatial split ---
    half = path_length // 2
    mid = board_cols // 2

    def in_bounds_right(col, row, o):
        return all(ok(cc, rr) and cc > mid for cc, rr in occupied_cells(col, row, o))

    def in_bounds_left(col, row, o):
        return all(ok(cc, rr) and cc < mid - 1 for cc, rr in occupied_cells(col, row, o))

    gc = _randint(mid + 2, board_cols - BOARD_PAD - 1)
    gr = _randint(BOARD_PAD, board_rows - BOARD_PAD - 1)

    c, r, ori = gc, gr, UPRIGHT
    seg1 = set(occupied_cells(c, r, ori))
    for _ in range(half):
        _shuffle(dirs)
        for d in dirs:
            nc, nr, no = move(c, r, ori, d)
            if in_bounds_right(nc, nr, no):
                seg1.update(occupied_cells(nc, nr, no))
                c, r, ori = nc, nr, no
                break

    for _ in range(board_cols * 2):
        if any(tc == mid + 1 for tc, _ in seg1):
            break
        left_first = sorted(dirs, key=lambda d: d[0])
        for d in left_first:
            nc, nr, no = move(c, r, ori, d)
            if in_bounds_right(nc, nr, no) and nc <= c + 1:
                seg1.update(occupied_cells(nc, nr, no))
                c, r, ori = nc, nr, no
                break

    boundary = [(tc, tr) for tc, tr in seg1 if tc == mid + 1]
    if not boundary:
        min_c = min(tc for tc, _ in seg1)
        boundary = [(tc, tr) for tc, tr in seg1 if tc == min_c]
    _, bridge_row = _choice(boundary)
    bridge_cell = (mid - 1, bridge_row)
    p2_start = (mid - 2, bridge_row)

    c, r, ori = p2_start[0], p2_start[1], UPRIGHT
    seg2 = set(occupied_cells(c, r, ori))
    seg2_path = [(c, r, ori)]
    for _ in range(half):
        _shuffle(dirs)
        for d in dirs:
            nc, nr, no = move(c, r, ori, d)
            if in_bounds_left(nc, nr, no):
                seg2.update(occupied_cells(nc, nr, no))
                c, r, ori = nc, nr, no
                seg2_path.append((c, r, ori))
                break
    start_col, start_row, start_ori = c, r, ori

    switch_cell = None
    mid_idx = max(1, len(seg2_path) // 3)
    for pc, pr, po in seg2_path[mid_idx:]:
        if po == UPRIGHT and (pc, pr) != p2_start:
            switch_cell = (pc, pr)
            break
    if switch_cell is None:
        for pc, pr, po in seg2_path:
            if po == UPRIGHT and (pc, pr) != p2_start:
                switch_cell = (pc, pr)
                break

    all_tiles = seg1 | seg2
    for src, keep_right in [(list(seg1), True), (list(seg2), False)]:
        for _ in range(branch_tiles // 2):
            if not src:
                break
            bc, br = _choice(src)
            dc, dr = _choice(dirs)
            nc, nr = bc + dc * 2, br + dr * 2
            if keep_right:
                valid = ok(nc, nr) and nc > mid
            else:
                valid = ok(nc, nr) and nc < mid - 1
            if valid and (nc, nr) not in ((mid - 1, bridge_row), (mid, bridge_row)):
                all_tiles.add((nc, nr))
                src.append((nc, nr))

    board = [[0] * board_cols for _ in range(board_rows)]
    for tc, tr in all_tiles:
        board[tr][tc] = 1
    board[gr][gc] = 2
    board[bridge_row][mid - 1] = 3
    board[bridge_row][mid] = 3
    if switch_cell:
        board[switch_cell[1]][switch_cell[0]] = 4

    return board, start_col, start_row, start_ori, switch_cell, bridge_cell


def occupied_cells(col, row, ori):
    if ori == UPRIGHT:
        return [(col, row)]
    elif ori == FLAT_H:
        return [(col, row), (col + 1, row)]
    else:  # FLAT_V
        return [(col, row), (col, row + 1)]


def is_valid(col, row, ori, board, gate_open=True):
    rows = len(board)
    cols = len(board[0])
    for c, r in occupied_cells(col, row, ori):
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return False
        cell = board[r][c]
        if cell == 0:
            return False
        if cell == 3 and not gate_open:  # closed gate
            return False
    return True


def move(col, row, ori, direction):
    """Return (new_col, new_row, new_ori) after rolling in direction."""
    dc, dr = direction  # (1,0)=right, (-1,0)=left, (0,1)=down, (0,-1)=up

    if ori == UPRIGHT:
        if dc == 1:   return col + 1, row, FLAT_H
        if dc == -1:  return col - 2, row, FLAT_H
        if dr == 1:   return col, row + 1, FLAT_V
        if dr == -1:  return col, row - 2, FLAT_V

    elif ori == FLAT_H:
        if dc == 1:   return col + 2, row, UPRIGHT
        if dc == -1:  return col - 1, row, UPRIGHT
        if dr == 1:   return col, row + 1, FLAT_H
        if dr == -1:  return col, row - 1, FLAT_H

    else:  # FLAT_V
        if dc == 1:   return col + 1, row, FLAT_V
        if dc == -1:  return col - 1, row, FLAT_V
        if dr == 1:   return col, row + 2, UPRIGHT
        if dr == -1:  return col, row - 1, UPRIGHT

    return col, row, ori


def is_win(col, row, ori, board):
    if ori != UPRIGHT:
        return False
    return board[row][col] == 2


def draw_board(surface, board, offset_x, offset_y, gate_open=True):
    for r, row in enumerate(board):
        for c, cell in enumerate(row):
            if cell == 0:
                continue
            x = offset_x + c * TILE_SIZE
            y = offset_y + r * TILE_SIZE
            if cell == 2:
                color, border = COLOR_GOAL, COLOR_GOAL_BORDER
            elif cell == 3:
                if gate_open:
                    color, border = COLOR_GATE_OPEN, COLOR_GATE_OPEN_BORDER
                else:
                    color, border = COLOR_GATE_CLOSED, COLOR_GATE_CLOSED_BORDER
            elif cell == 4:
                if gate_open:
                    color, border = (80, 220, 80), (40, 160, 40)
                else:
                    color, border = COLOR_SWITCH, COLOR_SWITCH_BORDER
            else:
                color, border = COLOR_TILE, COLOR_TILE_BORDER
            pygame.draw.rect(surface, color, (x + 2, y + 2, TILE_SIZE - 4, TILE_SIZE - 4))
            pygame.draw.rect(surface, border, (x + 2, y + 2, TILE_SIZE - 4, TILE_SIZE - 4), 2)


def draw_block(surface, col, row, ori, offset_x, offset_y, fall_progress=0.0):
    cells = occupied_cells(col, row, ori)
    min_c = min(c for c, _ in cells)
    min_r = min(r for _, r in cells)
    max_c = max(c for c, _ in cells)
    max_r = max(r for _, r in cells)

    shrink = int(fall_progress * (TILE_SIZE // 2 - 6))
    pad = 6 + shrink
    x = offset_x + min_c * TILE_SIZE + pad
    y = offset_y + min_r * TILE_SIZE + pad
    w = (max_c - min_c + 1) * TILE_SIZE - pad * 2
    h = (max_r - min_r + 1) * TILE_SIZE - pad * 2

    if w <= 0 or h <= 0:
        return

    color = COLOR_BLOCK_FALL if fall_progress > 0 else COLOR_BLOCK
    pygame.draw.rect(surface, color, (x, y, w, h), border_radius=6)
    pygame.draw.rect(surface, COLOR_BLOCK_BORDER, (x, y, w, h), 3, border_radius=6)


def parse_args():
    parser = argparse.ArgumentParser(description="Bloxorz — roll the block to the goal.")
    parser.add_argument("--cols", type=int, default=14,
                        help="Playable grid width in tiles (default: 14)")
    parser.add_argument("--rows", type=int, default=10,
                        help="Playable grid height in tiles (default: 10)")
    parser.add_argument("--no-gate", action="store_true",
                        help="Disable the gate/bridge mechanic")
    return parser.parse_args()


def new_level(grid_cols, grid_rows, gate):
    board, sc, sr, so, switch_cell, _ = generate_board(grid_cols, grid_rows, gate=gate)
    return board, sc, sr, so, switch_cell


def main():
    args = parse_args()
    grid_cols, grid_rows = args.cols, args.rows
    gate = not args.no_gate

    pygame.init()

    win_w = MARGIN * 2 + (grid_cols + 2 * BOARD_PAD) * TILE_SIZE
    win_h = MARGIN * 2 + (grid_rows + 2 * BOARD_PAD) * TILE_SIZE + 40

    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("Bloxorz")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 28)
    font_big = pygame.font.SysFont(None, 52)

    offset_x = MARGIN
    offset_y = MARGIN + 40

    board, start_col, start_row, start_ori, switch_cell = new_level(grid_cols, grid_rows, gate)
    col, row, ori = start_col, start_row, start_ori
    won = False
    fallen = False
    fall_progress = 0.0
    gate_open = False

    FALL_SPEED = 0.05

    key_to_dir = {
        pygame.K_RIGHT: (1, 0),
        pygame.K_LEFT:  (-1, 0),
        pygame.K_DOWN:  (0, 1),
        pygame.K_UP:    (0, -1),
    }

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_r:
                    col, row, ori = start_col, start_row, start_ori
                    won = False
                    fallen = False
                    fall_progress = 0.0
                    gate_open = False
                if event.key == pygame.K_n:
                    board, start_col, start_row, start_ori, switch_cell = new_level(grid_cols, grid_rows, gate)
                    col, row, ori = start_col, start_row, start_ori
                    won = False
                    fallen = False
                    fall_progress = 0.0
                    gate_open = False
                if not won and not fallen and event.key in key_to_dir:
                    d = key_to_dir[event.key]
                    nc, nr, no = move(col, row, ori, d)
                    if is_valid(nc, nr, no, board, gate_open):
                        col, row, ori = nc, nr, no
                        if switch_cell and ori == UPRIGHT and (col, row) == switch_cell:
                            gate_open = not gate_open
                        if is_win(col, row, ori, board):
                            won = True
                    else:
                        fallen = True
                        fall_progress = 0.0

        if fallen and fall_progress < 1.0:
            fall_progress = min(fall_progress + FALL_SPEED, 1.0)

        screen.fill(COLOR_BG)
        draw_board(screen, board, offset_x, offset_y, gate_open)
        draw_block(screen, col, row, ori, offset_x, offset_y, fall_progress if fallen else 0.0)

        hint = font.render("Arrows: move   R: reset   N: new board   Q: quit", True, COLOR_TEXT)
        screen.blit(hint, (MARGIN, 10))

        if won:
            msg = font_big.render("You Win!  N for next", True, COLOR_WIN)
            screen.blit(msg, (win_w // 2 - msg.get_width() // 2, win_h // 2 - msg.get_height() // 2))
        elif fallen and fall_progress >= 1.0:
            msg = font_big.render("You Fell!  R to retry", True, COLOR_FAIL)
            screen.blit(msg, (win_w // 2 - msg.get_width() // 2, win_h // 2 - msg.get_height() // 2))

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
