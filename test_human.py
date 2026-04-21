"""Human-playable test harness for BloxorzEnv.

Controls:
  Arrow keys  move the block
  R           reset (same board)
  N           new board
  Q / Escape  quit
"""

import numpy as np
import pygame
import gymnasium as gym
import bloxorz  # noqa: F401 — registers envs
from bloxorz import COLOR_BG, COLOR_TEXT, COLOR_WIN, COLOR_FAIL

KEY_TO_ACTION = {
    pygame.K_RIGHT: 0,
    pygame.K_LEFT:  1,
    pygame.K_DOWN:  2,
    pygame.K_UP:    3,
}

STATUS_H = 40  # pixels reserved above the board image for status text


def obs_to_surface(obs: np.ndarray) -> pygame.Surface:
    # obs is (H, W, 3); pygame.surfarray wants (W, H, 3)
    return pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))


def main():
    env = gym.make("Bloxorz-v0", render_mode=None)
    obs, _ = env.reset()

    img_h, img_w = obs.shape[:2]

    pygame.init()
    screen = pygame.display.set_mode((img_w, img_h + STATUS_H))
    pygame.display.set_caption("Bloxorz — Human Test")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 28)
    font_big = pygame.font.SysFont(None, 52)

    won = False
    fallen = False
    steps = 0
    total_reward = 0.0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

                elif event.key == pygame.K_r:
                    obs, _ = env.reset()
                    won = fallen = False
                    steps = 0
                    total_reward = 0.0

                elif event.key == pygame.K_n:
                    obs, _ = env.reset(seed=None)
                    won = fallen = False
                    steps = 0
                    total_reward = 0.0

                elif event.key in KEY_TO_ACTION and not won and not fallen:
                    action = KEY_TO_ACTION[event.key]
                    obs, reward, terminated, _, _ = env.step(action)
                    steps += 1
                    total_reward += float(reward)
                    if terminated:
                        won = reward > 0
                        fallen = not won

        # --- draw ---
        screen.fill(COLOR_BG)
        screen.blit(obs_to_surface(obs), (0, STATUS_H))

        # status bar
        inner = env.unwrapped
        gate_str = "open" if inner._gate_open else "closed"
        ori_names = {0: "upright", 1: "flat-H", 2: "flat-V"}
        status = font.render(
            f"steps: {steps}   reward: {total_reward:.2f}   ori: {ori_names[inner._ori]}   gate: {gate_str}"
            "   |   R: reset   N: new   Q: quit",
            True, COLOR_TEXT,
        )
        screen.blit(status, (10, 10))

        if won:
            msg = font_big.render("You Win!  N for next", True, COLOR_WIN)
            screen.blit(msg, (img_w // 2 - msg.get_width() // 2,
                               STATUS_H + img_h // 2 - msg.get_height() // 2))
        elif fallen:
            msg = font_big.render("You Fell!  R to retry", True, COLOR_FAIL)
            screen.blit(msg, (img_w // 2 - msg.get_width() // 2,
                               STATUS_H + img_h // 2 - msg.get_height() // 2))

        pygame.display.flip()
        clock.tick(60)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
