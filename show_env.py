"""Render a single static frame of the Bloxorz environment with matplotlib."""

import argparse

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import bloxorz  # noqa: F401  (registers the envs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="Bloxorz-NoGate-v0",
                        help="Registered env ID (e.g. Bloxorz-v0, Bloxorz-NoGate-v0)")
    parser.add_argument("--seed", type=int, default=0, help="Reset seed for the board")
    parser.add_argument("--save", default=None, help="Optional path to save the image instead of showing")
    args = parser.parse_args()

    env = gym.make(args.env_id, render_mode="rgb_array")
    obs, _ = env.reset(seed=args.seed)
    env.close()

    print(obs.shape)

    obs = np.asarray(Image.fromarray(obs).resize((64, 64), Image.NEAREST))
    print(obs.shape)

    plt.imshow(obs)
    plt.axis("off")
    plt.title(f"{args.env_id} (seed={args.seed})")

    if args.save:
        plt.savefig(args.save, bbox_inches="tight", dpi=150)
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
