# src/run_all.py
"""
One-shot runner:
1) Trains warm-start reward models for seeds (default: 0-4) using noisy + soft labels.
2) Trains the 4 policy conditions per seed:
   - standard (true reward)
   - ours_relabel (warm-start + online)
   - ours_norelabel (warm-start + online)
   - random

This produces the folders eval.py expects under src/results/.

Usage:
  python3 src/run_all.py
  python3 src/run_all.py --seeds 0 1 2 3 4 --episodes 2000 --eval_after

Notes:
- This script calls your existing train_reward.py and train.py via subprocess to avoid import/path headaches.
- Warm-start checkpoints are expected to live at:
    results/reward_noisy_soft_seed{SEED}/reward_net.pth
  We enforce that by passing --exp_name reward_noisy_soft_seed{SEED} to train_reward.py.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List

THIS_DIR = Path(__file__).resolve().parent
PY = sys.executable  # current python


def run(cmd: List[str], cwd: Path) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--episodes", type=int, default=2000)

    # Reward pretrain defaults (match your train_reward.py defaults, but force noisy_soft)
    p.add_argument("--total_queries", type=int, default=500)
    p.add_argument("--init_queries", type=int, default=100)
    p.add_argument("--acquire_per_round", type=int, default=40)
    p.add_argument("--round_epochs", type=int, default=50)
    p.add_argument("--final_epochs", type=int, default=50)
    p.add_argument("--random_steps", type=int, default=20_000)
    p.add_argument("--pref_margin", type=float, default=0.10)
    p.add_argument("--beta", type=float, default=0.20)
    p.add_argument("--flip_prob", type=float, default=0.10)
    p.add_argument("--label_smooth", type=float, default=0.05)

    # Optional: after training, run eval.py over all experiment folders
    p.add_argument("--eval_after", action="store_true")
    p.add_argument("--eval_episodes", type=int, default=500)

    args = p.parse_args()

    # Ensure we run from repo root (parent of src/)
    repo_root = THIS_DIR.parent
    src_dir = repo_root / "src"
    if not (src_dir / "train.py").exists():
        raise FileNotFoundError(f"Expected src/train.py at {src_dir}/train.py")

    # --- 1) Train warm-start reward model (ONLY SEED 0) ---
    # We train ONE reward model on seed 0, and use it as the starting point for ALL seeds.
    reward_seed = 0
    exp_name = f"reward_noisy_soft_seed{reward_seed}"
    # This ensures checkpoint path: src/results/reward_noisy_soft_seed0/reward_net.pth
    run(
        [
            PY, "src/train_reward.py",
            "--seed", str(reward_seed),
            "--pref_mode", "noisy",
            "--noisy_soft",
            "--exp_name", exp_name,
            "--total_queries", str(args.total_queries),
            "--init_queries", str(args.init_queries),
            "--acquire_per_round", str(args.acquire_per_round),
            "--round_epochs", str(args.round_epochs),
            "--final_epochs", str(args.final_epochs),
            "--random_steps", str(args.random_steps),
            "--pref_margin", str(args.pref_margin),
            "--beta", str(args.beta),
            "--flip_prob", str(args.flip_prob),
            "--label_smooth", str(args.label_smooth),
        ],
        cwd=repo_root,
    )

    ckpt = src_dir / "results" / exp_name / "reward_net.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"Reward checkpoint not found after training: {ckpt}")

    # --- 2) Train the 4 policy conditions per seed ---
    # All seeds (0..4) use the SAME reward checkpoint from seed 0 as initialization.
    reward_init = f"results/reward_noisy_soft_seed{reward_seed}/reward_net.pth"

    for seed in args.seeds:
        # (i) Standard DQN
        run(
            [PY, "src/train.py", "--algo", "standard", "--seed", str(seed), "--episodes", str(args.episodes)],
            cwd=repo_root,
        )

        # (ii) Warm-start + online + relabel
        run(
            [
                PY, "src/train.py",
                "--algo", "ours_relabel",
                "--pref_mode", "noisy",
                "--seed", str(seed),
                "--episodes", str(args.episodes),
                "--reward_init_path", reward_init,
            ],
            cwd=repo_root,
        )

        # (iii) Warm-start + online + no-relabel
        run(
            [
                PY, "src/train.py",
                "--algo", "ours_norelabel",
                "--pref_mode", "noisy",
                "--seed", str(seed),
                "--episodes", str(args.episodes),
                "--reward_init_path", reward_init,
            ],
            cwd=repo_root,
        )

        # (iv) Random
        run(
            [PY, "src/train.py", "--algo", "random", "--seed", str(seed), "--episodes", str(args.episodes)],
            cwd=repo_root,
        )

    # --- 3) Optional: run eval.py across everything we just created ---
    if args.eval_after:
        run(
            [
                PY, "src/eval.py",
                "--results_root", "src/results",
                "--match", "*_seed*",
                "--eval_episodes", str(args.eval_episodes),
            ],
            cwd=repo_root,
        )

    print("\nAll done. Results are under: src/results/")


if __name__ == "__main__":
    main()
