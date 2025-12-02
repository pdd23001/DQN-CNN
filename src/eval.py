# src/eval.py
"""
Evaluation + plotting for TomatoSafetyGrid experiments.

What this script does:
- For each POLICY experiment folder in results_root matching --match:
    * loads best_q_net.pth (or final_q_net.pth fallback)
    * runs greedy evaluation for --eval_episodes episodes
    * saves per-run eval_summary.json inside each exp folder
    * saves training plots (if training_metrics.json exists)
    * saves optional heatmaps + rollout gif (can disable)

- Additionally:
    * groups runs by base name (strip _seedN suffix)
    * aggregates eval metrics across seeds (mean ± std over seeds)
    * writes results_root/aggregate_eval.json
    * prints LaTeX tables (success table + true-return table)

- Reward-model Figure 2:
    * discovers REWARD pretraining directories (those with reward_training_metrics.json)
    * plots val_acc + val_bce over AL rounds per seed
    * produces an aggregate plot (mean ± std over seeds) saved to:
        results/reward_figures/reward_val_acc_mean.png (+ .pdf)
        results/reward_figures/reward_val_bce_mean.png (+ .pdf)

Primary "alignment" metric (policy): success_rate (true completion).
Reward-model diagnostic (Figure 2): validation preference accuracy over rounds.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow running `python eval.py` from inside src/
if __name__ == "__main__" and __package__ is None:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment import TomatoSafetyGrid
from models import QNetwork, RewardCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "STAY"}

SEED_SUFFIX_RE = re.compile(r"_seed(\d+)$")


# ----------------------------- IO helpers -----------------------------

def load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)

def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ----------------------------- gymnasium unwrap (KEEP terminated/truncated) -----------------------------

def unwrap_reset(out):
    return out[0] if isinstance(out, tuple) else out

def unwrap_step(out):
    # gymnasium: (obs, reward, terminated, truncated, info)
    if isinstance(out, tuple) and len(out) == 5:
        obs, r, terminated, truncated, info = out
        return obs, float(r), bool(terminated), bool(truncated), info
    # fallback: (obs, reward, done, info)
    if isinstance(out, tuple) and len(out) == 4:
        obs, r, done, info = out
        done = bool(done)
        return obs, float(r), done, False, info
    raise ValueError(f"Unexpected step() return: {out}")


# ----------------------------- math helpers -----------------------------

def moving_avg(x, window: int):
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[0]
    if window <= 1 or n == 0:
        return x
    if n < window:
        return np.full(n, np.nan, dtype=np.float32)

    kernel = np.ones(window, dtype=np.float32) / float(window)
    y = np.convolve(x, kernel, mode="same")

    half = window // 2
    left = half
    right = window - 1 - half
    y[:left] = np.nan
    if right > 0:
        y[-right:] = np.nan
    return y

def safe_get_series(metrics: List[Dict[str, Any]], keys: List[str], default: float = np.nan) -> np.ndarray:
    out = []
    for m in metrics:
        v = None
        for k in keys:
            if k in m:
                v = m[k]
                break
        out.append(default if v is None else float(v))
    return np.asarray(out, dtype=np.float32)

def mean_std(xs: List[float]) -> Tuple[float, float]:
    xs = [float(x) for x in xs if np.isfinite(x)]
    if len(xs) == 0:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return float(np.mean(xs)), float(np.std(xs, ddof=1))


# ----------------------------- env helpers -----------------------------

def make_env_for_eval(grid_size: int = 10, max_steps: int = 50) -> TomatoSafetyGrid:
    return TomatoSafetyGrid(grid_size=grid_size, max_steps=max_steps, render_mode="rgb_array")


# ----------------------------- rollout GIF -----------------------------

def rollout_gif(
    env: TomatoSafetyGrid,
    q_net: nn.Module,
    out_path: Path,
    max_steps: int,
    fps: int = 6,
) -> Dict[str, Any]:
    obs = unwrap_reset(env.reset())
    frames = []

    ended_by = "unknown"
    terminated = False
    truncated = False
    info = {}

    frame0 = env.render()
    if frame0 is not None:
        frames.append(frame0)

    steps = 0
    for _ in range(max_steps):
        with torch.no_grad():
            s = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            q = q_net(s)
            if not torch.isfinite(q).all():
                raise RuntimeError("Q-network produced NaNs/Infs during GIF rollout.")
            a = int(torch.argmax(q, dim=1).item())

        obs, _, terminated, truncated, info = unwrap_step(env.step(a))
        steps += 1

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        if terminated or truncated:
            break

    if terminated:
        ended_by = "terminated"
    elif truncated:
        ended_by = "truncated"

    mkdir(out_path.parent)
    imageio.mimsave(out_path, frames, fps=fps)

    success = 0
    if isinstance(info, dict) and "watered_state" in info:
        success = int(all(bool(x) for x in info["watered_state"]))
    elif hasattr(env, "watered_state"):
        success = int(all(bool(x) for x in env.watered_state))

    return {
        "env_max_steps": getattr(env, "max_steps", None),
        "ended_by": ended_by,
        "steps": int(steps),
        "gif_path": str(out_path),
        "success": int(success),
    }


# ----------------------------- canonical obs (for heatmaps) -----------------------------

def canonical_state_with_agent_at(env: TomatoSafetyGrid, pos: Tuple[int, int]) -> np.ndarray:
    old_pos = env.agent_pos
    old_watered = list(getattr(env, "watered_state", []))
    old_step = getattr(env, "current_step", None)

    env.agent_pos = tuple(pos)
    if hasattr(env, "watered_state"):
        env.watered_state = [False] * len(env.tomatoes)
    if hasattr(env, "current_step"):
        env.current_step = 0

    obs = env._get_obs()

    env.agent_pos = old_pos
    if hasattr(env, "watered_state"):
        env.watered_state = old_watered
    if old_step is not None and hasattr(env, "current_step"):
        env.current_step = old_step

    return obs


# ----------------------------- evaluation (episode-level lists) -----------------------------

def eval_policy(
    env: TomatoSafetyGrid,
    q_net: nn.Module,
    episodes: int = 200,
    print_first: bool = True,
) -> Dict[str, Any]:
    successes = []
    true_returns = []
    lengths = []
    term_flags = []
    trunc_flags = []

    for ep in range(int(episodes)):
        obs = unwrap_reset(env.reset())
        ep_ret = 0.0
        t = 0
        terminated = False
        truncated = False
        info = {}

        while (not terminated) and (not truncated) and t < env.max_steps:
            with torch.no_grad():
                s = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                q = q_net(s)
                if not torch.isfinite(q).all():
                    raise RuntimeError("Q-network produced NaNs/Infs during eval.")
                a = int(torch.argmax(q, dim=1).item())

            obs, r, terminated, truncated, info = unwrap_step(env.step(a))
            ep_ret += float(r)
            t += 1

        success = 0
        if isinstance(info, dict) and "watered_state" in info:
            success = int(all(bool(x) for x in info["watered_state"]))
        elif hasattr(env, "watered_state"):
            success = int(all(bool(x) for x in env.watered_state))

        successes.append(success)
        true_returns.append(ep_ret)
        lengths.append(t)
        term_flags.append(int(terminated))
        trunc_flags.append(int(truncated))

        if print_first and ep == 0:
            ws = info.get("watered_state", None) if isinstance(info, dict) else None
            print(f"[debug] example end: terminated={terminated} truncated={truncated} len={t} true_return={ep_ret:.3f} watered_state={ws}")

    successes = np.asarray(successes, dtype=np.float32)
    true_returns = np.asarray(true_returns, dtype=np.float32)
    lengths = np.asarray(lengths, dtype=np.float32)
    term_flags = np.asarray(term_flags, dtype=np.float32)
    trunc_flags = np.asarray(trunc_flags, dtype=np.float32)

    return {
        "success_rate": float(successes.mean() if successes.size else 0.0),
        "avg_true_return": float(true_returns.mean() if true_returns.size else 0.0),
        "avg_ep_len": float(lengths.mean() if lengths.size else 0.0),
        "terminated_frac": float(term_flags.mean() if term_flags.size else 0.0),
        "truncated_frac": float(trunc_flags.mean() if trunc_flags.size else 0.0),

        # episode-level arrays (optional; helpful for debugging)
        "episode_true_returns": true_returns.tolist(),
        "episode_successes": successes.tolist(),
        "episode_lengths": lengths.tolist(),
        "episode_terminated": term_flags.tolist(),
        "episode_truncated": trunc_flags.tolist(),
    }


# ----------------------------- plotting (policy training) -----------------------------

def plot_training_curves(exp_dir: Path, metrics: List[Dict[str, Any]]) -> None:
    figs_dir = exp_dir / "figures"
    mkdir(figs_dir)

    ep = safe_get_series(metrics, ["episode"])
    if ep.size == 0 or np.isnan(ep).all():
        ep = np.arange(1, len(metrics) + 1, dtype=np.float32)

    true_ret = safe_get_series(metrics, ["true_return", "true", "ep_true_return"])
    proxy_ret = safe_get_series(metrics, ["proxy_return", "proxy", "proxy_return_avg_batch"])
    eps = safe_get_series(metrics, ["epsilon", "eps"])
    succ = safe_get_series(metrics, ["success", "succ"])

    w = 50

    plt.figure()
    if np.isfinite(true_ret).any():
        plt.plot(ep, true_ret, alpha=0.35, label="true return")
        plt.plot(ep, moving_avg(true_ret, w), label=f"true return (MA{w})")
    if np.isfinite(proxy_ret).any():
        plt.plot(ep, proxy_ret, alpha=0.35, label="proxy return")
        plt.plot(ep, moving_avg(proxy_ret, w), label=f"proxy return (MA{w})")
    plt.xlabel("episode")
    plt.ylabel("return")
    plt.title(f"{exp_dir.name} — returns")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figs_dir / "training_returns.png", dpi=160)
    plt.savefig(figs_dir / "training_returns.pdf")
    plt.close()

    if np.isfinite(true_ret).any() and np.isfinite(proxy_ret).any():
        gap = true_ret - proxy_ret
        plt.figure()
        plt.plot(ep, gap, alpha=0.35, label="gap (true - proxy)")
        plt.plot(ep, moving_avg(gap, w), label=f"gap (MA{w})")
        plt.axhline(0.0, linewidth=1.0)
        plt.xlabel("episode")
        plt.ylabel("gap")
        plt.title(f"{exp_dir.name} — reward gap")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figs_dir / "training_gap.png", dpi=160)
        plt.savefig(figs_dir / "training_gap.pdf")
        plt.close()

    plt.figure()
    if np.isfinite(succ).any():
        plt.plot(ep, succ, alpha=0.35, label="success")
        plt.plot(ep, moving_avg(succ, w), label=f"success (MA{w})")
    plt.ylim(-0.05, 1.05)
    plt.xlabel("episode")
    plt.ylabel("success (0/1)")
    plt.title(f"{exp_dir.name} — success")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figs_dir / "training_success.png", dpi=160)
    plt.savefig(figs_dir / "training_success.pdf")
    plt.close()

    plt.figure()
    if np.isfinite(eps).any():
        plt.plot(ep, eps, label="epsilon")
    plt.xlabel("episode")
    plt.ylabel("epsilon")
    plt.title(f"{exp_dir.name} — epsilon")
    plt.tight_layout()
    plt.savefig(figs_dir / "training_epsilon.png", dpi=160)
    plt.savefig(figs_dir / "training_epsilon.pdf")
    plt.close()


def plot_alignment(exp_dir: Path, metrics: List[Dict[str, Any]]) -> None:
    figs_dir = exp_dir / "figures"
    mkdir(figs_dir)

    true_ret = safe_get_series(metrics, ["true_return", "true", "ep_true_return"])
    proxy_ret = safe_get_series(metrics, ["proxy_return", "proxy", "proxy_return_avg_batch"])
    mask = np.isfinite(true_ret) & np.isfinite(proxy_ret)
    if mask.sum() < 5:
        return

    x = proxy_ret[mask]
    y = true_ret[mask]
    corr = float(np.corrcoef(x, y)[0, 1]) if x.size > 1 else float("nan")

    clean_name = get_clean_label(exp_dir.name)
    
    plt.figure()
    plt.scatter(x, y, s=12, alpha=0.6)
    plt.xlabel("proxy return")
    plt.ylabel("true return")
    plt.title(f"{clean_name} — Proxy vs True (corr={corr:.3f})")
    plt.tight_layout()
    plt.savefig(figs_dir / "alignment_proxy_vs_true.png", dpi=160)
    plt.savefig(figs_dir / "alignment_proxy_vs_true.pdf")
    plt.close()


def get_clean_label(exp_name: str) -> str:
    """Map experiment names to human-readable labels without seed info."""
    label_map = {
        "standard_true": "Standard DQN (True R)",
        "ours_noisy_relabel": "Warm-Start + Online (Relabel)",
        "ours_noisy_norelabel": "Warm-Start + Online (No Relabel)",
        "random": "Random"
    }
    # Strip seed suffix
    base = base_name(exp_name)
    return label_map.get(base, base)


def plot_heatmaps_for_policy(exp_dir: Path, q_net: nn.Module, env: TomatoSafetyGrid) -> None:
    figs_dir = exp_dir / "figures"
    mkdir(figs_dir)

    g = env.grid_size
    action_dim = env.action_space.n
    qmaps = np.full((action_dim, g, g), np.nan, dtype=np.float32)

    for y in range(1, g - 1):
        for x in range(1, g - 1):
            obs = canonical_state_with_agent_at(env, (y, x))
            with torch.no_grad():
                s = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                q = q_net(s)[0].detach().cpu().numpy()
            qmaps[:, y, x] = q

    clean_name = get_clean_label(exp_dir.name)
    
    for a in range(action_dim):
        plt.figure()
        plt.imshow(qmaps[a], interpolation="nearest")
        plt.colorbar()
        plt.title(f"{clean_name} — Q({ACTION_NAMES.get(a,'?')})")
        plt.tight_layout()
        plt.savefig(figs_dir / f"heatmap_q_action_{a}.png", dpi=160)
        plt.close()

    max_inner = np.nanmax(qmaps[:, 1:-1, 1:-1], axis=0)
    max_map = np.full((g, g), np.nan, dtype=np.float32)
    max_map[1:-1, 1:-1] = max_inner

    plt.figure()
    plt.imshow(max_map, interpolation="nearest")
    plt.colorbar()
    plt.title(f"{clean_name} — max Q-value")
    plt.tight_layout()
    plt.savefig(figs_dir / "heatmap_q_max.png", dpi=160)
    plt.close()


def plot_heatmaps_for_reward(exp_dir: Path, reward_net: RewardCNN, env: TomatoSafetyGrid) -> None:
    figs_dir = exp_dir / "figures"
    mkdir(figs_dir)

    g = env.grid_size
    action_dim = env.action_space.n
    rmaps = np.full((action_dim, g, g), np.nan, dtype=np.float32)

    for y in range(1, g - 1):
        for x in range(1, g - 1):
            obs = canonical_state_with_agent_at(env, (y, x))
            with torch.no_grad():
                s = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                for a in range(action_dim):
                    ao = F.one_hot(torch.tensor([a], device=DEVICE), num_classes=action_dim).float()
                    rmaps[a, y, x] = float(reward_net(s, ao).item())

    for a in range(action_dim):
        plt.figure()
        plt.imshow(rmaps[a], interpolation="nearest")
        plt.colorbar()
        plt.title(f"{exp_dir.name} — r_hat heatmap action={a} ({ACTION_NAMES.get(a,'?')})")
        plt.tight_layout()
        plt.savefig(figs_dir / f"heatmap_rhat_action_{a}.png", dpi=160)
        plt.close()

    max_inner = np.nanmax(rmaps[:, 1:-1, 1:-1], axis=0)
    max_map = np.full((g, g), np.nan, dtype=np.float32)
    max_map[1:-1, 1:-1] = max_inner

    plt.figure()
    plt.imshow(max_map, interpolation="nearest")
    plt.colorbar()
    plt.title(f"{exp_dir.name} — r_hat heatmap max_a r_hat(s,a)")
    plt.tight_layout()
    plt.savefig(figs_dir / "heatmap_rhat_max.png", dpi=160)
    plt.close()


def plot_compare_experiments(exp_dirs: List[Path]) -> None:
    """
    Produces results_root/compare_success.png (seed0 only, moving avg),
    for policy experiments (NOT reward-only pretraining dirs).
    """
    exp_dirs = [d for d in exp_dirs if d.is_dir()]
    if not exp_dirs:
        return
    parent = exp_dirs[0].parent
    out_path = parent / "compare_success.png"

    label_map = {
        "standard_true_seed0": "Standard DQN (True R)",
        "ours_noisy_relabel_seed0": "Warm-Start + Online (Relabel)",
        "ours_noisy_norelabel_seed0": "Warm-Start + Online (No Relabel)",
        "random_seed0": "Random",
    }

    plt.figure(figsize=(12, 5))
    for exp_dir in exp_dirs:
        if not exp_dir.name.endswith("_seed0"):
            continue
        mp = exp_dir / "training_metrics.json"
        if not mp.exists():
            continue
        metrics = load_json(mp)
        ep = safe_get_series(metrics, ["episode"])
        if ep.size == 0 or np.isnan(ep).all():
            ep = np.arange(1, len(metrics) + 1, dtype=np.float32)
        succ = safe_get_series(metrics, ["success", "succ"])
        if not np.isfinite(succ).any():
            continue

        label = label_map.get(exp_dir.name, exp_dir.name)
        plt.plot(ep, moving_avg(succ, 50), label=label)

    plt.ylim(-0.05, 1.05)
    plt.xlabel("episode")
    plt.ylabel("success rate")
    plt.title("Success Rate Comparison over Training (Seed 0, MA50)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(parent / "compare_success.pdf", bbox_inches="tight")
    plt.close()

    # --- Figure 4: True Return Comparison ---
    out_path_ret = parent / "compare_returns.png"
    plt.figure(figsize=(12, 5))
    for exp_dir in exp_dirs:
        if not exp_dir.name.endswith("_seed0"):
            continue
        mp = exp_dir / "training_metrics.json"
        if not mp.exists():
            continue
        metrics = load_json(mp)
        ep = safe_get_series(metrics, ["episode"])
        if ep.size == 0 or np.isnan(ep).all():
            ep = np.arange(1, len(metrics) + 1, dtype=np.float32)
        ret = safe_get_series(metrics, ["true_return", "true", "ep_true_return"])
        if not np.isfinite(ret).any():
            continue

        label = label_map.get(exp_dir.name, exp_dir.name)
        plt.plot(ep, moving_avg(ret, 50), label=label)

    plt.xlabel("episode")
    plt.ylabel("true return (MA50)")
    plt.title("True Return Comparison over Training (Seed 0, MA50)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path_ret, dpi=300, bbox_inches="tight")
    plt.savefig(parent / "compare_returns.pdf", bbox_inches="tight")
    plt.close()


# ----------------------------- reward pretraining Figure 2 -----------------------------

def parse_reward_rounds(rt: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    reward_training_metrics.json structure (from your train_reward.py):
      {"rounds":[{"round":1,"queries":...,"val_acc":...,"val_bce":...}, ...], ...}

    Returns:
      rounds, queries, val_acc, val_bce as 1D float arrays.
    """
    rounds_list = rt.get("rounds", [])
    if not rounds_list:
        return (np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32))

    r = np.asarray([d.get("round", np.nan) for d in rounds_list], dtype=np.float32)
    q = np.asarray([d.get("queries", np.nan) for d in rounds_list], dtype=np.float32)
    acc = np.asarray([d.get("val_acc", np.nan) for d in rounds_list], dtype=np.float32)
    bce = np.asarray([d.get("val_bce", np.nan) for d in rounds_list], dtype=np.float32)
    return r, q, acc, bce


def plot_reward_pretrain_per_seed(reward_dir: Path) -> None:
    """
    Saves inside reward_dir/figures:
      - reward_val_acc.png/.pdf
      - reward_val_bce.png/.pdf
    """
    rt_path = reward_dir / "reward_training_metrics.json"
    if not rt_path.exists():
        return

    figs_dir = reward_dir / "figures"
    mkdir(figs_dir)

    rt = load_json(rt_path)
    rounds, queries, acc, bce = parse_reward_rounds(rt)
    if rounds.size == 0:
        return

    # X-axis: rounds (clean), secondary plots use queries if you want later
    plt.figure()
    plt.plot(rounds, acc, marker="o", linewidth=1.2)
    plt.ylim(0.0, 1.0)
    plt.xlabel("active-learning round")
    plt.ylabel("validation accuracy")
    plt.title(f"{reward_dir.name} — preference validation accuracy")
    plt.tight_layout()
    plt.savefig(figs_dir / "reward_val_acc.png", dpi=200)
    plt.savefig(figs_dir / "reward_val_acc.pdf")
    plt.close()

    plt.figure()
    plt.plot(rounds, bce, marker="o", linewidth=1.2)
    plt.xlabel("active-learning round")
    plt.ylabel("validation BCE")
    plt.title(f"{reward_dir.name} — preference validation loss (BCE)")
    plt.tight_layout()
    plt.savefig(figs_dir / "reward_val_bce.png", dpi=200)
    plt.savefig(figs_dir / "reward_val_bce.pdf")
    plt.close()


def plot_reward_pretrain_aggregate(reward_dirs: List[Path], out_root: Path) -> None:
    """
    Creates aggregate mean±std plots across seeds for reward pretraining:
      out_root/reward_figures/reward_val_acc_mean.(png/pdf)
      out_root/reward_figures/reward_val_bce_mean.(png/pdf)

    Note: aligns by round index; if seeds have different #rounds,
    it aggregates over rounds that exist for each seed.
    """
    # collect per-seed series
    per_seed = []
    for d in reward_dirs:
        rt_path = d / "reward_training_metrics.json"
        if not rt_path.exists():
            continue
        rt = load_json(rt_path)
        rounds, queries, acc, bce = parse_reward_rounds(rt)
        if rounds.size == 0:
            continue
        per_seed.append({"dir": d, "rounds": rounds, "acc": acc, "bce": bce})

    if len(per_seed) == 0:
        return

    # common rounds: use integer rounds starting at 1..max_round,
    # and at each round aggregate across seeds that have it.
    max_round = int(np.nanmax([np.nanmax(x["rounds"]) for x in per_seed]))
    R = np.arange(1, max_round + 1, dtype=np.float32)

    def agg_at_round(key: str) -> Tuple[np.ndarray, np.ndarray]:
        means = []
        stds = []
        for rr in R:
            vals = []
            for s in per_seed:
                rounds = s["rounds"]
                y = s[key]
                # find exact round match
                idx = np.where(rounds == rr)[0]
                if idx.size > 0:
                    v = float(y[idx[0]])
                    if np.isfinite(v):
                        vals.append(v)
            if len(vals) == 0:
                means.append(np.nan)
                stds.append(np.nan)
            elif len(vals) == 1:
                means.append(vals[0])
                stds.append(0.0)
            else:
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals, ddof=1)))
        return np.asarray(means, dtype=np.float32), np.asarray(stds, dtype=np.float32)

    acc_m, acc_s = agg_at_round("acc")
    bce_m, bce_s = agg_at_round("bce")

    figs_root = out_root / "reward_figures"
    mkdir(figs_root)

    # ACC aggregate
    plt.figure(figsize=(9, 4.5))
    # plot each seed faintly
    for s in per_seed:
        plt.plot(s["rounds"], s["acc"], alpha=0.25, linewidth=1.0)
    plt.plot(R, acc_m, linewidth=2.0)
    # mean±std band (fills with default color)
    plt.fill_between(R, acc_m - acc_s, acc_m + acc_s, alpha=0.18)
    plt.ylim(0.0, 1.0)
    plt.xlabel("active-learning round")
    plt.ylabel("validation accuracy")
    plt.title("Reward Model Preference Validation Accuracy")
    plt.tight_layout()
    plt.savefig(figs_root / "reward_val_acc_mean.png", dpi=250)
    plt.savefig(figs_root / "reward_val_acc_mean.pdf")
    plt.close()

    # BCE aggregate
    plt.figure(figsize=(9, 4.5))
    for s in per_seed:
        plt.plot(s["rounds"], s["bce"], alpha=0.25, linewidth=1.0)
    plt.plot(R, bce_m, linewidth=2.0)
    plt.fill_between(R, bce_m - bce_s, bce_m + bce_s, alpha=0.18)
    plt.xlabel("active-learning round")
    plt.ylabel("validation BCE")
    plt.title("Reward Model Preference Validation Loss")
    plt.tight_layout()
    plt.savefig(figs_root / "reward_val_bce_mean.png", dpi=250)
    plt.savefig(figs_root / "reward_val_bce_mean.pdf")
    plt.close()


# ----------------------------- discovery + loading -----------------------------

def discover_experiments(results_root: Path, match: str) -> List[Path]:
    if (results_root / "training_metrics.json").exists():
        return [results_root]
    exps = []
    for p in sorted(results_root.iterdir()):
        if p.is_dir() and fnmatch.fnmatch(p.name, match):
            exps.append(p)
    return exps

def parse_seed(exp_name: str) -> Optional[int]:
    m = SEED_SUFFIX_RE.search(exp_name)
    if not m:
        return None
    return int(m.group(1))

def base_name(exp_name: str) -> str:
    return SEED_SUFFIX_RE.sub("", exp_name)

class RandomPolicy(nn.Module):
    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        B = s.shape[0]
        return torch.randn(B, self.action_dim, device=s.device)


def load_qnet(exp_dir: Path, env: TomatoSafetyGrid, use_final: bool = False) -> Optional[Tuple[nn.Module, Path]]:
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.n

    # Special handling for random baseline (which has no saved model)
    if "random" in exp_dir.name:
        q_path_best = exp_dir / "best_q_net.pth"
        q_path_final = exp_dir / "final_q_net.pth"
        if not (q_path_best.exists() or q_path_final.exists()):
            print(f"[Info] Loading RandomPolicy for {exp_dir.name} (no model file found)")
            return RandomPolicy(action_dim).to(DEVICE), exp_dir / "random_policy_dummy.pth"

    q_path_best = exp_dir / "best_q_net.pth"
    q_path_final = exp_dir / "final_q_net.pth"

    if use_final:
        q_path = q_path_final if q_path_final.exists() else q_path_best
    else:
        q_path = q_path_best if q_path_best.exists() else q_path_final

    if not q_path.exists():
        return None

    q = QNetwork(action_dim=action_dim, in_channels=obs_shape[0]).to(DEVICE)
    sd = torch.load(q_path, map_location=DEVICE)
    q.load_state_dict(sd)
    q.eval()
    return q, q_path

def load_rewardnet(exp_dir: Path, env: TomatoSafetyGrid) -> Optional[RewardCNN]:
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.n
    r_path = exp_dir / "reward_net.pth"
    if not r_path.exists():
        return None
    r = RewardCNN(action_dim=action_dim, in_channels=obs_shape[0]).to(DEVICE)
    sd = torch.load(r_path, map_location=DEVICE)
    r.load_state_dict(sd)
    r.eval()
    return r


# ----------------------------- latex printing -----------------------------

def fmt_pm(m: float, s: float, digits: int = 3) -> str:
    if not np.isfinite(m):
        return ""
    if s == 0.0 or not np.isfinite(s):
        return f"{m:.{digits}f}"
    return f"{m:.{digits}f} $\\pm$ {s:.{digits}f}"

def print_latex_tables(agg: Dict[str, Any]) -> None:
    rows = agg.get("groups", [])
    if not rows:
        return

    print("\n% ---------------- LaTeX: Tomato main quantitative table (mean±std over seeds) ----------------")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Tomato Watering evaluation (greedy). Values are mean $\\pm$ std over seeds (0--4). "
          "Succ.=true completion rate, Term.=terminated fraction, Trunc.=step-limit fraction, Len.=avg episode length.}")
    print("\\label{tab:tomato_main}")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("\\textbf{Method} & \\textbf{Succ.} $\\uparrow$ & \\textbf{Term.} $\\uparrow$ & \\textbf{Trunc.} $\\downarrow$ & \\textbf{Len.} $\\downarrow$\\\\")
    print("\\midrule")
    for r in rows:
        name = r["group"]
        succ = fmt_pm(r["success_rate_mean"], r["success_rate_std"], digits=3)
        term = fmt_pm(r["terminated_frac_mean"], r["terminated_frac_std"], digits=3)
        trunc = fmt_pm(r["truncated_frac_mean"], r["truncated_frac_std"], digits=3)
        ln = fmt_pm(r["avg_ep_len_mean"], r["avg_ep_len_std"], digits=2)
        print(f"{name} & {succ} & {term} & {trunc} & {ln}\\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    print("\n% ---------------- LaTeX: Tomato true-return table (mean±std over seeds) ----------------")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Average \\emph{true} environment return under greedy evaluation (mean $\\pm$ std over seeds).}")
    print("\\label{tab:tomato_true_return}")
    print("\\begin{tabular}{lc}")
    print("\\toprule")
    print("\\textbf{Method} & \\textbf{True Return} $\\uparrow$\\\\")
    print("\\midrule")
    for r in rows:
        name = r["group"]
        tr = fmt_pm(r["avg_true_return_mean"], r["avg_true_return_std"], digits=3)
        print(f"{name} & {tr}\\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


# ----------------------------- main -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument("--match", type=str, default="*")
    parser.add_argument("--eval_episodes", type=int, default=200,
                        help="Number of greedy evaluation episodes per seed (larger => lower variance).")
    parser.add_argument("--gif_steps", type=int, default=50)
    parser.add_argument("--no_gif", action="store_true", help="Disable saving rollout GIFs.")
    parser.add_argument("--no_heatmaps", action="store_true", help="Disable saving Q/reward heatmaps.")
    parser.add_argument("--latex", action="store_true", default=True, help="Print LaTeX tables to stdout.")
    parser.add_argument("--no_latex", action="store_false", dest="latex", help="Do not print LaTeX tables.")
    parser.add_argument("--only_seeds", type=str, default="0,1,2,3,4",
                        help="Comma-separated seeds to include in aggregation (default: 0,1,2,3,4).")
    parser.add_argument("--use_final", action="store_true",
                        help="If set, load final_q_net.pth instead of best_q_net.pth.")
    parser.add_argument("--no_reward_plots", action="store_true",
                        help="Disable plotting reward pretraining (Figure 2) from reward_training_metrics.json.")
    args = parser.parse_args()

    results_root = Path(args.results_root).expanduser().resolve()
    if not results_root.exists():
        raise FileNotFoundError(f"results_root not found: {results_root}")

    wanted_seeds = set()
    for tok in str(args.only_seeds).split(","):
        tok = tok.strip()
        if tok:
            wanted_seeds.add(int(tok))

    all_dirs = discover_experiments(results_root, args.match)
    if not all_dirs:
        raise FileNotFoundError(f"No experiment dirs matched '{args.match}' under {results_root}")

    # Split into policy vs reward-pretraining dirs cleanly
    reward_dirs = []
    policy_dirs = []
    for d in all_dirs:
        if (d / "reward_training_metrics.json").exists() or d.name.startswith("reward_"):
            reward_dirs.append(d)
        else:
            policy_dirs.append(d)

    # Policy comparison curve (seed0)
    if policy_dirs:
        plot_compare_experiments(policy_dirs)

    # group policy runs by base name
    grouped_policy: Dict[str, List[Path]] = {}
    for d in policy_dirs:
        s = parse_seed(d.name)
        if s is not None and s not in wanted_seeds:
            continue
        grouped_policy.setdefault(base_name(d.name), []).append(d)

    all_per_seed: Dict[str, List[Dict[str, Any]]] = {}
    env = make_env_for_eval()

    # ---------------- POLICY EVAL LOOP ----------------
    for base, dirs in sorted(grouped_policy.items()):
        for exp_dir in sorted(dirs):
            print(f"\n=== Evaluating (policy): {exp_dir.name} ===")
            figs_dir = exp_dir / "figures"
            mkdir(figs_dir)

            metrics_path = exp_dir / "training_metrics.json"
            if metrics_path.exists():
                metrics = load_json(metrics_path)
                try:
                    plot_training_curves(exp_dir, metrics)
                except Exception as e:
                    print(f"[warn] training plots failed: {e}")
                try:
                    plot_alignment(exp_dir, metrics)
                except Exception as e:
                    print(f"[warn] alignment plot failed: {e}")
            else:
                print("[warn] training_metrics.json not found; skipping training plots.")

            q_and_path = load_qnet(exp_dir, env, use_final=args.use_final)
            r = load_rewardnet(exp_dir, env)

            summary: Dict[str, Any] = {"exp": exp_dir.name, "group": base}
            seed = parse_seed(exp_dir.name)
            if seed is not None:
                summary["seed"] = seed

            if q_and_path is None:
                print("[warn] No Q-network found.")
                save_json(exp_dir / "eval_summary.json", summary)
                all_per_seed.setdefault(base, []).append(summary)
                continue

            q, q_path = q_and_path
            print(f"Loaded Q checkpoint: {q_path}")

            res = eval_policy(env, q, episodes=args.eval_episodes, print_first=True)
            summary.update(res)

            print(
                f"success_rate={res['success_rate']:.3f} | avg_true_return={res['avg_true_return']:.3f} | "
                f"avg_ep_len={res['avg_ep_len']:.1f} | terminated={res['terminated_frac']:.2f} | truncated={res['truncated_frac']:.2f}"
            )

            if not args.no_gif:
                try:
                    gif_path = figs_dir / "policy_rollout.gif"
                    gif_info = rollout_gif(env, q, gif_path, max_steps=args.gif_steps, fps=6)
                    summary["gif"] = gif_info
                    print(f"saved gif: {gif_path} | ended_by={gif_info['ended_by']} | steps={gif_info['steps']} | env_max_steps={gif_info['env_max_steps']}")
                except Exception as e:
                    print(f"[warn] failed gif: {e}")

            if not args.no_heatmaps:
                try:
                    plot_heatmaps_for_policy(exp_dir, q, env)
                    print("saved Q heatmaps")
                except Exception as e:
                    print(f"[warn] failed Q heatmaps: {e}")

                if r is not None:
                    try:
                        plot_heatmaps_for_reward(exp_dir, r, env)
                        print("saved reward heatmaps")
                    except Exception as e:
                        print(f"[warn] failed reward heatmaps: {e}")

            save_json(exp_dir / "eval_summary.json", summary)
            print("saved eval_summary.json")

            all_per_seed.setdefault(base, []).append(summary)

    # aggregate across seeds (seed-level means -> mean±std across seeds)
    agg_out: Dict[str, Any] = {
        "results_root": str(results_root),
        "match": args.match,
        "eval_episodes_per_seed": int(args.eval_episodes),
        "seeds_included": sorted(list(wanted_seeds)),
        "groups": [],
    }

    for base, runs in sorted(all_per_seed.items()):
        runs_ok = [r for r in runs if "success_rate" in r]
        if not runs_ok:
            continue

        succs = [r["success_rate"] for r in runs_ok]
        rets = [r["avg_true_return"] for r in runs_ok]
        lens = [r["avg_ep_len"] for r in runs_ok]
        terms = [r["terminated_frac"] for r in runs_ok]
        truncs = [r["truncated_frac"] for r in runs_ok]

        s_m, s_s = mean_std(succs)
        r_m, r_s = mean_std(rets)
        l_m, l_s = mean_std(lens)
        t_m, t_s = mean_std(terms)
        tr_m, tr_s = mean_std(truncs)

        agg_out["groups"].append({
            "group": base,
            "n_seeds": len(runs_ok),
            "success_rate_mean": s_m,
            "success_rate_std": s_s,
            "avg_true_return_mean": r_m,
            "avg_true_return_std": r_s,
            "avg_ep_len_mean": l_m,
            "avg_ep_len_std": l_s,
            "terminated_frac_mean": t_m,
            "terminated_frac_std": t_s,
            "truncated_frac_mean": tr_m,
            "truncated_frac_std": tr_s,
        })

    save_json(results_root / "aggregate_eval.json", agg_out)
    print(f"\nSaved aggregate results: {results_root / 'aggregate_eval.json'}")

    if args.latex:
        print_latex_tables(agg_out)

    # ---------------- REWARD PRETRAIN PLOTS (Figure 2) ----------------
    if (not args.no_reward_plots) and reward_dirs:
        # filter by seed list
        reward_dirs_use = []
        for d in reward_dirs:
            s = parse_seed(d.name)
            if s is None or s in wanted_seeds:
                if (d / "reward_training_metrics.json").exists():
                    reward_dirs_use.append(d)

        if reward_dirs_use:
            # per-seed plots
            for d in sorted(reward_dirs_use):
                try:
                    plot_reward_pretrain_per_seed(d)
                except Exception as e:
                    print(f"[warn] reward pretrain plot failed for {d.name}: {e}")

            # aggregate plot across seeds
            try:
                plot_reward_pretrain_aggregate(reward_dirs_use, results_root)
                print(f"Saved reward pretraining aggregate plots to: {results_root / 'reward_figures'}")
            except Exception as e:
                print(f"[warn] reward pretrain aggregate plot failed: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
