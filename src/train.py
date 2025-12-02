# src/train.py
"""
Train one experiment at a time.

Supported algos:
  - standard        : DQN trained on TRUE reward (oracle RL upper bound)
  - ours_relabel    : RewardCNN from preference queries + DQN on proxy reward + replay RELABELING
  - ours_norelabel  : same but NO relabeling (ablation)
  - random          : random policy baseline (no training)

Preference labels (ours_* only):
  - oracle : deterministic from true returns
  - noisy  : simulated human (Bradley-Terry sampling + flips) via utils.label_noisy()

Reward initialization:
  - Initialize reward ensemble from pretrained reward_net.pth via --reward_init_path.
  - Either keep learning reward online (default) or freeze reward via --freeze_reward.

Proxy reward stabilization (recommended):
  - Bradley–Terry preference training only cares about relative reward differences, so r_hat(s,a)
    can drift by a large constant offset and/or scale. Summing over 50 steps can yield huge proxy returns
    (e.g., -81) even though the policy is fine.
  - We fix this by estimating proxy stats (mean/std) on replay and transforming:
        r_proxy = (r_hat - mean) / std   (optional)
    plus optional clipping.

Outputs:
  results/<exp_name>/
    best_q_net.pth
    final_q_net.pth
    reward_net.pth (ours_* only if it exists/used)
    training_metrics.json
"""

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from environment import TomatoSafetyGrid
from models import QNetwork, RewardCNN
from utils import (
    DEVICE,
    ReplayBuffer,
    EpsilonScheduler,
    preference_batch_loss,
    sample_preference_queries,
    select_uncertain_queries,
    set_seed,
    save_json,
)

# ---------------- defaults (paper-ish) ----------------

NUM_EPISODES = 2000
MAX_STEPS_PER_EP = 50
GAMMA = 0.99

BUFFER_CAPACITY = 50_000
BATCH_SIZE = 64

LR_DQN = 1e-3
LR_REWARD = 1e-4

TARGET_UPDATE_EVERY = 1000  # grad steps
BURN_IN_STEPS = 2000

SEG_LEN = 8
REWARD_UPDATE_EVERY_STEPS = 2000
REWARD_TRAIN_EPOCHS = 3

# active learning
ENSEMBLE_SIZE = 3
CANDIDATE_MULTIPLIER = 10
PREF_QUERIES_PER_UPDATE = 50

# epsilon
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 30_000


# ---------------- gymnasium-safe unwrap ----------------

def unwrap_reset(out):
    # gymnasium reset -> (obs, info)
    return out[0] if isinstance(out, tuple) else out


def unwrap_step(out):
    # gymnasium step -> (obs, reward, terminated, truncated, info)
    if isinstance(out, tuple) and len(out) == 5:
        obs, r, terminated, truncated, info = out
        return obs, float(r), bool(terminated), bool(truncated), info
    # fallback legacy -> (obs, reward, done, info)
    if isinstance(out, tuple) and len(out) == 4:
        obs, r, done, info = out
        return obs, float(r), bool(done), False, info
    raise ValueError(f"Unexpected step() return: {out}")


# ---------------- helpers ----------------

def make_env(seed: Optional[int] = None):
    return TomatoSafetyGrid(grid_size=10, max_steps=MAX_STEPS_PER_EP, seed=seed)


def pack_preference_batch(batch):
    """
    batch: list of (seg1, seg2, mu)
    returns numpy arrays:
      seg1_s: [B,L,C,H,W]
      seg1_a: [B,L]
      seg2_s: [B,L,C,H,W]
      seg2_a: [B,L]
      mu:     [B]
    """
    seg1_s = np.stack([[t.s for t in seg1] for (seg1, _, _) in batch], axis=0)
    seg1_a = np.stack([[t.a for t in seg1] for (seg1, _, _) in batch], axis=0)
    seg2_s = np.stack([[t.s for t in seg2] for (_, seg2, _) in batch], axis=0)
    seg2_a = np.stack([[t.a for t in seg2] for (_, seg2, _) in batch], axis=0)
    mu = np.array([mu for (_, _, mu) in batch], dtype=np.float32)
    return seg1_s, seg1_a, seg2_s, seg2_a, mu


def train_reward_epoch(
    model: RewardCNN,
    opt: optim.Optimizer,
    pref_dataset: List,
    action_dim: int,
    batch_size: int,
) -> float:
    """
    Trains on preference dataset. If dataset is smaller than batch_size, trains on it anyway
    (your earlier 0.0000 losses often came from "return early" behavior).
    """
    if len(pref_dataset) == 0:
        return 0.0

    model.train()
    losses = []
    np.random.shuffle(pref_dataset)

    effective_bs = min(int(batch_size), len(pref_dataset))
    for i in range(0, len(pref_dataset), effective_bs):
        batch = pref_dataset[i:i + effective_bs]
        seg1_s, seg1_a, seg2_s, seg2_a, mu = pack_preference_batch(batch)

        s1 = torch.tensor(seg1_s, dtype=torch.float32, device=DEVICE)
        a1 = torch.tensor(seg1_a, dtype=torch.long, device=DEVICE)
        s2 = torch.tensor(seg2_s, dtype=torch.float32, device=DEVICE)
        a2 = torch.tensor(seg2_a, dtype=torch.long, device=DEVICE)
        mu_t = torch.tensor(mu, dtype=torch.float32, device=DEVICE)

        loss = preference_batch_loss(model, s1, a1, s2, a2, mu_t, action_dim)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else 0.0


def dqn_update(q, tq, q_opt, batch, gamma: float) -> float:
    s = torch.tensor(batch["states"], dtype=torch.float32, device=DEVICE)
    a = torch.tensor(batch["actions"], dtype=torch.long, device=DEVICE).unsqueeze(1)
    r = torch.tensor(batch["rewards"], dtype=torch.float32, device=DEVICE).unsqueeze(1)
    ns = torch.tensor(batch["next_states"], dtype=torch.float32, device=DEVICE)
    d = torch.tensor(batch["dones"], dtype=torch.float32, device=DEVICE).unsqueeze(1)

    # Double DQN target
    q_sa = q(s).gather(1, a)
    with torch.no_grad():
        next_a = torch.argmax(q(ns), dim=1, keepdim=True)  # online selects
        next_q = tq(ns).gather(1, next_a)                  # target evaluates
        target = r + gamma * (1.0 - d) * next_q

    loss = F.smooth_l1_loss(q_sa, target)
    q_opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q.parameters(), 1.0)
    q_opt.step()
    return float(loss.item())


def resolve_ckpt_path(results_root: str, p: str) -> str:
    p = os.path.expanduser(p)
    if os.path.isabs(p):
        return p
    cand1 = os.path.join(results_root, p)
    if os.path.exists(cand1):
        return cand1
    return p


# ---------------- Reward / Proxy utilities ----------------

def build_rhat_raw_batch(reward_models, action_dim: int):
    """
    Returns a function raw_rhat_batch(states_np, actions_np) -> np.ndarray [N]
    where raw rhat is ensemble mean of RewardCNN outputs.
    """
    def raw_rhat_batch(states_np: np.ndarray, actions_np: np.ndarray) -> np.ndarray:
        s = torch.tensor(states_np, dtype=torch.float32, device=DEVICE)
        a = torch.tensor(actions_np, dtype=torch.long, device=DEVICE)
        aoh = F.one_hot(a, num_classes=action_dim).float()
        with torch.no_grad():
            vals = []
            for m in reward_models:
                vals.append(m(s, aoh))  # [N]
            stacked = torch.stack(vals, dim=0)  # [E,N]
            mean = stacked.mean(dim=0)
        return mean.detach().cpu().numpy().astype(np.float32)

    return raw_rhat_batch


def build_rhat_raw_single(raw_rhat_batch_fn, action_dim: int):
    """Wrap batch fn to single transition raw r_hat(s,a) -> float."""
    def raw_single(s_np: np.ndarray, a_int: int) -> float:
        out = raw_rhat_batch_fn(np.asarray(s_np, dtype=np.float32)[None, ...], np.asarray([a_int], dtype=np.int64))
        return float(out[0])
    return raw_single


@dataclass
class ProxyStats:
    mean: float = 0.0
    std: float = 1.0
    rmin: float = 0.0
    rmax: float = 0.0


class ProxyTransform:
    def __init__(self, center: bool, normalize: bool, clip: Optional[float], eps: float = 1e-6):
        self.center = bool(center)
        self.normalize = bool(normalize)
        self.clip = None if clip is None else float(clip)
        self.eps = float(eps)
        self.stats = ProxyStats()

    def update(self, mean: float, std: float, rmin: float, rmax: float):
        std = float(std)
        if not np.isfinite(std) or std < self.eps:
            std = 1.0
        self.stats = ProxyStats(mean=float(mean), std=std, rmin=float(rmin), rmax=float(rmax))

    def __call__(self, raw_r: float) -> float:
        r = float(raw_r)
        if self.center:
            r -= self.stats.mean
        if self.normalize:
            r /= max(self.eps, self.stats.std)
        if self.clip is not None and self.clip > 0:
            r = float(np.clip(r, -self.clip, self.clip))
        return r


def estimate_proxy_stats_from_replay(
    replay: ReplayBuffer,
    raw_rhat_batch_fn,
    sample_n: int,
) -> ProxyStats:
    """
    Estimate mean/std/min/max of raw r_hat over a sample from replay.
    """
    n = min(int(sample_n), len(replay))
    if n <= 0:
        return ProxyStats(mean=0.0, std=1.0, rmin=0.0, rmax=0.0)

    batch = replay.sample(n)
    states = batch["states"]       # [N,C,H,W]
    actions = batch["actions"]     # [N]
    raw = raw_rhat_batch_fn(states, actions)  # [N]
    raw = np.asarray(raw, dtype=np.float32)

    mean = float(np.mean(raw))
    std = float(np.std(raw))
    rmin = float(np.min(raw))
    rmax = float(np.max(raw))
    return ProxyStats(mean=mean, std=std, rmin=rmin, rmax=rmax)


# ---------------- main ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["standard", "ours_relabel", "ours_norelabel", "random"], default="ours_relabel")
    parser.add_argument("--pref_mode", choices=["oracle", "noisy"], default="noisy",
                        help="How preferences are labeled (only used for ours_*)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results_root", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)

    # reward init + freeze/online
    parser.add_argument(
        "--reward_init_path",
        type=str,
        default=None,
        help="Path to pretrained reward_net.pth (e.g. results/reward_noisy_soft_seed0/reward_net.pth)."
    )
    parser.add_argument(
        "--ensemble_noise_std",
        type=float,
        default=0.0,
        help="Optional Gaussian noise applied after loading to diversify ensemble (try 0.001 to 0.01)."
    )
    parser.add_argument(
        "--freeze_reward",
        action="store_true",
        help="If set: do NOT update reward online (no new preferences, no reward training)."
    )

    # proxy stabilization (addresses crazy proxy returns like -81 over 50 steps)
    parser.add_argument("--proxy_center", action="store_true", default=False,
                        help="Center proxy reward by subtracting mean(raw r_hat) estimated from replay (default: OFF).")
    parser.add_argument("--no_proxy_center", action="store_false", dest="proxy_center",
                        help="Disable proxy centering.")
    parser.add_argument("--proxy_norm", action="store_true", default=False,
                        help="Normalize proxy reward by dividing by std(raw r_hat) (default: OFF).")
    parser.add_argument("--proxy_clip", type=float, default=0.0,
                        help="If >0, clip transformed proxy reward to [-clip, clip]. Example: --proxy_clip 1.0")
    parser.add_argument("--proxy_stats_n", type=int, default=5000,
                        help="How many replay samples to use when estimating proxy mean/std/min/max.")
    args = parser.parse_args()

    set_seed(args.seed)
    env = make_env(seed=args.seed)
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.n

    # ---------- stable exp names for table ----------
    if args.algo == "standard":
        exp_name = f"standard_true_seed{args.seed}"
    elif args.algo == "ours_relabel":
        exp_name = f"ours_{args.pref_mode}_relabel_seed{args.seed}"
    elif args.algo == "ours_norelabel":
        exp_name = f"ours_{args.pref_mode}_norelabel_seed{args.seed}"
    else:
        exp_name = f"random_seed{args.seed}"

    this_dir = os.path.dirname(os.path.abspath(__file__))
    results_root = args.results_root or os.path.join(this_dir, "results")
    exp_dir = os.path.join(results_root, exp_name)
    fig_dir = os.path.join(exp_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # ---------- random baseline (no training) ----------
    if args.algo == "random":
        metrics: List[Dict] = []
        for ep in range(1, args.episodes + 1):
            obs = unwrap_reset(env.reset())
            ep_true = 0.0
            ep_len = 0
            terminated = False
            truncated = False
            info = {}
            while (not terminated) and (not truncated) and ep_len < MAX_STEPS_PER_EP:
                a = env.action_space.sample()
                obs, r, terminated, truncated, info = unwrap_step(env.step(a))
                ep_true += r
                ep_len += 1

            success = 0
            if isinstance(info, dict) and "watered_state" in info:
                success = int(all(bool(x) for x in info["watered_state"]))

            metrics.append({
                "episode": ep,
                "true_return": float(ep_true),
                "proxy_return": float(ep_true),
                "gap": 0.0,
                "success": int(success),
                "epsilon": 1.0,
                "terminated": int(terminated),
                "truncated": int(truncated),
                "ep_len": int(ep_len),
            })

            if ep % 20 == 0 or ep == 1:
                print(f"Ep {ep:4d}/{args.episodes} | True {ep_true:6.2f} | Succ {success}")

        save_json(os.path.join(exp_dir, "training_metrics.json"), metrics)
        print(f"\nSaved random baseline metrics to {exp_dir}")
        return

    # ---------- replay ----------
    replay = ReplayBuffer(capacity=BUFFER_CAPACITY, obs_shape=obs_shape)

    # ---------- Q networks ----------
    q = QNetwork(action_dim=action_dim, in_channels=obs_shape[0]).to(DEVICE)
    tq = QNetwork(action_dim=action_dim, in_channels=obs_shape[0]).to(DEVICE)
    tq.load_state_dict(q.state_dict())
    tq.eval()
    q_opt = optim.Adam(q.parameters(), lr=LR_DQN)

    # ---------- reward models (ours_*) ----------
    reward_models = None
    reward_optims = None
    pref_dataset: List = []
    reward_ckpt_loaded = False

    raw_rhat_batch = None
    raw_rhat_single = None
    proxy_tx = ProxyTransform(
        center=args.proxy_center,
        normalize=args.proxy_norm,
        clip=(None if args.proxy_clip <= 0 else args.proxy_clip),
    )

    if args.algo != "standard":
        reward_models = [RewardCNN(action_dim=action_dim, in_channels=obs_shape[0]).to(DEVICE) for _ in range(ENSEMBLE_SIZE)]
        reward_optims = [optim.Adam(m.parameters(), lr=LR_REWARD) for m in reward_models]

        raw_rhat_batch = build_rhat_raw_batch(reward_models, action_dim=action_dim)
        raw_rhat_single = build_rhat_raw_single(raw_rhat_batch, action_dim=action_dim)

        if args.reward_init_path is not None:
            ckpt = resolve_ckpt_path(results_root, args.reward_init_path)
            if os.path.exists(ckpt):
                sd = torch.load(ckpt, map_location=DEVICE)
                for m in reward_models:
                    m.load_state_dict(sd)
                    if args.ensemble_noise_std > 0:
                        with torch.no_grad():
                            for p in m.parameters():
                                p.add_(args.ensemble_noise_std * torch.randn_like(p))
                reward_ckpt_loaded = True
                print(f"[Reward] initialized ensemble from: {ckpt}")
            else:
                print(f"[warn] reward_init_path not found: {ckpt} (will learn reward online)")

    eps_sched = EpsilonScheduler(EPS_START, EPS_END, EPS_DECAY_STEPS)

    # ---------- burn-in ----------
    # IMPORTANT: store TD-done = terminated only (bootstrap through truncation)
    ep_id = 0
    obs = unwrap_reset(env.reset())

    for _ in range(BURN_IN_STEPS):
        a = env.action_space.sample()
        next_obs, true_r, terminated, truncated, info = unwrap_step(env.step(a))
        done_any = terminated or truncated
        done_td = terminated

        if args.algo == "standard":
            proxy_r = true_r
        else:
            if reward_ckpt_loaded:
                raw = float(raw_rhat_single(obs, a))
                proxy_r = float(proxy_tx(raw))
            else:
                proxy_r = 0.0

        replay.push(obs, a, proxy_r, next_obs, done_td, true_r=true_r, ep_id=ep_id)
        obs = next_obs

        if done_any:
            ep_id += 1
            obs = unwrap_reset(env.reset())

    reward_updates = 0

    def refresh_proxy_stats_and_maybe_relabel(do_relabel: bool):
        """
        Update proxy transform stats from replay. If do_relabel=True, relabel all stored proxy rewards.
        """
        if args.algo == "standard":
            return

        stats = estimate_proxy_stats_from_replay(replay, raw_rhat_batch, sample_n=args.proxy_stats_n)
        proxy_tx.update(stats.mean, stats.std, stats.rmin, stats.rmax)

        def proxy_reward_fn(s_np, a_int):
            raw = float(raw_rhat_single(s_np, a_int))
            return float(proxy_tx(raw))

        if do_relabel:
            replay.relabel_rewards(proxy_reward_fn)

        return stats

    # If we loaded a reward ckpt, synchronize the entire buffer immediately (+ update proxy stats)
    if args.algo != "standard" and reward_ckpt_loaded:
        stats = refresh_proxy_stats_and_maybe_relabel(do_relabel=True)
        reward_path = os.path.join(exp_dir, "reward_net.pth")
        torch.save(reward_models[0].state_dict(), reward_path)
        print(
            f"[Reward:init] ckpt loaded -> relabeled replay once | saved snapshot to {reward_path} | "
            f"proxy_stats mean={stats.mean:.3f} std={stats.std:.3f} min={stats.rmin:.3f} max={stats.rmax:.3f} "
            f"| center={int(proxy_tx.center)} norm={int(proxy_tx.normalize)} clip={args.proxy_clip if args.proxy_clip>0 else 0}"
        )

    # If we DIDN'T load ckpt: bootstrap reward once (even if frozen, otherwise proxy stays 0 forever)
    if args.algo != "standard" and (not reward_ckpt_loaded):
        if args.freeze_reward:
            print("[warn] --freeze_reward set but no ckpt provided; doing ONE bootstrap reward fit so proxy isn't all zeros.")
        # one bootstrap batch of prefs + some epochs
        candidates = sample_preference_queries(
            replay,
            num_queries=PREF_QUERIES_PER_UPDATE * CANDIDATE_MULTIPLIER,
            seg_len=SEG_LEN,
            mode=args.pref_mode,
        )
        chosen = select_uncertain_queries(
            reward_ensemble=reward_models,
            candidate_queries=candidates,
            action_dim=action_dim,
            top_k=PREF_QUERIES_PER_UPDATE,
        )
        pref_dataset.extend(chosen)

        reward_losses = []
        for m, opt in zip(reward_models, reward_optims):
            loss_mean = 0.0
            for _ in range(10):  # stronger initial fit
                loss_mean = train_reward_epoch(m, opt, pref_dataset, action_dim, BATCH_SIZE)
            reward_losses.append(loss_mean)

        reward_updates = 1
        stats = refresh_proxy_stats_and_maybe_relabel(do_relabel=True)

        reward_path = os.path.join(exp_dir, "reward_net.pth")
        torch.save(reward_models[0].state_dict(), reward_path)
        print(
            f"[Reward:init] bootstrapped from scratch | prefs={len(pref_dataset)} | loss~{np.mean(reward_losses):.4f} | saved {reward_path} | "
            f"proxy_stats mean={stats.mean:.3f} std={stats.std:.3f} min={stats.rmin:.3f} max={stats.rmax:.3f}"
        )

    print(f"Replay burn-in complete: {len(replay)} transitions")
    if args.algo != "standard":
        print(
            f"[Reward] freeze_reward={bool(args.freeze_reward)} | ckpt_loaded={bool(reward_ckpt_loaded)} | "
            f"proxy_center={int(proxy_tx.center)} proxy_norm={int(proxy_tx.normalize)} proxy_clip={args.proxy_clip if args.proxy_clip>0 else 0}"
        )

    # ---------- training loop ----------
    metrics: List[Dict] = []
    best_true_return = -1e9
    best_success = -1
    grad_steps = 0
    global_steps = 0

    for ep in range(1, args.episodes + 1):
        ep_id += 1
        obs = unwrap_reset(env.reset())
        ep_true = 0.0
        ep_proxy = 0.0
        ep_len = 0

        terminated = False
        truncated = False
        info = {}

        while (not terminated) and (not truncated) and ep_len < MAX_STEPS_PER_EP:
            global_steps += 1
            ep_len += 1

            eps = eps_sched.step()
            if np.random.rand() < eps:
                a = np.random.randint(action_dim)
            else:
                with torch.no_grad():
                    s_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    a = int(torch.argmax(q(s_t), dim=1).item())

            next_obs, true_r, terminated, truncated, info = unwrap_step(env.step(a))
            done_td = terminated  # IMPORTANT

            # proxy reward
            if args.algo == "standard":
                proxy_r = true_r
            else:
                raw = float(raw_rhat_single(obs, a))
                proxy_r = float(proxy_tx(raw))

            replay.push(obs, a, proxy_r, next_obs, done_td, true_r=true_r, ep_id=ep_id)

            ep_true += true_r
            ep_proxy += proxy_r
            obs = next_obs

            # ----- reward updates (ours_*), only if NOT frozen -----
            if (args.algo != "standard") and (not args.freeze_reward) and (global_steps % REWARD_UPDATE_EVERY_STEPS == 0):
                reward_updates += 1

                candidates = sample_preference_queries(
                    replay,
                    num_queries=PREF_QUERIES_PER_UPDATE * CANDIDATE_MULTIPLIER,
                    seg_len=SEG_LEN,
                    mode=args.pref_mode,
                )
                chosen = select_uncertain_queries(
                    reward_ensemble=reward_models,
                    candidate_queries=candidates,
                    action_dim=action_dim,
                    top_k=PREF_QUERIES_PER_UPDATE,
                )
                pref_dataset.extend(chosen)

                reward_losses = []
                for m, opt in zip(reward_models, reward_optims):
                    loss_mean = 0.0
                    for _ in range(REWARD_TRAIN_EPOCHS):
                        loss_mean = train_reward_epoch(m, opt, pref_dataset, action_dim, BATCH_SIZE)
                    reward_losses.append(loss_mean)

                # refresh stats; relabel only for ours_relabel
                do_relabel = (args.algo == "ours_relabel")
                stats = refresh_proxy_stats_and_maybe_relabel(do_relabel=do_relabel)

                reward_path = os.path.join(exp_dir, "reward_net.pth")
                torch.save(reward_models[0].state_dict(), reward_path)

                print(
                    f"[Reward] update#{reward_updates:02d} step={global_steps} "
                    f"| prefs={len(pref_dataset)} | loss~{np.mean(reward_losses):.4f} "
                    f"| relabel={'YES' if do_relabel else 'NO'} "
                    f"| proxy_stats mean={stats.mean:.3f} std={stats.std:.3f} min={stats.rmin:.3f} max={stats.rmax:.3f}"
                )

            # ----- DQN update -----
            if len(replay) >= BATCH_SIZE:
                batch = replay.sample(BATCH_SIZE)
                _ = dqn_update(q, tq, q_opt, batch, gamma=GAMMA)

                grad_steps += 1
                if grad_steps % TARGET_UPDATE_EVERY == 0:
                    tq.load_state_dict(q.state_dict())

        # success: only if terminated and watered all
        ep_success = 0
        watered = info.get("watered_state", None) if isinstance(info, dict) else None
        if terminated and watered is not None:
            ep_success = int(all(bool(x) for x in watered))

        gap = ep_true - ep_proxy
        m: Dict = dict(
            episode=ep,
            true_return=float(ep_true),
            proxy_return=float(ep_proxy),
            gap=float(gap),
            success=int(ep_success),
            epsilon=float(eps_sched.current()),
            terminated=int(terminated),
            truncated=int(truncated),
            ep_len=int(ep_len),
        )

        if args.algo != "standard":
            m.update(dict(
                pref_dataset_size=int(len(pref_dataset)),
                reward_updates=int(reward_updates),
                reward_frozen=int(bool(args.freeze_reward)),
                reward_ckpt_loaded=int(bool(reward_ckpt_loaded)),
                proxy_center=int(proxy_tx.center),
                proxy_norm=int(proxy_tx.normalize),
                proxy_clip=float(args.proxy_clip),
                proxy_stats_mean=float(proxy_tx.stats.mean),
                proxy_stats_std=float(proxy_tx.stats.std),
                proxy_stats_min=float(proxy_tx.stats.rmin),
                proxy_stats_max=float(proxy_tx.stats.rmax),
            ))

        metrics.append(m)

        if ep % 20 == 0 or ep == 1:
            print(
                f"Ep {ep:4d}/{args.episodes} | "
                f"True {ep_true:6.2f} | Proxy {ep_proxy:7.2f} | Gap {gap:7.2f} | "
                f"Succ {ep_success} | term {int(terminated)} | trunc {int(truncated)} | eps {eps_sched.current():.3f}"
            )

        # save best: prioritize SUCCESS, then true return
        best_path = os.path.join(exp_dir, "best_q_net.pth")
        if (ep_success > best_success) or (ep_success == best_success and ep_true > best_true_return):
            best_success = ep_success
            best_true_return = ep_true
            torch.save(q.state_dict(), best_path)

    # save final
    torch.save(q.state_dict(), os.path.join(exp_dir, "final_q_net.pth"))
    save_json(os.path.join(exp_dir, "training_metrics.json"), metrics)

    print("\nSaved:")
    print(" ", os.path.join(exp_dir, "best_q_net.pth"))
    print(" ", os.path.join(exp_dir, "final_q_net.pth"))
    print(" ", os.path.join(exp_dir, "training_metrics.json"))
    if args.algo != "standard":
        rp = os.path.join(exp_dir, "reward_net.pth")
        if os.path.exists(rp):
            print(" ", rp)


if __name__ == "__main__":
    main()
