# src/train_reward.py
import argparse
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.optim as optim

from environment import TomatoSafetyGrid
from models import RewardCNN
from utils import (
    DEVICE,
    ReplayBuffer,
    preference_batch_loss,
    select_uncertain_queries,
    set_seed,
    save_json,
    preference_batch_logits,  # make sure utils.py exports this (it does in your pasted version)
)


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.path.join(THIS_DIR, "results")
os.makedirs(RESULTS_ROOT, exist_ok=True)


# ---------------- gymnasium unwrap ----------------
def unwrap_reset(out):
    return out[0] if isinstance(out, tuple) else out


def unwrap_step(out):
    # gymnasium: (obs, r, terminated, truncated, info)
    if isinstance(out, tuple) and len(out) == 5:
        obs, r, term, trunc, info = out
        return obs, r, bool(term), bool(trunc), info
    # gym fallback
    if isinstance(out, tuple) and len(out) == 4:
        obs, r, done, info = out
        return obs, r, bool(done), False, info
    raise ValueError(f"Unexpected step() output: {out}")


# ---------------- data collection ----------------
def collect_random_data(env: TomatoSafetyGrid, buffer: ReplayBuffer, steps: int = 20_000):
    obs = unwrap_reset(env.reset())
    ep_id = 0
    for _ in range(steps):
        a = env.action_space.sample()
        nxt, true_r, terminated, truncated, info = unwrap_step(env.step(a))
        done = terminated or truncated
        buffer.push(obs, a, 0.0, nxt, done, true_r=float(true_r), ep_id=ep_id)
        obs = nxt
        if done:
            ep_id += 1
            obs = unwrap_reset(env.reset())


# ---------------- preference query sampling (margin + noisy) ----------------
def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def label_oracle(r1: float, r2: float) -> float:
    if r1 == r2:
        return 0.5
    return float(r1 > r2)


def label_noisy(r1: float, r2: float, beta: float, flip_prob: float, soft: bool) -> float:
    # Bradley–Terry probability from true-return gap (simulated "human")
    p = sigmoid((r1 - r2) / max(1e-6, beta))
    if soft:
        # soft target = probability (stabilizes BCE a lot)
        y = p
    else:
        y = float(np.random.rand() < p)
        if np.random.rand() < flip_prob:
            y = 1.0 - y
    return float(y)


def segment_return_true(seg) -> float:
    return float(sum(t.true_r for t in seg))


def sample_preference_queries_filtered(
    buffer: ReplayBuffer,
    num_queries: int,
    seg_len: int,
    mode: str,
    beta: float,
    flip_prob: float,
    noisy_soft: bool,
    margin: float,
    max_tries: int = 200_000,
):
    """
    Samples preference comparisons (seg1, seg2, mu) with an OPTIONAL margin filter:
      abs(R1-R2) >= margin
    This prevents active learning from over-focusing on near-ties (which are noisy).
    """
    out = []
    tries = 0
    while len(out) < num_queries and tries < max_tries:
        tries += 1
        s1 = buffer.get_segment(seg_len)
        s2 = buffer.get_segment(seg_len)
        r1, r2 = segment_return_true(s1), segment_return_true(s2)

        if margin > 0.0 and abs(r1 - r2) < margin:
            continue

        if mode == "oracle":
            mu = label_oracle(r1, r2)
        elif mode == "noisy":
            mu = label_noisy(r1, r2, beta=beta, flip_prob=flip_prob, soft=noisy_soft)
        else:
            raise ValueError(f"Unknown pref mode: {mode}")

        out.append((s1, s2, mu))

    if len(out) < num_queries:
        print(
            f"[warn] Only collected {len(out)}/{num_queries} queries after {tries} tries. "
            f"Consider lowering --pref_margin."
        )
    return out


# ---------------- batching ----------------
def pack_pref_batch(batch, seg_len: int):
    seg1_s = np.stack([[t.s for t in seg1] for seg1, _, _ in batch], axis=0)
    seg1_a = np.stack([[t.a for t in seg1] for seg1, _, _ in batch], axis=0)
    seg2_s = np.stack([[t.s for t in seg2] for _, seg2, _ in batch], axis=0)
    seg2_a = np.stack([[t.a for t in seg2] for _, seg2, _ in batch], axis=0)
    mu = np.array([mu for _, _, mu in batch], dtype=np.float32)

    return (
        torch.tensor(seg1_s, dtype=torch.float32, device=DEVICE),
        torch.tensor(seg1_a, dtype=torch.long, device=DEVICE),
        torch.tensor(seg2_s, dtype=torch.float32, device=DEVICE),
        torch.tensor(seg2_a, dtype=torch.long, device=DEVICE),
        torch.tensor(mu, dtype=torch.float32, device=DEVICE),
    )


@torch.no_grad()
def eval_pref(reward_net: RewardCNN, val_set, action_dim: int, seg_len: int, batch_size: int = 64):
    reward_net.eval()
    total = 0
    correct = 0
    bces = []

    for i in range(0, len(val_set), batch_size):
        batch = val_set[i:i + batch_size]
        s1, a1, s2, a2, mu = pack_pref_batch(batch, seg_len)

        # BCE
        loss = preference_batch_loss(reward_net, s1, a1, s2, a2, mu, action_dim)
        bces.append(float(loss.item()))

        # accuracy: threshold at 0.5 on sigmoid(logits)
        logits = preference_batch_logits(reward_net, s1, a1, s2, a2, action_dim)
        pred = (torch.sigmoid(logits) > 0.5).float()
        # for soft labels, treat "correct" as closest to 0/1
        hard_mu = (mu >= 0.5).float()
        correct += int((pred.cpu() == hard_mu.cpu()).sum().item())
        total += mu.shape[0]

    reward_net.train()
    return (correct / max(1, total)), float(np.mean(bces))


def train_epochs(
    reward_net: RewardCNN,
    opt: optim.Optimizer,
    train_set,
    val_set,
    action_dim: int,
    seg_len: int,
    epochs: int,
    batch_size: int,
    label_smooth: float,
    verbose_prefix: str = "[Reward]",
):
    for ep in range(1, epochs + 1):
        np.random.shuffle(train_set)
        losses = []
        reward_net.train()

        for i in range(0, len(train_set), batch_size):
            batch = train_set[i:i + batch_size]
            s1, a1, s2, a2, mu = pack_pref_batch(batch, seg_len)

            # label smoothing prevents overconfident BCE explosion when labels are noisy
            if label_smooth > 0.0:
                mu = mu * (1.0 - 2.0 * label_smooth) + label_smooth

            loss = preference_batch_loss(reward_net, s1, a1, s2, a2, mu, action_dim)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reward_net.parameters(), 1.0)
            opt.step()

            losses.append(float(loss.item()))

        val_acc, val_bce = eval_pref(reward_net, val_set, action_dim, seg_len, batch_size=batch_size)
        if ep == 1 or ep % 10 == 0 or ep == epochs:
            print(
                f"{verbose_prefix} epoch {ep:03d}/{epochs} | "
                f"loss={np.mean(losses):.4f} | val_acc={val_acc:.3f} | val_bce={val_bce:.4f}"
            )


# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)

    # target query budget
    parser.add_argument("--total_queries", type=int, default=500)
    parser.add_argument("--seg_len", type=int, default=8)

    # training
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--round_epochs", type=int, default=50)
    parser.add_argument("--final_epochs", type=int, default=50)

    # active learning
    parser.add_argument("--init_queries", type=int, default=100)
    parser.add_argument("--acquire_per_round", type=int, default=40)
    parser.add_argument("--candidates_multiplier", type=int, default=10)
    parser.add_argument("--ensemble_size", type=int, default=3)
    parser.add_argument("--ensemble_epochs", type=int, default=10)

    # preferences
    parser.add_argument("--pref_mode", choices=["oracle", "noisy"], default="oracle")
    parser.add_argument("--beta", type=float, default=0.20)
    parser.add_argument("--flip_prob", type=float, default=0.10)
    parser.add_argument("--noisy_soft", action="store_true",
                        help="If set, use soft labels p instead of sampling a hard label (stabilizes noisy training).")
    parser.add_argument("--label_smooth", type=float, default=0.05,
                        help="Smooth preference labels toward 0.5 to reduce overconfidence.")

    # IMPORTANT: avoid ambiguous comparisons
    parser.add_argument("--pref_margin", type=float, default=0.10,
                        help="Only keep comparisons with abs(R1-R2) >= margin. Helps noisy labels a lot.")

    # data
    parser.add_argument("--random_steps", type=int, default=20_000)

    # validation: use a FRACTION of total, not a huge fixed value
    parser.add_argument("--val_frac", type=float, default=0.20,
                        help="Validation fraction of total_queries (e.g., 0.2 => 100 val when total=500).")

    # output directory
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Optional subfolder name under results/. If not set, auto-named.")

    args = parser.parse_args()

    set_seed(args.seed)

    env = TomatoSafetyGrid(seed=args.seed)
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.n

    # experiment dir
    exp_name = args.exp_name or f"reward_{args.pref_mode}{'_soft' if args.noisy_soft else ''}_seed{args.seed}"
    exp_dir = os.path.join(RESULTS_ROOT, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # buffer
    buffer = ReplayBuffer(capacity=max(50_000, args.random_steps + 10_000), obs_shape=obs_shape)
    collect_random_data(env, buffer, steps=args.random_steps)

    # compute val size sanely
    val_size = int(max(50, min(args.total_queries * 0.4, round(args.total_queries * args.val_frac))))
    val_size = min(val_size, args.total_queries)  # safety

    # fixed val set (small-ish; stable)
    val_set = sample_preference_queries_filtered(
        buffer,
        num_queries=val_size,
        seg_len=args.seg_len,
        mode=args.pref_mode,
        beta=args.beta,
        flip_prob=args.flip_prob,
        noisy_soft=args.noisy_soft,
        margin=args.pref_margin,
    )

    # initial train set
    pref_set = sample_preference_queries_filtered(
        buffer,
        num_queries=args.init_queries,
        seg_len=args.seg_len,
        mode=args.pref_mode,
        beta=args.beta,
        flip_prob=args.flip_prob,
        noisy_soft=args.noisy_soft,
        margin=args.pref_margin,
    )

    print(f"Initial pref set: train={len(pref_set)}, val={len(val_set)}")

    # main reward model + persistent optimizer (IMPORTANT FIX)
    reward_net = RewardCNN(action_dim=action_dim, in_channels=obs_shape[0]).to(DEVICE)
    opt_main = optim.Adam(reward_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ensemble (for query selection)
    ensemble = [RewardCNN(action_dim=action_dim, in_channels=obs_shape[0]).to(DEVICE) for _ in range(args.ensemble_size)]

    metrics = {
        "pref_mode": args.pref_mode,
        "beta": args.beta,
        "flip_prob": args.flip_prob,
        "noisy_soft": bool(args.noisy_soft),
        "pref_margin": args.pref_margin,
        "val_size": len(val_set),
        "rounds": [],
    }

    round_id = 0
    while len(pref_set) < args.total_queries:
        round_id += 1

        # train main model
        train_epochs(
            reward_net, opt_main,
            train_set=pref_set,
            val_set=val_set,
            action_dim=action_dim,
            seg_len=args.seg_len,
            epochs=args.round_epochs,
            batch_size=args.batch_size,
            label_smooth=args.label_smooth,
            verbose_prefix="[Reward]",
        )
        val_acc, val_bce = eval_pref(reward_net, val_set, action_dim, args.seg_len, batch_size=args.batch_size)
        print(f"[AL {round_id:02d}] val_acc={val_acc:.3f} | val_bce={val_bce:.4f} | queries={len(pref_set)}")

        # (re)build ensemble from main weights + bootstrap for diversity
        # NOTE: this makes disagreement meaningful; pure random reinit each round is noisy.
        for k in range(len(ensemble)):
            ensemble[k].load_state_dict(reward_net.state_dict())
            opt_e = optim.Adam(ensemble[k].parameters(), lr=args.lr, weight_decay=args.weight_decay)

            # bootstrap sample from pref_set
            boot = [pref_set[i] for i in np.random.randint(0, len(pref_set), size=len(pref_set))]
            train_epochs(
                ensemble[k], opt_e,
                train_set=boot,
                val_set=val_set,
                action_dim=action_dim,
                seg_len=args.seg_len,
                epochs=args.ensemble_epochs,
                batch_size=args.batch_size,
                label_smooth=args.label_smooth,
                verbose_prefix=f"[Ens {k}]",
            )

        # acquire more queries via uncertainty sampling
        remaining = args.total_queries - len(pref_set)
        k = min(args.acquire_per_round, remaining)

        num_candidates = k * args.candidates_multiplier
        candidates = sample_preference_queries_filtered(
            buffer,
            num_queries=num_candidates,
            seg_len=args.seg_len,
            mode=args.pref_mode,
            beta=args.beta,
            flip_prob=args.flip_prob,
            noisy_soft=args.noisy_soft,
            margin=args.pref_margin,
        )

        chosen = select_uncertain_queries(
            reward_ensemble=ensemble,
            candidate_queries=candidates,
            action_dim=action_dim,
            top_k=k,
        )
        pref_set.extend(chosen)
        print(f"[AL {round_id:02d}] acquired {k} queries (candidates={num_candidates}); now total={len(pref_set)}")

        metrics["rounds"].append(
            {"round": round_id, "queries": len(pref_set), "val_acc": float(val_acc), "val_bce": float(val_bce)}
        )

    print("\nFinal training pass...")
    train_epochs(
        reward_net, opt_main,
        train_set=pref_set,
        val_set=val_set,
        action_dim=action_dim,
        seg_len=args.seg_len,
        epochs=args.final_epochs,
        batch_size=args.batch_size,
        label_smooth=args.label_smooth,
        verbose_prefix="[FINAL]",
    )
    val_acc, val_bce = eval_pref(reward_net, val_set, action_dim, args.seg_len, batch_size=args.batch_size)
    print(f"[FINAL] val_acc={val_acc:.3f} | val_bce={val_bce:.4f}")

    # save into exp_dir
    out_path = os.path.join(exp_dir, "reward_net.pth")
    torch.save(reward_net.state_dict(), out_path)
    save_json(os.path.join(exp_dir, "reward_training_metrics.json"), metrics)

    print(f"\nSaved reward model to: {out_path}")
    print(f"Saved metrics to:      {os.path.join(exp_dir, 'reward_training_metrics.json')}")


if __name__ == "__main__":
    main()

