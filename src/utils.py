# src/utils.py
import json
import os
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ---------------- misc ----------------

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # reproducibility (safe defaults)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def save_json(path: str, obj):
    _ensure_parent_dir(path)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def unwrap_reset(out):
    # gymnasium reset -> (obs, info)
    return out[0] if isinstance(out, tuple) else out


def unwrap_step(out):
    # gymnasium step -> (obs, reward, terminated, truncated, info)
    if isinstance(out, tuple) and len(out) == 5:
        obs, r, terminated, truncated, info = out
        return obs, r, bool(terminated) or bool(truncated), info
    # old gym -> (obs, reward, done, info)
    obs, r, done, info = out
    return obs, r, bool(done), info


# ---------------- replay buffer ----------------

@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float          # proxy reward (can be relabeled)
    ns: np.ndarray
    done: bool
    true_r: float     # true env reward (for oracle/noisy labels + logging)
    ep_id: int        # episode id (for contiguous segments)


class ReplayBuffer:
    """
    Ring buffer that can:
      - sample i.i.d. transitions (for DQN)
      - sample contiguous segments w/o crossing episode boundaries (for preferences)
      - relabel all stored proxy rewards using current reward model (paper’s key stabilization)
    """
    def __init__(self, capacity: int, obs_shape: Tuple[int, ...]):
        self.capacity = int(capacity)
        self.obs_shape = tuple(obs_shape)
        self._data: List[Transition] = []
        self._ptr = 0  # next write index (only meaningful once buffer is full)
        self._default_ep_id = 0  # fallback for old code that doesn't pass ep_id

    def __len__(self):
        return len(self._data)

    def push(
        self,
        s,
        a,
        r,
        ns,
        done,
        true_r: float,
        ep_id: Optional[int] = None,  # <-- backward compatible
    ):
        if ep_id is None:
            ep_id = self._default_ep_id

        tr = Transition(
            s=np.asarray(s, dtype=np.float32),
            a=int(a),
            r=float(r),
            ns=np.asarray(ns, dtype=np.float32),
            done=bool(done),
            true_r=float(true_r),
            ep_id=int(ep_id),
        )

        if len(self._data) < self.capacity:
            self._data.append(tr)
            # when filling (not full yet), _ptr stays 0; once full, oldest is at index 0
        else:
            self._data[self._ptr] = tr
            self._ptr = (self._ptr + 1) % self.capacity

    def set_default_ep_id(self, ep_id: int):
        """Optional: lets you avoid passing ep_id in every push()."""
        self._default_ep_id = int(ep_id)

    def _chronological(self) -> List[Transition]:
        """Return transitions in chronological order (oldest -> newest) even after wrap."""
        n = len(self._data)
        if n < self.capacity:
            return self._data
        # when full, _ptr points to where the next write will go => oldest is at _ptr
        return self._data[self._ptr:] + self._data[:self._ptr]

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        idxs = np.random.randint(0, len(self._data), size=batch_size)
        s = np.stack([self._data[i].s for i in idxs], axis=0)
        a = np.array([self._data[i].a for i in idxs], dtype=np.int64)
        r = np.array([self._data[i].r for i in idxs], dtype=np.float32)
        ns = np.stack([self._data[i].ns for i in idxs], axis=0)
        d = np.array([self._data[i].done for i in idxs], dtype=np.float32)
        tr = np.array([self._data[i].true_r for i in idxs], dtype=np.float32)
        return {"states": s, "actions": a, "rewards": r, "next_states": ns, "dones": d, "true_rewards": tr}

    def get_segment(self, length: int, max_tries: int = 2000) -> List[Transition]:
        """
        Sample a contiguous segment of length `length` that:
          - is contiguous in time (chronological order)
          - does NOT cross episode boundary (ep_id constant)
          - does NOT include done except possibly at the final element
        """
        seq = self._chronological()
        n = len(seq)
        if n < length:
            raise ValueError("Not enough data to sample a segment yet.")

        for _ in range(max_tries):
            start = np.random.randint(0, n - length + 1)
            seg = seq[start:start + length]

            ep0 = seg[0].ep_id
            if any(t.ep_id != ep0 for t in seg):
                continue
            if any(t.done for t in seg[:-1]):
                continue
            return seg

        # fallback scan
        for start in range(0, n - length + 1):
            seg = seq[start:start + length]
            ep0 = seg[0].ep_id
            if any(t.ep_id != ep0 for t in seg):
                continue
            if any(t.done for t in seg[:-1]):
                continue
            return seg

        raise RuntimeError("Could not find a valid contiguous segment. Collect longer episodes or increase buffer.")

    def relabel_rewards(self, reward_fn: Callable[[np.ndarray, int], float], batch_size: int = 4096):
        """
        Iterative Replay Buffer Relabeling:
        overwrite proxy reward for ALL transitions with current reward model's r_hat(s,a).
        """
        n = len(self._data)
        for i0 in range(0, n, batch_size):
            i1 = min(n, i0 + batch_size)
            for i in range(i0, i1):
                tr = self._data[i]
                new_r = float(reward_fn(tr.s, tr.a))
                self._data[i] = Transition(tr.s, tr.a, new_r, tr.ns, tr.done, tr.true_r, tr.ep_id)


class EpsilonScheduler:
    def __init__(self, start: float, end: float, decay_steps: int):
        self.start = float(start)
        self.end = float(end)
        self.decay_steps = int(decay_steps)
        self.t = 0

    def step(self) -> float:
        self.t += 1
        return self.current()

    def current(self) -> float:
        frac = min(1.0, self.t / max(1, self.decay_steps))
        return self.start + frac * (self.end - self.start)


# ---------------- preference helpers ----------------

def one_hot(actions: torch.Tensor, action_dim: int) -> torch.Tensor:
    return F.one_hot(actions.long(), num_classes=action_dim).float()


def preference_batch_logits(
    reward_model,
    seg1_s: torch.Tensor, seg1_a: torch.Tensor,
    seg2_s: torch.Tensor, seg2_a: torch.Tensor,
    action_dim: int
) -> torch.Tensor:
    """
    logits = R(seg1) - R(seg2) for Bradley–Terry.
    seg*_s: [B,L,C,H,W], seg*_a: [B,L]
    """
    B, L = seg1_a.shape

    s1 = seg1_s.reshape(B * L, *seg1_s.shape[2:])
    s2 = seg2_s.reshape(B * L, *seg2_s.shape[2:])
    a1 = seg1_a.reshape(B * L)
    a2 = seg2_a.reshape(B * L)

    r1 = reward_model(s1, one_hot(a1, action_dim)).reshape(B, L).sum(dim=1)
    r2 = reward_model(s2, one_hot(a2, action_dim)).reshape(B, L).sum(dim=1)
    return (r1 - r2)


def preference_batch_loss(
    reward_model,
    seg1_s: torch.Tensor, seg1_a: torch.Tensor,
    seg2_s: torch.Tensor, seg2_a: torch.Tensor,
    mu: torch.Tensor,
    action_dim: int
) -> torch.Tensor:
    """
    Bradley–Terry / logistic preference loss.
    mu in {0,1} or soft in [0,1].
    """
    logits = preference_batch_logits(reward_model, seg1_s, seg1_a, seg2_s, seg2_a, action_dim)
    return F.binary_cross_entropy_with_logits(logits, mu.float())


def pack_preference_batch(batch: List[Tuple[List[Transition], List[Transition], int]]):
    """
    batch: list of (seg1, seg2, mu)
    returns numpy arrays:
      seg1_s: [B,L,C,H,W], seg1_a: [B,L], seg2_s..., mu: [B]
    """
    seg1_s = np.stack([[t.s for t in seg1] for seg1, _, _ in batch], axis=0)
    seg1_a = np.stack([[t.a for t in seg1] for seg1, _, _ in batch], axis=0)
    seg2_s = np.stack([[t.s for t in seg2] for _, seg2, _ in batch], axis=0)
    seg2_a = np.stack([[t.a for t in seg2] for _, seg2, _ in batch], axis=0)
    mu = np.array([mu for _, _, mu in batch], dtype=np.float32)
    return seg1_s, seg1_a, seg2_s, seg2_a, mu


def segment_return(seg: List[Transition]) -> float:
    return float(sum(t.true_r for t in seg))


def label_oracle(r1: float, r2: float) -> int:
    if r1 == r2:
        return int(np.random.rand() < 0.5)
    return int(r1 > r2)


def _stable_sigmoid(x: float) -> float:
    # numerically stable sigmoid for large |x|
    if x >= 0:
        z = np.exp(-x)
        return float(1.0 / (1.0 + z))
    else:
        z = np.exp(x)
        return float(z / (1.0 + z))


def label_noisy(r1: float, r2: float, beta: float = 0.25, flip_prob: float = 0.10) -> int:
    """
    Simulated noisy human:
      - preference sampled from sigmoid((r1-r2)/beta)
      - plus occasional mistake (flip_prob)
    """
    x = (r1 - r2) / max(1e-6, float(beta))
    p = _stable_sigmoid(x)
    y = int(np.random.rand() < p)
    if np.random.rand() < flip_prob:
        y = 1 - y
    return y


def sample_preference_queries(
    buffer: ReplayBuffer,
    num_queries: int,
    seg_len: int,
    mode: str = "oracle",        # "oracle" or "noisy"
    beta: float = 0.25,
    flip_prob: float = 0.10,
) -> List[Tuple[List[Transition], List[Transition], int]]:
    out = []
    for _ in range(num_queries):
        s1 = buffer.get_segment(seg_len)
        s2 = buffer.get_segment(seg_len)
        r1, r2 = segment_return(s1), segment_return(s2)

        if mode == "oracle":
            mu = label_oracle(r1, r2)
        elif mode == "noisy":
            mu = label_noisy(r1, r2, beta=beta, flip_prob=flip_prob)
        else:
            raise ValueError(f"Unknown preference mode: {mode}")

        out.append((s1, s2, mu))
    return out


def ensemble_uncertainty_score(probs: np.ndarray) -> np.ndarray:
    # probs: [E, N] => disagreement score per query
    return probs.var(axis=0)


def select_uncertain_queries(
    reward_ensemble,
    candidate_queries: List[Tuple[List[Transition], List[Transition], int]],
    action_dim: int,
    top_k: int,
) -> List[Tuple[List[Transition], List[Transition], int]]:
    """
    Ensemble-based uncertainty metric for query selection.
    """
    N = len(candidate_queries)
    if N == 0:
        return []
    L = len(candidate_queries[0][0])

    def pack(seg_list):
        s = np.stack([[t.s for t in seg] for seg in seg_list], axis=0)  # [N,L,C,H,W]
        a = np.stack([[t.a for t in seg] for seg in seg_list], axis=0)  # [N,L]
        return (
            torch.tensor(s, dtype=torch.float32, device=DEVICE),
            torch.tensor(a, dtype=torch.long, device=DEVICE),
        )

    seg1s = [q[0] for q in candidate_queries]
    seg2s = [q[1] for q in candidate_queries]
    s1, a1 = pack(seg1s)
    s2, a2 = pack(seg2s)

    probs = []
    with torch.no_grad():
        for m in reward_ensemble:
            logits = preference_batch_logits(m, s1, a1, s2, a2, action_dim)  # [N]
            p = torch.sigmoid(logits).cpu().numpy()
            probs.append(p)

    probs = np.stack(probs, axis=0)  # [E,N]
    score = ensemble_uncertainty_score(probs)
    idx = np.argsort(-score)[:top_k]
    return [candidate_queries[i] for i in idx]
