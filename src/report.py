# src/report.py
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio

from environment import TomatoSafetyGrid
from models import QNetwork, RewardCNN
from utils import DEVICE, set_seed

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(THIS_DIR, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

ACTION_NAMES = {0:"UP",1:"DOWN",2:"LEFT",3:"RIGHT",4:"STAY"}


def unwrap_reset(out):
    return out[0] if isinstance(out, tuple) else out

def unwrap_step(out):
    if len(out) == 5:
        obs, r, term, trunc, info = out
        return obs, r, bool(term) or bool(trunc), info
    obs, r, done, info = out
    return obs, r, bool(done), info


@torch.no_grad()
def greedy_action(q_net, obs):
    s = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    q = q_net(s)
    return int(torch.argmax(q, dim=1).item())


def eval_success_rate(policy_fn, n_episodes=200, seed=0):
    env = TomatoSafetyGrid()
    rng = np.random.default_rng(seed)
    successes = 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 10**9)))
        done = False
        while not done:
            a = policy_fn(obs)
            obs, r, done, info = unwrap_step(env.step(a))
        # success == watered all tomatoes => terminated was True (your env uses terminated for that)
        # if you stored it in info, you can check info["watered_state"]
        if all(info.get("watered_state", ())):
            successes += 1
    return successes / n_episodes


def save_rollout_gif(policy_fn, out_path, max_steps=50, seed=0, fps=6):
    env = TomatoSafetyGrid(render_mode="rgb_array")

    obs, _ = env.reset(seed=seed)
    frames = [env.render()]
    done = False
    t = 0
    while not done and t < max_steps:
        a = policy_fn(obs)
        obs, r, done, info = unwrap_step(env.step(a))
        frames.append(env.render())
        t += 1

    imageio.mimsave(out_path, frames, fps=fps)


def reward_heatmaps(reward_net: RewardCNN, action_dim: int):
    """
    Two heatmaps:
      1) max_a r_hat(s,a) when placing agent at each location (tomatoes unwatered)
      2) r_hat(s, STAY) same state
    This visualizes "tomatoes high, sprinkler ~ neutral" described in paper :contentReference[oaicite:9]{index=9}.
    """
    env = TomatoSafetyGrid()
    g = env.grid_size

    # make a base state: tomatoes unwatered
    env.reset()
    env.watered_state = [False] * len(env.tomatoes)

    max_map = np.zeros((g, g), dtype=np.float32)
    stay_map = np.zeros((g, g), dtype=np.float32)

    @torch.no_grad()
    def r_hat(obs, a):
        s = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        a_oh = torch.nn.functional.one_hot(torch.tensor([a], device=DEVICE), action_dim).float()
        return float(reward_net(s, a_oh).item())

    for y in range(g):
        for x in range(g):
            # skip border walls (in your env borders are walls)
            if y == 0 or x == 0 or y == g - 1 or x == g - 1:
                max_map[y, x] = np.nan
                stay_map[y, x] = np.nan
                continue

            env.agent_pos = (y, x)
            obs = env._get_obs()

            vals = [r_hat(obs, a) for a in range(action_dim)]
            max_map[y, x] = np.max(vals)
            stay_map[y, x] = vals[4]  # STAY

    return max_map, stay_map


def plot_heatmap(mat, title, out_path):
    plt.figure()
    plt.title(title)
    plt.imshow(mat, interpolation="nearest")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_alignment_curves(metrics_path, out_path):
    with open(metrics_path, "r") as f:
        m = json.load(f)

    ep = [d["episode"] for d in m]
    true_r = [d["true_return"] for d in m]
    proxy_r = [d["proxy_return"] for d in m]
    gap = [d["gap"] for d in m]

    plt.figure()
    plt.plot(ep, true_r, label="True return")
    plt.plot(ep, proxy_r, label="Proxy return")
    plt.plot(ep, gap, label="Gap (true - proxy)")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    # ---- load trained models ----
    env = TomatoSafetyGrid()
    action_dim = env.action_space.n

    # Q-networks you trained should be here:
    q_best_path = os.path.join(RESULTS_DIR, "best_q_net.pth")
    q_final_path = os.path.join(RESULTS_DIR, "final_q_net.pth")
    reward_path = os.path.join(RESULTS_DIR, "reward_net.pth")
    metrics_path = os.path.join(RESULTS_DIR, "training_metrics.json")

    # policy: learned
    q_net = QNetwork(action_dim=action_dim, in_channels=env.observation_space.shape[0]).to(DEVICE)
    q_net.load_state_dict(torch.load(q_best_path, map_location=DEVICE))
    q_net.eval()
    learned_policy = lambda obs: greedy_action(q_net, obs)

    # policy: random
    rnd_env = TomatoSafetyGrid()
    random_policy = lambda obs: rnd_env.action_space.sample()

    # ---- success rates (table values) ----
    # paper reports mean±std over 5 seeds :contentReference[oaicite:10]{index=10}
    seeds = [0, 1, 2, 3, 4]
    n_eval = 200

    learned_rates = []
    random_rates = []
    for s in seeds:
        learned_rates.append(eval_success_rate(learned_policy, n_episodes=n_eval, seed=s))
        random_rates.append(eval_success_rate(random_policy, n_episodes=n_eval, seed=s))

    def mean_std(xs):
        xs = np.array(xs, dtype=np.float32)
        return float(xs.mean()), float(xs.std(ddof=1))

    learned_mu, learned_sd = mean_std(learned_rates)
    rnd_mu, rnd_sd = mean_std(random_rates)

    table = {
        "LearnedPolicy_from_best_q_net": {"mean": learned_mu, "std": learned_sd, "per_seed": learned_rates},
        "Random": {"mean": rnd_mu, "std": rnd_sd, "per_seed": random_rates},
        "n_eval_per_seed": n_eval,
        "seeds": seeds,
    }

    out_table = os.path.join(FIG_DIR, "success_table.json")
    with open(out_table, "w") as f:
        json.dump(table, f, indent=2)
    print("Wrote:", out_table)

    # ---- alignment curves (true vs proxy vs gap) ----
    if os.path.exists(metrics_path):
        plot_alignment_curves(metrics_path, os.path.join(FIG_DIR, "alignment_curves.png"))
        print("Wrote:", os.path.join(FIG_DIR, "alignment_curves.png"))

    # ---- reward heatmaps ----
    if os.path.exists(reward_path):
        r_net = RewardCNN(action_dim=action_dim, in_channels=env.observation_space.shape[0]).to(DEVICE)
        r_net.load_state_dict(torch.load(reward_path, map_location=DEVICE))
        r_net.eval()

        max_map, stay_map = reward_heatmaps(r_net, action_dim)
        plot_heatmap(max_map, "RewardCNN heatmap: max_a r_hat(s,a)", os.path.join(FIG_DIR, "reward_heatmap_max.png"))
        plot_heatmap(stay_map, "RewardCNN heatmap: r_hat(s, STAY)", os.path.join(FIG_DIR, "reward_heatmap_stay.png"))
        print("Wrote heatmaps into figures/")

    # ---- rollout GIF ----
    save_rollout_gif(learned_policy, os.path.join(FIG_DIR, "rollout_learned.gif"), seed=0)
    save_rollout_gif(random_policy, os.path.join(FIG_DIR, "rollout_random.gif"), seed=0)
    print("Wrote GIFs into figures/")


if __name__ == "__main__":
    main()
