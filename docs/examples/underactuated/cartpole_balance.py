#!/usr/bin/env python3
"""Prescriptive policy learning (PPL) on Gymnasium CartPole-v1: keep the pole upright.

A pole stands on a cart that is pushed left or right with 10 N at every step. The push changes the pole angular velocity
directly in one step, so the pole angular velocity is the key quantity, and one principle gives its reference.
  Stay upright  turn back toward upright, fast when far and slowing down on the way: the reference is
                sqrt(2 a |theta'|) toward upright, with theta' the pole angle after this step and a a quarter of the
                angular acceleration the push gives the pole there. The cart position is not part of the principle.
Each action is credited by how much its own transition brings the pole angular velocity closer to the reference,
    delta = 0.5 (w_ref - w)^2 - 0.5 (w_ref - w')^2,
and the policy is updated with this coefficient in place of the return.

Usage:
    python cartpole_balance.py                  train until every evaluation episode succeeds (at most 1e5 steps)
    python cartpole_balance.py --resume DIR     continue an interrupted run
    python cartpole_balance.py --gif DIR        write a GIF of the trained policy
    python cartpole_balance.py --export DIR F   write episodes of the trained policy to JSON (data of the web player)
Writes runs/<time>_cartpole_balance/: eval.csv (one row per evaluation), models/ (after every update), resume.pt.
Success: 500 steps without the pole passing 12 degrees or the cart leaving the track.
"""
import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

G, M_CART, M_POLE, HALF, FORCE, DT = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02   # CartPole-v1 constants
M_TOTAL = M_CART + M_POLE
DECEL = 0.25                            # planned share of the angular acceleration the push can give
EVAL_SEEDS = np.arange(10_000, 10_020)
T_MAX = 500


# ---------------------------------------------------------------------------
# Principle and reference
# ---------------------------------------------------------------------------
def reference(obs):
    th, w = obs[:, 2], obs[:, 3]
    th_next = th + DT * w                               # the environment's Euler step; the push does not move it
    push = np.abs(np.cos(th_next) * FORCE / M_TOTAL / (HALF * (4 / 3 - M_POLE * np.cos(th_next) ** 2 / M_TOTAL)))
    return -np.sign(th_next) * np.sqrt(2 * DECEL * push * np.abs(th_next))


def coefficient(s, nx):
    ref = reference(s)                                  # fixed in the state before the action
    return 0.5 * (ref - s[:, 3]) ** 2 - 0.5 * (ref - nx[:, 3]) ** 2, ref - s[:, 3]


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------
def mlp(inp, out, hidden=(64, 64), out_gain=1.0):
    layers, prev = [], inp
    for h in hidden:
        lin = nn.Linear(prev, h)
        nn.init.orthogonal_(lin.weight, np.sqrt(2))
        nn.init.zeros_(lin.bias)
        layers += [lin, nn.Tanh()]
        prev = h
    last = nn.Linear(prev, out)
    nn.init.orthogonal_(last.weight, out_gain)
    nn.init.zeros_(last.bias)
    return nn.Sequential(*layers, last)


class CategoricalPolicy(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.logits = mlp(obs_dim, n_actions, out_gain=0.01)

    def dist(self, obs):
        return torch.distributions.Categorical(logits=self.logits(obs))


class RunningNorm:
    def __init__(self, dim):
        self.mean, self.var, self.count = np.zeros(dim), np.ones(dim), 1e-4

    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        delta, tot = bm - self.mean, self.count + x.shape[0]
        self.mean = self.mean + delta * bc / tot
        self.var = (self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10, 10)


# ---------------------------------------------------------------------------
# Environment, evaluation, files
# ---------------------------------------------------------------------------
def make_env(render=False):
    return gym.make("CartPole-v1", render_mode="rgb_array" if render else None)


def evaluate(envs, act):
    obs = np.stack([env.reset(seed=int(s))[0] for env, s in zip(envs, EVAL_SEEDS)])
    done = np.zeros(len(envs), dtype=bool)
    steps = np.full(len(envs), T_MAX)
    for t in range(T_MAX):
        if done.all():
            break
        a = act(obs)
        for i, env in enumerate(envs):
            if done[i]:
                continue
            o, _, term, trunc, _ = env.step(int(a[i]))
            obs[i] = o
            if term:
                done[i], steps[i] = True, t + 1
            elif trunc:
                done[i] = True
    return dict(success=float((steps == T_MAX).mean()), mean_steps=float(steps.mean()))


def append_row(path, row):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def save(state, path):
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load(run):
    ck = torch.load(sorted((run / "models").glob("update_*.pt"))[-1], weights_only=False)
    policy, norm = CategoricalPolicy(4, 2), RunningNorm(4)
    policy.load_state_dict(ck["policy"])
    norm.mean, norm.var = ck["norm_mean"], ck["norm_var"]

    def act(o):
        with torch.no_grad():
            return policy.dist(torch.as_tensor(norm(o), dtype=torch.float32)).probs.argmax(-1).numpy()
    return act


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(out, seed=0, n_env=20, horizon=100, iters=50, lr=3e-4, epochs=10, clip=0.2, minibatch=64,
          grad_clip=0.5, eval_every=1):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    envs = [make_env() for _ in range(n_env)]
    envs_eval = [make_env() for _ in EVAL_SEEDS]
    policy, norm = CategoricalPolicy(4, 2), RunningNorm(4)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    base = mlp(5, 1)                                    # learned baseline on the state and the deviation
    bopt = torch.optim.Adam(base.parameters(), lr=1e-3)
    (out / "models").mkdir(parents=True, exist_ok=True)
    start = 0
    if (out / "resume.pt").exists():
        ck = torch.load(out / "resume.pt", weights_only=False)
        policy.load_state_dict(ck["policy"]); opt.load_state_dict(ck["optimizer"])
        base.load_state_dict(ck["baseline"]); bopt.load_state_dict(ck["baseline_optimizer"])
        norm.mean, norm.var, norm.count = ck["norm_mean"], ck["norm_var"], ck["norm_count"]
        rng.bit_generator.state = ck["numpy_rng"]; torch.set_rng_state(ck["torch_rng"])
        start = ck["iteration"]
    obs = np.stack([env.reset(seed=int(rng.integers(1 << 31)))[0] for env in envs])
    t0 = time.time()

    def act(o):
        with torch.no_grad():
            return policy.dist(torch.as_tensor(norm(o), dtype=torch.float32)).probs.argmax(-1).numpy()

    for it in range(start, iters + 1):
        if it % eval_every == 0 or it == iters:
            ev = evaluate(envs_eval, act)
            append_row(out / "eval.csv", dict(update=it, steps=it * n_env * horizon, **ev))
            print(f"steps {it * n_env * horizon:6d}  success {ev['success']:.2f}  "
                  f"steps kept up {ev['mean_steps']:.0f}  ({time.time() - t0:.0f} s)", flush=True)
            if ev["success"] == 1.0:
                print("every evaluation episode succeeds: training stops", flush=True)
                break
        if it == iters:
            break
        S, A, S2 = [], [], []
        for _ in range(horizon):
            with torch.no_grad():
                a = policy.dist(torch.as_tensor(norm(obs), dtype=torch.float32)).sample().numpy()
            nxt, before = obs.copy(), obs.copy()
            for i, env in enumerate(envs):
                o, _, term, trunc, _ = env.step(int(a[i]))       # the reward is never used
                nxt[i] = o
                obs[i] = env.reset(seed=int(rng.integers(1 << 31)))[0] if term or trunc else o
            S.append(before); A.append(a); S2.append(nxt)
        s, nx, a = np.concatenate(S), np.concatenate(S2), np.concatenate(A)
        norm.update(s)
        c, e = coefficient(s, nx)
        s_t = torch.as_tensor(norm(s), dtype=torch.float32)
        a_t = torch.as_tensor(a, dtype=torch.long)
        c_t = torch.as_tensor(c[:, None], dtype=torch.float32)
        x = torch.cat([s_t, torch.as_tensor(e[:, None] / 10.0, dtype=torch.float32)], 1)
        scale = c_t.std() + 1e-8
        for _ in range(20):
            bopt.zero_grad()
            ((base(x) - c_t / scale) ** 2).mean().backward()
            bopt.step()
        with torch.no_grad():
            adv = c_t - base(x) * scale
            adv = (adv / (adv.std() + 1e-8))[:, 0]
            old_lp = policy.dist(s_t).log_prob(a_t)
        for _ in range(epochs):
            perm = torch.randperm(len(adv))
            for k in range(0, len(adv), minibatch):
                idx = perm[k:k + minibatch]
                ratio = (policy.dist(s_t[idx]).log_prob(a_t[idx]) - old_lp[idx]).exp()
                loss = -torch.min(ratio * adv[idx], ratio.clamp(1 - clip, 1 + clip) * adv[idx]).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
                opt.step()
        model = dict(policy=policy.state_dict(), norm_mean=norm.mean, norm_var=norm.var, norm_count=norm.count)
        save(model, out / "models" / f"update_{it + 1:04d}.pt")
        save(dict(**model, iteration=it + 1, optimizer=opt.state_dict(), baseline=base.state_dict(),
                  baseline_optimizer=bopt.state_dict(), numpy_rng=rng.bit_generator.state,
                  torch_rng=torch.get_rng_state()), out / "resume.pt")
    for env in envs + envs_eval:
        env.close()


def gif(run, seed=int(EVAL_SEEDS[0])):
    import imageio.v2 as imageio
    act = load(run)
    env = make_env(render=True)
    o, frames = env.reset(seed=seed)[0][None], []
    for _ in range(T_MAX):
        frames.append(env.render())
        o, _, term, trunc, _ = env.step(int(act(o)[0]))
        o = o[None]
        if term or trunc:
            frames.append(env.render())
            break
    env.close()
    imageio.mimsave(run / "episode.gif", frames, duration=0.1, loop=0)
    print(run / "episode.gif")


def export(run, path):
    """Three episodes of the last policy, for the web player."""
    import json
    act = load(run)
    env = make_env()
    episodes = []
    for k, seed in enumerate(EVAL_SEEDS[:3]):
        o = env.reset(seed=int(seed))[0][None]
        x, th, u = [], [], []
        for _ in range(T_MAX):
            x.append(round(float(o[0, 0]), 4)); th.append(round(float(o[0, 2]), 5))
            action = int(act(o)[0])
            u.append(1.0 if action == 1 else -1.0)
            o, _, term, trunc, _ = env.step(action)
            o = o[None]
            if term or trunc:
                break
        episodes.append(dict(label=f"Episode {k + 1}", position=x, angle=th, torque=u, rule=[0] * len(th)))
    env.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(dict(kind="cartpole", dt=DT, max_torque=1.0, track=2.4, episodes=episodes))
    if path.suffix == ".js":                        # loaded by a script tag, so the page also works opened as a file
        data = f"(window.PPL_EPISODES = window.PPL_EPISODES || {{}})[{json.dumps(path.stem)}] = {data};\n"
    path.write_text(data)
    print(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", type=Path, help="an existing run directory to continue")
    p.add_argument("--gif", type=Path, help="a run directory: write a GIF of its last policy")
    p.add_argument("--export", nargs=2, type=Path, metavar=("RUN", "JSON"),
                   help="write episodes of the last policy of RUN to a .json file, or a .js file for the web player")
    args = p.parse_args()
    if args.gif:
        gif(args.gif)
        return
    if args.export:
        export(*args.export)
        return
    torch.set_num_threads(1)
    out = args.resume or Path("runs") / f"{datetime.now():%y%m%d_%H%M%S}_cartpole_balance"
    print(out, flush=True)
    train(out, seed=args.seed)


if __name__ == "__main__":
    main()
