#!/usr/bin/env python3
"""Prescriptive policy learning (PPL) on Gymnasium Pendulum-v1: swing up and stay upright.

The torque changes the angular velocity directly in one step, so the angular velocity is the key quantity, and two
principles give its reference. The threshold speed at an angle is the speed with which the pendulum would just coast
to the top and stop there, sqrt(3 g / l * (1 - cos theta)) with theta = 0 upright. Each reference is the threshold
speed at the angle the pendulum reaches after this step without torque, so that it describes the moment the next
angular velocity is compared with it.
  Accumulate energy  (where the torque cannot hold the pendulum against gravity):
                     the threshold speed in the direction of the present rotation.
  Stay upright       (within the angle where the largest torque balances gravity, 15 sin(theta) = 6, about 0.41 rad):
                     the threshold speed toward the top, zero only at the top.
Each action is credited by how much its own transition brings the angular velocity closer to the reference,
    delta = 0.5 (w_ref - w)^2 - 0.5 (w_ref - w')^2,
and the policy is updated with this coefficient in place of the return.

Usage:
    python pendulum_swing_up.py                 train (1e5 steps, about ten seconds on a laptop CPU)
    python pendulum_swing_up.py --resume DIR    continue an interrupted run
    python pendulum_swing_up.py --gif DIR       write a GIF of the trained policy
    python pendulum_swing_up.py --export DIR F  write episodes of the trained policy to JSON (data of the web player)
Writes runs/<time>_pendulum_swing_up/: eval.csv (one row per evaluation), models/ (after every update), resume.pt.
Success: the last 2.5 s (50 steps) of an episode within 0.1 rad of upright.
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

G, L, MAX_TORQUE, DT = 10.0, 1.0, 2.0, 0.05            # Pendulum-v1 defaults (mass 1)
GRAVITY = 3 * G / (2 * L)                               # angular acceleration from gravity: GRAVITY * sin(theta)
CAPTURE = np.arcsin(3 / L ** 2 * MAX_TORQUE / GRAVITY)  # the largest torque balances gravity here, about 0.41 rad
EVAL_SEEDS = np.arange(10_000, 10_020)
T_MAX, UPRIGHT, HOLD = 200, 0.1, 50


# ---------------------------------------------------------------------------
# Principles and reference
# ---------------------------------------------------------------------------
def angle(obs):
    return np.arctan2(obs[:, 1], obs[:, 0])


def reference(obs):
    th, w = angle(obs), obs[:, 2]
    nxt = th + (w + GRAVITY * np.sin(th) * DT) * DT    # angle after this step without torque, as the environment integrates
    need = np.sqrt(2 * GRAVITY * (1 - np.cos(nxt)))    # threshold speed there
    toward = np.where(th >= 0, -1.0, 1.0)              # nearer way to the top
    along = np.where(w != 0, np.sign(w), toward)       # present rotation; standing still: the nearer way to the top
    accumulate = along * need
    upright = np.where(nxt >= 0, -1.0, 1.0) * need
    return np.where(np.abs(th) < CAPTURE, upright, accumulate)


def coefficient(s, nx):
    ref = reference(s)                                  # fixed in the state before the action
    return 0.5 * (ref - s[:, 2]) ** 2 - 0.5 * (ref - nx[:, 2]) ** 2, ref - s[:, 2]


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


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, log_std_init=-0.5):
        super().__init__()
        self.mu = mlp(obs_dim, act_dim, out_gain=0.01)
        self.log_std = nn.Parameter(torch.full((act_dim,), log_std_init))

    def dist(self, obs):
        return torch.distributions.Normal(self.mu(obs), self.log_std.exp())


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
    env = gym.make("Pendulum-v1", g=G, render_mode="rgb_array" if render else None)
    return gym.wrappers.RescaleAction(env, -1.0, 1.0)


def evaluate(envs, act):
    obs = np.stack([env.reset(seed=int(s))[0] for env, s in zip(envs, EVAL_SEEDS)])
    th = np.zeros((T_MAX, len(envs)))
    for t in range(T_MAX):
        a = act(obs)
        for i, env in enumerate(envs):
            obs[i] = env.step(a[i])[0]
        th[t] = angle(obs)
    near = np.abs(th) < UPRIGHT
    return dict(success=float(near[-HOLD:].all(0).mean()), reached=float(near.any(0).mean()),
                final_dev=float(np.abs(th[-HOLD:]).mean()))


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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(out, seed=0, n_env=20, horizon=100, iters=50, lr=3e-4, epochs=10, clip=0.2, minibatch=64,
          grad_clip=0.5, eval_every=5):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    envs = [make_env() for _ in range(n_env)]
    envs_eval = [make_env() for _ in EVAL_SEEDS]
    policy, norm = GaussianPolicy(3, 1), RunningNorm(3)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    base = mlp(4, 1)                                    # learned baseline on the state and the deviation
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
            return np.clip(policy.dist(torch.as_tensor(norm(o), dtype=torch.float32)).mean.numpy(), -1, 1)

    for it in range(start, iters + 1):
        if it % eval_every == 0 or it == iters:
            ev = evaluate(envs_eval, act)
            append_row(out / "eval.csv", dict(update=it, steps=it * n_env * horizon, **ev))
            print(f"steps {it * n_env * horizon:6d}  success {ev['success']:.2f}  reached top {ev['reached']:.2f}  "
                  f"final deviation {ev['final_dev']:.3f} rad  ({time.time() - t0:.0f} s)", flush=True)
        if it == iters:
            break
        S, A, S2 = [], [], []
        for _ in range(horizon):
            with torch.no_grad():
                a = policy.dist(torch.as_tensor(norm(obs), dtype=torch.float32)).sample().numpy()
            nxt, before = obs.copy(), obs.copy()
            for i, env in enumerate(envs):
                o, _, term, trunc, _ = env.step(np.clip(a[i], -1, 1))   # the reward is never used
                nxt[i] = o
                obs[i] = env.reset(seed=int(rng.integers(1 << 31)))[0] if term or trunc else o
            S.append(before); A.append(a); S2.append(nxt)
        s, nx, a = np.concatenate(S), np.concatenate(S2), np.concatenate(A)
        norm.update(s)
        c, e = coefficient(s, nx)
        s_t = torch.as_tensor(norm(s), dtype=torch.float32)
        a_t = torch.as_tensor(a, dtype=torch.float32)
        c_t = torch.as_tensor(c[:, None], dtype=torch.float32)
        x = torch.cat([s_t, torch.as_tensor(e[:, None] / 10.0, dtype=torch.float32)], 1)
        scale = c_t.std() + 1e-8
        for _ in range(20):
            bopt.zero_grad()
            ((base(x) - c_t / scale) ** 2).mean().backward()
            bopt.step()
        with torch.no_grad():
            adv = c_t - base(x) * scale
            adv = adv / (adv.std() + 1e-8)
            old_lp = policy.dist(s_t).log_prob(a_t)
        for _ in range(epochs):
            perm = torch.randperm(len(adv))
            for k in range(0, len(adv), minibatch):
                idx = perm[k:k + minibatch]
                ratio = (policy.dist(s_t[idx]).log_prob(a_t[idx]) - old_lp[idx]).exp()
                loss = -torch.min(ratio * adv[idx], ratio.clamp(1 - clip, 1 + clip) * adv[idx]).sum(1).mean()
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
    ck = torch.load(sorted((run / "models").glob("update_*.pt"))[-1], weights_only=False)
    policy, norm = GaussianPolicy(3, 1), RunningNorm(3)
    policy.load_state_dict(ck["policy"])
    norm.mean, norm.var = ck["norm_mean"], ck["norm_var"]
    env = make_env(render=True)
    o, frames = env.reset(seed=seed)[0][None], []
    for _ in range(T_MAX):
        frames.append(env.render())
        with torch.no_grad():
            a = np.clip(policy.dist(torch.as_tensor(norm(o), dtype=torch.float32)).mean.numpy(), -1, 1)
        o = env.step(a[0])[0][None]
    env.close()
    imageio.mimsave(run / "episode.gif", frames[::2], duration=0.1, loop=0)
    print(run / "episode.gif")


def export(run, path):
    """Episodes of the last policy from three starts (lowest, sideways, near the top), for the web player."""
    import json
    ck = torch.load(sorted((run / "models").glob("update_*.pt"))[-1], weights_only=False)
    policy, norm = GaussianPolicy(3, 1), RunningNorm(3)
    policy.load_state_dict(ck["policy"])
    norm.mean, norm.var = ck["norm_mean"], ck["norm_var"]
    env = make_env()
    starts = np.array([abs(angle(env.reset(seed=int(s))[0][None])[0]) for s in EVAL_SEEDS])
    picks = [("Starts low", np.argmax(starts)), ("Starts sideways", np.argmin(np.abs(starts - np.pi / 2))),
             ("Starts near the top", np.argmin(starts))]
    episodes = []
    for label, k in picks:
        o = env.reset(seed=int(EVAL_SEEDS[k]))[0][None]
        th, w, u, rule = [], [], [], []
        for _ in range(T_MAX + 1):
            th.append(round(float(angle(o)[0]), 4)); w.append(round(float(o[0, 2]), 3))
            rule.append(int(abs(th[-1]) < CAPTURE))
            with torch.no_grad():
                a = np.clip(policy.dist(torch.as_tensor(norm(o), dtype=torch.float32)).mean.numpy(), -1, 1)
            u.append(round(float(a[0, 0] * MAX_TORQUE), 3))
            o = env.step(a[0])[0][None]
        episodes.append(dict(label=label, angle=th, velocity=w, torque=u, rule=rule))
    env.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(dict(dt=DT, max_torque=MAX_TORQUE, episodes=episodes))
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
    out = args.resume or Path("runs") / f"{datetime.now():%y%m%d_%H%M%S}_pendulum_swing_up"
    print(out, flush=True)
    train(out, seed=args.seed)


if __name__ == "__main__":
    main()
