#!/usr/bin/env python3
"""Prescriptive policy learning (PPL) on the gym-pybullet-drones hover task (HoverAviary, Crazyflie 2.X).

The quadrotor starts at rest 0.1 m above the ground and has to fly to the point 1 m above its start and stay there.
The four motor commands change the vertical velocity v_z (all together) and the angular velocities w_x, w_y
(one side against the other) within one step, so these are the key quantities. The horizontal velocity changes only
through the tilt of the thrust axis, so the horizontal goal enters through a target tilt. Three principles hold
together in every state, with references computed from the next state without action (d' = d - v dt, n' = n + n_dot dt;
d is the target minus the position, n the horizontal part of the thrust axis). Positions are approached in one form,
    toward(e, a) = sgn(e) min(sqrt(2 a |e|), |e| / tau):
fast when far, slowing down on arrival.
  Climb to the target        v_z toward toward(d_z', a_z).
  Lean toward the target     target tilt n~ = sin(theta_l) clip((u toward(rho', a_h) - v_h) / v_c), the lean the remaining
                             horizontal velocity still needs (rho' horizontal distance, u its direction).
  Turn to that lean          n_dot toward (n~ - n') / |n~ - n'| sqrt(2 a_w |n~ - n'|), expressed as w_x, w_y.
Each action is credited by how much its own transition brings the key quantities closer to their references,
    delta = sum_k 0.5 ((q_ref,k - q_k) / s_k)^2 - 0.5 ((q_ref,k - q'_k) / s_k)^2,
with s_k the change one step of planned effort makes, and the policy is updated with this coefficient in place of the return.

Usage:
    python quadrotor_hover.py                  train until the evaluation episode succeeds (at most 3e5 steps)
    python quadrotor_hover.py --resume DIR     continue an interrupted run
    python quadrotor_hover.py --export DIR F   write the episode of the trained policy (data of the web player)
Writes runs/<time>_quadrotor_hover/: eval.csv (one row per evaluation), models/ (after every update), resume.pt.
Success: the full 8 s without leaving the box or tipping over, and every step of the last 2 s within 0.1 m of the target.
The environment has no randomness, so one evaluation episode with the mean action is enough.
"""
import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

G = 9.8
DT = 1 / 30
TARGET = np.array([0.0, 0.0, 1.0])
LAST = 60                 # steps in the last 2 s
SUCCESS_DIST = 0.1        # m

# Capability (Crazyflie model: m 0.027 kg, motor lever 0.028 m, I_xx 1.4e-5):
#   all motors +1: thrust (1.05)^2 = 1.1025 weight -> +1.00 m/s^2; two sides opposite -> 53 rad/s^2.
# Each motor serves the thrust and the rotation, half to each, and half of that is held back.
A_M, ALPHA_M = 1.0, 53.0
A_Z = A_M / 4
A_W = ALPHA_M / 4
F_HALF = (1 + 0.05 * 0.5) ** 2                              # thrust / weight with half of the common command
THETA_L = float(np.arccos((1 + F_HALF) / (2 * F_HALF)))     # holding height at theta_l uses half of the upward margin
A_H = 0.5 * G * np.tan(THETA_L)                             # half the horizontal acceleration at theta_l
TAU = 1.0
V_C = A_H * TAU                                             # speed difference that asks for the full target tilt
SCALE = np.array([A_Z * DT, A_W * DT, A_W * DT])            # v_z, w_x, w_y: planned change in one step


def make_env():
    return HoverAviary(obs=ObservationType.KIN, act=ActionType.RPM)


def state(env):
    """pos, rpy, vel (world), angular velocity (world), thrust axis n (world)."""
    s = env._getDroneStateVector(0)
    pos, quat, rpy, vel, w = s[0:3], s[3:7], s[7:10], s[10:13], s[13:16]
    import pybullet as p
    n = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)[:, 2]
    return pos, rpy, vel, w, n


def observe(env):
    pos, rpy, vel, w, _ = state(env)
    return np.concatenate([pos - TARGET, rpy, vel, w]).astype(np.float32)


def toward(e, a):
    return np.sign(e) * min(np.sqrt(2 * a * abs(e)), abs(e) / TAU)


def references(env):
    """References of (v_z, w_x, w_y) at the current state."""
    pos, _, vel, w, n = state(env)
    d1 = TARGET - (pos + vel * DT)
    vz_ref = toward(d1[2], A_Z)
    rho = np.linalg.norm(d1[:2])
    vh_ref = d1[:2] / max(rho, 1e-9) * toward(rho, A_H)
    t = (vh_ref - vel[:2]) / V_C
    t = t / max(1.0, np.linalg.norm(t))
    n_tgt = np.sin(THETA_L) * t
    ndot = np.array([w[1] * n[2] - w[2] * n[1], w[2] * n[0] - w[0] * n[2]])
    n1 = n[:2] + ndot * DT
    e = n_tgt - n1
    m = np.linalg.norm(e)
    ndot_ref = e / max(m, 1e-9) * np.sqrt(2 * A_W * m)
    nz = max(n[2], 1e-3)
    wy_ref = (ndot_ref[0] + w[2] * n[1]) / nz
    wx_ref = (w[2] * n[0] - ndot_ref[1]) / nz
    return np.array([vz_ref, wx_ref, wy_ref])


def quantities(env):
    _, _, vel, w, _ = state(env)
    return np.array([vel[2], w[0], w[1]])


def coefficient(ref, q, q1):
    return float((0.5 * ((ref - q) / SCALE) ** 2 - 0.5 * ((ref - q1) / SCALE) ** 2).sum())


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.mu = nn.Sequential(nn.Linear(12, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 4), nn.Tanh())
        self.log_std = nn.Parameter(torch.full((4,), -1.0))

    def dist(self, o):
        return torch.distributions.Normal(self.mu(o), self.log_std.exp())


def baseline_net():
    return nn.Sequential(nn.Linear(12, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))


def run_episode(env, act, record=False):
    env.reset()
    o = observe(env)
    dist, traj, end = [], [], "time"
    while True:
        a = act(o)
        pos, rpy, *_ = state(env)
        if record:
            traj.append(np.concatenate([pos, rpy]))
        _, _, term, trunc, _ = env.step(np.clip(a, -1, 1).reshape(1, 4))
        o = observe(env)
        dist.append(float(np.linalg.norm(o[:3])))
        if term or trunc:
            if env.step_counter / env.PYB_FREQ <= env.EPISODE_LEN_SEC:
                end = "early"
            break
    last = np.array(dist[-LAST:])
    ok = end == "time" and len(last) == LAST and bool((last < SUCCESS_DIST).all())
    return dict(success=float(ok), end=end, ep_steps=len(dist), last_dist=float(last.mean()),
                final_dist=dist[-1]), np.array(traj)


def collect(env, policy, episodes):
    obs, acts, coefs = [], [], []
    for _ in range(episodes):
        env.reset()
        o = observe(env)
        while True:
            with torch.no_grad():
                a = policy.dist(torch.as_tensor(o)).sample().numpy()
            ref, q = references(env), quantities(env)
            _, _, term, trunc, _ = env.step(np.clip(a, -1, 1).reshape(1, 4))
            obs.append(o); acts.append(a); coefs.append(coefficient(ref, q, quantities(env)))
            o = observe(env)
            if term or trunc:
                break
    return np.array(obs), np.array(acts), np.array(coefs, dtype=np.float32)


def append_row(path, row):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def save(obj, path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def train(out, seed=0, episodes=4, budget=300_000, lr=3e-4, epochs=30, clip=0.2, minibatch=256, grad_clip=0.5):
    torch.manual_seed(seed)
    policy, base = Policy(), baseline_net()
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    bopt = torch.optim.Adam(base.parameters(), lr=1e-3)
    (out / "models").mkdir(parents=True, exist_ok=True)
    steps = it = 0
    if (out / "resume.pt").exists():
        ck = torch.load(out / "resume.pt", weights_only=False)
        policy.load_state_dict(ck["policy"]); opt.load_state_dict(ck["optimizer"])
        base.load_state_dict(ck["baseline"]); bopt.load_state_dict(ck["baseline_optimizer"])
        torch.set_rng_state(ck["torch_rng"]); steps, it = ck["steps"], ck["iteration"]
    env = make_env()
    t0 = time.time()

    def act(o):
        with torch.no_grad():
            return policy.mu(torch.as_tensor(o)).numpy()

    while True:
        ev, _ = run_episode(env, act)
        append_row(out / "eval.csv", dict(update=it, steps=steps, **ev))
        print(f"update {it:4d}  steps {steps:6d}  success {ev['success']:.0f}  end {ev['end']}  "
              f"last-2s dist {ev['last_dist']:.3f}  ({time.time() - t0:.0f} s)", flush=True)
        if ev["success"] == 1.0 or steps >= budget:
            break
        o, a, c = collect(env, policy, episodes)
        steps += len(c)
        it += 1
        s_t, a_t, c_t = (torch.as_tensor(x) for x in (o, a, c))
        scale = c_t.std(unbiased=False).clamp_min(1e-8)
        for _ in range(20):
            bopt.zero_grad()
            ((base(s_t).flatten() - c_t / scale) ** 2).mean().backward()
            bopt.step()
        with torch.no_grad():
            adv = c_t - base(s_t).flatten() * scale
            adv = adv / adv.std(unbiased=False).clamp_min(1e-8)
            old_lp = policy.dist(s_t).log_prob(a_t).sum(-1)
        for _ in range(epochs):
            perm = torch.randperm(len(adv))
            for k in range(0, len(adv), minibatch):
                idx = perm[k:k + minibatch]
                ratio = (policy.dist(s_t[idx]).log_prob(a_t[idx]).sum(-1) - old_lp[idx]).exp()
                loss = -torch.min(ratio * adv[idx], ratio.clamp(1 - clip, 1 + clip) * adv[idx]).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
                opt.step()
        save(dict(policy=policy.state_dict()), out / "models" / f"update_{it:04d}.pt")
        save(dict(policy=policy.state_dict(), optimizer=opt.state_dict(), baseline=base.state_dict(),
                  baseline_optimizer=bopt.state_dict(), torch_rng=torch.get_rng_state(),
                  steps=steps, iteration=it), out / "resume.pt")
    env.close()


def export(run, path):
    """The episode of the last policy (position and attitude) for the web player."""
    import json
    policy = Policy()
    policy.load_state_dict(torch.load(sorted((run / "models").glob("update_*.pt"))[-1])["policy"])
    env = make_env()
    env.reset()
    o = observe(env)
    x, y, z, roll, pitch = [], [], [], [], []
    while True:
        pos, rpy, *_ = state(env)
        x.append(round(float(pos[0]), 4)); y.append(round(float(pos[1]), 4)); z.append(round(float(pos[2]), 4))
        roll.append(round(float(rpy[0]), 4)); pitch.append(round(float(rpy[1]), 4))
        with torch.no_grad():
            a = policy.mu(torch.as_tensor(o)).numpy()
        _, _, term, trunc, _ = env.step(np.clip(a, -1, 1).reshape(1, 4))
        o = observe(env)
        if term or trunc:
            break
    env.close()
    episode = dict(label="Episode 1", x=x, y=y, z=z, roll=roll, pitch=pitch, angle=pitch, rule=[0] * len(x))
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(dict(kind="quadrotor", dt=DT, target=TARGET.tolist(), episodes=[episode]))
    if path.suffix == ".js":                        # loaded by a script tag, so the page also works opened as a file
        data = f"(window.PPL_EPISODES = window.PPL_EPISODES || {{}})[{json.dumps(path.stem)}] = {data};\n"
    path.write_text(data)
    print(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", type=Path, help="an existing run directory to continue")
    p.add_argument("--export", nargs=2, type=Path, metavar=("RUN", "JSON"),
                   help="write the episode of the last policy of RUN to a .json file, or a .js file for the web player")
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.export:
        export(*args.export)
        return
    out = args.resume or Path("runs") / f"{datetime.now():%y%m%d_%H%M%S}_quadrotor_hover"
    print(out, flush=True)
    train(out, seed=args.seed)


if __name__ == "__main__":
    main()
