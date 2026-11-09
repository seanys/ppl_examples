#!/usr/bin/env python3
"""Prescriptive policy learning (PPL) on the Fetch robot arm: move the gripper along the straight line from A to B on
the table at a set speed, reaching B at that speed.

The first three actions move the gripper's position target by up to 5 cm per step (dt = 0.04 s) and change the gripper
velocity v within one step, so v is the key quantity, each action dimension credited by its own component. The fingers
are not used (their action is held at 0). One principle: keep pace with a point p(t) = A + u v_target t that leaves A
at the target speed and moves along the line (u = (B - A) / |B - A|, past B without stopping). The reference, at the
next state without action (x' = x + v dt, t' = t + dt), is
    v_ref = u v_target + toward(p(t') - x', A_x),   toward(e, a) = e / |e| min(sqrt(2 a |e|), |e| / tau, V):
move with the target velocity and close any gap to the pace point, whether behind, ahead or off the line. Keeping pace,
the gripper passes B after |B - A| / v_target at v_target, so the average speed and the speed at B both equal v_target.
Each action is credited by delta_k = 0.5 ((v_ref,k - v_k) / s)^2 - 0.5 ((v_ref,k - v'_k) / s)^2 (s = A_x dt), in place of
the return (one learned baseline and one clipped ratio per action dimension).

Episodes: FetchPickAndPlace-v4 with the block moved out of reach; A and B random points on the table at the height of
the block centre (0.425 m), within 0.2 m horizontally of the gripper's reset position and at least 10 cm apart; v_target
uniform in [0.1, 0.4] m/s. The gripper is first commanded to A (not part of learning); the episode ends once it passes
B along the line (at most 200 steps). Policy input: gripper position and velocity, the target velocity u v_target and
the gap p(t) - x. Success: when B is passed, within 2 cm of the line, and both the speed at B (distance moved along the
line in the last step / dt) and the average speed within 20 % of v_target.

Usage:
    python fetch_line.py                  train for 100,000 steps
    python fetch_line.py --resume DIR     continue an interrupted run
Writes runs/<time>_fetch_line/: eval.csv (one row per update), models/ (after every update), resume.pt.
Each update collects at least 2,000 steps; after every update the mean action is evaluated on 100 fixed episodes.
"""
import argparse
import csv
import inspect
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import gymnasium_robotics.utils.mujoco_utils as mu
import mujoco
import numpy as np
import torch
from torch import nn

# MuJoCo >= 3.3 no longer compares joint-type enums with numpy integers inside a tuple; read them as Python ints.
for _name in ("set_joint_qpos", "set_joint_qvel", "get_joint_qpos", "get_joint_qvel"):
    _src = inspect.getsource(getattr(mu, _name))
    _src = _src.replace("joint_type = model.jnt_type[joint_id]", "joint_type = int(model.jnt_type[joint_id])")
    _src = _src.replace("(mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)",
                        "(int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))")
    exec(compile(_src, mu.__file__, "exec"), mu.__dict__)
gym.register_envs(gymnasium_robotics)

DT, TAU = 0.04, 0.2
A_X, V = 4.5, 0.4          # planned acceleration (a quarter of the measured ~18 m/s^2) and top speed (half of ~0.8 m/s)
SCALE = np.full(3, A_X * DT)


def make_env():
    return gym.make("FetchPickAndPlace-v4", max_episode_steps=1000)


def state(env):
    u = env.unwrapped
    ut, m, d = u._utils, u.model, u.data
    x = ut.get_site_xpos(m, d, "robot0:grip").copy()
    v = ut.get_site_xvelp(m, d, "robot0:grip").copy()
    fingers = ("robot0:r_gripper_finger_joint", "robot0:l_gripper_finger_joint")
    w = ut.get_joint_qpos(m, d, fingers[0])[0] + ut.get_joint_qpos(m, d, fingers[1])[0]
    wd = ut.get_joint_qvel(m, d, fingers[0])[0] + ut.get_joint_qvel(m, d, fingers[1])[0]
    return x, v, float(w), float(wd)


def toward(e, a, cap=np.inf):
    n = np.linalg.norm(e)
    return np.zeros_like(e) if n < 1e-9 else e / n * min(np.sqrt(2 * a * n), n / TAU, cap)


V_LOW, V_HIGH = 0.10, 0.40
MIN_LEN, MAX_STEPS = 0.10, 200
TOL_LINE, TOL_SPEED = 0.02, 0.20
BATCH, BUDGET = 2000, 100_000
EVAL_SEEDS = range(10000, 10100)
N_IN = 12
TABLE = (np.array([1.142, 0.549, 0.425]), np.array([1.542, 0.949, 0.425]))   # on the table, block-centre height


def drive(env, p, steps):
    """Command the gripper straight to p (sets up the start; not part of learning)."""
    o = None
    for _ in range(steps):
        o, *_ = env.step(np.r_[np.clip((p - state(env)[0]) / 0.05, -1, 1), 0.0])
    return o


class Episode:
    def __init__(self, env, seed):
        self.env = env
        env.reset(seed=int(seed))
        rng = np.random.default_rng(int(seed) + 7)
        u = env.unwrapped
        u._utils.set_joint_qpos(u.model, u.data, "object0:joint", np.array([3.0, 0.75, 0.03, 1.0, 0.0, 0.0, 0.0]))
        mujoco.mj_forward(u.model, u.data)                       # the block out of reach
        while True:
            a, b = rng.uniform(*TABLE), rng.uniform(*TABLE)
            if np.linalg.norm(b - a) >= MIN_LEN:
                break
        while np.linalg.norm(state(env)[0] - a) >= 0.01:
            drive(env, a, 60)
            if np.linalg.norm(state(env)[0] - a) >= 0.01:
                a = rng.uniform(*TABLE)
        self.A, self.B = state(env)[0].copy(), b
        self.obs = drive(env, self.A, 3)
        self.L = np.linalg.norm(self.B - self.A)
        self.u = (self.B - self.A) / self.L
        self.vt = rng.uniform(V_LOW, V_HIGH)
        self.k = 0                                               # steps since the start
        self.x_prev = self.A.copy()

    def pace(self, k):
        return self.A + self.u * self.vt * k * DT

    def along(self, x):
        s = (x - self.A) @ self.u
        return s, (x - self.A) - s * self.u

    def inputs(self):
        ob = self.obs["observation"]
        x = state(self.env)[0]
        return np.concatenate([ob[:3], ob[20:23], self.u * self.vt, self.pace(self.k) - x]).astype(np.float32)

    def quantities(self):
        return state(self.env)[1]

    def reference(self):
        x, v = state(self.env)[:2]
        return self.u * self.vt + toward(self.pace(self.k + 1) - (x + v * DT), A_X, V)

    def step(self, a3):
        self.x_prev = state(self.env)[0].copy()
        self.obs, *_ = self.env.step(np.r_[np.clip(a3, -1, 1), 0.0])
        self.k += 1

    def passed(self):
        return self.along(state(self.env)[0])[0] >= self.L

    def result(self):
        x = state(self.env)[0]
        speed = (x - self.x_prev) @ self.u / DT
        return np.linalg.norm(self.along(x)[1]), speed / self.vt, self.L / (self.k * DT) / self.vt


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.mu = nn.Sequential(nn.Linear(N_IN, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 3), nn.Tanh())
        self.log_std = nn.Parameter(torch.full((3,), -1.0))

    def dist(self, x):
        return torch.distributions.Normal(self.mu(x), self.log_std.exp())


def make_bases(n_in=11, dims=4):
    nets = [nn.Sequential(nn.Linear(n_in + dims, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
            for _ in range(dims)]
    return nets, [torch.optim.Adam(n.parameters(), lr=1e-3) for n in nets]


def update(policy, opt, bases, bopts, s, a, C, E, epochs=30, clip=0.2, minibatch=256, grad_clip=0.5):
    """Per action dimension: a baseline on (state, the other actions, the dimension's deviation from its reference),
    normalised coefficients, and a clipped ratio; the dimensions' objectives are summed."""
    dims = C.shape[1]
    s_t, a_t = torch.as_tensor(s, dtype=torch.float32), torch.as_tensor(a, dtype=torch.float32)
    a_c = np.clip(a, -1, 1)
    adv = np.zeros_like(C)
    for d in range(dims):
        ct = torch.as_tensor(C[:, d:d + 1], dtype=torch.float32)
        x = torch.cat([s_t, torch.as_tensor(np.delete(a_c, d, axis=1), dtype=torch.float32),
                       torch.as_tensor(E[:, d:d + 1], dtype=torch.float32)], 1)
        cs = ct.std() + 1e-8
        for _ in range(20):
            bopts[d].zero_grad()
            ((bases[d](x) - ct / cs) ** 2).mean().backward()
            bopts[d].step()
        with torch.no_grad():
            b = (bases[d](x) * cs).numpy()[:, 0]
        ad = C[:, d] - b
        adv[:, d] = ad / (ad.std() + 1e-8)
    adv_t = torch.as_tensor(adv, dtype=torch.float32)
    with torch.no_grad():
        old = policy.dist(s_t).log_prob(a_t)
    n = len(adv_t)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for k in range(0, n, minibatch):
            i = perm[k:k + minibatch]
            ratio = (policy.dist(s_t[i]).log_prob(a_t[i]) - old[i]).exp()
            loss = -torch.min(ratio * adv_t[i], ratio.clamp(1 - clip, 1 + clip) * adv_t[i]).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            opt.step()


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


def collect(env, policy, rng):
    S, A, C, E = [], [], [], []
    while len(C) < BATCH:
        ep = Episode(env, rng.integers(100_000, 2 ** 31))
        for _ in range(MAX_STEPS):
            ref, x = ep.reference(), ep.inputs()
            with torch.no_grad():
                a = policy.dist(torch.as_tensor(x)).sample().numpy()
            q = ep.quantities()
            ep.step(a)
            S.append(x); A.append(a)
            C.append(0.5 * ((ref - q) / SCALE) ** 2 - 0.5 * ((ref - ep.quantities()) / SCALE) ** 2); E.append(ref - q)
            if ep.passed():
                break
    f = lambda z: np.array(z, dtype=np.float32)
    return f(S), f(A), f(C), f(E)


def evaluate(env, policy):
    ok, dev, speed, avg = [], [], [], []
    for seed in EVAL_SEEDS:
        ep = Episode(env, seed)
        passed = False
        for _ in range(MAX_STEPS):
            with torch.no_grad():
                ep.step(policy.mu(torch.as_tensor(ep.inputs())).numpy())
            if ep.passed():
                passed = True
                break
        d, r, ra = ep.result()
        ok.append(passed and d < TOL_LINE and abs(r - 1) < TOL_SPEED and abs(ra - 1) < TOL_SPEED)
        dev.append(d); speed.append(r); avg.append(ra)
    return dict(success=float(np.mean(ok)), median_dev_at_B_cm=float(100 * np.median(dev)),
                median_speed_at_B_over_target=float(np.median(speed)), median_average_speed_over_target=float(np.median(avg)))


def train(out, seed=0, lr=3e-4):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    env = make_env()
    policy = Policy()
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    bases, bopts = make_bases(N_IN, dims=3)
    (out / "models").mkdir(parents=True, exist_ok=True)
    steps = it = 0
    if (out / "resume.pt").exists():
        ck = torch.load(out / "resume.pt", weights_only=False)
        policy.load_state_dict(ck["policy"]); opt.load_state_dict(ck["optimizer"])
        for n, sd in zip(bases, ck["baseline"]): n.load_state_dict(sd)
        for o, sd in zip(bopts, ck["baseline_optimizer"]): o.load_state_dict(sd)
        rng.bit_generator.state = ck["numpy_rng"]; torch.set_rng_state(ck["torch_rng"])
        steps, it = ck["steps"], ck["iteration"]
    t0 = time.time()
    while steps < BUDGET:
        s, a, c, e = collect(env, policy, rng)
        steps += len(c)
        it += 1
        update(policy, opt, bases, bopts, s, a, c, e)
        save(dict(policy=policy.state_dict()), out / "models" / f"update_{it:04d}.pt")
        save(dict(policy=policy.state_dict(), optimizer=opt.state_dict(), baseline=[n.state_dict() for n in bases],
                  baseline_optimizer=[o.state_dict() for o in bopts], numpy_rng=rng.bit_generator.state,
                  torch_rng=torch.get_rng_state(), steps=steps, iteration=it), out / "resume.pt")
        ev = evaluate(env, policy)
        append_row(out / "eval.csv", dict(update=it, steps=steps, **ev))
        print(f"update {it:3d}  steps {steps:6d}  success {ev['success']:.2f}  speed at B / target "
              f"{ev['median_speed_at_B_over_target']:.2f}  off the line {ev['median_dev_at_B_cm']:.2f} cm"
              f"  ({time.time() - t0:.0f} s)", flush=True)
    env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    torch.set_num_threads(1)
    out = args.resume or Path("runs") / f"{datetime.now():%y%m%d_%H%M%S}_fetch_line"
    print(out, flush=True)
    train(out)


if __name__ == "__main__":
    main()
