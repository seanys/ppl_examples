#!/usr/bin/env python3
"""Prescriptive policy learning (PPL) on FetchPush-v4 by reusing two skills: push the block straight toward the goal,
re-aiming at every step, and back off and push again whenever it slides off.

Skills (each trained from scratch here, then reused without further training):
  move  "Move the gripper to a target": key quantity the gripper velocity v (and the finger rate), reference
        v -> toward(target - x', A_x), toward(e, a) = e / |e| min(sqrt(2 a |e|), |e| / tau, V); trained for 40,000 steps.
  line  "Move along a line at a set speed": keep pace with a point p(t) leaving A at v_target along the line,
        v -> u v_target + toward(p(t') - x', A_x); trained for 10 updates (about 20,000 steps).
Both credit each action dimension by delta_k = 0.5 ((q_ref,k - q_k) / s_k)^2 - 0.5 ((q_ref,k - q'_k) / s_k)^2 in place of
the return.
Reuse on FetchPush-v4 (fingers locked closed): approach safely behind the block with the move skill (up to 15 cm above
the block, over the start point 6 cm behind it, down), then push with the line skill at 0.1 m/s, its pace point the
contact point 3.5 cm behind the block. If the block slides sideways off the gripper (more than 1.5 cm from the push
axis), approach again and push again. Success, the environment's: the block ends within 5 cm of the
goal.
Push: along the line from the block to the goal, re-aimed at the block at every step, until the block reaches the
goal along the direction it started with.

Usage:
    python fetch_push_retry.py     train both skills, then push the block in 200 new episodes
Writes runs/<time>_fetch_push_retry/: move/ and line/ (models after every update), episodes.csv (one row per episode).
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
A_W, W_OPEN = 12.5, 0.10   # planned finger acceleration and the open width (m)


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


def drive(env, p, steps, fingers):
    """Command the gripper straight to p (sets up a training start; not part of learning)."""
    o = None
    for _ in range(steps):
        o, *_ = env.step(np.r_[np.clip((p - state(env)[0]) / 0.05, -1, 1), fingers])
    return o


# ---- skill 1: move the gripper to a target (as fetch_move.py), trained for 40,000 steps
MOVE_STEPS, MOVE_EPISODES, MOVE_BUDGET = 100, 4, 40_000
MOVE_SCALE = np.array([A_X * DT] * 3 + [A_W * DT])
BOX = (np.array([1.142, 0.549, 0.425]), np.array([1.542, 0.949, 0.85]))
GRID = np.array([0.75, 0.20, 0.42])      # origin of the 5 cm cells
OUT_OF_REACH = {(14, 6, 8), (14, 7, 8), (14, 13, 8), (14, 14, 8), (15, 6, 6), (15, 6, 7), (15, 6, 8), (15, 7, 7),
                (15, 7, 8), (15, 8, 7), (15, 8, 8), (15, 9, 7), (15, 9, 8), (15, 10, 7), (15, 10, 8), (15, 11, 7),
                (15, 11, 8), (15, 12, 7), (15, 12, 8), (15, 13, 7), (15, 13, 8), (15, 14, 6), (15, 14, 7), (15, 14, 8)}

# ---- skill 2: move along a line at a set speed (as fetch_line.py), trained for 10 updates
LINE_V_LOW, LINE_V_HIGH = 0.10, 0.40
LINE_MIN_LEN, LINE_MAX_STEPS, LINE_BATCH, LINE_UPDATES = 0.10, 200, 2000, 10
LINE_SCALE = np.full(3, A_X * DT)
LINE_N_IN = 12
TABLE = (np.array([1.142, 0.549, 0.425]), np.array([1.542, 0.949, 0.425]))   # on the table, block-centre height

def sample(rng):
    while True:
        p = rng.uniform(*BOX)
        if tuple(np.floor((p - GRID) / 0.05).astype(int)) not in OUT_OF_REACH:
            return p


class MoveEpisode:
    def __init__(self, env, seed):
        self.env = env
        env.reset(seed=int(seed))
        rng = np.random.default_rng(int(seed) + 7)
        u = env.unwrapped
        u._utils.set_joint_qpos(u.model, u.data, "object0:joint", np.array([3.0, 0.75, 0.03, 1.0, 0.0, 0.0, 0.0]))
        mujoco.mj_forward(u.model, u.data)                       # the block out of reach
        while True:
            a = sample(rng)
            drive(env, a, 60, 1.0)
            if np.linalg.norm(state(env)[0] - a) < 0.01:
                break
        self.B = sample(rng)
        self.obs = drive(env, state(env)[0], 3, 1.0)                  # come to rest

    def inputs(self):
        ob = self.obs["observation"]       # gripper position, velocity (per step), finger width and rate, target
        return np.concatenate([ob[:3], ob[20:23], [ob[9] + ob[10], ob[23] + ob[24]], self.B]).astype(np.float32)

    def quantities(self):
        x, v, w, wd = state(self.env)
        return np.r_[v, wd]

    def reference(self):
        x, v, w, wd = state(self.env)
        e_w = W_OPEN - (w + wd * DT)
        return np.r_[toward(self.B - (x + v * DT), A_X, V), np.sign(e_w) * min(np.sqrt(2 * A_W * abs(e_w)), abs(e_w) / TAU)]

    def step(self, a):
        self.obs, *_ = self.env.step(a)


class MovePolicy(nn.Module):
    def __init__(self, n_in=11):
        super().__init__()
        self.mu = nn.Sequential(nn.Linear(n_in, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 4), nn.Tanh())
        self.log_std = nn.Parameter(torch.full((4,), -1.0))

    def dist(self, x):
        return torch.distributions.Normal(self.mu(x), self.log_std.exp())


def move_collect(env, policy, seeds):
    S, A, C, E = [], [], [], []
    for seed in seeds:
        ep = MoveEpisode(env, seed)
        for _ in range(MOVE_STEPS):
            ref = ep.reference()
            x = ep.inputs()
            with torch.no_grad():
                a = policy.dist(torch.as_tensor(x)).sample().numpy()
            q = ep.quantities()
            ep.step(np.clip(a, -1, 1))
            q1 = ep.quantities()
            S.append(x); A.append(a)
            C.append(0.5 * ((ref - q) / MOVE_SCALE) ** 2 - 0.5 * ((ref - q1) / MOVE_SCALE) ** 2); E.append(ref - q)
    return np.array(S), np.array(A, dtype=np.float32), np.array(C, dtype=np.float32), np.array(E, dtype=np.float32)


class LineEpisode:
    def __init__(self, env, seed):
        self.env = env
        env.reset(seed=int(seed))
        rng = np.random.default_rng(int(seed) + 7)
        u = env.unwrapped
        u._utils.set_joint_qpos(u.model, u.data, "object0:joint", np.array([3.0, 0.75, 0.03, 1.0, 0.0, 0.0, 0.0]))
        mujoco.mj_forward(u.model, u.data)                       # the block out of reach
        while True:
            a, b = rng.uniform(*TABLE), rng.uniform(*TABLE)
            if np.linalg.norm(b - a) >= LINE_MIN_LEN:
                break
        while np.linalg.norm(state(env)[0] - a) >= 0.01:
            drive(env, a, 60, 0.0)
            if np.linalg.norm(state(env)[0] - a) >= 0.01:
                a = rng.uniform(*TABLE)
        self.A, self.B = state(env)[0].copy(), b
        self.obs = drive(env, self.A, 3, 0.0)
        self.L = np.linalg.norm(self.B - self.A)
        self.u = (self.B - self.A) / self.L
        self.vt = rng.uniform(LINE_V_LOW, LINE_V_HIGH)
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


class LinePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.mu = nn.Sequential(nn.Linear(LINE_N_IN, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 3), nn.Tanh())
        self.log_std = nn.Parameter(torch.full((3,), -1.0))

    def dist(self, x):
        return torch.distributions.Normal(self.mu(x), self.log_std.exp())


def line_collect(env, policy, rng):
    S, A, C, E = [], [], [], []
    while len(C) < LINE_BATCH:
        ep = LineEpisode(env, rng.integers(100_000, 2 ** 31))
        for _ in range(LINE_MAX_STEPS):
            ref, x = ep.reference(), ep.inputs()
            with torch.no_grad():
                a = policy.dist(torch.as_tensor(x)).sample().numpy()
            q = ep.quantities()
            ep.step(a)
            S.append(x); A.append(a)
            C.append(0.5 * ((ref - q) / LINE_SCALE) ** 2 - 0.5 * ((ref - ep.quantities()) / LINE_SCALE) ** 2); E.append(ref - q)
            if ep.passed():
                break
    f = lambda z: np.array(z, dtype=np.float32)
    return f(S), f(A), f(C), f(E)


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


def train_skill(make_policy, make_bases_n, collect, stop, out, seed=0, lr=3e-4):
    """Train one skill from scratch (its own seeds), saving the model after every update; stop(steps, it) ends it."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    env = make_env()
    policy = make_policy()
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    bases, bopts = make_bases_n()
    out.mkdir(parents=True, exist_ok=True)
    steps = it = 0
    while not stop(steps, it):
        s, a, c, e = collect(env, policy, rng)
        steps += len(c)
        it += 1
        update(policy, opt, bases, bopts, s, a, c, e)
        save(dict(policy=policy.state_dict()), out / f"update_{it:04d}.pt")
    env.close()
    print(f"{out.name} trained: {it} updates, {steps} steps", flush=True)
    return policy


def train_move(out):
    return train_skill(lambda: MovePolicy(11), lambda: make_bases(11, 4),
                       lambda env, pol, rng: move_collect(env, pol, rng.integers(100_000, 2 ** 31, size=MOVE_EPISODES)),
                       lambda steps, it: steps >= MOVE_BUDGET, out)


def train_line(out):
    return train_skill(LinePolicy, lambda: make_bases(LINE_N_IN, 3), line_collect,
                       lambda steps, it: it >= LINE_UPDATES, out)


# ---- reusing the skills on FetchPush-v4 (fingers locked closed)
D_START = 0.06       # m behind the block centre where a push starts
D_CONTACT = 0.035    # m from the gripper to the block centre while pushing
V_PUSH = 0.10        # m/s
APPROACH = 0.15      # m above the block for the approach (above the reset height)
SLIP = 0.015         # m: block centre this far sideways from the push axis = slid off the gripper
MAX_TRIES = 1000     # no practical limit: back off and push again whenever the block slides
TEST_SEEDS = range(20000, 20200)


def go(env, move, o, target, steps, stop):
    """Move skill toward target() for up to `steps` steps (stop: once within 1.5 cm and slower than 2 cm/s)."""
    for _ in range(steps):
        ob = o["observation"]
        xin = np.concatenate([ob[:3], ob[20:23], [0.0, 0.0], target()]).astype(np.float32)   # closed-finger readings
        with torch.no_grad():
            a = move.mu(torch.as_tensor(xin)).numpy()
        o, *_ = env.step(np.r_[a[:3], -1.0])
        x, v = state(env)[:2]
        if stop and np.linalg.norm(x - target()) < 0.015 and np.linalg.norm(v) < 0.02:
            break
    return o


def block_goal(env):
    u = env.unwrapped
    return u._utils.get_site_xpos(u.model, u.data, "object0").copy(), u.goal.copy()


def push_step(env, line, o, u):
    """One step of the line skill pushing along u: its pace point is the contact point behind the block."""
    x = state(env)[0]
    block, _ = block_goal(env)
    pace = np.r_[block[:2] - D_CONTACT * u[:2], block[2]]
    ob = o["observation"]
    xin = np.concatenate([ob[:3], ob[20:23], u * V_PUSH, pace - x]).astype(np.float32)
    with torch.no_grad():
        a = line.mu(torch.as_tensor(xin)).numpy()
    o, *_ = env.step(np.r_[np.clip(a, -1, 1), -1.0])
    return o


def slid(env, u):
    x = state(env)[0]
    r = block_goal(env)[0][:2] - x[:2]
    return abs(u[0] * r[1] - u[1] * r[0]) > SLIP


def approach(env, move, o, u):
    """Safely behind the block: up, over the start point S = block - D_START u, down."""
    x, block = state(env)[0], block_goal(env)[0]
    if x[2] < block[2] + APPROACH - 0.02:
        up = np.r_[x[:2], block[2] + APPROACH]
        o = go(env, move, o, lambda: up, 100, True)
    block = block_goal(env)[0]
    S = np.r_[block[:2] - D_START * u[:2], block[2]]
    o = go(env, move, o, lambda: S + [0, 0, APPROACH], 100, True)
    return go(env, move, o, lambda: S, 100, True)


def push(env, move, line, seed):
    """Push along the line from the block to the goal, re-aimed at every step; back off and push again on a slide."""
    o, _ = env.reset(seed=int(seed))
    block, goal = block_goal(env)
    u = np.r_[(goal - block)[:2] / max(np.linalg.norm((goal - block)[:2]), 1e-9), 0.0]
    o = approach(env, move, o, u)
    retries = 0
    for attempt in range(MAX_TRIES):
        block, goal = block_goal(env)
        u0 = np.r_[(goal - block)[:2] / max(np.linalg.norm((goal - block)[:2]), 1e-9), 0.0]
        slipped = False
        for k in range(LINE_MAX_STEPS):
            block, goal = block_goal(env)
            d = goal[:2] - block[:2]
            if d @ u0[:2] < 0.005:
                break
            u = np.r_[d / np.linalg.norm(d), 0.0]
            if slid(env, u):
                slipped = True
                break
            o = push_step(env, line, o, u)
        if not slipped:
            break
        retries += 1
        block, goal = block_goal(env)
        o = approach(env, move, o, np.r_[(goal - block)[:2] / max(np.linalg.norm((goal - block)[:2]), 1e-9), 0.0])
    hold = state(env)[0].copy()
    o = go(env, move, o, lambda: hold, 25, False)
    block, goal = block_goal(env)
    d = np.linalg.norm(block - goal)
    return dict(seed=int(seed), retries=retries, block_goal_cm=round(100 * d, 2), success=int(d < 0.05))


def main():
    torch.set_num_threads(1)
    out = Path("runs") / f"{datetime.now():%y%m%d_%H%M%S}_fetch_push_retry"
    print(out, flush=True)
    t0 = time.time()
    move = train_move(out / "move")
    line = train_line(out / "line")
    env = gym.make("FetchPush-v4", max_episode_steps=10_000)
    ok = []
    for s in TEST_SEEDS:
        row = push(env, move, line, s)
        append_row(out / "episodes.csv", row)
        ok.append(row["success"])
    print(f"block pushed to the goal in {sum(ok)} of {len(ok)} episodes ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
