#!/usr/bin/env python3
"""Prescriptive policy learning (PPL) on FetchPickAndPlace-v4: pick up the block and place it at the goal with the
move skill alone (the skill of the card "Move the gripper to a target").

The move skill is trained once, for 40,000 steps (as in fetch_move.py, without evaluation). Its principles: the
action moves the gripper's position target by up to 5 cm per step and drives the two fingers, so the key quantities
are the gripper velocity v and the finger opening rate w_dot, each action dimension credited by its own quantity;
references are computed from the next state without action (x' = x + v dt, w' = w + w_dot dt), with
    toward(e, a) = e / |e| min(sqrt(2 a |e|), |e| / tau, V)
  Go to the target           v toward toward(B - x', A_x),
  Keep the fingers open      w_dot toward sgn(e_w) min(sqrt(2 A_w |e_w|), |e_w| / tau), e_w = w_open - w',
and the coefficient delta_k = 0.5 ((q_ref,k - q_k) / s_k)^2 - 0.5 ((q_ref,k - q'_k) / s_k)^2 in place of the return.
It is trained with the block moved out of reach, A and B random in the task region (within 0.2 m horizontally of the
gripper's reset position, from the block-centre height to 0.45 m above the table, without the cells in OUT_OF_REACH).

The trained skill is then used for every motion of the task without further training; the fingers are not part of
the skill and are set directly:
  1. to 5 cm above the block, fingers open, until within 1.5 cm and slower than 2 cm/s (at most 100 steps);
  2. down to the block centre, fingers open, likewise (at most 100 steps);
  3. close the fingers (5 steps, the gripper held at the block centre);
  4. to the goal, fingers closed on the block (100 steps).
The move policy is given its usual closed-finger readings. Success, the environment's: the block centre ends within
5 cm of the goal.

Usage:
    python fetch_pick_and_place.py     train the move skill, then pick and place the block in 200 new episodes
Writes runs/<time>_fetch_pick_and_place/: move/ (models after every update) and episodes.csv (one row per episode).
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
SCALE = np.array([A_X * DT] * 3 + [A_W * DT])
STEPS, EPISODES, BUDGET = 100, 4, 40_000
GOAL_RADIUS = 0.05         # m: FetchReach's success distance
EVAL_SEEDS = range(10000, 10100)
BOX = (np.array([1.142, 0.549, 0.425]), np.array([1.542, 0.949, 0.85]))
GRID = np.array([0.75, 0.20, 0.42])      # origin of the 5 cm cells
OUT_OF_REACH = {(14, 6, 8), (14, 7, 8), (14, 13, 8), (14, 14, 8), (15, 6, 6), (15, 6, 7), (15, 6, 8), (15, 7, 7),
                (15, 7, 8), (15, 8, 7), (15, 8, 8), (15, 9, 7), (15, 9, 8), (15, 10, 7), (15, 10, 8), (15, 11, 7),
                (15, 11, 8), (15, 12, 7), (15, 12, 8), (15, 13, 7), (15, 13, 8), (15, 14, 6), (15, 14, 7), (15, 14, 8)}


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


def drive(env, p, steps):
    """Command the gripper straight to p with the fingers open (sets up the start; not part of learning)."""
    o = None
    for _ in range(steps):
        o, *_ = env.step(np.r_[np.clip((p - state(env)[0]) / 0.05, -1, 1), 1.0])
    return o


def sample(rng):
    while True:
        p = rng.uniform(*BOX)
        if tuple(np.floor((p - GRID) / 0.05).astype(int)) not in OUT_OF_REACH:
            return p


class Episode:
    def __init__(self, env, seed):
        self.env = env
        env.reset(seed=int(seed))
        rng = np.random.default_rng(int(seed) + 7)
        u = env.unwrapped
        u._utils.set_joint_qpos(u.model, u.data, "object0:joint", np.array([3.0, 0.75, 0.03, 1.0, 0.0, 0.0, 0.0]))
        mujoco.mj_forward(u.model, u.data)                       # the block out of reach
        while True:
            a = sample(rng)
            drive(env, a, 60)
            if np.linalg.norm(state(env)[0] - a) < 0.01:
                break
        self.B = sample(rng)
        self.obs = drive(env, state(env)[0], 3)                  # come to rest

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


class Policy(nn.Module):
    def __init__(self, n_in=11):
        super().__init__()
        self.mu = nn.Sequential(nn.Linear(n_in, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 4), nn.Tanh())
        self.log_std = nn.Parameter(torch.full((4,), -1.0))

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


def collect(env, policy, seeds):
    S, A, C, E = [], [], [], []
    for seed in seeds:
        ep = Episode(env, seed)
        for _ in range(STEPS):
            ref = ep.reference()
            x = ep.inputs()
            with torch.no_grad():
                a = policy.dist(torch.as_tensor(x)).sample().numpy()
            q = ep.quantities()
            ep.step(np.clip(a, -1, 1))
            q1 = ep.quantities()
            S.append(x); A.append(a)
            C.append(0.5 * ((ref - q) / SCALE) ** 2 - 0.5 * ((ref - q1) / SCALE) ** 2); E.append(ref - q)
    return np.array(S), np.array(A, dtype=np.float32), np.array(C, dtype=np.float32), np.array(E, dtype=np.float32)


def evaluate(env, policy):
    err = []
    for seed in EVAL_SEEDS:
        ep = Episode(env, seed)
        for _ in range(STEPS):
            with torch.no_grad():
                ep.step(policy.mu(torch.as_tensor(ep.inputs())).numpy())
        err.append(np.linalg.norm(state(env)[0] - ep.B))
    err = np.array(err)
    return dict(success=float(np.mean(err < GOAL_RADIUS)), mean_error_cm=float(100 * err.mean()),
                max_error_cm=float(100 * err.max()))


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


def train(out, seed=0, lr=3e-4, evaluate_each=True):
    """Train the reach policy for BUDGET steps; evaluate before every update unless evaluate_each is False
    (evaluation does not change the training, which is the same either way). Returns the policy."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    env = make_env()
    policy = Policy()
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    bases, bopts = make_bases()
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
    while True:
        if evaluate_each:
            ev = evaluate(env, policy)
            append_row(out / "eval.csv", dict(update=it, steps=steps, **ev))
            print(f"update {it:4d}  steps {steps:6d}  within 5 cm {ev['success']:.2f}  mean distance {ev['mean_error_cm']:.2f} cm"
                  f"  largest {ev['max_error_cm']:.2f} cm  ({time.time() - t0:.0f} s)", flush=True)
        if steps >= BUDGET:
            break
        s, a, c, e = collect(env, policy, rng.integers(100_000, 2 ** 31, size=EPISODES))
        steps += len(c)
        it += 1
        update(policy, opt, bases, bopts, s, a, c, e)
        save(dict(policy=policy.state_dict()), out / "models" / f"update_{it:04d}.pt")
        save(dict(policy=policy.state_dict(), optimizer=opt.state_dict(), baseline=[n.state_dict() for n in bases],
                  baseline_optimizer=[o.state_dict() for o in bopts], numpy_rng=rng.bit_generator.state,
                  torch_rng=torch.get_rng_state(), steps=steps, iteration=it), out / "resume.pt")
    env.close()
    return policy


TEST_SEEDS = range(20000, 20200)


def block(env):
    u = env.unwrapped
    return u._utils.get_site_xpos(u.model, u.data, "object0").copy()


def go(env, move, o, target, fingers, steps, stop):
    """Move the gripper toward target() with the move policy, the fingers set to `fingers` (+1 open, -1 closed)."""
    for _ in range(steps):
        ob = o["observation"]
        xin = np.concatenate([ob[:3], ob[20:23], [0.0, 0.0], target()]).astype(np.float32)   # closed-finger readings
        with torch.no_grad():
            a = move.mu(torch.as_tensor(xin)).numpy()
        a[3] = fingers
        o, *_ = env.step(a)
        x, v, _, _ = state(env)
        if stop and np.linalg.norm(x - target()) < 0.015 and np.linalg.norm(v) < 0.02:
            break
    return o


def pick_and_place(env, move, seed):
    o, _ = env.reset(seed=int(seed))
    goal = env.unwrapped.goal.copy()
    o = go(env, move, o, lambda: block(env) + [0, 0, 0.05], 1.0, 100, True)    # above the block
    o = go(env, move, o, lambda: block(env), 1.0, 100, True)                   # down to the block
    o = go(env, move, o, lambda: block(env), -1.0, 5, False)                   # close the fingers
    o = go(env, move, o, lambda: goal, -1.0, 100, False)                       # to the goal
    d = np.linalg.norm(block(env) - goal)
    return dict(seed=int(seed), block_goal_cm=round(100 * d, 2), placed=int(d < 0.05))


def main():
    torch.set_num_threads(1)
    out = Path("runs") / f"{datetime.now():%y%m%d_%H%M%S}_fetch_pick_and_place"
    print(out, flush=True)
    t0 = time.time()
    move = train(out / "move", evaluate_each=False)
    print(f"move skill trained ({time.time() - t0:.0f} s)", flush=True)
    env = make_env()
    placed = []
    for s in TEST_SEEDS:
        row = pick_and_place(env, move, s)
        append_row(out / "episodes.csv", row)
        placed.append(row["placed"])
    print(f"block placed at the goal in {sum(placed)} of {len(placed)} episodes ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
