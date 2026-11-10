#!/usr/bin/env python3
"""Prescriptive policy learning (PPL) on the Adroit hand (Gymnasium-Robotics, AdroitHandRelocate-v1): move each finger
to a given target.

The actions are absolute joint targets (position actuators, 10 ms per step); each one changes the angular velocity of
its own joint within a step, so these are the key quantities and each action dimension is credited by its own joint.
A finger's target is given by the quantities it can set on its own:
    index, middle, ring   sideways, base bend (MCP), middle bend (PIP)
    little                palm cup, sideways, base bend, middle bend
    thumb                 its five joint angles
The last joint of each of the four fingers (DIP) is tied to the middle bend by a tendon in the model, so its target is
the middle-bend target times the tendon ratio k_c (read from the model). Every joint turns toward its target angle:
    omega_ref = sgn(d) min(sqrt(2 a_plan |d|), |d| / t_settle),   d = target - angle,
fast when far, slowing down on arrival; a_plan is a rough quarter of what the joint gives (k_p * control range /
joint inertia, from the model) and t_settle = 0.05 s. How to get there is learned. Each action is credited by
    delta_k = 0.5 (omega_ref,k - omega_k)^2 - 0.5 (omega_ref,k - omega'_k)^2,
and the policy is updated with this coefficient in place of the return (one learned baseline and one clipped ratio per
action dimension). Three small networks: one shared by the index, middle and ring fingers (same build), one for the
little finger, one for the thumb; each sees only its finger's joint angles and target. Wrist and arm stay still.

Each episode: every finger joint starts at a random angle and gets a random target, both within its range; 50 steps.
Usage:
    python adroit_fingers.py                  train for 100,000 steps
    python adroit_fingers.py --resume DIR     continue an interrupted run
Writes runs/<time>_adroit_fingers/: eval.csv and eval_alone.csv (one row per update), models/ (after every update),
resume.pt. After every update the mean actions are evaluated on fixed seeds 10000-10019: eval.csv with all fingers
moving at once (a finger reaches its target when every joint is within 0.1 rad), eval_alone.csv with one finger moving
at a time and the others held flat (100 targets: reached, median and largest joint error).
"""
import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import mujoco
import numpy as np
import torch
from torch import nn

gym.register_envs(gymnasium_robotics)
FINGERS = {"index": [8, 9, 10, 11], "middle": [12, 13, 14, 15], "ring": [16, 17, 18, 19],
           "little": [20, 21, 22, 23, 24], "thumb": [25, 26, 27, 28, 29]}
NET_OF = {"index": "four", "middle": "four", "ring": "four", "little": "little", "thumb": "thumb"}
DIP = {"index": 11, "middle": 15, "ring": 19, "little": 24}    # tied to the joint before it (PIP)
T_EP, T_SETTLE, TOL = 50, 0.05, 0.1
BATCH, BUDGET = 2000, 100_000
EVAL_SEEDS = range(10000, 10020)


def make_env():
    return gym.make("AdroitHandRelocate-v1")


def coupling_ratio(env):
    """DIP / PIP from the tendon that ties the last two joints of the index finger (the same build in all four)."""
    m = env.unwrapped.model
    t = [i for i in range(m.ntendon) if m.tendon(i).name == "T_FFJ1c"][0]
    adr = m.tendon_adr[t]
    coef = {m.joint(m.wrap_objid[k]).name: m.wrap_prm[k] for k in range(adr, adr + m.tendon_num[t])}
    return -coef["FFJ1"] / coef["FFJ0"]


def plan_deceleration(env):
    """A quarter of what each joint gives: k_p times the control range over the joint inertia at the start pose."""
    u = env.unwrapped
    m, d = u.model, u.data
    d.qpos[:30] = 0.0
    mujoco.mj_forward(m, d)
    M = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, M)
    span = m.actuator_ctrlrange[:, 1] - m.actuator_ctrlrange[:, 0]
    return m.actuator_gainprm[:30, 0] * span / np.diag(M)[:30] / 4


class Episode:
    def __init__(self, env, seed, a_plan, ratio):
        self.env, self.u, self.a_plan = env, env.unwrapped, a_plan
        env.reset(seed=int(seed))
        m, d = self.u.model, self.u.data
        rng = np.random.default_rng(int(seed) + 11)
        J = np.concatenate(list(FINGERS.values()))
        d.qpos[:30] = 0.0
        d.qpos[J] = rng.uniform(m.jnt_range[J, 0], m.jnt_range[J, 1])
        self.target = np.zeros(30)
        self.target[J] = rng.uniform(m.jnt_range[J, 0], m.jnt_range[J, 1])
        for f, j in DIP.items():
            self.target[j] = np.clip(ratio * self.target[j - 1], *m.jnt_range[j])
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        self.hold = (0.0 - self.u.act_mean) / self.u.act_rng       # wrist and arm stay at their start

    def inputs(self, f):
        j = FINGERS[f]
        code = [k for k in j if k != DIP.get(f)]                     # the quantities the finger sets on its own
        return np.concatenate([self.u.data.qpos[j], self.target[code]]).astype(np.float32)

    def quantities(self, f):
        return self.u.data.qvel[FINGERS[f]].copy()

    def reference(self, f):
        j = FINGERS[f]
        d = self.target[j] - self.u.data.qpos[j]
        return np.sign(d) * np.minimum(np.sqrt(2 * self.a_plan[j] * np.abs(d)), np.abs(d) / T_SETTLE)

    def error(self, f):
        j = FINGERS[f]
        return np.abs(self.target[j] - self.u.data.qpos[j])

    def step(self, actions):
        a = self.hold.copy()
        for f, af in actions.items():
            a[FINGERS[f]] = af
        self.env.step(np.clip(a, -1, 1))


class Policy(nn.Module):
    def __init__(self, n, n_in=None):
        super().__init__()
        self.mu = nn.Sequential(nn.Linear(n_in or 2 * n, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, n), nn.Tanh())
        self.log_std = nn.Parameter(torch.full((n,), -1.0))

    def dist(self, x):
        return torch.distributions.Normal(self.mu(x), self.log_std.exp())


def make_nets():
    return {"four": Policy(4, 7), "little": Policy(5, 9), "thumb": Policy(5)}


def make_bases(obs_dim, dims):
    """One learned baseline per action dimension, on (state, the other actions, the dimension's deviation)."""
    nets = [nn.Sequential(nn.Linear(obs_dim + dims, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
            for _ in range(dims)]
    return nets, [torch.optim.Adam(n.parameters(), lr=1e-3) for n in nets]


def coefficient(ref, q, q1):
    return 0.5 * (ref - q) ** 2 - 0.5 * (ref - q1) ** 2


def per_dim_update(policy, opt, bases, bopts, s, a, C, E, epochs=10, clip=0.2, minibatch=256, grad_clip=0.5):
    """Each action dimension is updated by its own coefficient: its own baseline and normalized advantage, its own
    clipped ratio, summed over dimensions."""
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
        old_lp = policy.dist(s_t).log_prob(a_t)
    n = len(adv_t)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for k in range(0, n, minibatch):
            idx = perm[k:k + minibatch]
            ratio = (policy.dist(s_t[idx]).log_prob(a_t[idx]) - old_lp[idx]).exp()
            ab = adv_t[idx]
            loss = -torch.min(ratio * ab, ratio.clamp(1 - clip, 1 + clip) * ab).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            opt.step()


def collect(env, nets, rng, a_plan, ratio):
    data = {k: ([], [], [], []) for k in nets}
    steps = 0
    while steps < BATCH:
        ep = Episode(env, rng.integers(100_000, 2 ** 31), a_plan, ratio)
        for _ in range(T_EP):
            x = {f: ep.inputs(f) for f in FINGERS}
            ref = {f: ep.reference(f) for f in FINGERS}
            q = {f: ep.quantities(f) for f in FINGERS}
            with torch.no_grad():
                act = {f: nets[NET_OF[f]].dist(torch.as_tensor(x[f])).sample().numpy() for f in FINGERS}
            ep.step(act)
            steps += 1
            for f in FINGERS:
                S, A, C, E = data[NET_OF[f]]
                S.append(x[f]); A.append(act[f])
                C.append(coefficient(ref[f], q[f], ep.quantities(f))); E.append(ref[f] - q[f])
    f32 = lambda z: np.array(z, dtype=np.float32)
    return steps, {k: tuple(f32(z) for z in v) for k, v in data.items()}


def evaluate(env, nets, a_plan, ratio):
    """All fingers moving at once."""
    errs = []
    for s in EVAL_SEEDS:
        ep = Episode(env, s, a_plan, ratio)
        for _ in range(T_EP):
            with torch.no_grad():
                ep.step({f: nets[NET_OF[f]].mu(torch.as_tensor(ep.inputs(f))).numpy() for f in FINGERS})
        errs.append({f: ep.error(f) for f in FINGERS})
    row = {f"{f}_reached": float(np.mean([e[f].max() < TOL for e in errs])) for f in FINGERS}
    row["hand_reached"] = float(np.mean([all(e[f].max() < TOL for f in FINGERS) for e in errs]))
    row["mean_err"] = float(np.mean([np.concatenate(list(e.values())).mean() for e in errs]))
    return row


def evaluate_alone(env, nets, a_plan, ratio):
    """One finger moving at a time, the others held flat: 5 fingers x 20 seeds = 100 targets."""
    u = env.unwrapped
    worst = []
    for f, j in FINGERS.items():
        others = [k for g, jj in FINGERS.items() if g != f for k in jj]
        for s in EVAL_SEEDS:
            ep = Episode(env, s, a_plan, ratio)
            u.data.qpos[others] = 0.0; u.data.qvel[:] = 0.0
            mujoco.mj_forward(u.model, u.data)
            flat = (0.0 - u.act_mean[others]) / u.act_rng[others]
            for _ in range(T_EP):
                with torch.no_grad():
                    a_f = nets[NET_OF[f]].mu(torch.as_tensor(ep.inputs(f))).numpy()
                a = ep.hold.copy(); a[others] = flat; a[j] = a_f
                env.step(np.clip(a, -1, 1))
            worst.append(ep.error(f).max())
    worst = np.array(worst)
    return dict(reached=int((worst < TOL).sum()), median_err=float(np.median(worst)), max_err=float(worst.max()))


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


def train(out, seed=0, lr=3e-4):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    env = make_env()
    a_plan, ratio = plan_deceleration(env), coupling_ratio(env)
    nets = make_nets()
    opts = {k: torch.optim.Adam(n.parameters(), lr=lr) for k, n in nets.items()}
    bases = {k: make_bases(n.mu[0].in_features, n.log_std.numel()) for k, n in nets.items()}
    (out / "models").mkdir(parents=True, exist_ok=True)
    steps = it = 0
    if (out / "resume.pt").exists():
        ck = torch.load(out / "resume.pt", weights_only=False)
        for k in nets:
            nets[k].load_state_dict(ck["nets"][k]); opts[k].load_state_dict(ck["opts"][k])
            for n, sd in zip(bases[k][0], ck["bases"][k]): n.load_state_dict(sd)
            for o, sd in zip(bases[k][1], ck["bopts"][k]): o.load_state_dict(sd)
        rng.bit_generator.state = ck["numpy_rng"]; torch.set_rng_state(ck["torch_rng"])
        steps, it = ck["steps"], ck["iteration"]
    else:
        append_row(out / "eval.csv", dict(update=0, steps=0, **evaluate(env, nets, a_plan, ratio)))
        append_row(out / "eval_alone.csv", dict(update=0, steps=0, **evaluate_alone(env, nets, a_plan, ratio)))
    t0 = time.time()
    while steps < BUDGET:
        n_steps, data = collect(env, nets, rng, a_plan, ratio)
        steps += n_steps
        it += 1
        for k, (s, a, c, e) in data.items():
            per_dim_update(nets[k], opts[k], bases[k][0], bases[k][1], s, a, c, e)
        save({k: n.state_dict() for k, n in nets.items()}, out / "models" / f"update_{it:04d}.pt")
        save(dict(nets={k: n.state_dict() for k, n in nets.items()}, opts={k: o.state_dict() for k, o in opts.items()},
                  bases={k: [n.state_dict() for n in b[0]] for k, b in bases.items()},
                  bopts={k: [o.state_dict() for o in b[1]] for k, b in bases.items()},
                  numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), steps=steps, iteration=it),
             out / "resume.pt")
        ev = evaluate(env, nets, a_plan, ratio)
        al = evaluate_alone(env, nets, a_plan, ratio)
        append_row(out / "eval.csv", dict(update=it, steps=steps, **ev))
        append_row(out / "eval_alone.csv", dict(update=it, steps=steps, **al))
        print(f"update {it:3d}  steps {steps:6d}  one finger at a time: {al['reached']}/100 reached, "
              f"median error {al['median_err']:.3f} rad  ({time.time() - t0:.0f} s)", flush=True)
    env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    torch.set_num_threads(1)
    out = args.resume or Path("runs") / f"{datetime.now():%y%m%d_%H%M%S}_adroit_fingers"
    print(out, flush=True)
    train(out, seed=args.seed)


if __name__ == "__main__":
    main()
