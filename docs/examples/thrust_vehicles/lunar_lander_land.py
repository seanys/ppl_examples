#!/usr/bin/env python3
"""Prescriptive policy learning (PPL) on Gymnasium LunarLander-v3: land on the pad from 2 m.

The lander starts 2 m above the pad, close to its centre and nearly upright, descending at about 2 m/s. It has to come
to rest between the flags. The main engine changes the vertical velocity v_y directly in one step and the side engines
change the angular velocity w, so v_y, w and the horizontal velocity v_x are the key quantities. References are computed
from the next state without engines (x' = x + v_x dt, h' = h + v_y dt, theta' = theta + w dt). In flight three
principles hold together:
  Descend safely     v_y toward -sqrt(v_td^2 + 2 a_y h'): the fastest descent that still brakes to v_td = 0.8 m/s at
                     the ground.
  Return to centre   v_x toward sgn(-x') min(sqrt(2 a_x |x'|), |x'| / tau): fast when far, slowing down on arrival.
  Lean as needed     w toward sgn(e) sqrt(2 a_w |e|), e = target tilt - theta': the lander moves sideways only when
                     tilted, so the target tilt follows the remaining horizontal velocity and shrinks to upright before
                     touchdown.
  Settle             once a leg touches, the ground brings the lander to rest, so each reference is the next velocity
                     with the engines off, and an action is credited for leaving them off.
Each action is credited by how much its own transition brings the key quantities closer to their references,
    delta = sum_k 0.5 ((q_ref,k - q_k) / s_k)^2 - 0.5 ((q_ref,k - q'_k) / s_k)^2,
with scales s_k, and the policy is updated with this coefficient in place of the return.

Usage:
    python lunar_lander_land.py                  train until every evaluation episode succeeds (at most 2e5 steps)
    python lunar_lander_land.py --resume DIR     continue an interrupted run
    python lunar_lander_land.py --export DIR F   write episodes of the trained policy to JSON (data of the web player)
Writes runs/<time>_lunar_lander_land/: eval.csv (one row per evaluation), models/ (after every update), resume.pt.
Success: the episode ends with the lander at rest between the flags.
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
from gymnasium.envs.box2d import lunar_lander as ll

EVAL_SEEDS = np.arange(10_000, 10_020)
FPS, DT = ll.FPS, 1 / ll.FPS
HALF_W, HALF_H = ll.VIEWPORT_W / ll.SCALE / 2, ll.VIEWPORT_H / ll.SCALE / 2
SCALE_OBS = np.array([HALF_W, HALF_H, FPS / HALF_W, FPS / HALF_H, 1.0, FPS / 20, 1.0, 1.0])   # observation -> m, m/s, rad
MAX_STEPS, PAD_HALF = 1000, 2.0
# start: height, and ranges of x, tilt, v_x; mean and range of v_y (downward)
START = dict(height=2.0, x=0.5, tilt=0.05, vx=0.3, vy=-2.0, vy_spread=0.3)

# general knowledge of the lander: main engine and gravity (m/s^2), side engine angular and sideways acceleration
A_MAIN, GRAVITY, SIDE_ALPHA, SIDE_ACCEL = 23.83, 9.609, 4.83, 2.4
V_TD = 0.8                                  # touchdown speed (m/s), below the largest speed the lander survives
A_W, A_X = SIDE_ALPHA / 4, SIDE_ACCEL / 4   # each engine fires in half of the steps, half of that is held back
THETA_L = np.arcsin(2 * A_X / A_MAIN)       # largest target tilt: the main engine pushes sideways at 2 a_x
TAU = 1.0                                   # near the centre, 1 m off asks for 1 m/s
FLIGHT_SCALE = np.array([1.0, 0.3, 0.5])    # deviations of 1 m/s in v_y, 0.3 rad/s in w, 0.5 m/s in v_x weigh alike
# engines off on the ground (measured once): vertical and sideways deceleration, and one leg down while descending
A_Y_GROUND, A_X_GROUND, ONE_LEG_DVY, ONE_LEG_DW = 7.05, 1.25, -0.155, 0.17
GROUND_SCALE = np.array([0.2, 0.3, 0.04])


# ---------------------------------------------------------------------------
# Principles and references (key quantities in the order v_y, w, v_x)
# ---------------------------------------------------------------------------
def flight_reference(o):
    x, h, vx, vy, theta, w = o[0], max(o[1], 0.0), o[2], o[3], o[4], o[5]
    x1, h1, theta1 = x + vx * DT, max(h + vy * DT, 0.0), theta + w * DT
    a_y = (0.5 * A_MAIN - GRAVITY) / 2 * max(A_MAIN * np.cos(theta) - GRAVITY, 0.1) / (A_MAIN - GRAVITY)
    vy_ref = -np.sqrt(V_TD ** 2 + 2 * a_y * h1)
    vx_ref = np.sign(-x1) * min(np.sqrt(2 * A_X * abs(x1)), abs(x1) / TAU)
    return_time = 2 * np.sqrt(THETA_L / A_W)
    upright = np.clip(h1 / abs(vy_ref) / (2 * return_time), 0.0, 1.0)   # back to upright before touchdown
    e = THETA_L * np.clip((vx - vx_ref) / 0.5, -1.0, 1.0) * upright - theta1
    return np.array([vy_ref, np.sign(e) * np.sqrt(2 * A_W * abs(e)), vx_ref]), FLIGHT_SCALE


def ground_reference(o, left, right):
    vy, w, vx = o[3], o[5], o[2]
    if left and right:                                      # the ground removes the descent, friction the sideways speed
        return np.array([min(vy + A_Y_GROUND * DT, 0.0) if vy < 0 else 0.0, w,
                         vx - np.sign(vx) * min(abs(vx), A_X_GROUND * DT)]), GROUND_SCALE
    if vy < -0.1:                                           # one leg down: the free side keeps falling and turns down
        return np.array([vy + ONE_LEG_DVY, w + (1.0 if left else -1.0) * ONE_LEG_DW, vx]), GROUND_SCALE
    return np.array([vy, w, vx]), GROUND_SCALE


def coefficient(obs, nxt):
    o, n = obs * SCALE_OBS, nxt * SCALE_OBS
    ref, scale = ground_reference(o, bool(obs[6]), bool(obs[7])) if obs[6] or obs[7] else flight_reference(o)
    q, qn = o[[3, 5, 2]], n[[3, 5, 2]]
    return float((0.5 * ((ref - q) / scale) ** 2 - 0.5 * ((ref - qn) / scale) ** 2).sum())


# ---------------------------------------------------------------------------
# Policy, environment, evaluation, files
# ---------------------------------------------------------------------------
def network(out):
    return nn.Sequential(nn.Linear(8, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, out))


def make_env():
    return gym.make("LunarLander-v3", max_episode_steps=MAX_STEPS)


def reset(env, seed):
    """The environment's reset (terrain, engine noise), then the start state above replaces position and velocity."""
    obs = env.reset(seed=int(seed))[0]
    u = env.unwrapped
    rng = np.random.default_rng(int(seed) + 7_777_777)
    dx = float(rng.uniform(-START["x"], START["x"]) - obs[0] * HALF_W)
    dy = float(START["height"] - obs[1] * HALF_H)
    tilt = float(rng.uniform(-START["tilt"], START["tilt"]))
    v = (float(rng.uniform(-START["vx"], START["vx"])),
         float(START["vy"] + rng.uniform(-START["vy_spread"], START["vy_spread"])))
    for body in [u.lander] + u.legs:
        body.position = (float(body.position.x) + dx, float(body.position.y) + dy)
        body.linearVelocity = v
        body.angularVelocity = 0.0
    u.lander.angle = tilt
    return env.step(0)[0]


def landed(env, term, obs):
    return bool(term and not env.unwrapped.lander.awake and abs(obs[0] * HALF_W) < PAD_HALF)


def evaluate(act):
    envs = [make_env() for _ in EVAL_SEEDS]
    obs = [reset(env, s) for env, s in zip(envs, EVAL_SEEDS)]
    active, success = list(range(len(envs))), np.zeros(len(envs))
    while active:
        actions, left = act(np.array([obs[i] for i in active])), []
        for i, a in zip(active, actions):
            obs[i], _, term, trunc, _ = envs[i].step(int(a))
            if term or trunc:
                success[i] = landed(envs[i], term, obs[i])
            else:
                left.append(i)
        active = left
    for env in envs:
        env.close()
    return dict(success=float(success.mean()))


def collect(policy, seeds):
    """Whole episodes; transitions that end the episode, or in which a leg first touches, carry no coefficient."""
    envs = [make_env() for _ in seeds]
    obs = [reset(env, s) for env, s in zip(envs, seeds)]
    active, O, A, C, steps = list(range(len(envs))), [], [], [], 0
    while active:
        with torch.no_grad():
            p = policy(torch.as_tensor(np.array([obs[i] for i in active]), dtype=torch.float32)).softmax(-1)
            actions = torch.multinomial(p, 1).flatten().tolist()
        left = []
        for i, a in zip(active, actions):
            nxt, _, term, trunc, _ = envs[i].step(a)         # the reward is never used
            steps += 1
            body = bool(envs[i].unwrapped.game_over)
            if not (term or trunc or body) and (obs[i][6] or obs[i][7] or not (nxt[6] or nxt[7])):
                O.append(obs[i]); A.append(a); C.append(coefficient(obs[i], nxt))
            obs[i] = nxt
            if not (term or trunc):
                left.append(i)
        active = left
    for env in envs:
        env.close()
    return np.array(O), np.array(A), np.array(C), steps


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
    policy = network(4)
    policy.load_state_dict(torch.load(sorted((run / "models").glob("update_*.pt"))[-1], weights_only=False)["policy"])

    def act(o):
        with torch.no_grad():
            return policy(torch.as_tensor(o, dtype=torch.float32)).argmax(-1).numpy()
    return act


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(out, seed=0, episodes=4, budget=200_000, lr=3e-4, epochs=30, clip=0.2, minibatch=256, grad_clip=0.5):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    policy, base = network(4), network(1)                   # policy over the four actions; learned baseline
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    bopt = torch.optim.Adam(base.parameters(), lr=1e-3)
    (out / "models").mkdir(parents=True, exist_ok=True)
    steps = it = 0
    if (out / "resume.pt").exists():
        ck = torch.load(out / "resume.pt", weights_only=False)
        policy.load_state_dict(ck["policy"]); opt.load_state_dict(ck["optimizer"])
        base.load_state_dict(ck["baseline"]); bopt.load_state_dict(ck["baseline_optimizer"])
        rng.bit_generator.state = ck["numpy_rng"]; torch.set_rng_state(ck["torch_rng"])
        steps, it = ck["steps"], ck["iteration"]
    t0 = time.time()

    def act(o):
        with torch.no_grad():
            return policy(torch.as_tensor(o, dtype=torch.float32)).argmax(-1).numpy()

    while True:
        ev = evaluate(act)
        append_row(out / "eval.csv", dict(update=it, steps=steps, **ev))
        print(f"steps {steps:6d}  success {ev['success']:.2f}  ({time.time() - t0:.0f} s)", flush=True)
        if ev["success"] == 1.0:
            print("every evaluation episode succeeds: training stops", flush=True)
            break
        if steps >= budget:
            break
        o, a, c, n = collect(policy, rng.integers(100_000, 2 ** 31, size=episodes))
        steps += n
        it += 1
        s_t = torch.as_tensor(o, dtype=torch.float32)
        a_t = torch.as_tensor(a, dtype=torch.long)
        c_t = torch.as_tensor(c, dtype=torch.float32)
        scale = c_t.std(unbiased=False).clamp_min(1e-8)
        for _ in range(20):
            bopt.zero_grad()
            ((base(s_t).flatten() - c_t / scale) ** 2).mean().backward()
            bopt.step()
        with torch.no_grad():
            adv = c_t - base(s_t).flatten() * scale
            adv = adv / adv.std(unbiased=False).clamp_min(1e-8)
            old_lp = policy(s_t).log_softmax(-1).gather(1, a_t[:, None]).flatten()
        for _ in range(epochs):
            perm = torch.randperm(len(adv))
            for k in range(0, len(adv), minibatch):
                idx = perm[k:k + minibatch]
                lp = policy(s_t[idx]).log_softmax(-1).gather(1, a_t[idx, None]).flatten()
                ratio = (lp - old_lp[idx]).exp()
                loss = -torch.min(ratio * adv[idx], ratio.clamp(1 - clip, 1 + clip) * adv[idx]).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
                opt.step()
        save(dict(policy=policy.state_dict()), out / "models" / f"update_{it:04d}.pt")
        save(dict(policy=policy.state_dict(), optimizer=opt.state_dict(), baseline=base.state_dict(),
                  baseline_optimizer=bopt.state_dict(), numpy_rng=rng.bit_generator.state,
                  torch_rng=torch.get_rng_state(), steps=steps, iteration=it), out / "resume.pt")


def export(run, path):
    """Three episodes of the last policy in world coordinates, with the terrain and the flags, for the web player."""
    import json
    act = load(run)
    episodes = []
    for k, seed in enumerate(EVAL_SEEDS[:3]):
        env = make_env()
        o = reset(env, seed)
        u = env.unwrapped
        terrain = [[round(float(p[0][0]), 3), round(float(p[0][1]), 3)] for p in u.sky_polys]
        terrain.append([round(float(u.sky_polys[-1][1][0]), 3), round(float(u.sky_polys[-1][1][1]), 3)])
        x, y, th, actions, rule = [], [], [], [], []
        while True:
            pos = u.lander.position
            x.append(round(float(pos.x), 3)); y.append(round(float(pos.y), 3)); th.append(round(float(u.lander.angle), 4))
            rule.append(int(bool(o[6] or o[7])))
            a = int(act(o[None])[0])
            actions.append(a)
            o, _, term, trunc, _ = env.step(a)
            if term or trunc:
                pos = u.lander.position
                x.append(round(float(pos.x), 3)); y.append(round(float(pos.y), 3)); th.append(round(float(u.lander.angle), 4))
                actions.append(0); rule.append(1)
                break
        episodes.append(dict(label=f"Episode {k + 1}", x=x, y=y, angle=th, action=actions, rule=rule, terrain=terrain,
                             pad=[round(float(u.helipad_x1), 3), round(float(u.helipad_x2), 3), round(float(u.helipad_y), 3)]))
        env.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(dict(kind="lander", dt=DT, width=2 * HALF_W, height=2 * HALF_H, episodes=episodes))
    if path.suffix == ".js":                        # loaded by a script tag, so the page also works opened as a file
        data = f"(window.PPL_EPISODES = window.PPL_EPISODES || {{}})[{json.dumps(path.stem)}] = {data};\n"
    path.write_text(data)
    print(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", type=Path, help="an existing run directory to continue")
    p.add_argument("--export", nargs=2, type=Path, metavar=("RUN", "JSON"),
                   help="write episodes of the last policy of RUN to a .json file, or a .js file for the web player")
    args = p.parse_args()
    if args.export:
        export(*args.export)
        return
    torch.set_num_threads(1)
    out = args.resume or Path("runs") / f"{datetime.now():%y%m%d_%H%M%S}_lunar_lander_land"
    print(out, flush=True)
    train(out, seed=args.seed)


if __name__ == "__main__":
    main()
