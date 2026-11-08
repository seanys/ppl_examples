#!/usr/bin/env python3
"""Prescriptive policy learning (PPL) on Gymnasium LunarLander-v3: land and come to rest on the pad.

The main engine changes the vertical velocity directly in one step and the side engines change the angular velocity,
so the vertical velocity v_y, the angular velocity w and the horizontal velocity v_x are the key quantities. In flight,
three principles give their references from the present state (h: height above the pad, x: offset from its centre):
  Descend safely       v_y toward  -min( sqrt(v_td^2 + 2 a_y h), max(v_td, h / T_x) ): slow down near the ground, and
                       do not reach the ground before the lander is above the pad (T_x: time the approach still needs).
  Approach the pad     v_x toward  -sgn(x) sqrt(2 a_x |x|): move toward the pad, slowing down on arrival.
  Lean as needed       w toward sgn(e) sqrt(2 a_w |e|), e = target tilt - tilt: the lander moves sideways only when
                       tilted, so it tilts only as far as the remaining horizontal velocity needs and is upright again
                       before touchdown.
  Settle               once a leg touches, the ground brings the lander to rest, so each reference is the next velocity
                       with the engines off, and an action is credited for leaving them off.
Each action is credited by how much its own transition brings the key quantities closer to their references,
    delta = sum_k 0.5 ((q_ref,k - q_k) / s_k)^2 - 0.5 ((q_ref,k - q'_k) / s_k)^2,
with scales s_k, and the policy is updated with this coefficient in place of the return. These are the principles and
the update of the paper, without wind.

Usage:
    python lunar_lander_landing.py                  train until every evaluation episode succeeds (at most 2e5 steps)
    python lunar_lander_landing.py --resume DIR     continue an interrupted run
    python lunar_lander_landing.py --gif DIR        write a GIF of the trained policy
    python lunar_lander_landing.py --export DIR F   write episodes of the trained policy to JSON (data of the web player)
Writes runs/<time>_lunar_lander_landing/: eval.csv (one row per evaluation), models/ (after every update), resume.pt.
Success: the episode ends with the lander at rest.
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

# general knowledge of the lander: main engine and gravity (m/s^2), side engine angular and sideways acceleration
A_MAIN, GRAVITY, SIDE_ALPHA, SIDE_ACCEL = 23.83, 9.609, 4.83, 2.4
V_TD = 0.8                                  # touchdown speed (m/s), below the largest speed the lander survives
A_W, A_X = SIDE_ALPHA / 4, SIDE_ACCEL / 4   # each engine fires in half of the steps, half of that is held back
FLIGHT_SCALE = np.array([1.0, 0.3, 0.5])    # deviations of 1 m/s in v_y, 0.3 rad/s in w, 0.5 m/s in v_x weigh alike
# engines off on the ground (measured once): vertical and sideways deceleration, and one leg down while descending
A_Y_GROUND, A_X_GROUND, ONE_LEG_DVY, ONE_LEG_DW = 7.05, 1.25, -0.155, 0.17
GROUND_SCALE = np.array([0.2, 0.3, 0.04])


# ---------------------------------------------------------------------------
# Principles and references (key quantities in the order v_y, w, v_x)
# ---------------------------------------------------------------------------
def flight_reference(o):
    x, h, vx, vy, theta = o[0], max(o[1], 0.0), o[2], o[3], o[4]
    a_y = (0.5 * A_MAIN - GRAVITY) / 2 * max(A_MAIN * np.cos(theta) - GRAVITY, 0.1) / (A_MAIN - GRAVITY)
    if x + vx * abs(vx) / (2 * A_X) > 0:                   # time the horizontal approach still needs
        t_x = (vx + 2 * np.sqrt(vx * vx / 2 + A_X * x)) / A_X
    else:
        t_x = (-vx + 2 * np.sqrt(vx * vx / 2 - A_X * x)) / A_X
    braking = np.sqrt(V_TD ** 2 + 2 * a_y * h)
    waiting = max(V_TD, h / max(t_x, 1e-6))
    vy_ref = -min(braking, waiting)
    vx_ref = -np.sign(x) * np.sqrt(2 * A_X * abs(x))
    theta_l = np.arcsin(2 * A_X / A_MAIN)                  # largest tilt: the main engine pushes sideways at 2 a_x
    return_time = 2 * np.sqrt(theta_l / A_W)
    upright = np.clip(h / abs(vy_ref) / (2 * return_time), 0.0, 1.0)   # back to upright before touchdown
    theta_target = theta_l * np.clip((vx - vx_ref) / 0.5, -1.0, 1.0) * upright
    e = theta_target - theta
    w_ref = np.sign(e) * np.sqrt(2 * A_W * abs(e))
    return np.array([vy_ref, w_ref, vx_ref]), FLIGHT_SCALE


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


def make_env(render=False):
    return gym.make("LunarLander-v3", render_mode="rgb_array" if render else None)


def landed(env, term):
    return bool(term and not env.unwrapped.lander.awake)


def evaluate(act):
    envs = [make_env() for _ in EVAL_SEEDS]
    obs = [env.reset(seed=int(s))[0] for env, s in zip(envs, EVAL_SEEDS)]
    active, success, steps = list(range(len(envs))), np.zeros(len(envs)), np.zeros(len(envs))
    while active:
        actions, left = act(np.array([obs[i] for i in active])), []
        for i, a in zip(active, actions):
            obs[i], _, term, trunc, _ = envs[i].step(int(a))
            steps[i] += 1
            if term or trunc:
                success[i] = landed(envs[i], term)
            else:
                left.append(i)
        active = left
    for env in envs:
        env.close()
    return dict(success=float(success.mean()), mean_steps=float(steps.mean()))


def collect(policy, seeds):
    """Whole episodes; transitions that end the episode, or in which a leg first touches, carry no coefficient."""
    envs = [make_env() for _ in seeds]
    obs = [env.reset(seed=int(s))[0] for env, s in zip(envs, seeds)]
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
        print(f"steps {steps:6d}  success {ev['success']:.2f}  episode length {ev['mean_steps']:.0f}  "
              f"({time.time() - t0:.0f} s)", flush=True)
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


def gif(run, seed=int(EVAL_SEEDS[0])):
    import imageio.v2 as imageio
    act = load(run)
    env = make_env(render=True)
    o, frames = env.reset(seed=seed)[0], []
    while True:
        frames.append(env.render())
        o, _, term, trunc, _ = env.step(int(act(o[None])[0]))
        if term or trunc:
            break
    env.close()
    imageio.mimsave(run / "episode.gif", frames[::2], duration=0.04, loop=0)
    print(run / "episode.gif")


def export(run, path):
    """Three episodes of the last policy in world coordinates, with the terrain, for the web player."""
    import json
    act = load(run)
    episodes = []
    for k, seed in enumerate(EVAL_SEEDS[:3]):
        env = make_env()
        o = env.reset(seed=int(seed))[0]
        u = env.unwrapped
        terrain = [[round(float(p[0][0]), 3), round(float(p[0][1]), 3)] for p in u.sky_polys]
        terrain.append([round(float(u.sky_polys[-1][1][0]), 3), round(float(u.sky_polys[-1][1][1]), 3)])
        x, y, th, act_k, rule = [], [], [], [], []
        while True:
            pos = u.lander.position
            x.append(round(float(pos.x), 3)); y.append(round(float(pos.y), 3)); th.append(round(float(u.lander.angle), 4))
            rule.append(int(bool(o[6] or o[7])))
            a = int(act(o[None])[0])
            act_k.append(a)
            o, _, term, trunc, _ = env.step(a)
            if term or trunc:
                pos = u.lander.position
                x.append(round(float(pos.x), 3)); y.append(round(float(pos.y), 3)); th.append(round(float(u.lander.angle), 4))
                rule.append(1); act_k.append(0)
                break
        episodes.append(dict(label=f"Episode {k + 1}", x=x, y=y, angle=th, action=act_k, rule=rule, terrain=terrain,
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
    out = args.resume or Path("runs") / f"{datetime.now():%y%m%d_%H%M%S}_lunar_lander_landing"
    print(out, flush=True)
    train(out, seed=args.seed)


if __name__ == "__main__":
    main()
