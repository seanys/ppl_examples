#!/usr/bin/env python3
"""Grip an upright object with the Adroit hand, learned with Deterministic PPL.

Each episode a cylinder or square column of random width (4-6 cm) and height (13-20 cm) stands before the hand.
Principles: close toward a grip posture set by the object's height and width (each joint's angular velocity toward
its posture angle, fast when far, slow near); hold where a finger touched; once the thumb and two of index, middle
and ring touch, press every touching segment to the target force F_target (2-6 N, drawn per episode).
Deterministic PPL: the policy has no noise; the next-step key quantity (joint angular velocity, segment force) is
predicted by the small networks in models/ (fitted to a quick test of the hand), so the principle coefficient is
differentiated with respect to the action and the policy follows that gradient.

Run: python adroit_grip.py. Trains UPDATES updates (BATCH steps each, 5 grasps per update), then
  out/force.png  : each finger's grip force against the target force, on 20 random objects;
  out/grip_lift.gif : nine objects gripped and lifted 10 cm (0.2 kg), in a 3 x 3 grid.
"""
import csv
import itertools
import multiprocessing as mp
import os
import time
from collections import namedtuple
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn

gym.register_envs(gymnasium_robotics)
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

# ---------------------------------------------------------------- scene and reference posture
FINGERS = {"index": [8, 9, 10, 11], "middle": [12, 13, 14, 15], "ring": [16, 17, 18, 19],
           "little": [20, 21, 22, 23, 24], "thumb": [25, 26, 27, 28, 29]}
JOINTS = np.concatenate(list(FINGERS.values()))
PREFIX = {"index": "ff", "middle": "mf", "ring": "rf", "little": "lf", "thumb": "th"}
ARM = np.arange(8)
R_CYL, H_CYL = 0.03, 0.08
D_ALONG, D_GAP = 0.055, 0.04
D_RAND = 0.01                               # cylinder moved by up to this per horizontal axis per episode
START = np.load(HERE / "models" / "start_posture.npy")
START[FINGERS["thumb"][0]] = -0.45          # thumb base turned away: 2 cm from the cylinder after the hand is moved
POSTURE = np.load(HERE / "models" / "grip_posture.npy")
# numbering from the outside in: joint 1 (fingertip), 2, 3 (at the palm); 0 = outside the numbering (sideways joint,
# little finger's palm joint, thumb base rotations); segments 1 (distal), 2 (middle), 3 (proximal, thumb with its hub)
NUM = {"index": [0, 3, 2, 1], "middle": [0, 3, 2, 1], "ring": [0, 3, 2, 1], "little": [0, 0, 3, 2, 1],
       "thumb": [0, 0, 0, 2, 1]}
SEGS = {f: {1: [p + "distal"], 2: [p + "middle"], 3: [p + "proximal"]} for f, p in PREFIX.items()}
SEGS["thumb"][3] = ["thproximal", "thhub"]
DT, T_EP, T_CHECK = 0.01, 400, 20
D_NEAR, A_FAR = 0.003, 0.1                 # near zone (m); far deceleration as a fraction of a_plan
W_NEAR = {0: 0.1, 1: 0.8, 2: 1.0, 3: 0.1}  # near speed by joint number (rad/s)
F_MIN, F_MAX = 2.0, 6.0                    # range of the target force (N)
OPPOSE = ("index", "middle", "ring")
POS_RANGE = (0.0, 0.8)                     # contact positions sampled for the force model (segment middle to end)
SIGN = np.sign(POSTURE - np.load(HERE / "models" / "start_posture.npy"))    # direction of an increment towards the grip
# section 13: objects standing upright, cylinder or square column; r = radius or half width, h = half height
Obj = namedtuple("Obj", "kind r h")
R_RANGE, H_RANGE = (0.02, 0.03), (0.065, 0.10)         # r from 2 to 3 cm, full height from 13 to 20 cm (centred hand clear of the table)
SHIFT_MAX = 0.01           # section 13: horizontal random shift of the object, D_RAND (3 cm - r) / 1 cm: 1 cm at r = 2 cm, 0 at 3 cm
SIDE = {"index": 0, "middle": 0, "ring": 0, "little": 1}   # sideways joint within the finger
MCP = {"index": 1, "middle": 1, "ring": 1, "little": 2}   # knuckle bending joint within the finger
SPREAD = {"index": 0.44, "middle": 0.15, "ring": -0.15, "little": -0.44}   # sideways joints, fingers fanned apart (rad)
MCP_LESS = 0.35            # the knuckle bends this much less (20 deg) when fully fanned: at 90 deg the sideways axis
                           # lies along the first finger bone and turning it only twists the finger
DQ_MAX = 0.4
NET_OF = {f: f for f in FINGERS}          # one policy network per finger
BATCH, UPDATES, LR, EPOCHS, MINIBATCH, GRAD_CLIP = 2000, 18, 3e-4, 1, 256, 0.5
REUSE = 10       # samples of the last REUSE updates are trained on (the loss uses only the state, section 12.10)
EVAL_SEEDS = range(10000, 10020)
N_WORKERS = min(12, os.cpu_count() or 1)
OMEGA_MODEL = HERE / "models" / "joint_speed.pt"
FORCE_MODEL = HERE / "models" / "finger_force.pt"
THUMB_FORCE_MODEL = HERE / "models" / "thumb_force.pt"
# numbering of the joints that carry a force in the force phase: as NUM, except that the thumb's base rotation THJ4
# and lift THJ3 also press (counted with the proximal segment); THJ2 stays at its limit
FORCE_NUM = {**NUM, "thumb": [3, 3, 0, 2, 1]}
THUMB_MOVED = [0, 1, 3, 4]                # thumb joints whose angle and increment enter its force model

def set_object(m, obj):
    """The object geom becomes obj (standing upright)."""
    g = m.geom("sphere").id
    if obj.kind == "cylinder":
        m.geom_type[g] = mujoco.mjtGeom.mjGEOM_CYLINDER
        m.geom_size[g] = [obj.r, obj.h, 0]
    else:
        m.geom_type[g] = mujoco.mjtGeom.mjGEOM_BOX
        m.geom_size[g] = [obj.r, obj.r, obj.h]
    m.geom_rbound[g] = np.linalg.norm([obj.r, obj.r, obj.h])
    m.geom_aabb[g] = [0, 0, 0, obj.r, obj.r, obj.h]

def posture_for(obj):
    """Reference posture. obj None: the fixed grip posture of section 12. Otherwise (section 13): the four fingers
    fanned by s = min(s_h, s_r), s_h = (h - h_min) / (h_max - h_min), s_r likewise for r, both clipped to [0, 1]:
    sideways joints s SPREAD and the knuckle bent s MCP_LESS less than in section 12 (s = 0, short or thin: the
    fingers side by side as in section 12; s = 1, tall and thick: fanned from the knuckles); the middle and tip joints
    as in section 12; the thumb as close to the four fingers as it can get (thumb_for)."""
    q = POSTURE.copy()
    if obj is None:
        return q
    s = min(np.clip((obj.h - H_RANGE[0]) / (H_RANGE[1] - H_RANGE[0]), 0, 1),
            np.clip((obj.r - R_RANGE[0]) / (R_RANGE[1] - R_RANGE[0]), 0, 1))
    s = round(float(s), 2)
    for f, k in SIDE.items():
        q[FINGERS[f][k]] = s * SPREAD[f]
        q[FINGERS[f][MCP[f]]] = POSTURE[FINGERS[f][MCP[f]]] - s * MCP_LESS
    q[FINGERS["thumb"]] = thumb_for(s)
    return q

_THUMB = {}

def thumb_for(s):
    """Thumb angles for the four-finger posture of spread s: THJ2 at its limit; THJ4, THJ3, THJ1, THJ0 on a grid within
    their ranges, keeping those whose thumb tip is within 1 mm of the closest it gets to any segment of the four
    fingers, and of these the one nearest the section 12 thumb. Geometry only (the object is ignored)."""
    if s in _THUMB:
        return _THUMB[s]
    env = _THUMB.get("env") or make_env()
    _THUMB["env"] = env
    m = env.unwrapped.model
    d = mujoco.MjData(m)
    q = POSTURE.copy()
    for f, k in SIDE.items():
        q[FINGERS[f][k]] = s * SPREAD[f]
        q[FINGERS[f][MCP[f]]] = POSTURE[FINGERS[f][MCP[f]]] - s * MCP_LESS
    j = FINGERS["thumb"]
    tip = m.geom("C_thdistal").id
    four = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name[:2] in ("ff", "mf", "rf", "lf")
            and m.geom_contype[g] + m.geom_conaffinity[g] > 0]
    lo, hi = m.jnt_range[j, 0], m.jnt_range[j, 1]
    grid = [np.linspace(lo[i], hi[i], 9) if i != 2 else [hi[2]] for i in range(5)]
    best = []
    for a in itertools.product(*grid):
        d.qpos[:len(q)] = q
        d.qpos[j] = a
        mujoco.mj_kinematics(m, d)
        best.append((abs(min(mujoco.mj_geomDistance(m, d, tip, g, 0.3, None) for g in four)), a))
    dmin = min(b[0] for b in best)
    close = [np.array(a) for dist, a in best if dist <= dmin + 0.001]
    _THUMB[s] = min(close, key=lambda a: np.linalg.norm(a - POSTURE[j]))
    return _THUMB[s]

def make_env():
    env = gym.make("AdroitHandRelocate-v1")
    m = env.unwrapped.model
    ball = m.body("Object").id
    set_object(m, Obj("cylinder", R_CYL, H_CYL))
    j0 = m.body_jntadr[ball]
    m.dof_damping[m.jnt_dofadr[j0]:m.jnt_dofadr[j0] + 6] = 1e6
    m.dof_damping[ARM] = 1e6
    m.jnt_range[m.joint("ARRz").id] = [-1.6, 1.6]
    m.jnt_range[m.joint("ARTy").id] = [-0.3, 0.2]
    m.actuator_ctrlrange[m.actuator("A_ARRz").id] = [-1.6, 1.6]
    m.site_rgba[m.site("target").id, 3] = 0.0       # the Relocate goal marker, unused here: not drawn
    return env

def place(env, seed=0, q_fingers=START, obj=None):
    """Reset, stand the object (obj None: the cylinder of section 12), turn the hand on its side and move the arm so
    the object sits where it should: the palm point at the object's mid height (the hand centred whatever the height)."""
    env.reset(seed=int(seed))
    m, d = env.unwrapped.model, env.unwrapped.data
    o = obj or Obj("cylinder", R_CYL, H_CYL)
    set_object(m, o)
    ball = m.body("Object").id
    bq = m.jnt_qposadr[m.body_jntadr[ball]]
    d.qpos[:] = 0.0
    d.qpos[bq:bq + 3] = [0.0, 0.0, o.h - 0.035]
    d.qpos[5] = np.pi / 2
    d.qpos[JOINTS] = q_fingers[JOINTS]
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    palm = m.body("palm").id
    pg = m.geom("C_palm0").id
    pm = d.xmat[palm].reshape(3, 3)
    p = d.geom_xpos[pg] + pm[:, 2] * D_ALONG - pm[:, 1] * (m.geom_size[pg][1] + o.r + D_GAP)
    jp = np.zeros((3, m.nv))
    mujoco.mj_jac(m, d, jp, None, p, palm)
    d.qpos[:3] += np.linalg.solve(jp[:, :3], d.xpos[ball] - p)
    # move the arm horizontally so that, in the reference posture, the mean of the five finger pads is on the axis
    d.qpos[JOINTS] = posture_for(obj)[JOINTS]
    mujoco.mj_forward(m, d)
    p = pads(m, d).mean(0)
    mujoco.mj_jac(m, d, jp, None, p, palm)
    shift = d.xpos[ball] - p
    shift[2] = 0.0
    d.qpos[:3] += np.linalg.solve(jp[:, :3], shift)
    d.qpos[JOINTS] = q_fingers[JOINTS]
    mujoco.mj_forward(m, d)

def pads(m, d):
    """Centres of the five finger pads (distal segment collision geoms)."""
    return np.array([d.geom_xpos[m.geom(f"C_{p}distal").id] for p in PREFIX.values()])

def plan_deceleration(m, d):
    """Half of a quarter of what each joint's actuator gives over its whole command span, per unit inertia."""
    d.qpos[:30] = 0.0
    mujoco.mj_forward(m, d)
    M = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, M)
    span = m.actuator_ctrlrange[:30, 1] - m.actuator_ctrlrange[:30, 0]
    return m.actuator_gainprm[:30, 0] * span / np.diag(M)[:30] / 4 / 2

class Hand:
    """One episode: commands joint target angles, reads the force on each segment and the fingertip distance."""

    def __init__(self, env, seed=0, d_rand=D_RAND, only=None, obj=None):
        self.only = only                          # single-finger training: the finger trained, others at START
        self.obj = obj or Obj("cylinder", R_CYL, H_CYL)
        self.posture = posture_for(obj)
        self.env, self.u = env, env.unwrapped
        self.f_target = np.random.default_rng(int(seed) + 31).uniform(F_MIN, F_MAX)
        self.m, self.d = self.u.model, self.u.data
        self.a_plan = plan_deceleration(self.m, self.d)
        place(env, seed, START, obj)
        m = self.m
        self.ball = m.body("Object").id
        bq = m.jnt_qposadr[m.body_jntadr[self.ball]]
        self.d.qpos[bq:bq + 2] += np.random.default_rng(int(seed) + 47).uniform(-d_rand, d_rand, 2)
        mujoco.mj_forward(m, self.d)
        self.q_arm = self.d.qpos[ARM].copy()
        self.pad = {f: m.geom(f"C_{p}distal").id for f, p in PREFIX.items()}
        self.seg_body = {f: {s: {m.body(n).id for n in names} for s, names in SEGS[f].items()} for f in FINGERS}
        self.seg_geom = {f: {s: m.geom("C_" + names[0]).id for s, names in SEGS[f].items()} for f in FINGERS}
        self.seg_joint = {f: {n: FINGERS[f][i] for i, n in enumerate(NUM[f]) if n > 0} for f in FINGERS}
        self.palm = m.body("palm").id
        self.force_on = False
        self.base = {f: np.full(len(j), np.nan) for f, j in FINGERS.items()}   # angle at contact of held/pressing joints
        self.observe()

    def step(self, cmd):
        """cmd: target angles of all finger joints (JOINTS order); the arm is held where it was placed."""
        m, d, u = self.m, self.d, self.u
        a = np.zeros(30)
        ctrl = (d.qfrc_bias[ARM] - m.actuator_biasprm[ARM, 1] * self.q_arm) / m.actuator_gainprm[ARM, 0]
        a[ARM] = (ctrl - u.act_mean[ARM]) / u.act_rng[ARM]
        a[JOINTS] = (cmd - u.act_mean[JOINTS]) / u.act_rng[JOINTS]
        self.env.step(np.clip(a, -1, 1))
        self.observe()

    def observe(self):
        m, d = self.m, self.d
        self.seg_F = {f: {1: 0.0, 2: 0.0, 3: 0.0} for f in FINGERS}
        pos = {f: {1: np.zeros(3), 2: np.zeros(3), 3: np.zeros(3)} for f in FINGERS}
        nrm = {1: np.zeros(3), 2: np.zeros(3), 3: np.zeros(3)}    # thumb: contact normal towards the cylinder
        w = np.zeros(6)
        for c in range(d.ncon):
            con = d.contact[c]
            b1, b2 = m.geom_bodyid[con.geom1], m.geom_bodyid[con.geom2]
            if self.ball not in (b1, b2):
                continue
            other = b2 if b1 == self.ball else b1
            for f, segs in self.seg_body.items():
                for s, bodies in segs.items():
                    if other in bodies:
                        mujoco.mj_contactForce(m, d, c, w)
                        self.seg_F[f][s] += abs(w[0])
                        pos[f][s] += abs(w[0]) * con.pos
                        if f == "thumb":
                            nrm[s] += abs(w[0]) * (con.frame[:3] if b2 == self.ball else -con.frame[:3])
        self.seg_pos = {f: {} for f in FINGERS}           # contact position along the segment, from its middle (0)
        for f in FINGERS:                                 # towards its outer end (1)
            for s, F in self.seg_F[f].items():
                if F > 0 and s in self.seg_joint[f]:
                    g, k = self.seg_geom[f][s], self.seg_joint[f][s]
                    gz = d.geom_xmat[g].reshape(3, 3)[:, 2]
                    out = gz if np.dot(gz, d.geom_xpos[g] - d.xanchor[k]) > 0 else -gz
                    x = np.dot(pos[f][s] / F - d.geom_xpos[g], out) / m.geom_size[g][1]
                    self.seg_pos[f][s] = float(np.clip(x, *POS_RANGE))
        self.thumb_e = {}                                 # per thumb joint: how far it moves the contact point
        jp = np.zeros((3, m.nv))                          # along the normal towards the cylinder (section 12.7)
        for s in (1, 2):
            F = self.seg_F["thumb"][s]
            if F > 0:
                n = nrm[s] / max(np.linalg.norm(nrm[s]), 1e-9)
                mujoco.mj_jac(m, d, jp, None, pos["thumb"][s] / F, next(iter(self.seg_body["thumb"][s])))
                self.thumb_e[s] = np.array([jp[:, m.jnt_dofadr[k]] @ n for k in FINGERS["thumb"]])
        self.touched = {f: any(F > 0 for F in self.seg_F[f].values()) for f in FINGERS}
        if self.only:                                     # single finger: its own contact starts the force phase
            self.force_on = self.force_on or self.touched[self.only]
        else:
            self.force_on = self.force_on or (self.touched["thumb"] and sum(self.touched[g] for g in OPPOSE) >= 2)
        c = d.xpos[self.ball]
        self.d_pad = {f: float(np.hypot(*(d.geom_xpos[g] - c)[:2]) - self.obj.r - m.geom_size[g][0])
                      for f, g in self.pad.items()}

    def inputs(self, f):
        d = self.d
        j = FINGERS[f]
        largest = [max(self.seg_F[g].values()) / self.f_target for g in FINGERS]
        own = [self.seg_F[f][s] / self.f_target for s in (1, 2, 3)]
        centre = d.xmat[self.palm].reshape(3, 3).T @ (d.xpos[self.ball] - d.xpos[self.palm])
        flags = [self.f_target, float(self.touched[f]), float(self.force_on), float(self.d_pad[f] <= D_NEAR)]
        return np.concatenate([d.qpos[j], d.qvel[j], self.posture[j], largest, own, flags, centre]).astype(np.float32)

def omega_reference(h, f):
    """Reference angular velocity of each joint of finger f, from the state at the start of the step.
    Each joint turns towards its angle in the reference posture at v = sqrt(2 a |gap|), gap = posture - angle,
    a = A_FAR a_plan (fast when far from its posture angle, zero at it).
    Not in contact: far (d_pad > D_NEAR) sgn(gap) v; near sgn(gap) min(w_near, v).
    In contact (o = outermost segment in contact): joints beyond it sgn(gap) min(w_near, v); the others 0."""
    d = h.d
    j = FINGERS[f]
    num = np.array(NUM[f])
    w_near = np.array([W_NEAR[n] for n in num])
    gap = h.posture[j] - d.qpos[j]
    v = np.sqrt(2 * A_FAR * h.a_plan[j] * np.abs(gap))
    slow = np.sign(gap) * np.minimum(w_near, v)
    segs = [s for s, F in h.seg_F[f].items() if F > 0]
    if segs:
        return np.where((num > 0) & (num < min(segs)), slow, 0.0)
    if h.d_pad[f] <= D_NEAR:
        return slow
    return np.sign(gap) * v

def references(h, f):
    """Per joint of finger f: held (command kept at the angle it had when held, no learning), assigned to a segment force, or
    angular velocity. Returns the angular-velocity references (NaN where not used), the force entries (segment, mask
    of its joints, force reference) and the mask of held joints.
    In contact before the force phase: the joints at or inside the outermost segment in contact and the joints outside
    the numbering are held. In the force phase: every numbered joint with a segment in contact at or beyond it is
    assigned to the nearest such segment; the joints outside the numbering of a finger in contact are held."""
    num = np.array(NUM[f])
    w_ref = omega_reference(h, f)
    forces = []
    hold = np.zeros(len(num), bool)
    segs = [s for s, F in h.seg_F[f].items() if F > 0]
    if not segs:
        return w_ref, forces, hold
    if not h.force_on:
        hold = (num == 0) | (num >= min(segs))
        w_ref[hold] = np.nan
        return w_ref, forces, hold
    segs = [s for s in segs if s in h.seg_pos[f]]
    num = np.array(FORCE_NUM[f])
    owner = np.array([max([s for s in segs if s <= n], default=0) if n > 0 else 0 for n in num])
    for s in sorted(set(owner) - {0}):
        forces.append((s, owner == s, h.f_target))
    hold = num == 0
    w_ref[(owner > 0) | hold] = np.nan
    return w_ref, forces, hold

def command(h, f, dq, hold, press):
    """Target angles of finger f. Joints held or pressing a segment: the angle at contact (kept from the moment the
    joint became held or pressing) + dq, dq = 0 when held; other joints: angle + dq."""
    j = FINGERS[f]
    q = h.d.qpos[j]
    fixed = hold | press
    h.base[f] = np.where(fixed, np.where(np.isnan(h.base[f]), q, h.base[f]), np.nan)
    return np.where(fixed, h.base[f] + np.where(hold, 0.0, dq), q + dq)

def pressing_mask(forces, n):
    """Joints assigned to a segment force."""
    m = np.zeros(n, bool)
    for _, mask, _ in forces:
        m |= mask
    return m

def mlp(n_in, n_out, out=None):
    layers = [nn.Linear(n_in, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, n_out)]
    return nn.Sequential(*layers, *([out] if out else []))

def make_policy():
    n_in = {k: 3 * len(FINGERS[f]) + 15 for f, k in NET_OF.items()}
    return {k: mlp(n_in[k], len(FINGERS[f]), nn.Tanh()) for f, k in NET_OF.items()}

def act(nets, x, f):
    return DQ_MAX * nets[NET_OF[f]](x)

def pressing(f, s):
    """Indices (within the finger) of the joints between the palm and segment s."""
    return [i for i, n in enumerate(NUM[f]) if n >= s and n > 0]

class OneStep:
    """Section 11 models: omega' per joint; steady force per segment, with increments from -0.2 to 0.4 rad; thumb
    segments: the model of section 12.7 (inputs e, x, thumb angles)."""

    def __init__(self):
        W = torch.load(OMEGA_MODEL, weights_only=False)
        Fm = torch.load(FORCE_MODEL, weights_only=False)
        self.w, self.F = {}, {}
        for key, v in W.items():
            net = mlp(3, 1); net.load_state_dict(v["state"]); net.requires_grad_(False)
            self.w[key] = (net, torch.as_tensor(v["mu"]), torch.as_tensor(v["sd"]))
        Fm = {k: v for k, v in Fm.items() if k[0] != "thumb"}
        Fm.update({("thumb", s): v for s, v in torch.load(THUMB_FORCE_MODEL, weights_only=False).items()})
        for key, v in Fm.items():
            n_in = len(v["mu"])
            assert n_in == (5 + 2 * len(THUMB_MOVED) if key[0] == "thumb" else 2 * len(pressing(*key)) + 1), key
            net = mlp(n_in, 1); net.load_state_dict(v["state"]); net.requires_grad_(False)
            self.F[key] = (net, torch.as_tensor(v["mu"]), torch.as_tensor(v["sd"]), float(v["ys"]))

    def omega(self, f, i, th, w, dq):
        net, mu, sd = self.w[(f, i)]
        return net((torch.stack([th, w, dq], 1) - mu) / sd)[:, 0]

    def force(self, f, s, ang, x, z):
        """z: the contact position along the segment, or for the thumb e of its five joints."""
        net, mu, sd, ys = self.F[(f, s)]
        inp = [z, x[:, THUMB_MOVED], ang[:, THUMB_MOVED]] if f == "thumb" else [ang, x, z]
        return net((torch.cat(inp, 1) - mu) / sd)[:, 0] * ys

_W = {}

def _init_worker():
    torch.set_num_threads(1)
    _W["env"] = make_env()

def _episode(args):
    """One episode with the deterministic policy. collect: per finger the samples for the update; otherwise the
    evaluation over the last T_CHECK steps."""
    states, seed, collect, only, shapes = args
    nets = make_policy()
    for k, n in nets.items():
        n.load_state_dict(states[k])
    if shapes:
        obj = obj_for(seed)
        h = Hand(_W["env"], seed, d_rand=shift_max(obj), only=only, obj=obj)
    else:
        h = Hand(_W["env"], seed, only=only)
    active = [only] if only else list(FINGERS)
    data = {f: dict(x=[], th=[], w=[], w_ref=[], sig=[], fs=[]) for f in active}
    hist = []
    for t in range(T_EP):
        cmd = np.zeros(len(JOINTS))
        k0 = 0
        for f, j in FINGERS.items():
            if f not in active:
                cmd[k0:k0 + len(j)] = START[j]
                k0 += len(j)
                continue
            x = h.inputs(f)
            with torch.no_grad():
                dq = act(nets, torch.as_tensor(x), f).numpy()
            w_ref, forces, hold = references(h, f)
            cmd[k0:k0 + len(j)] = command(h, f, dq, hold, pressing_mask(forces, len(j)))
            k0 += len(j)
            if collect:
                r = data[f]
                r["x"].append(x); r["th"].append(h.d.qpos[j].copy()); r["w"].append(h.d.qvel[j].copy())
                r["w_ref"].append(w_ref); r["sig"].append(h.a_plan[j] * DT)
                for s, mask, F_ref in forces:
                    z = h.thumb_e[s] if f == "thumb" else [h.seg_pos[f][s]]
                    r["fs"].append((t, s, mask, F_ref, z, h.f_target))
        h.step(cmd)
        hist.append([[h.seg_F[f][s] for s in (1, 2, 3)] for f in FINGERS])
    if collect:
        return data
    H = np.array(hist[-T_CHECK:]).sum(2)                  # (T_CHECK, finger) force of each finger
    touch = (H > 0).all(0)
    mean = H.mean(0)
    row = dict(fingers_touching=float(touch.sum()), f_target=h.f_target)
    for i, f in enumerate(FINGERS):
        if f in active:
            row[f"{f}_touch"] = float(touch[i])
            row[f"{f}_force"] = float(mean[i])
            row[f"{f}_error"] = float(abs(mean[i] - h.f_target)) if touch[i] else np.nan   # |force - target| when touching
    return row

def finger_loss(nets, M, f, D, idx):
    """1/2 sum of ((reference - model prediction) / sigma)^2 over the key quantities of the samples idx of finger f."""
    X = torch.as_tensor(D["x"][idx])
    dq = act(nets, X, f)
    loss = torch.zeros(())
    th, w = torch.as_tensor(D["th"][idx]), torch.as_tensor(D["w"][idx])
    ref, sig = torch.as_tensor(D["w_ref"][idx]), torch.as_tensor(D["sig"][idx])
    for i in range(len(FINGERS[f])):
        ok = ~torch.isnan(ref[:, i])
        if ok.any():
            pred = M.omega(f, i, th[ok, i], w[ok, i], dq[ok, i])
            loss = loss + 0.5 * (((ref[ok, i] - pred) / sig[ok, i]) ** 2).sum()
    fs = D["fs"]
    if len(fs["t"]):
        sel = np.isin(fs["t"], idx)
        pos_in = {t: k for k, t in enumerate(idx)}
        rows = np.array([pos_in[t] for t in fs["t"][sel]], dtype=int)
        for s in np.unique(fs["s"][sel]):
            e = np.flatnonzero(fs["s"][sel] == s)
            r = torch.as_tensor(rows[e])
            pr = list(range(5)) if f == "thumb" else pressing(f, s)
            mask = torch.as_tensor(fs["mask"][sel][e][:, pr]).float()
            sign = 1.0 if f == "thumb" else torch.as_tensor(SIGN[np.array(FINGERS[f])[pr]]).float()
            x = dq[r][:, pr] * sign * mask
            pred = M.force(f, s, th[r][:, pr], x, torch.as_tensor(fs["z"][sel][e]))
            ref_f, Ft = torch.as_tensor(fs["ref"][sel][e]), torch.as_tensor(fs["Ft"][sel][e])
            loss = loss + 0.5 * (((ref_f - pred) / Ft) ** 2).sum()
    return loss / len(idx)

def stack(episodes):
    """Per finger: arrays over all collected steps; force entries indexed by their step in the stacked arrays."""
    out = {}
    for f in episodes[0]:
        D = {k: [] for k in ("x", "th", "w", "w_ref", "sig")}
        fs = {k: [] for k in ("t", "s", "mask", "ref", "z", "Ft")}
        n = 0
        for ep in episodes:
            r = ep[f]
            for k in D:
                D[k] += r[k]
            for t, s, mask, F_ref, z, Ft in r["fs"]:
                for k, v in zip(fs, (t + n, s, mask, F_ref, z, Ft)):
                    fs[k].append(v)
            n += len(r["x"])
        D = {k: np.array(v, np.float32) for k, v in D.items()}
        D["fs"] = {k: np.array(v, np.float32 if k in ("ref", "z", "Ft") else None) for k, v in fs.items()}
        D["fs"]["t"] = D["fs"]["t"].astype(int) if len(fs["t"]) else np.zeros(0, int)
        D["fs"]["s"] = D["fs"]["s"].astype(int) if len(fs["t"]) else np.zeros(0, int)
        out[f] = D
    return out

def update(nets, opts, M, data, rng):
    for _ in range(EPOCHS):
        for f in data:
            D = data[f]
            perm = rng.permutation(len(D["x"]))
            for k in range(0, len(perm), MINIBATCH):
                key = NET_OF[f]
                loss = finger_loss(nets, M, f, D, np.sort(perm[k:k + MINIBATCH]))
                opts[key].zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(nets[key].parameters(), GRAD_CLIP)
                opts[key].step()

def evaluate(pool, nets, only=None, shapes=False):
    states = {k: n.state_dict() for k, n in nets.items()}
    rows = pool.map(_episode, [(states, s, False, only, shapes) for s in EVAL_SEEDS])
    return {k: float(np.nanmean([r[k] for r in rows])) if not all(np.isnan(r[k]) for r in rows) else np.nan
            for k in rows[0]}

def obj_for(seed):
    """Section 13 training: the object of an episode, from its seed: cylinder or square column, r uniform in R_RANGE,
    half height uniform in H_RANGE."""
    rng = np.random.default_rng(int(seed) + 61)
    return Obj(("cylinder", "box")[rng.integers(2)], rng.uniform(*R_RANGE), rng.uniform(*H_RANGE))

def shift_max(obj):
    """Largest horizontal random shift per axis for obj (section 13): thin objects move, the thickest stay put."""
    return SHIFT_MAX * float(np.clip((R_RANGE[1] - obj.r) / (R_RANGE[1] - R_RANGE[0]), 0, 1))

def train():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    nets = make_policy()
    opts = {k: torch.optim.Adam(n.parameters(), lr=LR) for k, n in nets.items()}
    M = OneStep()
    pool = mp.get_context("spawn").Pool(N_WORKERS, _init_worker)
    pool_eps, seed, t0 = [], 100, time.time()
    for it in range(1, UPDATES + 1):
        states = {k: n.state_dict() for k, n in nets.items()}
        eps = pool.map(_episode, [(states, seed + e, True, None, True) for e in range(BATCH // T_EP)])
        seed += BATCH // T_EP
        pool_eps = (pool_eps + [eps])[-REUSE:]
        update(nets, opts, M, stack([e for b in pool_eps for e in b]), rng)
        ev = evaluate(pool, nets, None, True)
        print(f"update {it:2d}/{UPDATES}  fingers touching {ev['fingers_touching']:.2f}/5  "
              + " ".join(f"{f} {ev[f + '_error']:.2f}N" for f in FINGERS) + f"  ({time.time() - t0:.0f} s)", flush=True)
    pool.close(); pool.join()
    torch.save({k: n.state_dict() for k, n in nets.items()}, OUT / "policy.pt")
    return nets


# ---------------------------------------------------------------- after training: force, and grip and lift
LIFT, MASS, ARM_DAMPING, UP = 0.10, 0.2, 20.0, 1     # 10 cm up in 1 s; ARTy moves the hand straight up here
T_GRIP, T_LIFT, T_HOLD = 1.0, 1.0, 0.5
SCENES = (10016, 10011, 10001, 10007, 10003, 10006, 10009, 10014, 10018)   # nine of the 20 evaluation objects
SIZE, EVERY = 240, 3


def grip_and_lift(env, nets, seed, t_grip, draw=None):
    """Grip the episode's object for t_grip s as in training, then release it (MASS kg) and raise the arm LIFT m."""
    obj = obj_for(seed)
    h = Hand(env, seed, d_rand=shift_max(obj), obj=obj)
    m, d, ball = h.m, h.d, h.ball
    j0 = m.body_jntadr[ball]
    dof = slice(m.jnt_dofadr[j0], m.jnt_dofadr[j0] + 6)
    mass0 = m.body_mass[ball]
    n_grip, n_lift, n_hold = (int(round(x / DT)) for x in (t_grip, T_LIFT, T_HOLD))
    grip_F, frames = [], []
    for t in range(n_grip + n_lift + n_hold):
        if t == n_grip:
            m.dof_damping[dof] = 0.0
            m.body_mass[ball] = MASS
            m.dof_damping[UP] = ARM_DAMPING
        if n_grip <= t < n_grip + n_lift:
            h.q_arm[UP] += LIFT / n_lift
        cmd = []
        for f, j in FINGERS.items():
            with torch.no_grad():
                dq = act(nets, torch.as_tensor(h.inputs(f)), f).numpy()
            _, forces, hold = references(h, f)
            cmd.append(command(h, f, dq, hold, pressing_mask(forces, len(j))))
        h.step(np.concatenate(cmd))
        if n_grip - T_CHECK <= t < n_grip:
            grip_F.append([sum(h.seg_F[f].values()) for f in FINGERS])
        if draw and t % EVERY == 0:
            frames.append(draw(h))
    m.dof_damping[dof], m.dof_damping[UP], m.body_mass[ball] = 1e6, 1e6, mass0
    return h.f_target, np.array(grip_F).mean(0), frames


def plot_force(nets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    env = make_env()
    rows = [grip_and_lift(env, nets, s, 2.0)[:2] for s in EVAL_SEEDS]
    cols = ("#3b6ea5", "#e07b39", "#5b9e5b", "#8c6bb1", "#c44e52")
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    lo, hi = F_MIN - 0.5, F_MAX + 0.5
    ax.fill_between([lo, hi], [lo - 1, hi - 1], [lo + 1, hi + 1], color="0.92", lw=0, label="target \u00b1 1 N")
    ax.plot([lo, hi], [lo, hi], color="0.4", lw=0.8)
    for k, (f, c) in enumerate(zip(FINGERS, cols)):
        ax.scatter([ft for ft, _ in rows], [F[k] for _, F in rows], s=12, color=c, label=f)
    ax.set_xlabel("Target force (N)"); ax.set_ylabel("Grip force (N)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "force.png", dpi=200)
    err = np.abs(np.array([F for _, F in rows]) - np.array([[ft] for ft, _ in rows]))
    print(f"grip force within 1 N of the target: {100 * (err <= 1).mean():.0f}% of finger forces")


def grid_gif(nets):
    env = make_env()
    m = env.unwrapped.model
    r = mujoco.Renderer(m, SIZE, SIZE)
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0

    def draw(h):
        cam.distance, cam.azimuth, cam.elevation = 0.42, 0, -15
        cam.lookat[:] = h.d.xpos[h.ball]
        r.update_scene(h.d, camera=cam, scene_option=opt)
        return r.render().copy()

    clips = [grip_and_lift(env, nets, s, T_GRIP, draw)[2] for s in SCENES]
    frames = []
    for k in range(min(len(c) for c in clips)):
        rows = [np.concatenate([clips[3 * i + j][k] for j in range(3)], 1) for i in range(3)]
        frames.append(Image.fromarray(np.concatenate(rows, 0)))
    frames[0].save(OUT / "grip_lift.gif", save_all=True, append_images=frames[1:], duration=int(EVERY * DT * 1000), loop=0)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    nets = train()
    plot_force(nets)
    grid_gif(nets)
    print(f"saved {OUT / 'force.png'} and {OUT / 'grip_lift.gif'}")
