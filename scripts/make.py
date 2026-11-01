# -*- coding: utf-8 -*-
"""七宗罪 · 粒子人像卡点视频生成器（18s / 16:9 / 90 BPM 节拍网格）。
usage: python3 make.py <sin_key> [--preview t1,t2,...] [--out path]
"""
import os
import sys
import math
import time
import subprocess
import numpy as np
import cv2

import geom
import render as R
import textfx as T
import fx2d as FX
from sins_data import SINS, FOOTNOTE

W, H = R.W, R.H
FPS = 30
DUR = 18.0
NF = int(round(DUR * FPS))
SCR = os.environ.get("SINS_WORK", os.path.join(os.getcwd(), "work"))  # cache + previews + default output
os.makedirs(SCR, exist_ok=True)
CACHE = os.path.join(SCR, "cache")
os.makedirs(CACHE, exist_ok=True)

# ---- fixed beat grid (~90.25 BPM) ----
BEAT = 0.6648
B0 = -0.055


def beat(n):
    return B0 + BEAT * n


T_PUSH, T_C, T_D, T_G1, T_G2, T_F = beat(3), beat(4), beat(8), beat(12), beat(13), beat(16)
T_WORDS = [beat(n) for n in range(19, 25)]
T_PULSE = beat(25)
T_FADE0, T_FADE1 = 17.25, 18.0

ORDER = ["pride", "envy", "wrath", "sloth", "greed", "gluttony", "lust"]

# ---- music-driven FX ----
import audio_fx
import sfx
AF = audio_fx.load(SCR)  # audio_fx.npz from your BGM, else synthetic beat-grid envelopes
T_PREDROP = 4.40


def af(name, t):
    a = AF[name]
    i = t * FPS
    i0 = int(np.clip(math.floor(i), 0, len(a) - 1)); i1 = min(i0 + 1, len(a) - 1)
    f = i - math.floor(i)
    return float(a[i0] * (1 - f) + a[i1] * f)


def sec(t):
    """section intensity: intro / riser / breakdown / DROP / transformation / title"""
    if t < 2.0:
        return 0.25
    if t < T_PREDROP:
        return 0.45
    if t < T_D:
        return 0.2
    if t < T_G1:
        return 1.45
    if t < T_F:
        return 1.25
    return 1.15


D_CUTS = [beat(9), beat(10), beat(11)]
# (time, roll, yaw, pitch, is_snap) — head-only rotation, snapped on kicks
TW_KEYS = [(0.0, 0, 0, 0, 0), (3.97, 0.22, 0.10, 0.0, 1), (4.62, -0.24, -0.12, 0.05, 1), (5.06, 0.0, 0.0, 0.0, 0),
           (9.18, 0.32, 0.05, -0.06, 1), (9.93, -0.20, 0.16, 0.0, 1), (10.42, 0.0, 0.0, 0.0, 0),
           (13.32, 0.16, -0.08, 0.0, 1), (14.00, -0.14, 0.10, 0.0, 1), (14.66, 0.0, 0.0, 0.0, 0),
           (16.18, 0.20, 0.0, 0.0, 1), (16.58, -0.12, 0.0, 0.0, 1), (17.00, 0.0, 0.0, 0.0, 0)]


def twist(t):
    k = 0
    for i, kk in enumerate(TW_KEYS):
        if t >= kk[0]:
            k = i
    if k == 0:
        return 0.0, 0.0, 0.0
    p, c = TW_KEYS[k - 1], TW_KEYS[k]
    u = clamp01((t - c[0]) / (0.13 if c[4] else 0.30))
    if c[4]:
        c1 = 2.6; c3 = c1 + 1
        e = 1 + c3 * (u - 1) ** 3 + c1 * (u - 1) ** 2
    else:
        e = u * u * (3 - 2 * u)
    return tuple(p[j] + (c[j] - p[j]) * e for j in (1, 2, 3))
GLINTS = [(T_D + 0.04, 1.3), (6.12, 0.9), (6.76, 0.9), (beat(11) + 0.02, 1.3), (7.74, 0.9),
          (T_G2 + 0.35, 1.1), (T_F + 0.28, 1.1), (12.06, 0.9), (14.38, 0.9), (beat(25), 1.2), (17.2, 0.8)]
EYE_GLINT = np.array([[-0.26, 0.055, 0.765]], np.float32)
HITS = [(T_C, 0.45), (T_D, 1.0), (T_G1, 0.6), (T_G2, 0.95), (T_F, 0.85)] + [(tw, 0.5) for tw in T_WORDS] + \
       [(tc, 0.6) for tc in D_CUTS] + [(kk[0], 0.4) for kk in TW_KEYS if kk[4]]
SHOCKS = [(T_D, 1.0), (T_G2, 0.9), (T_F, 0.7)]


def hit(t, tau=0.22, min_s=0.0):
    m = 0.0
    for th, s_ in HITS:
        if s_ < min_s:
            continue
        d = t - th
        if d >= -0.5 / FPS:
            m = max(m, s_ * math.exp(-max(d, 0) / tau))
    return m


def blast(t, th, amp, tau=0.13):
    d = t - th
    if d < 0 or d > 1.5:
        return 0.0
    return amp * (math.exp(-d / tau) + 0.12 * math.exp(-d / 0.35) * math.sin(d * 34))
SIN = {s["key"]: s for s in SINS}


# ------------------------------------------------------------------ utils
def clamp01(x):
    return min(1.0, max(0.0, x))


def ease_io(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def ease_out(x, p=3):
    x = clamp01(x)
    return 1 - (1 - x) ** p


def ease_in(x, p=3):
    x = clamp01(x)
    return x ** p


def lerp(a, b, u):
    return np.asarray(a, float) * (1 - u) + np.asarray(b, float) * u


def env(t, t0, a=0.03, d=0.25):
    """attack/decay pulse around t0"""
    if t < t0 - a or t > t0 + d:
        return 0.0
    if t < t0:
        return (t - (t0 - a)) / a
    return (1 - (t - t0) / d) ** 2


# ------------------------------------------------------------------ clouds
def cloud(name, sdf, n, bbox, cell):
    f = os.path.join(CACHE, name + ".npz")
    if os.path.exists(f):
        d = np.load(f)
        return d["P"].astype(np.float32), d["N"].astype(np.float32)
    t = time.time()
    P, N = geom.sample_surface(sdf, n, bbox)
    P, N = geom.thin_uniform(P, N, cell)
    np.savez(f, P=P.astype(np.float32), N=N.astype(np.float32))
    print(f"[cloud] {name}: {len(P)} pts in {time.time() - t:.0f}s", flush=True)
    return P.astype(np.float32), N.astype(np.float32)


BUST_BOX = ((-1.3, -2.4, -1.1), (1.3, 1.25, 1.1))
FACE_BOX = ((-0.95, -1.05, -0.15), (0.95, 0.95, 1.05))


def get_bust(mouth):
    tag = {0.0: "0", 0.35: "35", 1.0: "100"}[mouth]
    return cloud(f"bust{tag}", lambda p: geom.head_sdf(p, mouth, True), 1_000_000, BUST_BOX, 0.0085)


def get_face():
    return cloud("face0", lambda p: geom.head_sdf(p, 0.0, False), 1_500_000, FACE_BOX, 0.0046)


# ------------------------------------------------------------------ pose
PIVOT = np.array([0, -0.95, -0.1], np.float32)


def pose_mat(pitch=0.0, yaw=0.0, roll=0.0):
    return (R.rot_y(yaw) @ R.rot_x(pitch) @ R.rot_z(roll)).astype(np.float32)


def head_w(P):
    return np.clip((P[:, 1] + 1.40) / 0.50, 0, 1).astype(np.float32)


def apply_pose(P, N, M, w=None, offset=(0, 0, 0)):
    Q = (P - PIVOT) @ M.T + PIVOT
    if w is None:
        P2 = Q
        N2 = N @ M.T if N is not None else None
    else:
        P2 = P + (Q - P) * w[:, None]
        if N is not None:
            NQ = N @ M.T
            N2 = N + (NQ - N) * w[:, None]
            N2 /= np.linalg.norm(N2, axis=1, keepdims=True) + 1e-6
        else:
            N2 = None
    return P2 + np.asarray(offset, np.float32), N2


# ------------------------------------------------------------------ attribute geometry (per sin)
class Attr:
    def __init__(self, P, N=None, w=None, t=None, a=None, attach=True, kind="static", data=None, lit=True):
        self.P = P.astype(np.float32)
        self.N = None if N is None else N.astype(np.float32)
        n = len(P)
        self.w = np.zeros(n, np.float32) if w is None else np.broadcast_to(np.asarray(w, np.float32), (n,)).copy()
        self.t = np.zeros(n, np.float32) if t is None else np.broadcast_to(np.asarray(t, np.float32), (n,)).copy()
        self.a = np.zeros(n, np.float32) if a is None else np.broadcast_to(np.asarray(a, np.float32), (n,)).copy()
        self.attach = attach
        self.kind = kind
        self.data = data or {}
        self.lit = lit
        rr = np.random.default_rng(n)
        self.scatter = rr.normal(0, 1, (n, 3)).astype(np.float32)
        self.scatter /= np.linalg.norm(self.scatter, axis=1, keepdims=True) + 1e-6
        self.scatter *= (2.0 + 3.0 * rr.random((n, 1))).astype(np.float32)
        self.delay = rr.random(n).astype(np.float32)

    def subset(self, frac, seed=0):
        rr = np.random.default_rng(seed)
        m = rr.random(len(self.P)) < frac
        b = Attr.__new__(Attr)
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray) and len(v) == len(self.P):
                setattr(b, k, v[m])
            else:
                setattr(b, k, v)
        b.data = dict(self.data)
        for k, v in self.data.items():
            if isinstance(v, np.ndarray) and len(v) == len(self.P):
                b.data[k] = v[m]
        return b


def halo_attr(kind, R0=1.32, z=-0.95, n=26000, cy=0.15):
    rr = np.random.default_rng(42)
    if kind == "ring":
        th = rr.random(n) * 2 * np.pi
        r = R0 + rr.normal(0, 0.010, n)
        P = np.stack([r * np.cos(th), cy + r * np.sin(th), np.full(n, z)], 1)
    elif kind == "crescent":
        th = rr.random(n * 2) * 2 * np.pi
        r = R0 * np.sqrt(0.80 + 0.20 * rr.random(n * 2))
        x = r * np.cos(th); y = r * np.sin(th)
        keep = (x + R0 * 0.42) ** 2 + (y - R0 * 0.18) ** 2 > (R0 * 0.88) ** 2
        x, y = x[keep][:n], y[keep][:n]
        P = np.stack([x, cy + y, np.full(len(x), z)], 1)
    elif kind == "heart":
        th = rr.random(n) * 2 * np.pi
        hx = 16 * np.sin(th) ** 3
        hy = 13 * np.cos(th) - 5 * np.cos(2 * th) - 2 * np.cos(3 * th) - np.cos(4 * th)
        s = R0 / 15.5
        P = np.stack([hx * s + rr.normal(0, 0.01, n), cy + 0.12 + hy * s + rr.normal(0, 0.01, n), np.full(n, z)], 1)
    elif kind == "ouroboros":
        th = rr.random(n) * 2 * np.pi
        # tube radius varies: thick at head (th~0), thin at tail (th~2pi)
        u = th / (2 * np.pi)
        tr = 0.012 + 0.07 * (1 - u) ** 0.7
        phi = rr.random(n) * 2 * np.pi
        r = R0 + tr * np.cos(phi)
        zz = z + tr * np.sin(phi)
        P = np.stack([r * np.cos(th + 1.2), cy + r * np.sin(th + 1.2), zz], 1)
        # head blob
        m = n // 8
        q = rr.normal(0, 1, (m, 3)); q /= np.linalg.norm(q, axis=1, keepdims=True)
        hc = np.array([R0 * np.cos(1.2), cy + R0 * np.sin(1.2), z])
        P = np.concatenate([P, hc + q * np.array([0.13, 0.09, 0.08])])
    elif kind == "fire":
        th = rr.random(n) * 2 * np.pi
        r = R0 + rr.normal(0, 0.012, n)
        P = np.stack([r * np.cos(th), cy + r * np.sin(th), np.full(n, z)], 1)
    elif kind == "coins":
        th = rr.random(n) * 2 * np.pi
        r = R0 + rr.normal(0, 0.012, n)
        P = np.stack([r * np.cos(th), cy + r * np.sin(th), np.full(n, z)], 1)
    return Attr(P, None, w=0.10, t=1.0, attach=False, kind="halo", data={"halo": kind}, lit=False)


def build_attrs(key):
    A = []
    if key == "pride":
        P, N, J = geom.imperial_crown()
        A.append(Attr(P, N, w=0.30, a=0.95))
        A.append(Attr(J, None, w=1.2, a=2.4, lit=False))
        (Ps, Ns), Pr, Pe = geom.collar_parts(30000)
        A.append(Attr(Ps, Ns, w=0.06, t=0.32, attach=False))
        A.append(Attr(Pr, None, w=0.12, t=0.32, attach=False, lit=False))
        A.append(Attr(Pe, None, w=0.25, t=0.6, a=0.12, attach=False, lit=False))
        ch, med, star = geom.necklace()
        A.append(Attr(ch, None, w=0.55, a=1.3, attach=False, lit=False))
        A.append(Attr(med, None, w=0.7, a=1.1, attach=False, lit=False))
        A.append(Attr(star, None, w=1.4, a=1.6, attach=False, lit=False))
        A.append(halo_attr("ring"))
    elif key == "envy":
        P, N = geom.serpent(42000)
        A.append(Attr(P, N, w=0.25, t=0.9, attach=False, kind="serpent"))
        A.append(halo_attr("ouroboros", n=30000))
    elif key == "wrath":
        P, N = geom.horns(30000)
        A.append(Attr(P, N, w=0.45, t=0.6, a=0.3))
        rr = np.random.default_rng(9)
        n = 26000
        # flame emitters: shoulders + crown of head
        src = np.concatenate([
            np.stack([rr.uniform(-1.1, 1.1, n // 2), rr.uniform(-1.75, -1.55, n // 2), rr.uniform(-0.4, 0.3, n // 2)], 1),
            np.stack([rr.uniform(-0.55, 0.55, n // 2), rr.uniform(0.75, 1.0, n // 2), rr.uniform(-0.5, 0.3, n // 2)], 1),
        ])
        A.append(Attr(src, None, w=0.15, t=0.9, a=0.8, attach=False, kind="flame",
                      data={"phase": rr.random(n).astype(np.float32), "spd": (0.8 + 0.8 * rr.random(n)).astype(np.float32)},
                      lit=False))
        A.append(halo_attr("fire"))
    elif key == "sloth":
        A.append(halo_attr("crescent"))
    elif key == "greed":
        cp, cn_, cid = geom.coins(40, 420)
        A.append(Attr(cp, cn_, w=0.30, a=1.2, attach=False, kind="coins", data={"cid": cid}))
        # necklace beads
        rr = np.random.default_rng(4)
        beads = []
        for k in range(26):
            th = -np.pi * 0.95 + k / 25 * np.pi * 0.95 * 2
            c = np.array([0.62 * np.sin(th), -1.42 - 0.22 * np.cos(th * 0.5) ** 2 + 0.0, 0.05 + 0.48 * np.cos(th)])
            q = rr.normal(0, 1, (160, 3)); q /= np.linalg.norm(q, axis=1, keepdims=True)
            beads.append(c + q * 0.045)
        A.append(Attr(np.concatenate(beads), None, w=0.5, a=1.2, attach=False, lit=False))
        A.append(halo_attr("coins"))
    elif key == "gluttony":
        rr = np.random.default_rng(6)
        n = 30000
        A.append(Attr(np.zeros((n, 3)), None, w=0.35, t=0.4, a=1.0, attach=True, kind="vortex",
                      data={"phase": rr.random(n).astype(np.float32), "ang": (rr.random(n) * 2 * np.pi).astype(np.float32)},
                      lit=False))
        A.append(halo_attr("ring"))
    elif key == "lust":
        pp, pid = geom.petals(70, 300)
        rr = np.random.default_rng(8)
        A.append(Attr(pp, None, w=0.25, t=1.1, attach=False, kind="petals",
                      data={"pid": pid, "ph": (rr.random(70) * 2 * np.pi).astype(np.float32),
                            "rad": (1.05 + 0.5 * rr.random(70)).astype(np.float32),
                            "tilt": (rr.normal(0, 0.5, 70)).astype(np.float32),
                            "yo": (rr.uniform(-1.2, 0.9, 70)).astype(np.float32)}, lit=False))
        A.append(halo_attr("heart"))
    return A


def mask_attr(a, m):
    b = Attr.__new__(Attr)
    n = len(a.P)
    for k, v in a.__dict__.items():
        setattr(b, k, v[m] if isinstance(v, np.ndarray) and len(v) == n else v)
    b.data = {k: (v[m] if isinstance(v, np.ndarray) and len(v) == n else v) for k, v in a.data.items()}
    return b


def lineup_attr(a):
    if a.kind == "flame":
        return mask_attr(a, a.P[:, 1] > 0)
    if a.kind in ("static", "serpent"):
        return mask_attr(a, a.P[:, 1] > -1.5)
    return a


def animate_attr(a, t, M, hw_offset, intro=1.0):
    """returns (P, N) for attribute at time t (figure local coords, pose applied if attached)."""
    P = a.P; N = a.N
    if a.kind == "flame":
        ph = (a.data["phase"] + t * 0.55 * a.data["spd"]) % 1.0
        up = ph * 1.3
        P = a.P.copy()
        P[:, 1] += up
        P[:, 0] += 0.12 * np.sin(t * 3 + a.data["phase"] * 30) * ph
        P[:, 2] += 0.08 * np.cos(t * 2.3 + a.data["phase"] * 17) * ph
        a._fade = (np.sin(np.pi * ph) ** 1.5).astype(np.float32)
    elif a.kind == "serpent":
        P = a.P.copy()
        P[:, 0] += 0.025 * np.sin(t * 2.0 + a.P[:, 1] * 7)
        P[:, 2] += 0.025 * np.cos(t * 2.0 + a.P[:, 1] * 7)
    elif a.kind == "coins":
        cid = a.data["cid"]
        k = cid.astype(np.float32)
        ringi = (cid % 2).astype(np.float32)
        nper = 20
        th = (k // 2) / nper * 2 * np.pi + t * (0.45 + 0.15 * ringi) * (1 - 2 * ringi)
        Rr = 1.30 + 0.32 * ringi
        tilt = 0.35 - 0.7 * ringi
        cx = Rr * np.sin(th); cz = Rr * np.cos(th) - 0.1
        cy = 0.15 + cz * np.sin(tilt) * 0.5 + 0.25 * np.sin(th * 2 + ringi)
        spin = t * 3.0 + k
        lx = a.P[:, 0] * np.cos(spin); lz = a.P[:, 0] * np.sin(spin); ly = a.P[:, 1]
        P = np.stack([cx + lx, cy + ly, cz + lz], 1).astype(np.float32)
        N = np.stack([np.sin(spin + 0 * cx) * 0 + np.cos(spin), 0 * cx, np.sin(spin)], 1).astype(np.float32)
        # depth fade for coins behind head
    elif a.kind == "vortex":
        ph = (a.data["phase"] - t * 0.45) % 1.0  # 1 far -> 0 mouth
        s = ph
        ang = a.data["ang"] + s * 9.0 + t * 2.0
        r = 0.04 + 1.25 * s ** 1.4
        mouth = np.array([0, -0.66, 0.80], np.float32)
        P = np.stack([mouth[0] + r * np.cos(ang), mouth[1] + r * np.sin(ang) * 0.8,
                      mouth[2] + 1.6 * s ** 1.2 - 0.05], 1).astype(np.float32)
        a._fade = (np.clip(s * 6, 0, 1) * np.clip((1 - s) * 3, 0, 1)).astype(np.float32)
    elif a.kind == "petals":
        pid = a.data["pid"]
        ph = a.data["ph"][pid]; rad = a.data["rad"][pid]; tilt = a.data["tilt"][pid]; yo = a.data["yo"][pid]
        th = ph + t * 0.5
        cx = rad * np.sin(th); cz = rad * np.cos(th) * 0.8
        cy = yo + 0.25 * np.sin(th * 2 + ph) + cz * tilt * 0.3
        spin = t * 1.7 + ph * 3
        lx = a.P[:, 0] * np.cos(spin) - a.P[:, 1] * np.sin(spin)
        ly = a.P[:, 0] * np.sin(spin) + a.P[:, 1] * np.cos(spin)
        lz = a.P[:, 2]
        P = np.stack([cx + lx, cy + ly, cz + lz], 1).astype(np.float32)
    elif a.kind == "halo":
        P = a.P.copy()
        if a.data["halo"] == "fire":
            th = np.arctan2(P[:, 1] - 0.15, P[:, 0])
            fl = 0.08 * (0.5 + 0.5 * np.sin(th * 14 + t * 9)) * (0.5 + 0.5 * np.sin(th * 5 - t * 4))
            rr_ = np.hypot(P[:, 0], P[:, 1] - 0.15)
            sc = (rr_ + fl) / (rr_ + 1e-6)
            P[:, 0] = P[:, 0] * sc; P[:, 1] = 0.15 + (P[:, 1] - 0.15) * sc
        elif a.data["halo"] in ("ring", "coins", "ouroboros", "crescent", "heart"):
            ang = t * 0.12
            c, s_ = math.cos(ang), math.sin(ang)
            if a.data["halo"] in ("ring", "coins", "ouroboros"):
                x = P[:, 0].copy(); y = P[:, 1] - 0.15
                P[:, 0] = x * c - y * s_; P[:, 1] = 0.15 + x * s_ + y * c
        sc = intro
        P[:, 0] *= sc; P[:, 1] = 0.15 + (P[:, 1] - 0.15) * sc
    if a.attach and M is not None:
        P, N = apply_pose(P, N, M)
    return P, N


# ------------------------------------------------------------------ occlusion (cheap ray vs ellipsoids)
OCC_HEAD = (np.array([0, 0.08, -0.02], np.float32), np.array([0.70, 0.95, 0.80], np.float32))
OCC_BODY = [(np.array([0, -1.85, -0.08], np.float32), np.array([1.05, 0.42, 0.45], np.float32)),
            (np.array([0, -1.15, -0.10], np.float32), np.array([0.32, 0.55, 0.34], np.float32))]


def _hit(o, d, e, r):
    oo = (o - e) / r
    dd = d / r
    a = np.einsum("ij,ij->i", dd, dd)
    b = 2 * np.einsum("ij,ij->i", oo, dd) if oo.ndim == 2 else 2 * dd @ oo
    c = (oo * oo).sum(-1) - 1
    disc = b * b - 4 * a * c
    ok = disc > 0
    s1 = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a + 1e-9)
    return ok & (s1 > 0) & (s1 < 0.97)


def occlusion(P, cam_pos, M=None, offset=None, k=0.12):
    Pw = P if offset is None else P - offset
    c = cam_pos if offset is None else cam_pos - offset
    if M is not None:
        Pl = (Pw - PIVOT) @ M + PIVOT
        cl = (c - PIVOT) @ M + PIVOT
    else:
        Pl, cl = Pw, c
    d = Pl - cl[None]
    occ = _hit(cl[None].repeat(1, 0), d, *OCC_HEAD)
    for e, r in OCC_BODY:
        occ |= _hit(c[None], Pw - c[None], e, r)
    return np.where(occ, k, 1.0).astype(np.float32)


# ------------------------------------------------------------------ eyes & cracks
def eye_points(n=14000, openness=1.0, look=(0.0, 0.0), seed=3):
    """almond eyes with pupils (big on x<0, small on x>0) and catchlights. returns (P, E) where E holds layout info."""
    rr = np.random.default_rng(seed)
    Ps, xs, ys, sides, catch = [], [], [], [], []
    hh = 0.040 * openness
    for sx in (-1, 1):
        u = rr.uniform(-1, 1, n // 2); v = rr.uniform(-1, 1, n // 2)
        x = u * 0.10
        y = v * hh * (1 - u ** 2) ** 0.8
        z = 0.744 - 1.4 * x ** 2
        Ps.append(np.stack([sx * 0.26 + x, 0.050 + y, z], 1)); xs.append(x); ys.append(y)
        sides.append(np.full(len(x), sx)); catch.append(np.zeros(len(x), bool))
        # catchlight cluster (upper-left of the pupil)
        rp = 0.030 if sx < 0 else 0.011
        m = 500
        ang = rr.random(m) * 2 * np.pi; rad = np.sqrt(rr.random(m)) * rp * 0.32
        cx = look[0] - rp * 0.38 + rad * np.cos(ang); cy = look[1] + rp * 0.38 + rad * np.sin(ang)
        Ps.append(np.stack([sx * 0.26 + cx, 0.050 + cy, 0.749 - 1.4 * cx ** 2], 1)); xs.append(cx); ys.append(cy)
        sides.append(np.full(m, sx)); catch.append(np.ones(m, bool))
    P = np.concatenate(Ps).astype(np.float32)
    E = dict(x=np.concatenate(xs).astype(np.float32), y=np.concatenate(ys).astype(np.float32),
             side=np.concatenate(sides), catch=np.concatenate(catch), look=look, open=openness)
    return P, E


def eye_weights(E, kick=0.0):
    """tint / white weights: dark pupil, bright iris ring, glowing sclera, twinkling catchlight."""
    side = E["side"]
    rp = np.where(side < 0, 0.030, 0.011) * (1 + 0.18 * kick * np.where(side < 0, -1, 1))
    dx = E["x"] - E["look"][0]; dy = E["y"] - E["look"][1]
    r = np.sqrt(dx * dx + dy * dy)
    pupil = r < rp
    ring = np.exp(-((r - rp - 0.005) / 0.005) ** 2)
    scl = np.where(pupil, 0.0, 0.16)
    wt = scl + ring * 1.0
    ww = np.where(pupil, 0.0, 0.05) + ring * 0.55
    c = E["catch"]
    wt = np.where(c, 0.3, wt); ww = np.where(c, 2.2 * (1 + 1.5 * kick), ww)
    return wt.astype(np.float32), ww.astype(np.float32)


def crack_points(Pf, Nf, seed=11):
    f = os.path.join(CACHE, f"cracks_{seed}.npz")
    if os.path.exists(f):
        d = np.load(f)
        return d["P"], d["s"], d["wid"]
    starts = [(-0.37, 0.05, 0.66), (0.37, 0.05, 0.66), (-0.15, 0.04, 0.73), (0.15, 0.04, 0.73),
              (-0.22, 0.24, 0.68), (0.22, 0.24, 0.68), (-0.3, -0.12, 0.7), (0.3, -0.12, 0.7),
              (0.0, 0.25, 0.75), (-0.45, 0.2, 0.5), (0.45, 0.2, 0.5)]
    walks = geom.crack_walks(Pf, Nf, starts, n_walks=34, steps=80, step=0.016, seed=seed)
    Ps, ss, ws = [], [], []
    rr = np.random.default_rng(seed)
    for wi, wk in enumerate(walks):
        seg = np.diff(wk, axis=0)
        L = np.linalg.norm(seg, axis=1)
        cum = np.concatenate([[0], np.cumsum(L)])
        tot = cum[-1]
        if tot < 0.05:
            continue
        m = int(tot / 0.0022)
        d = np.sort(rr.random(m)) * tot
        i = np.clip(np.searchsorted(cum, d) - 1, 0, len(seg) - 1)
        fr = ((d - cum[i]) / np.maximum(L[i], 1e-6))[:, None]
        p = wk[i] + seg[i] * fr + rr.normal(0, 0.0025, (m, 3))
        Ps.append(p); ss.append(d / tot); ws.append(np.full(m, wi))
    P = np.concatenate(Ps).astype(np.float32); s = np.concatenate(ss).astype(np.float32); wid = np.concatenate(ws)
    np.savez(f, P=P, s=s, wid=wid)
    return P, s, wid


# ------------------------------------------------------------------ 2D layers: nebula, spotlights, rays, dust
def make_nebula(theme, seed):
    rr = np.random.default_rng(seed)
    w, h = 720, 405
    def field(scales):
        acc = np.zeros((h, w), np.float32)
        for sc, amp in scales:
            n = rr.random((h, w)).astype(np.float32)
            acc += cv2.GaussianBlur(n, (0, 0), sc) * amp
        acc -= acc.min(); acc /= acc.max() + 1e-6
        return acc
    f1 = field([(40, 1.0), (18, 0.6), (7, 0.25)])
    f2 = field([(60, 1.0), (25, 0.5), (10, 0.3)])
    th = np.asarray(theme, np.float32)
    deep = th * np.array([0.35, 0.25, 0.55], np.float32)
    img = (f1[:, :, None] ** 3.2) * th[None, None] * 0.55 + (f2[:, :, None] ** 4) * deep[None, None] * 0.6
    img += th[None, None] * 0.012
    # bright wisp band
    yy = np.linspace(-1, 1, h)[:, None]; xx = np.linspace(-1, 1, w)[None, :]
    band = np.exp(-((yy + 0.25 * xx) / 0.55) ** 2)
    img *= (0.35 + 0.9 * band)[:, :, None]
    return img.astype(np.float32)


def nebula_frame(neb, t, k=1.0):
    h, w = neb.shape[:2]
    dx = 18 * math.sin(t * 0.21) + t * 3.5
    dy = 8 * math.cos(t * 0.17)
    M = np.float32([[1.06, 0, -dx - 20], [0, 1.06, -dy - 10]])
    small = cv2.warpAffine(neb, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR) * k


def make_cone(width_top=50, width_bot=300, height=820):
    y = np.linspace(0, 1, height)[:, None]
    wdt = width_top + (width_bot - width_top) * y
    x = np.arange(-width_bot, width_bot)[None, :]
    g = np.exp(-(x / (wdt * 0.5)) ** 2) * (0.25 + 0.75 * (1 - y) ** 0.6) * np.exp(-((y - 1.0) / 0.25) ** 2 * 0.0)
    g *= np.clip(y * 8, 0, 1) * (1 - y) ** 1.3
    return g.astype(np.float32)


CONE = make_cone()


def add_cone(img, cx, top, k, col):
    h, w = CONE.shape
    x0 = int(cx - w / 2)
    xs0 = max(0, -x0); xd0 = max(0, x0); xd1 = min(W, x0 + w)
    if xd1 <= xd0:
        return
    y1 = min(H, top + h)
    img[top:y1, xd0:xd1] += CONE[: y1 - top, xs0: xs0 + xd1 - xd0, None] * (k * np.asarray(col, np.float32))[None, None]


def make_rays(seed=2, n=720):
    rr = np.random.default_rng(seed)
    prof = np.zeros(n, np.float32)
    for f_, amp in ((7, 0.5), (19, 0.35), (43, 0.3), (97, 0.25)):
        prof += amp * (0.5 + 0.5 * np.sin(np.arange(n) / n * 2 * np.pi * f_ + rr.random() * 6.28))
    prof = (prof / prof.max()) ** 4
    return prof


RAYS = make_rays()
_rays_cache = {}


def rays_frame(cx, cy, t, k, col, r0=260):
    key = (int(cx), int(cy))
    if len(_rays_cache) > 12:
        _rays_cache.clear()
    if key not in _rays_cache:
        w, h = W // 3, H // 3
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        dx = xx - cx / 3; dy = yy - cy / 3
        ang = (np.arctan2(dy, dx) / (2 * np.pi) % 1.0 * len(RAYS)).astype(np.int32)
        rad = np.sqrt(dx ** 2 + dy ** 2) * 3
        fall = np.exp(-((rad - r0) / 520) ** 2) * np.clip((rad - r0 * 0.6) / 140, 0, 1)
        _rays_cache[key] = (ang, fall.astype(np.float32))
    ang, fall = _rays_cache[key]
    sh = int(t * 6) % len(RAYS)
    v = RAYS[(ang + sh) % len(RAYS)] * fall
    v = cv2.resize(v, (W, H), interpolation=cv2.INTER_LINEAR)
    return v[:, :, None] * (k * np.asarray(col, np.float32))[None, None]


def make_vignette():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    return np.clip(1.08 - 0.42 * d ** 2.2, 0.25, 1.0).astype(np.float32)[:, :, None]


VIG = make_vignette()
GRAIN = [np.random.default_rng(i).normal(0, 0.018, (H // 2, W // 2)).astype(np.float32) for i in range(6)]


def glitch(img, amt, seed):
    rr = np.random.default_rng(seed)
    out = img.copy()
    sh = int(6 + amt * 26)
    out[:, :, 0] = np.roll(img[:, :, 0], sh, axis=1)
    out[:, :, 2] = np.roll(img[:, :, 2], -sh, axis=1)
    for _ in range(int(4 + amt * 16)):
        y = int(rr.integers(0, H)); h = int(rr.integers(4, 70)); dx = int(rr.integers(-110, 110) * amt)
        out[y:y + h] = np.roll(out[y:y + h], dx, axis=1)
        if rr.random() < 0.3:
            out[y:y + h] *= 1.0 + 0.6 * amt
    out[::3] *= (1 - 0.22 * amt)
    return out


# ------------------------------------------------------------------ main renderer
class Video:
    def __init__(self, key):
        self.s = SIN[key]
        s = self.s
        self.key = key
        self.idx = ORDER.index(key)
        self.theme = np.asarray(s["theme"], np.float32)
        self.accent = np.asarray(s["accent"], np.float32)
        self.white = np.array([0.93, 0.93, 0.98], np.float32) * 0.9 + self.theme * 0.1
        self.kw = T.hex2rgb(s["kw"])
        # clouds
        self.B0P, self.B0N = get_bust(0.0)
        m = s["mouth"]
        if m > 0:
            self.BmP, self.BmN = get_bust(m)
        else:
            self.BmP, self.BmN = self.B0P, self.B0N
        self.FP, self.FN = get_face()
        rr = np.random.default_rng(1)
        self.spB0 = (0.55 + rr.random(len(self.B0P)) ** 9 * 4.5).astype(np.float32)
        self.spBm = (0.55 + rr.random(len(self.BmP)) ** 9 * 4.5).astype(np.float32)
        self.spF = (0.55 + rr.random(len(self.FP)) ** 9 * 4.5).astype(np.float32)
        self.twB0 = (rr.random(len(self.B0P)) * 6.28).astype(np.float32)
        self.fadeB0 = np.clip((self.B0P[:, 1] + 2.30) / 0.65, 0, 1).astype(np.float32)
        self.fadeBm = np.clip((self.BmP[:, 1] + 2.30) / 0.65, 0, 1).astype(np.float32)
        self.hwB0 = head_w(self.B0P); self.hwBm = head_w(self.BmP)
        self.fadeF = np.clip((self.FP[:, 1] + 1.05) / 0.5, 0, 1).astype(np.float32) * \
            np.clip((self.FP[:, 2] + 0.15) / 0.25, 0, 1).astype(np.float32)
        # sloth: dissolving points
        if key == "sloth":
            self.dis = (rr.random(len(self.BmP)) < 0.16) & (self.BmP[:, 1] < 0.6)
            self.dis_ph = rr.random(len(self.BmP)).astype(np.float32)
        # attributes
        self.attrs = build_attrs(key)
        self.eyeP, self.eyeE = eye_points(openness=s["eyes"], look=s.get("look", (0, 0)))
        self.crP, self.crS, self.crW = crack_points(self.FP, self.FN)
        rr2 = np.random.default_rng(5)
        self.crStart = (T_D + 0.10 + rr2.random(self.crW.max() + 1) * 1.6).astype(np.float32)
        # lineup heads (unposed; pose/expression applied per frame)
        self.lineup = []
        for i, k in enumerate(ORDER):
            sk = SIN[k]
            r3 = np.random.default_rng(100 + i)
            sel = (self.B0P[:, 1] > -1.55) & (r3.random(len(self.B0P)) < 0.22)
            P0 = self.B0P[sel]; N0 = self.B0N[sel]
            hd = dict(k=k, P0=P0, N0=N0, hw0=head_w(P0), fade0=np.clip((P0[:, 1] + 1.55) / 0.5, 0, 1).astype(np.float32),
                      sp0=(0.55 + r3.random(len(P0)) ** 9 * 4).astype(np.float32), pose=sk["pose"],
                      col=np.asarray(sk["theme"], np.float32), acc=np.asarray(sk["accent"], np.float32))
            if sk["mouth"] > 0:
                bp, bn = get_bust(sk["mouth"])
                selm = (bp[:, 1] > -1.55) & (r3.random(len(bp)) < 0.22)
                Pm = bp[selm]; Nm = bn[selm]
                hd.update(Pm=Pm, Nm=Nm, hwm=head_w(Pm), fadem=np.clip((Pm[:, 1] + 1.55) / 0.5, 0, 1).astype(np.float32),
                          spm=(0.55 + r3.random(len(Pm)) ** 9 * 4).astype(np.float32))
            hd["eyeP"], hd["eyeE"] = eye_points(n=5000, openness=sk["eyes"], look=sk.get("look", (0, 0)), seed=40 + i)
            src_attrs = build_attrs(k) if k != self.key else self.attrs
            at = [lineup_attr(a.subset(0.22, seed=i)) for a in src_attrs if a.kind != "halo"]
            hd["attrs"] = [a for a in at if len(a.P)]
            self.lineup.append(hd)
        # full-density target head for the push-in (full expression)
        tP, tN = (self.BmP, self.BmN)
        tsel = tP[:, 1] > -1.55
        self.tgtP = tP[tsel]; self.tgtN = tN[tsel]
        self.tgtF = np.clip((self.tgtP[:, 1] + 1.55) / 0.5, 0, 1).astype(np.float32)
        self.tgtSp = (self.spBm if self.BmP is not self.B0P else self.spB0)[tsel]
        self.tgtHW = head_w(self.tgtP)
        # roulette timing
        self.steps = [max(0.0, beat(0.5 * k_)) for k_ in range(6)]

        # background
        self.neb = make_nebula(self.theme, seed=10 + self.idx)
        # dust
        rd = np.random.default_rng(77)
        self.dust = np.stack([rd.random(1400) * W, rd.random(1400) * H, rd.random(1400) * 6.28,
                              0.3 + rd.random(1400)], 1).astype(np.float32)
        self.L = R.Layers(["w", "t", "a"])
        # ---- FX precompute: fuzz shell, blast directions, embers
        rf = np.random.default_rng(21)

        def fx_for(P, N, fz):
            n = len(P)
            fl = (rf.random(n) ** 2 * fz).astype(np.float32)
            jit = (rf.normal(0, 1, (n, 3)) * fz * 0.12).astype(np.float32)
            d = N + 0.9 * rf.normal(0, 1, (n, 3))
            d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-6
            d *= (0.3 + 1.3 * rf.random((n, 1)) ** 2)
            return fl, jit, d.astype(np.float32)
        self.fxB0 = fx_for(self.B0P, self.B0N, 0.035)
        self.fxBm = self.fxB0 if self.BmP is self.B0P else fx_for(self.BmP, self.BmN, 0.035)
        self.fxF = fx_for(self.FP, self.FN, 0.022)
        ne = 3200
        self.emb = dict(x0=np.stack([(rf.random(ne) - 0.5) * 3.4, -2.3 + rf.random(ne) * 1.2, (rf.random(ne) - 0.5) * 1.8], 1).astype(np.float32),
                        v=np.stack([(rf.random(ne) - 0.5) * 0.35, 0.7 + 1.2 * rf.random(ne), (rf.random(ne) - 0.5) * 0.25], 1).astype(np.float32),
                        life=(1.8 + 1.6 * rf.random(ne)).astype(np.float32), ph=rf.random(ne).astype(np.float32),
                        tw=(rf.random(ne) * 6.28).astype(np.float32))
        nb = 2600
        self.bursts = []
        for k_, (th, st_) in enumerate(SHOCKS):
            dd = rf.normal(0, 1, (nb, 3)); dd /= np.linalg.norm(dd, axis=1, keepdims=True)
            self.bursts.append(dict(t=th, s=st_, d=dd.astype(np.float32), spd=(1.5 + 5.5 * rf.random(nb) ** 2).astype(np.float32)))
        self._texts()

    # -------------------------------------------------------------- text assets
    def _texts(self):
        s = self.s
        wh = (0.97, 0.96, 0.94)
        kw = self.kw
        sub = 62
        def segs(line):
            return [(txt, kw if c == "kw" else wh) for txt, c in line]
        self.tx = {}
        sh = 0.95
        self.tx["l1"] = T.make_text([("我同时发布了", wh), ("七宗罪", kw), ("，", wh)], "serif_b", sub, shadow=sh)
        self.tx["l2"] = T.make_text([("你被分配到了——", wh)], "serif_b", sub, shadow=sh)
        self.tx["l2b"] = T.make_text([("你被分配到了——", wh), (s["cn"], kw)], "serif_b", sub, shadow=sh)
        self.tx["A"] = T.make_text(segs(s["lineA"]), "serif_b", sub, shadow=sh)
        self.tx["B"] = T.make_text(segs(s["lineB"]), "serif_b", sub, shadow=sh)
        self.tx["fin"] = T.make_text([("这就是", wh), (s["cn"], kw), ("。", wh)], "serif_b", sub, shadow=sh)
        # HUD
        grey = (0.75, 0.74, 0.78)
        self.tx["hud"] = T.make_text([("七宗罪  ·  SEPTEM PECCATA", grey)], "sans", 21, tracking=2, shadow=0.4)
        self.tx["num"] = T.make_text([(s["roman"], (0.95, 0.95, 0.95))], "serif_b", 34, shadow=0.4)
        self.tx["num2"] = T.make_text([("/  VII", grey)], "serif", 22, tracking=2, shadow=0.4)
        # lineup labels
        self.tx["lab"] = []
        for k in ORDER:
            sk = SIN[k]
            on = T.make_text([(sk["cn"], T.hex2rgb(sk["kw"]))], "serif_b", 26, tracking=4, glow=tuple(sk["theme"]), glow_r=8, glow_k=0.6)
            off = T.make_text([(sk["cn"], (0.62, 0.62, 0.66))], "serif_b", 26, tracking=4)
            lat_on = T.make_text([(sk["latin"], T.hex2rgb(sk["kw"]))], "serif", 14, tracking=6)
            lat = T.make_text([(sk["latin"], (0.45, 0.45, 0.5))], "serif", 14, tracking=6)
            self.tx["lab"].append((on, off, lat_on, lat))
        # title card
        cream = (0.96, 0.90, 0.74)
        th = tuple(self.theme)
        self.tx["t_small"] = T.make_text([(f"七宗罪  ·  {s['nth']}", T.hex2rgb(s["kw"]))], "serif", 28, tracking=6, shadow=0.5)
        self.tx["t_big"] = T.make_text([(s["cn"], cream)], "serif_bl", 190, tracking=6, glow=th, glow_r=26, glow_k=0.55, shadow=0.6)
        self.tx["t_lat"] = T.make_text([(s["latin"], (0.93, 0.92, 0.9))], "serif_sb", 40, tracking=22, shadow=0.5)
        self.tx["t_meta"] = T.make_text([(f"{s['roman']} / VII   ·   对立美德  ", (0.70, 0.70, 0.74)),
                                         (s["virtue_cn"], T.hex2rgb(s["kw"])),
                                         (f"  {s['virtue_lat']}", (0.70, 0.70, 0.74))], "serif", 26, tracking=3, shadow=0.5)
        self.tx["words"] = [T.make_text([(w_, (0.98, 0.97, 0.95))], "serif_b", 50, tracking=2, glow=th, glow_r=14, glow_k=0.8)
                            for w_ in s["words"]]
        self.tx["wnum"] = [T.make_text([(f"0{i + 1}", T.hex2rgb(s["kw"]))], "serif", 18, tracking=3, shadow=0.3)
                           for i in range(6)]
        self.tx["foot"] = T.make_text([(FOOTNOTE, (0.55, 0.55, 0.6))], "serif", 19, tracking=1, shadow=0.3)

    def expr(self, i, t):
        """expression amount (0..1) and selection-pop for lineup head i"""
        e = 0.0; popv = 0.0
        st = self.steps
        for j in range(6):
            if (self.idx - 5 + j) % 7 != i:
                continue
            t_on = st[j]
            t_off = st[j + 1] if j < 5 else 1e9
            if t < t_on:
                continue
            if t < t_off:
                e = max(e, clamp01((t - t_on) / 0.07))
            else:
                e = max(e, math.exp(-(t - t_off) / 0.10))
            popv = max(popv, math.exp(-(t - t_on) / 0.14))
        return e, popv

    # -------------------------------------------------------------- point drawing
    def draw(self, cam, P, N, sp, wW, wT, wA=0.0, zref=7.4, rimk=1.0, extra_fade=None, light=(0.35, 0.7, 0.75), cmax=8.0):
        sx, sy, z, f = cam.project(P)
        comp = np.clip((z / zref) ** -2, 0.15, cmax).astype(np.float32)
        if N is not None:
            lit, rim = R.lighting(N, f, light)
        else:
            lit = np.ones(len(P), np.float32); rim = np.ones(len(P), np.float32)
        base = sp * comp
        if extra_fade is not None:
            base = base * extra_fade
        if np.any(wW):
            self.L.add("w", sx, sy, base * lit * wW)
        if np.any(wT):
            self.L.add("t", sx, sy, base * (rim * rimk + 0.0) * wT)
        if np.any(wA):
            self.L.add("a", sx, sy, base * (0.35 + 0.65 * lit) * wA)
        return sx, sy, z

    def draw_attr(self, cam, a, t, M, alpha=1.0, intro_t=None, zref=7.4, halo_intro=1.0):
        P, N = animate_attr(a, t, M, None, intro=halo_intro)
        fade = getattr(a, "_fade", None) if a.kind in ("flame", "vortex") else None
        oc = occlusion(P, cam.pos.astype(np.float32), M)
        fade = oc if fade is None else fade * oc
        if intro_t is not None and a.kind != "halo":
            # fly-in from scattered positions
            u = np.clip((intro_t - a.delay * 0.45) / 0.55, 0, 1)
            u = 1 - (1 - u) ** 3
            P = P + a.scatter * (1 - u)[:, None]
            fl = u ** 0.5
            fade = fl if fade is None else fade * fl
        sp = np.ones(len(P), np.float32) * alpha
        if a.lit and N is not None:
            self.draw(cam, P, N, sp, a.w, a.t, a.a, zref=zref, extra_fade=fade)
        else:
            sx, sy, z, f = cam.project(P)
            comp = np.clip((z / zref) ** -2, 0.15, 6.0).astype(np.float32) * sp
            if fade is not None:
                comp = comp * fade
            if a.kind == "coins":
                # dim coins behind the head
                comp = comp * np.where(P[:, 2] < -0.2, 0.45, 1.0)
            if np.any(a.w):
                self.L.add("w", sx, sy, comp * a.w)
            if np.any(a.t):
                self.L.add("t", sx, sy, comp * a.t)
            if np.any(a.a):
                self.L.add("a", sx, sy, comp * a.a)

    def compose(self, colors=None, glow_k=1.0):
        colors = colors or {"w": self.white, "t": self.theme, "a": self.accent}
        img = self.L.compose(colors)
        img = R.glow(img, strengths=((6, 0.55 * glow_k), (22, 0.45 * glow_k), (70, 0.35 * glow_k)))
        return img

    # -------------------------------------------------------------- figure (bust + attrs)
    def deform(self, Pp, Np, M, fx, breathe=0.0, blast_=0.0, tremble=0.0):
        fl, jit, dirs = fx
        if breathe:
            Pp = Pp + Np * np.float32(breathe)
        if blast_:
            Pp = Pp + (dirs @ M.T) * np.float32(blast_)
        if tremble:
            Pp = Pp + np.random.default_rng(int(tremble * 1e6) % 99991).normal(0, tremble, Pp.shape).astype(np.float32)
        return Pp

    def draw_fuzz(self, cam, Pp, Np, fx, sp, fade, zref=7.4, k=1.0):
        fl, jit, dirs = fx
        Pf = Pp + Np * fl[:, None] + jit
        self.draw(cam, Pf, Np, sp * 0.55 * k, 0.32, 0.55, 0.0, zref=zref, extra_fade=fade, cmax=1.6)

    def figure(self, cam, t, M, mouth_mix=0.0, intro_t=None, eyes=0.0, cracks=0.0, halo=1.0, gain=1.0,
               attrs_on=True, dissolve=False, breathe=0.0, blast_=0.0, tremble=0.0, halo_pulse=0.0):
        use_m = mouth_mix > 0.5
        P = self.BmP if use_m else self.B0P
        N = self.BmN if use_m else self.B0N
        sp = (self.spBm if use_m else self.spB0) * gain
        fade = self.fadeBm if use_m else self.fadeB0
        hw = self.hwBm if use_m else self.hwB0
        fx = self.fxBm if use_m else self.fxB0
        Pp, Np = apply_pose(P, N, M, hw)
        Pp = self.deform(Pp, Np, M, fx, breathe, blast_, tremble)
        if dissolve and self.key == "sloth" and use_m:
            ph = (self.dis_ph + t * 0.16) % 1.0
            Pp = Pp.copy()
            Pp[self.dis, 1] -= ph[self.dis] * 1.6
            Pp[self.dis, 0] += 0.15 * np.sin(ph[self.dis] * 6 + self.dis_ph[self.dis] * 20) * ph[self.dis]
            fade = fade.copy()
            fade[self.dis] *= (1 - ph[self.dis]) ** 1.5
        self.draw(cam, Pp, Np, sp, 0.55, 0.60, 0.0, extra_fade=fade, cmax=1.8)
        self.draw_fuzz(cam, Pp, Np, fx, sp, fade)
        if attrs_on:
            for a in self.attrs:
                if a.kind == "halo":
                    if halo > 0.01:
                        self.draw_attr(cam, a, t, M, alpha=halo * (1 + 0.8 * halo_pulse), halo_intro=(0.6 + 0.4 * halo) * (1 + 0.035 * halo_pulse))
                else:
                    self.draw_attr(cam, a, t, M, intro_t=intro_t)
        if eyes > 0.01:
            Pe, _ = apply_pose(self.eyeP, None, M)
            sx, sy, z, f = cam.project(Pe)
            comp = np.clip((z / 7.4) ** -2, 0.15, 8.0) * eyes * 1.4 * np.clip(z / 1.6, 0.5, 4.0)
            wt, ww = eye_weights(self.eyeE, self._kick)
            self.L.add("t", sx, sy, comp * wt)
            self.L.add("w", sx, sy, comp * ww)
        if cracks > 0.01:
            self.draw_cracks(cam, t, M, cracks, zref=7.4 * 1.0)

    def draw_embers(self, cam, t, S, center=(0, 0.1, 0.25), zref=7.4):
        e = self.emb
        if S > 0.55:
            age = (t + e["ph"] * e["life"]) % e["life"]
            P = e["x0"] + e["v"] * age[:, None]
            P[:, 0] += 0.10 * np.sin(age * 2.7 + e["tw"])
            P[:, 2] += 0.10 * np.cos(age * 2.1 + e["tw"])
            a = np.sin(np.pi * age / e["life"]) ** 1.5 * (0.6 + 0.4 * np.sin(t * 9 + e["tw"])) * min(1.0, (S - 0.55) * 3)
            sx, sy, z, f = cam.project(P)
            comp = np.clip((z / zref) ** -2, 0.1, 4.0) * a
            self.L.add("a", sx, sy, comp * 1.4)
            self.L.add("w", sx, sy, comp * 0.5)
        c = np.asarray(center, np.float32)
        for b in self.bursts:
            age = t - b["t"]
            if age < 0 or age > 1.4:
                continue
            r = b["spd"] * (1 - math.exp(-age * 2.6)) / 2.6
            P = c + b["d"] * r[:, None]
            P[:, 1] -= 0.5 * age * age
            a = math.exp(-age / 0.45) * b["s"]
            sx, sy, z, f = cam.project(P)
            ok = z > 0.3
            comp = np.clip((z / zref) ** -2, 0.1, 6.0) * a * ok
            self.L.add("a", sx, sy, comp * 2.2)
            self.L.add("w", sx, sy, comp * 1.2)

    def draw_cracks(self, cam, t, M, k, zref=7.4):
        st = self.crStart[self.crW]
        grow = np.clip((t - st) / 0.9, 0, 1)
        vis = self.crS < grow
        if not vis.any():
            return
        P = self.crP[vis]
        tip = np.exp(-((grow[vis] - self.crS[vis]) / 0.06) ** 2)
        P, _ = apply_pose(P, None, M)
        sx, sy, z, f = cam.project(P)
        comp = np.clip((z / zref) ** -2, 0.15, 8.0) * k * 0.9
        self.L.add("t", sx, sy, comp * (1.2 + 3.0 * tip))
        self.L.add("w", sx, sy, comp * (0.25 + 1.5 * tip))

    # -------------------------------------------------------------- frame
    def frame(self, t):
        self.L.clear()
        s = self.s
        scene_ov = []    # drawn before camera FX (shakes with the picture)
        screen_ov = []   # subtitles, drawn after camera FX (lighter shake)
        flash = 0.0
        g_amt = 0.0
        neb_k = 1.0
        expo = 1.2
        hud_num = t >= T_C
        kick = af("kick", t); snare = af("snare", t)
        self._kick = kick
        S = sec(t)
        hv = hit(t)
        bob = kick * S
        head_xy = None
        shock_xy = None
        zb = 0.0
        speed_k = 0.0
        streak_k = 0.12 + 0.18 * bob
        self.so = (0.0, 0.0)
        self.dutch = 0.0
        self.eyecam = None

        def pop(t0, amp=0.25, d=0.18):
            return 1.0 + amp * (1 - ease_out((t - t0) / d, 3)) if t >= t0 else 1.0
        # ---------------- scene A/B: lineup + push-in
        if t < T_C:
            neb_k = 0.8
            expo = 1.15
            spacing = 2.0
            xs = [(i - 3) * spacing for i in range(7)]
            if t < T_PUSH:
                cam = R.Cam(pos=(0.25 * math.sin(t * 0.8), 0.10, 16.4 - 0.3 * t), look=(0, -0.05, 0), fov=30)
                push = 0.0
            else:
                lin = (t - T_PUSH) / (T_C - T_PUSH)
                u = ease_in(lin, 2.2)
                push = max(u, 1e-3)
                xt = xs[self.idx]
                p0 = (0.25 * math.sin(T_PUSH * 0.8), 0.10, 16.4 - 0.3 * T_PUSH)
                cam = R.Cam(pos=lerp(p0, (xt, 0.05, 3.6), u),
                            look=lerp((0, -0.05, 0), (xt, -0.02, 0), ease_out(min(1, lin * 1.6), 2)), fov=30)
                speed_k = 0.7 * lin ** 1.5
                zb = 0.07 * u ** 2
            steps = self.steps
            k_now = sum(1 for st in steps if t >= st) - 1
            hi = -1 if k_now < 0 else (self.idx - 5 + k_now) % 7
            landed = k_now >= 5
            add2d = []
            zr_s = 7.4 / math.sqrt(0.22)
            for i, hd in enumerate(self.lineup):
                e, popv = self.expr(i, t)
                on = (i == hi)
                lvl = 0.42 + 0.6 * e
                if landed:
                    lvl = (1.3 + 0.15 * math.sin(t * 12)) if on else 0.26
                off = np.array([xs[i], 0, 0], np.float32)
                M_i = pose_mat(**{kk: vv * (0.12 + 1.3 * e) for kk, vv in hd["pose"].items()})
                scl = np.float32(1 + 0.08 * popv)
                hc = np.array([0, 0.1, 0], np.float32)
                hd["_aproj"] = []
                if push > 0 and i == self.idx:
                    P, N = apply_pose(self.tgtP, self.tgtN, M_i, self.tgtHW)
                    sets = [(P, N, self.tgtSp * self.tgtF, 7.4, 1.0)]
                else:
                    sets = []
                    P, N = apply_pose(hd["P0"], hd["N0"], M_i, hd["hw0"])
                    w0 = 1.0 - e if "Pm" in hd else 1.0
                    if w0 > 0.01:
                        sets.append((P, N, hd["sp0"] * hd["fade0"] * w0, zr_s, 1.0))
                    if "Pm" in hd and e > 0.01:
                        Pm, Nm = apply_pose(hd["Pm"], hd["Nm"], M_i, hd["hwm"])
                        sets.append((Pm, Nm, hd["spm"] * hd["fadem"] * e, zr_s, 1.0))
                rbuf = []
                for (P, N, sp, zr, _) in sets:
                    P = (P - hc) * scl + hc + off
                    if e > 0:
                        P = P + N * np.float32(0.02 * kick * e)
                    colmix = 0.35 + 0.6 * e
                    self.draw(cam, P, N, sp * lvl, 0.55, 0.0, 0.0, zref=zr)
                    sx, sy, z, f = cam.project(P)
                    lit, rim = R.lighting(N, f)
                    comp = np.clip((z / zr) ** -2, 0.15, 8.0)
                    rbuf.append((sx, sy, comp * sp * rim * lvl * colmix))
                hd["_proj"] = rbuf
                # glowing eyes snap open when selected
                if e > 0.02 or (push > 0 and i == self.idx):
                    ee = 1.0 if (push > 0 and i == self.idx) else e
                    Pe, _ = apply_pose(hd["eyeP"], None, M_i)
                    Pe = (Pe - hc) * scl + hc + off
                    sxe, sye, ze, _ = cam.project(Pe)
                    ce = np.clip((ze / 7.4) ** -2, 0.15, 8.0) * ee * (1.0 + 1.5 * popv) * 3.0
                    wt, ww = eye_weights(hd["eyeE"], kick)
                    hd["_aproj"].append((sxe, sye, ce * wt * 2.0))
                    self.L.add("w", sxe, sye, ce * ww * 1.6)
                for a in hd["attrs"]:
                    Pa, Na = animate_attr(a, t, M_i, None)
                    Pa = (Pa - hc) * scl + hc + off
                    sxa, sya, za, fa = cam.project(Pa)
                    ca = np.clip((za / zr_s) ** -2, 0.15, 8.0) * lvl
                    fd = getattr(a, "_fade", None)
                    if fd is not None:
                        ca = ca * fd
                    ca = ca * occlusion(Pa, cam.pos.astype(np.float32), M_i, offset=off)
                    hd["_aproj"].append((sxa, sya, ca * (a.w + a.t * 0.8 + a.a * 0.8)))
                hx, hy, hz, _ = cam.project(np.array([[xs[i], 1.6, 0.0], [xs[i], -1.35, 0.0]], np.float32))
                add2d.append((i, hx, hy, on, lvl))
            img = self.L.compose({"w": np.array([0.92, 0.92, 0.96], np.float32), "t": self.theme, "a": self.accent})
            rimimg = np.zeros((H, W, 3), np.float32)
            for i, hd in enumerate(self.lineup):
                buf = np.zeros(H * W, np.float64)
                for (sx, sy, w_) in hd.pop("_proj"):
                    m = (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
                    buf += np.bincount(sy[m].astype(np.int32) * W + sx[m].astype(np.int32), weights=w_[m], minlength=H * W)
                for (sxa, sya, ca) in hd.pop("_aproj", []):
                    m = (sxa >= 0) & (sxa < W) & (sya >= 0) & (sya < H)
                    buf += np.bincount(sya[m].astype(np.int32) * W + sxa[m].astype(np.int32), weights=ca[m], minlength=H * W)
                b = cv2.GaussianBlur(buf.reshape(H, W).astype(np.float32), (3, 3), 0.65)
                rimimg += b[:, :, None] * hd["col"][None, None]
            img = img + rimimg
            img = R.glow(img, strengths=((6, 0.65), (22, 0.55), (70, 0.4)))
            for (i, hx, hy, on, lvl) in add2d:
                kk = (0.05 if not on else 0.24) * (1 - push)
                if landed and on:
                    kk = (0.34 + 0.15 * kick) * (1 - push)
                col = np.asarray(self.lineup[i]["col"]) * 0.5 + 0.5
                add_cone(img, hx[0], 0, kk, col)
                if on:
                    head_xy = (hx[0], (hx[0] * 0 + hy[0] + hy[1]) / 2)
            if push == 0:
                def lab_overlay(im, add2d=add2d):
                    for (i, hx, hy, on, lvl) in add2d:
                        on_t, off_t, lat_on, lat = self.tx["lab"][i]
                        sc = pop(steps[max(0, k_now)], 0.3, 0.15) if on else 1.0
                        T.blit_center(im, on_t if on else off_t, hx[1], hy[1] + 22, 1.0, scale=sc)
                        T.blit_center(im, lat_on if on else lat, hx[1], hy[1] + 52, 1.0)
                scene_ov.append(lab_overlay)
            flash += 0.35 * env(t, beat(2.5), a=0.02, d=0.25)
            for st in steps[:-1]:
                flash += 0.08 * env(t, st, a=0.01, d=0.12)

            def subs(im):
                if t >= T_C - 0.05:
                    return
                ox, oy = self.so
                T.blit_center(im, self.tx["l1"], W / 2 + ox, 835 + oy, 1.0)
                if t >= T_PUSH:
                    T.blit_center(im, self.tx["l2b"], W / 2 + ox, 925 + oy, 1.0, scale=pop(T_PUSH, 0.35, 0.2))
                else:
                    T.blit_center(im, self.tx["l2"], W / 2 + ox, 925 + oy, 1.0)
            screen_ov.append(subs)
        # ---------------- scene C: calm face, line A, pre-drop tension
        elif t < T_D:
            u = (t - T_C) / (T_D - T_C)
            pre = clamp01((t - T_PREDROP) / (T_D - T_PREDROP))
            cam = R.Cam(pos=(0.0, 0.05, 4.4 - 0.45 * u - 0.55 * ease_in(pre, 2)), look=(0, -0.08, 0), fov=30)
            tr, ty, tp = twist(t)
            M = pose_mat(yaw=-0.14 + 0.2 * ease_io(u) + ty, pitch=0.02 + tp, roll=tr)
            self.figure(cam, t, M, attrs_on=False, eyes=0.0, gain=1.0,
                        breathe=0.012 * bob - 0.06 * ease_in(pre, 2), tremble=0.0001 + 0.012 * pre ** 2)
            img = self.compose(glow_k=1.25)
            expo = 1.25 * (1 - 0.42 * ease_in(pre, 1.5))
            flash += 0.7 * env(t, T_C, a=0.01, d=0.22)
            zb = 0.06 * math.exp(-(t - T_C) / 0.12) + 0.02 * pre ** 2
            hx, hy, _, _ = cam.project(np.array([[0, 0.1, 0]], np.float32))
            head_xy = (hx[0], hy[0])

            def subs(im):
                a_ = clamp01((t - T_C - 0.12) / 0.12)
                ox, oy = self.so
                if pre > 0:
                    rr_ = np.random.default_rng(int(t * FPS))
                    ox += rr_.normal() * 3 * pre; oy += rr_.normal() * 3 * pre
                T.blit_center(im, self.tx["A"], W / 2 + ox, 935 + oy, a_, scale=pop(T_C + 0.12, 0.18, 0.15))
            screen_ov.append(subs)
        # ---------------- scene D: THE DROP — beat-cut close-ups, eyes open, blasts, cracks
        elif t < T_G1:
            dt = t - T_D
            shot = sum(1 for tc in D_CUTS if t >= tc)
            t0 = [T_D] + D_CUTS
            us = (t - t0[shot]) / BEAT
            slam = math.exp(-(t - t0[shot]) / 0.08)
            dutch = 0.0
            if shot == 0:      # front, slam in
                cam = R.Cam(pos=(0.0, 0.15, 2.28 - 0.12 * us + 0.75 * math.exp(-dt / 0.09)), look=(0, 0.10, 0), fov=30)
            elif shot == 1:    # tight on the eyes, dutch angle
                cam = R.Cam(pos=(0.0, 0.10, 1.62 - 0.16 * us + 0.35 * slam), look=(0, 0.06, 0), fov=30)
                dutch = 6.0
            elif shot == 2:    # three-quarter view
                cam = R.Cam(pos=(1.05 - 0.12 * us, 0.28, 2.05 - 0.12 * us + 0.3 * slam), look=(0, 0.02, 0.2), fov=30)
                dutch = -5.0
            else:              # extreme close-up on one eye, rushing in
                cam = R.Cam(pos=(-0.26, 0.09, 1.38 - 0.30 * ease_in(us, 2) + 0.25 * slam), look=(-0.26, 0.05, 0.74), fov=30)
                dutch = 3.0
            M = pose_mat(yaw=0.06 - 0.03 * shot, pitch=0.0, roll=0.02 * math.sin(t * 1.5))
            bl = blast(t, T_D, 0.30) + sum(blast(t, tc, 0.10) for tc in D_CUTS)
            Pp, Np = apply_pose(self.FP, self.FN, M)
            Pp = self.deform(Pp, Np, M, self.fxF, breathe=0.045 * kick, blast_=bl + 0.035 * kick,
                             tremble=(0.004 + 0.010 * us ** 2) if shot == 3 else 0.0)
            zr = 7.4 * 0.0046 / 0.0085
            cm = 1.6 if shot == 0 else 3.2 - 1.6 * slam
            self.draw(cam, Pp, Np, self.spF, 0.60, 0.60, 0.0, zref=zr, rimk=0.6, extra_fade=self.fadeF, cmax=cm)
            self.draw_fuzz(cam, Pp, Np, self.fxF, self.spF, self.fadeF, zref=zr, k=0.8)
            eo = clamp01(dt / 0.06)
            Pe, _ = apply_pose(self.eyeP, None, M)
            sx, sy, z, f = cam.project(Pe)
            comp = np.clip((z / 4.0) ** -2, 0.15, 5.0) * eo * (1 + 1.2 * math.exp(-dt / 0.35) + 1.0 * kick + 0.6 * slam) * \
                np.clip(z / 1.6, 0.5, 4.0)
            wt, ww = eye_weights(self.eyeE, kick)
            self.L.add("t", sx, sy, comp * wt * 1.2)
            self.L.add("w", sx, sy, comp * ww * 1.1)
            self.draw_cracks(cam, t, M, 1.0 + 2.2 * kick, zref=4.0)
            self.draw_embers(cam, t, S, center=(0, 0.06, 0.65), zref=4.0)
            img = self.compose(glow_k=1.3 + 0.4 * kick)
            expo = 1.25 * (1 + 0.2 * kick)
            neb_k = 0.7
            flash += 0.85 * env(t, T_D, a=0.02, d=0.11) + sum(0.32 * env(t, tc, a=0.01, d=0.09) for tc in D_CUTS)
            zb = 0.08 * math.exp(-dt / 0.10) + 0.06 * slam * (shot > 0) + 0.025 * kick
            streak_k = 0.18 + 0.45 * kick + 0.5 * math.exp(-dt / 0.22) + 0.3 * slam * (shot > 0)
            ec, ey, _, _ = cam.project(np.array([[0 if shot < 3 else -0.26, 0.05, 0.74]], np.float32))
            head_xy = shock_xy = (float(np.clip(ec[0], 0, W)), float(np.clip(ey[0], 0, H)))
            if snare > 0.5:
                g_amt = max(g_amt, 0.32 * (snare - 0.4))
            if shot == 3 and us > 0.6:
                g_amt = max(g_amt, 0.25 * (us - 0.6) / 0.4)
            self.dutch = dutch
            self.eyecam = (cam, M)

            def subs(im):
                ox, oy = self.so
                T.blit_center(im, self.tx["B"], W / 2 + ox * 1.6, 935 + oy * 1.6, 1.0,
                              scale=pop(T_D, 0.5, 0.22) * (1 + 0.06 * kick))
            screen_ov.append(subs)
        # ---------------- scene E: transformation
        elif t < T_F:
            u1 = ease_out((t - T_G1) / (T_G2 + 0.35 - T_G1), 3)
            u2 = (t - T_G2) / (T_F - T_G2)
            cam = R.Cam(pos=lerp((0, 0.10, 2.4), (0, 0.18, 5.7), u1) + np.array([0, 0, -0.35 * ease_io(u2)]),
                        look=lerp((0, 0.04, 0), (0, 0.22, 0), u1), fov=30)
            pz = s["pose"]
            pu = ease_io((t - T_G2) / 0.8)
            tr, ty, tp = twist(t)
            M = pose_mat(pitch=pz["pitch"] * pu + tp, yaw=pz["yaw"] * pu + 0.03 * math.sin(t) + ty, roll=pz["roll"] * pu + tr)
            mouth_mix = 1.0 if t >= T_G2 else 0.0
            intro_t = (t - T_G1) / (T_G2 + 0.5 - T_G1) * 1.0
            halo = ease_out((t - T_G2) / 0.7)
            cr = 1.0 - clamp01((t - T_G2) / 0.4)
            self.figure(cam, t, M, mouth_mix=mouth_mix, intro_t=intro_t, eyes=(0.8 if t >= T_G2 else 0.55) * (1 + 0.5 * kick),
                        cracks=cr, halo=halo, dissolve=True, breathe=0.018 * bob,
                        blast_=blast(t, T_G2, 0.55) + blast(t, T_G1, 0.12),
                        tremble=0.006 if t < T_G2 else 0.0, halo_pulse=snare)
            self.draw_embers(cam, t, S, center=(0, 0.1, 0.2))
            img = self.compose(glow_k=1.3)
            hx, hy, _, _ = cam.project(np.array([[0, 0.15, -0.95]], np.float32))
            img += rays_frame(hx[0], hy[0], t, (0.32 + 0.25 * kick + 0.6 * math.exp(-max(0, t - T_G2) / 0.3)) * halo,
                              self.theme * 0.6 + 0.4)
            neb_k = 1.1
            expo = 1.12
            g_amt = max(env(t, T_G1, a=0.02, d=0.30), env(t, T_G2, a=0.02, d=0.40))
            flash += 0.35 * env(t, T_G1, a=0.02, d=0.10) + 0.75 * env(t, T_G2, a=0.02, d=0.10)
            zb = (0.06 * math.exp(-(t - T_G1) / 0.10)) + (0.08 * math.exp(-(t - T_G2) / 0.12) if t >= T_G2 else 0.0)
            streak_k = 0.2 + 0.25 * kick + (0.6 * math.exp(-(t - T_G2) / 0.25) if t >= T_G2 else 0.0)
            hc, hy2, _, _ = cam.project(np.array([[0, 0.1, 0.0]], np.float32))
            head_xy = shock_xy = (hc[0], hy2[0])
            self.eyecam = (cam, M)

            def subs(im):
                ox, oy = self.so
                T.blit_center(im, self.tx["B"], W / 2 + ox, 935 + oy, 1.0)
            screen_ov.append(subs)
        # ---------------- scene F: title card
        else:
            u = (t - T_F) / (DUR - T_F)
            cam = R.Cam(pos=(0, -0.14, 6.35 - 0.40 * u), look=(0, -0.14, 0), fov=30, shift=(300, 0))
            pz = s["pose"]
            tr, ty, tp = twist(t)
            M = pose_mat(pitch=pz["pitch"] + tp, yaw=pz["yaw"] + 0.05 * math.sin(t * 0.6) + ty, roll=pz["roll"] + tr)
            pulse = env(t, T_PULSE, a=0.02, d=0.5)
            self.figure(cam, t, M, mouth_mix=1.0, intro_t=None, eyes=(0.85 + 0.3 * pulse) * (1 + 0.5 * kick), cracks=0.0,
                        halo=1.0, gain=1.0 + 0.3 * pulse, dissolve=True, breathe=0.016 * bob,
                        blast_=blast(t, T_F, 0.22), halo_pulse=snare)
            self.draw_embers(cam, t, S, center=(0, 0.1, 0.2))
            img = self.compose(glow_k=1.2 + 0.3 * pulse)
            hx, hy, _, _ = cam.project(np.array([[0, 0.15, -0.95]], np.float32))
            img += rays_frame(hx[0], hy[0], t, 0.28 + 0.1 * pulse + 0.2 * kick, self.theme * 0.6 + 0.4)
            expo = 1.08
            flash += 0.55 * env(t, T_F, a=0.02, d=0.14)
            for tw in T_WORDS:
                flash += 0.14 * env(t, tw, a=0.01, d=0.15)
            zb = 0.06 * math.exp(-(t - T_F) / 0.10)
            streak_k = 0.18 + 0.22 * kick + 0.5 * math.exp(-(t - T_F) / 0.25)
            hc, hy2, _, _ = cam.project(np.array([[0, 0.1, 0.0]], np.float32))
            head_xy = shock_xy = (hc[0], hy2[0])
            self.eyecam = (cam, M)
            for tw in T_WORDS:
                if 0 <= t - tw < 0.12:
                    g_amt = max(g_amt, 0.20 * (1 - (t - tw) / 0.12))
            WPOS = [(965, 228), (1632, 208), (893, 528), (1704, 510), (975, 782), (1656, 806)]

            def card(im):
                a0 = clamp01((t - T_F) / 0.08)
                X = 144
                T.blit(im, self.tx["t_small"], X + 6 - 40 * (1 - ease_out((t - T_F) / 0.4)), 250, a0)
                sc = pop(T_F, 0.75, 0.22)
                tb = self.tx["t_big"]
                cx_ = X - 6 + (tb.w - 2 * tb.pad) / 2
                cy_ = 282 + (tb.h - 2 * tb.pad) / 2
                rj0 = np.random.default_rng(int(t * FPS) + 3)
                jx0, jy0 = rj0.normal(0, 1, 2) * (5 * kick + 22 * math.exp(-max(0, t - T_F) / 0.2))
                T.blit_center(im, tb, cx_ + jx0, cy_ + jy0, a0, scale=sc * (1 + 0.04 * kick))
                T.blit(im, self.tx["t_lat"], X + 4, 515, a0 * clamp01((t - T_F - 0.15) / 0.2))
                T.blit(im, self.tx["t_meta"], X + 4, 588, a0 * clamp01((t - T_F - 0.25) / 0.2))
                rj = np.random.default_rng(int(t * FPS) + 7)
                for i, tw in enumerate(T_WORDS):
                    if t >= tw - 0.02:
                        k_ = clamp01((t - tw + 0.02) / 0.06)
                        cx, cy = WPOS[i]
                        ja = 26 * math.exp(-max(0, t - tw) / 0.18) + 7 * kick
                        jx, jy = rj.normal(0, 1, 2) * ja
                        T.blit_center(im, self.tx["words"][i], cx + jx, cy + jy, k_, scale=pop(tw, 0.9, 0.16) * (1 + 0.05 * kick))
                        T.blit_center(im, self.tx["wnum"][i], cx - 85 + jx * 0.6, cy - 46 + jy * 0.6, k_ * 0.9)
                T.blit_center(im, self.tx["foot"], W / 2 + 300, 1035, clamp01((t - T_F - 0.6) / 0.6) * 0.9)
            scene_ov.append(card)

            def subs(im):
                if t < T_WORDS[0] - 0.05:
                    ox, oy = self.so
                    T.blit_center(im, self.tx["fin"], W / 2 + 300 + ox, 960 + oy, clamp01((t - T_F - 0.08) / 0.1),
                                  scale=pop(T_F + 0.08, 0.3, 0.18))
            screen_ov.append(subs)
        # ---------------- background, light, tonemap
        img = img + nebula_frame(self.neb, t, neb_k * (1 + 0.35 * bob))
        if head_xy is not None:
            FX.haze(img, head_xy[0], head_xy[1], 0.09 * (1 + 0.8 * bob + 0.4 * hv), self.theme * 0.85 + 0.15)
        d = self.dust
        dx_ = (d[:, 0] + t * 12 * d[:, 3]) % W
        dy_ = (d[:, 1] - t * 6 * d[:, 3]) % H
        tw_ = (0.5 + 0.5 * np.sin(t * 2.5 + d[:, 2])) * d[:, 3]
        dimg = np.bincount(dy_.astype(np.int32) * W + dx_.astype(np.int32), weights=tw_ * 0.6, minlength=H * W)
        dimg = cv2.GaussianBlur(dimg.reshape(H, W).astype(np.float32), (0, 0), 1.2)
        img += dimg[:, :, None] * (self.theme * 0.5 + 0.5)[None, None]
        if speed_k > 0:
            img = FX.speed_lines(img, t, speed_k, color=self.theme * 0.4 + 0.6)
        if flash > 0:
            img += flash * (0.55 + 0.45 * self.white)[None, None] * 0.9
        img = img * (expo * (1 + 0.15 * bob + 0.10 * hv))
        if self.eyecam is not None:
            cam_, M_ = self.eyecam
            for tg, kg in GLINTS:
                age = t - tg
                if -0.04 <= age <= 0.5:
                    a = clamp01((age + 0.04) / 0.05) * math.exp(-max(age, 0) / 0.12)
                    pe, _ = apply_pose(EYE_GLINT, None, M_)
                    ex, ey, ez, _ = cam_.project(pe)
                    if ez[0] > 0.2:
                        sc_ = (0.55 + 0.9 * clamp01(age / 0.25)) * float(np.clip(4.5 / ez[0], 0.8, 2.4)) * 0.7
                        img = FX.star(img, ex[0], ey[0], 6.0 * kg * a, scale=sc_, rot=12 + 40 * age,
                                      col=self.white * 0.6 + self.theme * 0.4)
        img = FX.streaks(img, streak_k, self.theme * 0.65 + 0.35, thr=1.7)
        img = 1.0 - np.exp(-img * 1.15)
        img *= VIG
        for f_ in scene_ov:
            f_(img)
        # ---------------- camera FX driven by the music
        if shock_xy is not None:
            for th, st_ in SHOCKS:
                age = t - th
                if 0 <= age <= 0.65:
                    img = FX.shockwave(img, shock_xy[0], shock_xy[1], age, strength=44 * st_,
                                       color=(self.theme * 0.7 + 0.3) * st_, ring_k=0.35)
        zc = head_xy if head_xy is not None else (W / 2, H / 2)
        img = FX.zoom_blur(img, zb, zc[0], zc[1])
        amp = 3.0 * S + 30 * bob + 60 * hv
        dx = FX.shake_noise(t, 1) * amp
        dy = FX.shake_noise(t, 2) * amp * 0.8
        rot = FX.shake_noise(t, 3) * (0.2 * S + 1.4 * bob + 3.0 * hv) + self.dutch
        zoom = (1.0 + 0.06 * bob + 0.14 * hv) * (1 + abs(rot) * 0.012 + (abs(dx) + abs(dy)) / W)
        img = FX.affine(img, dx, dy, rot, zoom)
        img = FX.chroma(img, 0.0010 + 0.005 * bob + 0.015 * hv)
        self.so = (dx * 0.3, dy * 0.3)
        self.hud(img, t, hud_num)
        for f_ in screen_ov:
            f_(img)
        if g_amt > 0.02:
            img = glitch(img, g_amt, seed=int(t * 60))
        g = cv2.resize(GRAIN[int(t * FPS) % len(GRAIN)], (W, H), interpolation=cv2.INTER_NEAREST)
        img += g[:, :, None]
        if t > T_FADE0:
            img *= 1 - ease_io((t - T_FADE0) / (T_FADE1 - T_FADE0))
        return np.clip(img, 0, 1)

    def hud(self, img, t, show_num):
        col = np.array([0.8, 0.8, 0.84], np.float32) * 0.8
        L = 34; m = 38; th = 2

        def rect(x0, y0, x1, y1):
            img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * 0.35 + col * 0.65
        rect(m, m, m + L, m + th); rect(m, m, m + th, m + L)
        rect(W - m - L, m, W - m, m + th); rect(W - m - th, m, W - m, m + L)
        rect(m, H - m - th, m + L, H - m); rect(m, H - m - L, m + th, H - m)
        rect(W - m - L, H - m - th, W - m, H - m); rect(W - m - th, H - m - L, W - m, H - m)
        T.blit(img, self.tx["hud"], 70, 56, 0.9)
        if show_num:
            T.blit(img, self.tx["num"], W - 215, 46, 1.0)
            T.blit(img, self.tx["num2"], W - 215 + self.tx["num"].w - 2 * self.tx["num"].pad + 16, 57, 0.9)


def main():
    key = sys.argv[1]
    args = sys.argv[2:]
    preview = None
    out = os.path.join(SCR, f"{key}.mp4")
    if "--preview" in args:
        preview = [float(x) for x in args[args.index("--preview") + 1].split(",")]
    if "--out" in args:
        out = args[args.index("--out") + 1]
    v = Video(key)
    if preview:
        for tt in preview:
            t0 = time.time()
            im = v.frame(tt)
            cv2.imwrite(os.path.join(SCR, f"pv_{key}_{tt:05.2f}.png"), (im[:, :, ::-1] * 255).astype(np.uint8))
            print(f"t={tt} {time.time() - t0:.2f}s", flush=True)
        return
    try:
        sfx_wav = sfx.ensure(SCR)
    except Exception as exc:
        print("[sfx] synthesis failed, falling back to silence:", exc, flush=True)
        sfx_wav = None
    audio_in = ["-i", sfx_wav] if sfx_wav else ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo"]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", *audio_in, "-t", f"{DUR}",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-pix_fmt", "yuv420p", "-threads", "2",
           "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    for i in range(NF):
        t = i / FPS
        im = v.frame(t)
        p.stdin.write((im * 255 + 0.5).astype(np.uint8).tobytes())
        if i % 60 == 0:
            print(f"[{key}] frame {i}/{NF}  {time.time() - t0:.0f}s", flush=True)
    p.stdin.close()
    p.wait()
    print(f"[{key}] done {out} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
