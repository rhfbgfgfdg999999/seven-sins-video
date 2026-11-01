"""Procedural particle geometry: SDF head/bust + per-sin attributes."""
import numpy as np

rng = np.random.default_rng(7)


# ---------------- SDF primitives (vectorised, p: (N,3)) ----------------
def ell(p, c, r):
    q = (p - np.asarray(c)) / np.asarray(r)
    k0 = np.linalg.norm(q, axis=1)
    k1 = np.linalg.norm(q / np.asarray(r), axis=1)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-6)


def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


def sym(p):
    q = p.copy()
    q[:, 0] = np.abs(q[:, 0])
    return q


def head_sdf(p, mouth_open=0.0, bust=True):
    s = sym(p)
    d = ell(p, (0, 0.28, -0.12), (0.74, 0.84, 0.86))          # cranium
    d = smin(d, ell(p, (0, -0.14, 0.14), (0.64, 0.72, 0.62)), 0.16)  # face mass
    d = smin(d, ell(p, (0, -0.56, 0.12), (0.55, 0.40, 0.52)), 0.14)  # jaw
    d = smin(d, ell(s, (0.47, -0.56, -0.06), (0.13, 0.20, 0.24)), 0.10)  # mandible corners
    d = smin(d, ell(p, (0, -0.88, 0.36), (0.22, 0.13, 0.17)), 0.10)  # chin
    d = smin(d, ell(s, (0.44, -0.12, 0.40), (0.18, 0.13, 0.17)), 0.10)  # cheekbones
    d = smin(d, ell(p, (0, 0.23, 0.60), (0.54, 0.10, 0.15)), 0.10)   # brow ridge
    d = smax(d, -ell(s, (0.26, 0.07, 0.78), (0.18, 0.11, 0.16)), 0.07)  # eye sockets
    d = smin(d, ell(s, (0.26, 0.06, 0.62), (0.13, 0.09, 0.105)), 0.035)  # eyeballs / lids
    d = smax(d, -ell(s, (0.26, 0.045, 0.73), (0.11, 0.006, 0.05)), 0.006)  # lid line
    # nose (bridge -> tip)
    for c, r, k in (((0, 0.10, 0.71), (0.045, 0.10, 0.05), 0.05), ((0, -0.04, 0.78), (0.052, 0.11, 0.06), 0.05),
                    ((0, -0.17, 0.85), (0.06, 0.10, 0.07), 0.05), ((0, -0.27, 0.885), (0.062, 0.055, 0.06), 0.05)):
        d = smin(d, ell(p, c, r), k)
    d = smin(d, ell(s, (0.07, -0.30, 0.785), (0.045, 0.035, 0.04)), 0.09)  # nostril wings
    # lips
    mo = mouth_open
    d = smin(d, ell(p, (0, -0.50 + 0.01 * mo, 0.70), (0.18, 0.042, 0.045)), 0.09)
    d = smin(d, ell(p, (0, -0.60 - 0.13 * mo, 0.675 - 0.04 * mo), (0.15, 0.048, 0.045)), 0.09)
    if mo > 0.02:
        d = smax(d, -ell(p, (0, -0.56 - 0.06 * mo, 0.80), (0.13 + 0.03 * mo, 0.012 + 0.11 * mo, 0.22)), 0.02)
    else:
        d = smax(d, -ell(p, (0, -0.545, 0.80), (0.18, 0.007, 0.08)), 0.01)
    # ears
    d = smin(d, ell(s, (0.76, 0.0, -0.04), (0.07, 0.21, 0.13)), 0.05)
    # neck
    d = smin(d, ell(p, (0, -1.18, -0.12), (0.38, 0.62, 0.40)), 0.15)
    if bust:
        d = smin(d, ell(p, (0, -1.80, -0.10), (1.18, 0.42, 0.52)), 0.25)  # shoulders
        d = smin(d, ell(p, (0, -2.12, 0.0), (0.88, 0.60, 0.50)), 0.20)   # chest
        d = np.maximum(d, -(p[:, 1] + 2.35))
    else:
        d = np.maximum(d, -(p[:, 1] + 1.55))
    return d


def sample_surface(sdf, n_target, bbox, band=0.012, iters=3, batch=1_500_000):
    lo, hi = np.asarray(bbox[0], float), np.asarray(bbox[1], float)
    out = []
    got = 0
    eps = 1e-3
    while got < n_target:
        p = lo + (hi - lo) * rng.random((batch, 3))
        d = sdf(p)
        p = p[np.abs(d) < band * 4]
        for _ in range(iters):
            d = sdf(p)
            g = np.stack([
                sdf(p + [eps, 0, 0]) - sdf(p - [eps, 0, 0]),
                sdf(p + [0, eps, 0]) - sdf(p - [0, eps, 0]),
                sdf(p + [0, 0, eps]) - sdf(p - [0, 0, eps]),
            ], 1) / (2 * eps)
            gn = np.linalg.norm(g, axis=1, keepdims=True) + 1e-9
            p = p - d[:, None] * g / gn
        d = sdf(p)
        keep = np.abs(d) < band * 0.25
        out.append((p[keep], (g / gn)[keep]))
        got += keep.sum()
    P = np.concatenate([o[0] for o in out])[:n_target]
    N = np.concatenate([o[1] for o in out])[:n_target]
    return P, N


def thin_uniform(P, N, cell):
    """Poisson-ish thinning so points are area-uniform (removes clumping from band sampling)."""
    key = np.floor(P / cell).astype(np.int64)
    key = key[:, 0] * 73856093 ^ key[:, 1] * 19349663 ^ key[:, 2] * 83492791
    _, idx = np.unique(key, return_index=True)
    return P[idx], N[idx]


# ---------------- tubes / curves ----------------
def tube(curve, radius, n):
    """curve: (M,3) polyline; radius: scalar or (M,) ; returns points on tube surface."""
    M = len(curve)
    t = rng.random(n) * (M - 1)
    i = np.floor(t).astype(int)
    f = (t - i)[:, None]
    i2 = np.minimum(i + 1, M - 1)
    c = curve[i] * (1 - f) + curve[i2] * f
    tang = curve[i2] - curve[i]
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    up = np.where(np.abs(tang[:, 1:2]) < 0.9, [[0, 1, 0]], [[1, 0, 0]])
    a = np.cross(tang, up); a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-9
    b = np.cross(tang, a)
    th = rng.random(n) * 2 * np.pi
    r = np.asarray(radius, float)
    if r.ndim:
        r = (r[i] * (1 - f[:, 0]) + r[i2] * f[:, 0])
    nrm = a * np.cos(th)[:, None] + b * np.sin(th)[:, None]
    return c + nrm * np.atleast_1d(r)[:, None], nrm, t / (M - 1)


def ring(center, radius, n, normal=(0, 0, 1), jitter=0.0):
    th = rng.random(n) * 2 * np.pi
    normal = np.asarray(normal, float); normal /= np.linalg.norm(normal)
    a = np.cross(normal, [0, 1, 0] if abs(normal[1]) < 0.9 else [1, 0, 0]); a /= np.linalg.norm(a)
    b = np.cross(normal, a)
    r = radius + rng.normal(0, jitter, n)
    p = np.asarray(center) + (a[None] * np.cos(th)[:, None] + b[None] * np.sin(th)[:, None]) * r[:, None]
    return p, np.tile(normal, (n, 1))


# ---------------- attributes ----------------
CROWN_K = 9
CROWN_H = (0.46, 0.30)
CROWN_R = 0.74


def crown(n=26000):
    nb = n // 3
    th = rng.random(nb) * 2 * np.pi
    y = 0.70 + rng.random(nb) * 0.11
    R = CROWN_R
    P1 = np.stack([R * np.sin(th), y, -0.12 + R * 1.02 * np.cos(th)], 1)
    N1 = np.stack([np.sin(th), 0 * th, np.cos(th)], 1)
    m = n - nb
    sid = rng.integers(0, CROWN_K, m)
    base = sid / CROWN_K * 2 * np.pi
    a = rng.random(m) ** 0.8
    b = rng.random(m) * 2 - 1
    half = np.pi / CROWN_K * 0.8
    h = np.where(sid % 2 == 0, CROWN_H[0], CROWN_H[1])
    th2 = base + b * half * (1 - a)
    yy = 0.80 + a * h
    rr = R * (1.0 + 0.06 * a)
    P2 = np.stack([rr * np.sin(th2), yy, -0.12 + rr * 1.02 * np.cos(th2)], 1)
    N2 = np.stack([np.sin(th2), 0 * th2, np.cos(th2)], 1)
    return np.concatenate([P1, P2]), np.concatenate([N1, N2])


def jewels_on_crown(n=1800):
    k = 9
    out = []
    for s in range(k):
        th = s / k * 2 * np.pi
        h = 0.62 if s % 2 == 0 else 0.40
        rr = 0.70 * 0.9
        c = np.array([rr * np.sin(th), 0.90 + h + 0.04, -0.10 + rr * 1.05 * np.cos(th)])
        q = rng.normal(0, 1, (n // k, 3)); q /= np.linalg.norm(q, axis=1, keepdims=True)
        out.append(c + q * 0.035)
    P = np.concatenate(out)
    return P, (P - P.mean(0)) / (np.linalg.norm(P - P.mean(0), axis=1, keepdims=True) + 1e-9)


def collar_parts(n=30000, seed=21):
    """flared standing collar behind the head: sparse surface, ribs and a scalloped edge."""
    r2 = np.random.default_rng(seed)
    span = 1.95
    K = 6

    def vmax(th):
        return 0.72 + 0.28 * np.abs(np.cos((th - np.pi) * K / 2 * 2))

    def pos(th, v):
        R = 0.60 + v * 0.80
        y = -1.42 + v * 1.55
        x = R * np.sin(th)
        z = -0.18 + R * np.cos(th) * 0.70
        return np.stack([x, y, z], 1)
    th = np.pi + (r2.random(n) * 2 - 1) * span
    v = r2.random(n) * vmax(th)
    Ps = pos(th, v)
    Ns = np.stack([-np.sin(th), 0 * th, -np.cos(th)], 1)
    ribs = []
    for i in range(2 * K + 1):
        tr = np.pi - span + i / (2 * K) * 2 * span
        m = n // 16
        vv = r2.random(m) * vmax(np.array([tr]))[0]
        q = pos(np.full(m, tr), vv) + r2.normal(0, 0.008, (m, 3))
        ribs.append(q)
    Pr = np.concatenate(ribs)
    te = np.pi + (r2.random(n // 2) * 2 - 1) * span
    Pe = pos(te, vmax(te)) + r2.normal(0, 0.007, (len(te), 3))
    return (Ps, Ns), Pr, Pe


def horns(n=24000):
    pts, nrs = [], []
    for sgn in (-1, 1):
        t = np.linspace(0, 1, 60)
        x = sgn * (0.42 + 0.55 * t - 0.15 * t ** 2)
        y = 0.80 + 0.95 * t - 0.2 * t ** 3
        z = 0.05 - 0.25 * t + 0.35 * t ** 2
        curve = np.stack([x, y, z], 1)
        r = 0.13 * (1 - t) ** 1.2 + 0.008
        p, nr, _ = tube(curve, r, n // 2)
        pts.append(p); nrs.append(nr)
    return np.concatenate(pts), np.concatenate(nrs)


def serpent(n=38000):
    t = np.linspace(0, 1, 260)
    ang = t * 2.3 * 2 * np.pi + 0.6
    R = 0.55 + 0.32 * np.sin(np.pi * t) + 0.2 * t
    y = -0.95 - 0.95 * t + 0.25 * np.sin(t * 9)
    curve = np.stack([R * np.sin(ang), y, -0.08 + R * 1.0 * np.cos(ang)], 1)
    # head of the snake at t=0 end rises toward the face's side
    head_t = np.linspace(0, 1, 40)
    hc = np.stack([0.45 + 0.35 * head_t, -0.60 + 0.35 * head_t, 0.55 + 0.1 * head_t], 1)
    curve = np.concatenate([hc[::-1], curve])
    r = np.concatenate([0.03 + 0.06 * np.sin(np.pi * np.linspace(0.1, 0.5, 40))[::-1],
                        0.085 * (1 - 0.7 * t) + 0.01])
    p, nr, _ = tube(curve, r, n)
    return p, nr


def coins(n_coins=36, per=420, seed=3):
    r2 = np.random.default_rng(seed)
    out_p, out_n, out_id = [], [], []
    for i in range(n_coins):
        rr = np.sqrt(r2.random(per)) * 0.075
        th = r2.random(per) * 2 * np.pi
        # rim heavier
        rim = r2.random(per) < 0.35
        rr[rim] = 0.075
        disc = np.stack([rr * np.cos(th), rr * np.sin(th), np.zeros(per)], 1)
        out_p.append(disc); out_n.append(np.tile([0, 0, 1.0], (per, 1)))
        out_id.append(np.full(per, i))
    return np.concatenate(out_p), np.concatenate(out_n), np.concatenate(out_id)


def petals(n_petal=60, per=260, seed=5):
    r2 = np.random.default_rng(seed)
    P, I = [], []
    for i in range(n_petal):
        u = r2.random(per); v = (r2.random(per) - 0.5) * 2
        # teardrop petal in local xy
        w = np.sin(np.pi * u) * (0.6 + 0.4 * u)
        x = v * w * 0.05
        y = u * 0.11
        z = 0.02 * (v ** 2)
        P.append(np.stack([x, y, z], 1)); I.append(np.full(per, i))
    return np.concatenate(P), np.concatenate(I)


def halo_ring(n=22000, R=1.55):
    th = rng.random(n) * 2 * np.pi
    r = R + rng.normal(0, 0.012, n)
    return np.stack([r * np.cos(th), r * np.sin(th)], 1)


def crack_walks(P, N, starts, n_walks=26, steps=70, step=0.018, seed=11):
    """random walks on the surface (approximated by nearest neighbours in P) from start points."""
    from scipy.spatial import cKDTree
    r2 = np.random.default_rng(seed)
    tree = cKDTree(P)
    walks = []
    for w in range(n_walks):
        s = np.asarray(starts[w % len(starts)]) + r2.normal(0, 0.03, 3)
        _, i = tree.query(s)
        pos = P[i].copy()
        d = r2.normal(0, 1, 3)
        path = [pos.copy()]
        for k in range(steps):
            nrm = N[i]
            d = d - nrm * np.dot(d, nrm)
            d += r2.normal(0, 0.55, 3)
            d = d - nrm * np.dot(d, nrm)
            d /= np.linalg.norm(d) + 1e-9
            pos = pos + d * step
            _, i = tree.query(pos)
            pos = P[i] + N[i] * 0.004
            path.append(pos.copy())
            if r2.random() < 0.015:
                break
        walks.append(np.array(path))
    return walks


def imperial_crown(seed=31):
    """band + fleurons + two pearled arches + orb + cross. returns (P, N) gold, jewels P."""
    r2 = np.random.default_rng(seed)
    R, RZ, CZ = 0.76, 0.78, -0.12
    y0, y1 = 0.66, 0.84
    pts, nrs, jw = [], [], []

    def ringpos(th, y, s=1.0):
        return np.stack([R * s * np.sin(th), y, CZ + RZ * s * np.cos(th)], -1)
    # band surface
    n = 14000
    th = r2.random(n) * 2 * np.pi
    y = y0 + r2.random(n) * (y1 - y0)
    pts.append(ringpos(th, y)); nrs.append(np.stack([np.sin(th), 0 * th, np.cos(th)], 1))
    # beaded rims (denser lines)
    for yy in (y0, y1):
        m = 5000
        th = r2.random(m) * 2 * np.pi
        pts.append(ringpos(th, yy + r2.normal(0, 0.006, m), 1.02)); nrs.append(np.stack([np.sin(th), 0 * th, np.cos(th)], 1))
    # gems on band
    for k in range(8):
        th = k / 8 * 2 * np.pi + np.pi / 8
        c = ringpos(np.array(th), (y0 + y1) / 2, 1.03)
        q = r2.normal(0, 1, (220, 3)); q /= np.linalg.norm(q, axis=1, keepdims=True)
        jw.append(c + q * np.array([0.045, 0.05, 0.03]))
    # fleurons / crosses pattee on top of band
    for k in range(8):
        th = k / 8 * 2 * np.pi
        m = 1600
        a = r2.random(m); b = r2.random(m) * 2 - 1
        h = 0.20 if k % 2 == 0 else 0.15
        wdt = 0.09 * (1 - a) + 0.05 * np.exp(-((a - 0.72) / 0.12) ** 2)  # leaf bulge
        th2 = th + b * wdt / R
        yy = y1 + a * h
        pts.append(ringpos(th2, yy, 1.0)); nrs.append(np.stack([np.sin(th2), 0 * th2, np.cos(th2)], 1))
        c = ringpos(np.array(th), y1 + h + 0.025, 1.0)
        q = r2.normal(0, 1, (120, 3)); q /= np.linalg.norm(q, axis=1, keepdims=True)
        jw.append(c + q * 0.025)
    # arches (front-back and left-right), slightly dipped at the top
    apex = 1.40
    for axis in (0, 1):
        t = np.linspace(0, np.pi, 160)
        lift = (apex - y1) * np.sin(t) - 0.05 * np.sin(t) ** 10
        if axis == 0:
            curve = np.stack([R * np.cos(t), y1 + lift, CZ + 0 * t], 1)
        else:
            curve = np.stack([0 * t, y1 + lift, CZ + RZ * np.cos(t)], 1)
        p, nr, u = tube(curve, 0.028, 7000)
        pts.append(p); nrs.append(nr)
        # pearls along the arch
        for s in np.linspace(0.06, 0.94, 15):
            i = int(s * (len(curve) - 1))
            q = r2.normal(0, 1, (60, 3)); q /= np.linalg.norm(q, axis=1, keepdims=True)
            jw.append(curve[i] + q * 0.022 + np.array([0, 0.03, 0]))
    # orb + cross
    oc = np.array([0, apex + 0.06, CZ])
    q = r2.normal(0, 1, (3500, 3)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    pts.append(oc + q * 0.075); nrs.append(q)
    for curve in (np.stack([np.zeros(40), np.linspace(apex + 0.13, apex + 0.40, 40), np.full(40, CZ)], 1),
                  np.stack([np.linspace(-0.10, 0.10, 30), np.full(30, apex + 0.31), np.full(30, CZ)], 1)):
        p, nr, _ = tube(curve, 0.022, 2600)
        pts.append(p); nrs.append(nr)
    return np.concatenate(pts), np.concatenate(nrs), np.concatenate(jw)


def necklace(seed=33):
    r2 = np.random.default_rng(seed)
    s = np.linspace(-1, 1, 200)
    curve = np.stack([0.47 * s, -1.42 - 0.40 * (1 - s ** 2), 0.16 + 0.36 * (1 - s ** 2)], 1)
    links = []
    for i in range(0, 200, 2):
        q = r2.normal(0, 1, (40, 3)); q /= np.linalg.norm(q, axis=1, keepdims=True)
        links.append(curve[i] + q * np.array([0.016, 0.012, 0.012]))
    chain = np.concatenate(links)
    mc = np.array([0, -1.90, 0.53])
    th = r2.random(5000) * 2 * np.pi
    rr = np.where(r2.random(5000) < 0.55, 0.13, np.sqrt(r2.random(5000)) * 0.10)
    nrm = np.array([0, 0.30, 1.0]); nrm /= np.linalg.norm(nrm)
    a = np.cross(nrm, [1, 0, 0]); a /= np.linalg.norm(a); b = np.cross(nrm, a)
    med = mc + (np.cos(th)[:, None] * a + np.sin(th)[:, None] * b) * rr[:, None]
    # star in the medallion
    k = 8
    st = []
    for i in range(k):
        ang = i / k * 2 * np.pi
        tt = r2.random(250)
        st.append(mc + (np.cos(ang) * a + np.sin(ang) * b)[None] * (tt[:, None] * 0.10) + nrm * 0.01)
    return chain, med, np.concatenate(st)
