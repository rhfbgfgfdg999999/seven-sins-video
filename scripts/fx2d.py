"""2D post effects: shake/zoom punch, chromatic aberration, zoom blur, shockwave, anamorphic streaks, speed lines, haze."""
import math
import numpy as np
import cv2

W, H = 1920, 1080
_YY, _XX = np.mgrid[0:H, 0:W].astype(np.float32)


def affine(img, dx=0.0, dy=0.0, rot_deg=0.0, zoom=1.0, cx=W / 2, cy=H / 2):
    if abs(dx) < 0.05 and abs(dy) < 0.05 and abs(rot_deg) < 0.01 and abs(zoom - 1) < 1e-4:
        return img
    M = cv2.getRotationMatrix2D((cx, cy), rot_deg, zoom)
    M[0, 2] += dx
    M[1, 2] += dy
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)


def chroma(img, k, cx=W / 2, cy=H / 2):
    """radial RGB split; k = relative scale difference (0.004 ~ 4px at the edge)."""
    if k < 1e-4:
        return img
    out = img.copy()
    for c, s in ((0, 1 + k), (2, 1 - k)):
        M = np.float32([[s, 0, cx * (1 - s)], [0, s, cy * (1 - s)]])
        out[:, :, c] = cv2.warpAffine(img[:, :, c], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    return out


def zoom_blur(img, amount, cx=W / 2, cy=H / 2, n=7):
    if amount < 0.004:
        return img
    acc = img.copy()
    for i in range(1, n):
        s = 1 + amount * i / (n - 1)
        M = np.float32([[s, 0, cx * (1 - s)], [0, s, cy * (1 - s)]])
        acc += cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    return acc / n


def shockwave(img, cx, cy, age, dur=0.65, strength=34.0, color=(1, 1, 1), ring_k=0.55):
    if age < 0 or age > dur:
        return img
    u = age / dur
    r = 40 + 1500 * u ** 0.7
    w = 55 + 90 * u
    s = strength * (1 - u) ** 1.5
    dx = _XX - np.float32(cx)
    dy = _YY - np.float32(cy)
    d = np.sqrt(dx * dx + dy * dy) + 1e-3
    prof = np.exp(-((d - r) / w) ** 2)
    disp = s * prof
    mx = (_XX - dx / d * disp).astype(np.float32)
    my = (_YY - dy / d * disp).astype(np.float32)
    out = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    ring = np.exp(-((d - r) / (w * 0.22)) ** 2) * ring_k * (1 - u) ** 2
    out += ring[:, :, None] * np.asarray(color, np.float32)[None, None]
    return out


def streaks(hdr, k, tint, thr=1.1):
    """anamorphic horizontal light streaks from HDR highlights (call before tonemapping)."""
    if k < 0.01:
        return hdr
    small = cv2.resize(hdr, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    lum = small.max(axis=2)
    b = np.maximum(lum - thr, 0)
    if not b.any():
        return hdr
    b = cv2.blur(b, (61, 1))
    b = cv2.blur(b, (121, 1))
    b = cv2.GaussianBlur(b, (0, 0), 0.8)
    up = cv2.resize(b, (W, H), interpolation=cv2.INTER_LINEAR)
    return hdr + up[:, :, None] * (k * np.asarray(tint, np.float32))[None, None]


def speed_lines(img, t, k, seed=3, n=160, color=(1, 1, 1), cx=W / 2, cy=H / 2):
    if k < 0.01:
        return img
    rr = np.random.default_rng(seed)
    ang = rr.random(n) * 2 * np.pi
    ph = rr.random(n)
    spd = 0.8 + rr.random(n) * 1.6
    ln = 0.10 + rr.random(n) * 0.25
    w2, h2 = W // 2, H // 2
    canvas = np.zeros((h2, w2), np.float32)
    R = math.hypot(W, H) / 2 / 2
    for i in range(n):
        p = (ph[i] + t * spd[i]) % 1.0
        r0 = (0.18 + p * 0.95) * R
        r1 = r0 + ln[i] * R * (0.5 + p)
        c, s = math.cos(ang[i]), math.sin(ang[i])
        a = (p ** 0.7) * (1 - p) * 4
        cv2.line(canvas, (int(cx / 2 + c * r0), int(cy / 2 + s * r0)), (int(cx / 2 + c * r1), int(cy / 2 + s * r1)),
                 float(a), 1, cv2.LINE_AA)
    canvas = cv2.GaussianBlur(canvas, (0, 0), 0.7)
    up = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_LINEAR)
    return img + up[:, :, None] * (k * np.asarray(color, np.float32))[None, None]


def _gauss_sprite(size=1600, sigma=0.22):
    yy, xx = np.mgrid[-1:1:complex(0, size // 4), -1:1:complex(0, size // 4)]
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2)).astype(np.float32)
    return cv2.resize(g, (size, size), interpolation=cv2.INTER_LINEAR)


_SPR = _gauss_sprite()


def haze(img, cx, cy, k, col, scale=1.0):
    if k < 0.005:
        return img
    spr = _SPR if abs(scale - 1) < 1e-3 else cv2.resize(_SPR, None, fx=scale, fy=scale)
    hs = spr.shape[0]
    x0 = int(cx - hs / 2); y0 = int(cy - hs / 2)
    sx0 = max(0, -x0); sy0 = max(0, -y0)
    dx0 = max(0, x0); dy0 = max(0, y0)
    dx1 = min(W, x0 + hs); dy1 = min(H, y0 + hs)
    if dx1 <= dx0 or dy1 <= dy0:
        return img
    img[dy0:dy1, dx0:dx1] += spr[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0, None] * (k * np.asarray(col, np.float32))[None, None]
    return img


def shake_noise(t, seed):
    rr = np.random.default_rng(seed)
    ph = rr.random(4) * 6.283
    return (0.55 * math.sin(2 * math.pi * 9.7 * t + ph[0]) + 0.30 * math.sin(2 * math.pi * 15.3 * t + ph[1])
            + 0.15 * math.sin(2 * math.pi * 23.1 * t + ph[2]))


def _star_sprite(size=1000):
    h = size // 2
    yy, xx = np.mgrid[-h:h, -h:h].astype(np.float32) / h
    r = np.sqrt(xx ** 2 + yy ** 2) + 1e-6

    def ray(a, b, L, w):
        return np.exp(-np.abs(b) / w) * np.exp(-np.abs(a) / L)
    d1 = (xx + yy) / 1.4142; d2 = (xx - yy) / 1.4142
    s = ray(xx, yy, 0.42, 0.006) + ray(yy, xx, 0.42, 0.006)
    s += 0.45 * (ray(d1, d2, 0.16, 0.005) + ray(d2, d1, 0.16, 0.005))
    s += 1.0 * np.exp(-(r / 0.022) ** 2) + 0.10 * np.exp(-(r / 0.09) ** 2)
    return s.astype(np.float32)


_STAR = _star_sprite()


def star(img, cx, cy, k, scale=1.0, rot=0.0, col=(1, 1, 1)):
    if k < 0.01:
        return img
    size = int(_STAR.shape[0] * scale)
    if size < 8:
        return img
    M = cv2.getRotationMatrix2D((_STAR.shape[1] / 2, _STAR.shape[0] / 2), rot, scale)
    M[0, 2] += size / 2 - _STAR.shape[1] / 2
    M[1, 2] += size / 2 - _STAR.shape[0] / 2
    spr = cv2.warpAffine(_STAR, M, (size, size), flags=cv2.INTER_LINEAR)
    x0 = int(cx - size / 2); y0 = int(cy - size / 2)
    sx0 = max(0, -x0); sy0 = max(0, -y0)
    dx0 = max(0, x0); dy0 = max(0, y0)
    dx1 = min(W, x0 + size); dy1 = min(H, y0 + size)
    if dx1 <= dx0 or dy1 <= dy0:
        return img
    img[dy0:dy1, dx0:dx1] += spr[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0, None] * (k * np.asarray(col, np.float32))[None, None]
    return img
