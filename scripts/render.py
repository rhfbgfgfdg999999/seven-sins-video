"""Point splatting renderer + post effects."""
import numpy as np
import cv2

W, H = 1920, 1080


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class Cam:
    def __init__(self, pos=(0, 0, 6.0), look=(0, 0, 0), fov=30.0, shift=(0.0, 0.0)):
        self.pos = np.asarray(pos, float)
        self.look = np.asarray(look, float)
        self.fov = fov
        self.shift = shift  # screen-space shift in px (for off-centre framing)

    def project(self, P):
        f = self.look - self.pos
        f /= np.linalg.norm(f)
        r = np.cross(f, [0, 1, 0]); r /= np.linalg.norm(r)
        u = np.cross(r, f)
        q = P - self.pos
        x = q @ r; y = q @ u; z = q @ f
        foc = 0.5 * H / np.tan(np.radians(self.fov) / 2)
        z = np.maximum(z, 1e-3)
        sx = W / 2 + self.shift[0] + foc * x / z
        sy = H / 2 + self.shift[1] - foc * y / z
        return sx, sy, z, f


def splat(buf, sx, sy, col, w=None):
    """additive bilinear splat. buf: (H,W,3) float32. col: (N,3)"""
    m = (sx >= 0) & (sx < W - 1) & (sy >= 0) & (sy < H - 1)
    sx = sx[m]; sy = sy[m]; col = col[m]
    x0 = sx.astype(np.int32); y0 = sy.astype(np.int32)
    fx = (sx - x0).astype(np.float32); fy = (sy - y0).astype(np.float32)
    base = y0 * W + x0
    flat = buf.reshape(-1, 3)
    n = H * W
    for dx, dy, wt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                       (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        idx = base + dy * W + dx
        for c in range(3):
            flat[:, c] += np.bincount(idx, weights=col[:, c] * wt, minlength=n)[:n]


def glow(img, strengths=((6, 0.55), (22, 0.45), (70, 0.35))):
    small = cv2.resize(img, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    acc = np.zeros_like(small)
    for sig, k in strengths:
        acc += cv2.GaussianBlur(small, (0, 0), sig / 4) * k
    return img + cv2.resize(acc, (W, H), interpolation=cv2.INTER_LINEAR)


def tonemap(img):
    return 1.0 - np.exp(-img * 1.15)


def shade_points(P, N, cam_f, base_col, rim_col, light=(0.25, 0.75, 0.6), sparkle=None, gain=1.0):
    L = np.asarray(light, float); L /= np.linalg.norm(L)
    ndl = np.clip(N @ L, 0, 1)
    ndv = np.abs(N @ (-cam_f))
    rim = (1 - ndv) ** 2.2
    b = 0.22 + 0.78 * ndl
    col = base_col[None] * b[:, None] + rim_col[None] * rim[:, None] * 1.3
    if sparkle is not None:
        col = col * sparkle[:, None]
    return col * gain


class Layers:
    """single-channel additive layers, each later tinted by its own colour."""
    def __init__(self, names):
        self.names = names
        self.buf = {k: np.zeros(H * W, np.float64) for k in names}

    def clear(self):
        for k in self.buf:
            self.buf[k][:] = 0

    def add(self, name, sx, sy, w):
        m = (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
        idx = sy[m].astype(np.int32) * W + sx[m].astype(np.int32)
        self.buf[name] += np.bincount(idx, weights=w[m], minlength=H * W)

    def compose(self, colors, blur=0.65):
        img = np.zeros((H, W, 3), np.float32)
        for k, c in colors.items():
            b = self.buf[k].reshape(H, W).astype(np.float32)
            if not b.any():
                continue
            if blur:
                b = cv2.GaussianBlur(b, (3, 3), blur)
            img += b[:, :, None] * np.asarray(c, np.float32)[None, None]
        return img


def lighting(N, cam_f, light=(0.35, 0.7, 0.75)):
    L = np.asarray(light, float); L /= np.linalg.norm(L)
    ndl = np.clip(N @ L, 0, 1)
    ndv = np.abs(N @ (-cam_f))
    rim = (1 - ndv) ** 3.2
    return 0.12 + 0.88 * ndl ** 1.3, rim
