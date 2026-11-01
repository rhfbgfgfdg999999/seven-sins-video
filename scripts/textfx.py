"""Text rendering to premultiplied RGBA float arrays + compositing."""
import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# 字体目录：默认 Linux Noto CJK 路径，其他平台用环境变量 SINS_FONT_DIR 覆盖
FD = os.environ.get("SINS_FONT_DIR", "/usr/share/fonts/opentype/noto/")
_fonts = {}


def font(kind, size):
    key = (kind, size)
    if key not in _fonts:
        file = {"serif_b": "NotoSerifCJK-Bold.ttc", "serif_bl": "NotoSerifCJK-Black.ttc",
                "serif_sb": "NotoSerifCJK-SemiBold.ttc", "serif": "NotoSerifCJK-Regular.ttc",
                "serif_l": "NotoSerifCJK-Light.ttc", "sans": "NotoSansCJK-Medium.ttc",
                "sans_l": "NotoSansCJK-Light.ttc"}[kind]
        _fonts[key] = ImageFont.truetype(FD + file, size, index=2)
    return _fonts[key]


def hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


class TextImg:
    """premultiplied rgb (h,w,3) float32 + alpha (h,w) float32; anchor = top-left offset of the ink box."""
    def __init__(self, rgb, a, pad):
        self.rgb, self.a, self.pad = rgb, a, pad
        self.h, self.w = a.shape


def make_text(segments, kind, size, tracking=0, shadow=0.75, glow=None, glow_r=10, glow_k=0.9, pad=40):
    """segments: list of (text, (r,g,b) 0..1)."""
    f = font(kind, size)
    # measure
    widths = []
    for txt, _ in segments:
        for ch in txt:
            widths.append(f.getlength(ch) + tracking)
    total = sum(widths) - (tracking if widths else 0)
    asc, desc = f.getmetrics()
    Wd = int(total + 2 * pad)
    Hd = int(asc + desc + 2 * pad)
    col_layers = np.zeros((Hd, Wd, 3), np.float32)
    alpha = np.zeros((Hd, Wd), np.float32)
    x = pad
    for txt, col in segments:
        im = Image.new("L", (Wd, Hd), 0)
        d = ImageDraw.Draw(im)
        xs = x
        for ch in txt:
            d.text((xs, pad), ch, font=f, fill=255)
            xs += f.getlength(ch) + tracking
        m = np.asarray(im, np.float32) / 255
        col_layers += m[:, :, None] * np.asarray(col, np.float32)[None, None]
        alpha = np.maximum(alpha, m)
        x = xs
    rgb = col_layers  # premultiplied (m * col)
    a = alpha
    out_rgb = np.zeros_like(rgb)
    out_a = np.zeros_like(a)
    if glow is not None:
        g = cv2.GaussianBlur(a, (0, 0), glow_r) * glow_k
        out_rgb += g[:, :, None] * np.asarray(glow, np.float32)[None, None]
        out_a = np.maximum(out_a, g * 0.0)
    if shadow:
        s = cv2.GaussianBlur(a, (0, 0), max(2, size / 14)) * shadow
        # shadow darkens: represented via alpha only (black)
        out_a = out_a + s * (1 - out_a)
        out_rgb = out_rgb * 1.0
    # text on top
    out_rgb = rgb + out_rgb * (1 - a[:, :, None])
    out_a = a + out_a * (1 - a)
    return TextImg(out_rgb.astype(np.float32), out_a.astype(np.float32), pad)


def blit(img, t, x, y, alpha=1.0, add_glow_only=False):
    """img: (H,W,3) float display-space [0,1]. (x,y) is the top-left of the text ink origin (excl. pad)."""
    if alpha <= 0.002:
        return
    H, W = img.shape[:2]
    x0 = int(round(x - t.pad)); y0 = int(round(y - t.pad))
    x1 = x0 + t.w; y1 = y0 + t.h
    sx0 = max(0, -x0); sy0 = max(0, -y0)
    dx0 = max(0, x0); dy0 = max(0, y0)
    dx1 = min(W, x1); dy1 = min(H, y1)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    sl = (slice(sy0, sy0 + dy1 - dy0), slice(sx0, sx0 + dx1 - dx0))
    a = t.a[sl] * alpha
    rgb = t.rgb[sl] * alpha
    reg = img[dy0:dy1, dx0:dx1]
    reg *= (1 - a[:, :, None])
    reg += rgb


def blit_center(img, t, cx, cy, alpha=1.0, scale=1.0):
    if scale != 1.0:
        rgb = cv2.resize(t.rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        a = cv2.resize(t.a, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        tt = TextImg(rgb, a, int(t.pad * scale))
    else:
        tt = t
    x = cx - tt.w / 2 + tt.pad
    y = cy - tt.h / 2 + tt.pad
    blit(img, tt, x, y, alpha)
