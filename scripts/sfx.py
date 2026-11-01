# -*- coding: utf-8 -*-
"""Procedural sound-effects track (18s, 44.1 kHz stereo), synthesized on the same
90 BPM beat grid as the visuals. No samples, no external audio: kick / sub-boom /
riser / cut whooshes / word zaps / shockwave / hats are built from oscillators and
filtered noise with numpy + scipy.
usage: python3 sfx.py [out.wav]
"""
import os
import numpy as np
import scipy.io.wavfile as wf
import scipy.signal as ss

SR = 44100
DUR = 18.0
N = int(SR * DUR)

BEAT, B0 = 0.6648, -0.055   # keep in sync with make.py / audio_fx.py


def beat(n):
    return B0 + BEAT * n


T_D, T_G1, T_F = beat(8), beat(12), beat(16)
D_CUTS = [beat(9), beat(10), beat(11)]
T_WORDS = [beat(n) for n in range(19, 25)]
T_PULSE = beat(25)
FADE0, FADE1 = 17.25, 18.0

RNG = np.random.default_rng(7)   # deterministic renders


def sec(t):
    """section intensity, mirrors make.py"""
    if t < 2.0:
        return 0.25
    if t < 4.40:
        return 0.45
    if t < T_D:
        return 0.20
    if t < T_G1:
        return 1.45
    if t < T_F:
        return 1.25
    return 1.15


def _exp(n, tau):
    return np.exp(-np.arange(n) / (tau * SR))


def kick(gain=1.0):
    n = int(0.35 * SR); t = np.arange(n) / SR
    f = 150 * np.exp(-t / 0.045) + 46
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * _exp(n, 0.13)
    click = RNG.normal(0, 1, n) * _exp(n, 0.004) * 0.5
    return (body + click) * gain


def boom(gain=1.0):
    n = int(2.2 * SR); t = np.arange(n) / SR
    f = 60 * np.exp(-t / 0.35) + 27
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * _exp(n, 0.75)
    b, a = ss.butter(2, 120 / (SR / 2), 'low')
    rumble = ss.lfilter(b, a, RNG.normal(0, 1, n)) * _exp(n, 0.5) * 0.7
    return (body + rumble) * gain


def whoosh(dur, gain=1.0):
    n = int(dur * SR)
    noise = RNG.normal(0, 1, n)
    step = 2048
    segs = []
    for i in range(0, n, step):
        c = 300 + (3600 - 300) * (i / n)
        b, a = ss.butter(2, [max(60, c * 0.5) / (SR / 2), min(c * 1.5, 20000) / (SR / 2)], 'band')
        segs.append(ss.lfilter(b, a, noise[i:i + step]))
    x = np.concatenate(segs)
    return x * ((np.arange(n) / SR) / dur) ** 2.2 * gain


def zap(pitch=1100.0, gain=1.0):
    n = int(0.16 * SR); t = np.arange(n) / SR
    f = pitch * np.exp(-t / 0.06) + 180
    ph = 2 * np.pi * np.cumsum(f) / SR
    tone = np.sign(np.sin(ph)) * 0.4 + np.sin(2 * ph) * 0.3
    body = tone * _exp(n, 0.05) + RNG.normal(0, 1, n) * _exp(n, 0.012) * 0.4
    return body * gain


def hat(gain=1.0):
    n = int(0.06 * SR); t = np.arange(n) / SR
    b, a = ss.butter(4, 7000 / (SR / 2), 'high')
    return ss.lfilter(b, a, RNG.normal(0, 1, n)) * _exp(n, 0.018) * gain


def clap(gain=1.0):
    n = int(0.2 * SR); t = np.arange(n) / SR
    b, a = ss.butter(2, [1500 / (SR / 2), 9000 / (SR / 2)], 'band')
    return ss.lfilter(b, a, RNG.normal(0, 1, n)) * _exp(n, 0.05) * gain


def stab(gain=1.0):
    n = int(0.9 * SR); t = np.arange(n) / SR
    x = sum(np.sin(2 * np.pi * f * t) for f in (220.0, 277.2, 329.6))
    return (x / 3) * _exp(n, 0.3) * gain


def sweep(gain=1.0):
    n = int(1.6 * SR); t = np.arange(n) / SR
    f = 95 * np.exp(-t / 0.3) + 24
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * _exp(n, 0.5) * gain


def place(buf, t, sig, gain=1.0, pan=0.0):
    i = int(t * SR)
    if i < 0:
        sig = sig[-i:]; i = 0
    n = min(len(sig), len(buf) - i)
    if n <= 0:
        return
    l = gain * np.sqrt(0.5 * (1 - pan)); r = gain * np.sqrt(0.5 * (1 + pan))
    buf[i:i + n, 0] += sig[:n] * l
    buf[i:i + n, 1] += sig[:n] * r


def build():
    buf = np.zeros((N, 2), np.float32)
    # beat-grid kicks, quiet in intro / pre-drop, strong after the drop
    for k in range(28):
        t = beat(k)
        if 0 <= t < FADE1 - 0.2:
            place(buf, t, kick(), gain=0.5 * sec(t))
    # offbeat claps through the drop & transformation
    for k in range(8, 24):
        t = beat(k) + BEAT / 2
        if T_D <= t < T_F:
            place(buf, t, clap(), gain=0.32)
    # hats every half beat from the drop on
    for k in range(16, 54):
        t = beat(k / 2)
        if t < T_D or t >= FADE1 - 0.1:
            continue
        place(buf, t, hat(), gain=0.22)
    # riser into the drop, then a short breath of silence
    place(buf, 2.0, whoosh(2.4), gain=0.5)
    # drop: sub boom
    place(buf, T_D, boom(), gain=1.0)
    # cut whoosh + slam on each drop cut
    for tc in D_CUTS:
        place(buf, tc - 0.10, whoosh(0.12), gain=0.35)
        place(buf, tc, kick(1.2), gain=0.8)
    # six word pops, alternating sides
    for i, tw in enumerate(T_WORDS):
        place(buf, tw, zap(1100.0 if i % 2 == 0 else 820.0), gain=0.5, pan=0.55 if i % 2 else -0.55)
    # title stab + shockwave pulse
    place(buf, T_F, stab(), gain=0.5)
    place(buf, T_PULSE, sweep(), gain=0.9)
    # master: fades + soft clip + normalize
    t = np.arange(N) / SR
    fade = np.clip((FADE1 - t) / max(FADE1 - FADE0, 1e-6), 0, 1) ** 1.5
    fade *= np.clip(t / 0.05, 0, 1)
    buf *= fade[:, None]
    buf = np.tanh(buf * 1.15)
    buf *= 0.88 / max(np.abs(buf).max(), 1e-9)
    return buf


def render(path):
    buf = build().astype(np.float32)
    wf.write(path, SR, (buf * 32767).astype(np.int16))
    return path


def ensure(work):
    d = os.path.join(work, "cache")
    os.makedirs(d, exist_ok=True)
    f = os.path.join(d, "sfx.wav")
    if not os.path.exists(f):
        render(f)
    return f


if __name__ == "__main__":
    import sys
    work = os.environ.get("SINS_WORK", os.path.join(os.getcwd(), "work"))
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(work, "sfx.wav")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    render(out)
    sr, x = wf.read(out)
    peak = np.abs(x / 32768.0).max()
    print(f"[sfx] {out} sr={sr} dur={len(x)/sr:.2f}s peak={peak:.2f}", flush=True)
