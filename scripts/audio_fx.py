"""Audio features of the BGM (shifted by 22ms to match the visual grid), sampled per video frame."""
import numpy as np, scipy.io.wavfile as wf, scipy.signal as ss, os

WAV = os.environ.get("SINS_WAV", "bgm.wav")  # your own BGM as 16-bit wav
SHIFT = 0.022
FPS = 30
NF = 540


def _onsets(y, sr, hop, thr, min_gap):
    h = int(sr * hop)
    e = np.array([np.sqrt((y[i:i + h] ** 2).mean()) for i in range(0, len(y) - h, h)])
    le = np.log(e + 1e-4)
    o = np.maximum(0, le - np.maximum.reduce([np.roll(le, k) for k in (1, 2, 3)]))
    o[:3] = 0
    o = o / (np.percentile(o[o > 0], 99) + 1e-9)
    pk, pr = ss.find_peaks(o, height=thr, distance=max(1, int(min_gap / hop)))
    return pk * hop, np.clip(pr["peak_heights"], 0, 1.5), e


def features():
    sr, x = wf.read(WAV)
    x = x.astype(np.float32).mean(1) / 32768
    x = x[int(SHIFT * sr):]
    def filt(kind, f):
        b, a = ss.butter(4, f, btype=kind, fs=sr)
        return ss.filtfilt(b, a, x)
    lo = filt("low", 130); mid = filt("band", [250, 2500]); hi = filt("high", 6000)
    out = {}
    t = np.arange(NF) / FPS
    for name, y, thr, gap, tau in (("kick", lo, 0.35, 0.18, 0.16), ("snare", mid, 0.40, 0.15, 0.12), ("hat", hi, 0.35, 0.07, 0.06)):
        ot, os_, e = _onsets(y, sr, 0.005, thr, gap)
        env = np.zeros(NF, np.float32)
        for tk, s in zip(ot, os_):
            d = t - tk
            m = (d >= -0.5 / FPS)
            env[m] = np.maximum(env[m], (s * np.exp(-np.maximum(d[m], 0) / tau)).astype(np.float32))
        out[name] = np.clip(env, 0, 1.2)
        out[name + "_t"] = ot
        out[name + "_s"] = os_
    # loudness (smoothed RMS)
    h = int(sr / FPS)
    rms = np.array([np.sqrt((x[i * h:(i + 1) * h] ** 2).mean()) if i * h < len(x) else 0 for i in range(NF)])
    rms = ss.savgol_filter(rms, 9, 2)
    out["rms"] = np.clip(rms / np.percentile(rms, 95), 0, 1.3).astype(np.float32)
    return out


BEAT, B0 = 0.6648, -0.055   # 90 BPM grid the timeline is built on (drop at beat 8)


def synth():
    """Envelopes from the beat grid only (no audio needed)."""
    t = np.arange(NF) / FPS
    out = {}
    for name, step, off, tau, gain in (("kick", 1, 0, 0.16, 1.0), ("snare", 2, 1, 0.12, 0.9), ("hat", 0.5, 0, 0.06, 0.6)):
        ot = np.array([B0 + BEAT * (off + step * k) for k in range(int(20 / (BEAT * step)))])
        ot = ot[(ot >= 0) & (ot < NF / FPS)]
        env = np.zeros(NF, np.float32)
        for tk in ot:
            d = t - tk; m = d >= -0.5 / FPS
            env[m] = np.maximum(env[m], (gain * np.exp(-np.maximum(d[m], 0) / tau)).astype(np.float32))
        out[name] = env; out[name + "_t"] = ot; out[name + "_s"] = np.full(len(ot), gain)
    out["rms"] = np.clip(np.interp(t, [0, 5.2, 5.3, 17, 18], [0.4, 0.5, 1.0, 1.0, 0.3]), 0, 1.3).astype(np.float32)
    return out


def load(work):
    f = os.path.join(work, "audio_fx.npz")
    if os.path.exists(f):
        return np.load(f)
    if os.path.exists(WAV):
        d = features(); np.savez(f, **d); return d
    print("[audio_fx] no BGM wav found -> synthetic beat-grid envelopes", flush=True)
    return synth()


if __name__ == "__main__":
    import sys
    work = os.environ.get("SINS_WORK", "work"); os.makedirs(work, exist_ok=True)
    if len(sys.argv) > 1:
        WAV = sys.argv[1]
    f = features()
    np.savez(os.path.join(work, "audio_fx.npz"), **f)
    for k in ("kick", "snare", "hat"):
        print(k, len(f[k + "_t"]), np.round(f[k + "_t"], 2).tolist())
