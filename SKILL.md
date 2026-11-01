---
name: seven-sins-video
description: Generate the 「七宗罪」(Seven Deadly Sins) series of 18-second 1080p30 particle-head short videos with beat-synced shake/glitch/shockwave FX, per-sin copy, and Douyin titles+tags. Use when asked to make, tweak or re-theme this series or similar procedural particle-face music videos.
---

# Seven Sins video generator

Pure procedural pipeline (numpy / OpenCV / Pillow / scipy + ffmpeg). No external assets. **Sound effects are synthesized procedurally on the beat grid** (kick / sub-boom / riser / word zaps) and muxed into every render — no BGM needed.

## Setup
```bash
pip install -r requirements.txt     # needs ffmpeg and Noto CJK fonts (/usr/share/fonts/opentype/noto/)
```

## Render
```bash
cd scripts
python3 make.py pride --preview 0.5,5.3,9      # stills -> work/pv_*.png (fast check)
python3 make.py pride --out work/pride.mp4     # one 18 s video (~3 min, silent)
./render_all.sh                                # all 7 -> out/, built-in SFX muxed
```
Keys: pride envy wrath sloth greed gluttony lust. Env: `SINS_WORK` (cache/output dir, default ./work), `SINS_WAV` (16-bit wav of your BGM for onset-driven FX; `python3 audio_fx.py my.wav` precomputes it).

## Files
- `sins_data.py` — all copy, colors, pose per sin. Edit to re-theme.
- `make.py` — timeline (scenes A–F: lineup/roulette → calm face → DROP close-ups → transformation → title card), expressions, neck twist, eye glints, FX tables.
- `geom.py` SDF head/bust, surface sampling, per-sin props (crown, serpent, horns, coins, vortex, petals). `render.py` point-splat renderer. `fx2d.py` post FX. `textfx.py` text. `audio_fx.py` beat/onset envelopes.
- `sfx.py` — procedural sound-effects track (kick/boom/riser/zap/shockwave), muxed into renders.
- `references/copy.md` — title + hashtags per video.

## Timing
Built on a 90 BPM grid (BEAT=0.6648 s, drop at beat 8 ≈ 5.26 s). Use music at ~90 BPM with the drop at ~5.3 s, or edit `BEAT/B0` in make.py. Mux with `-ss 0.022` for sync.

## Web player
`web/index.html` opens directly in a browser (no server). Put rendered mp4s in `web/videos/` (or drag them onto the page); it plays them, auto-advances, and has copy buttons for each title and hashtag set.
