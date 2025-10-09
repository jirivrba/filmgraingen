
# Film Grain Generator (35mm 70s, 80s Look)

This Python tool generates **4K film grain overlays** that emulate photochemical film stock from the 1970s or 1980s. 
The grain is **different in every frame** (no static texture), creating a realistic organic look suitable for overlay in editing software like Final Cut Pro X, DaVinci Resolve, Premiere Pro, etc.

---

## Features
- Emulates **35mm film grain** with independent ASA100 / ASA125 / ASA200 / ASA400 / ASA500 / ASA800 film sensitivities.
- Era styling via **LOOK80S** (default) or warm, flickery **LOOK70S**, combineable with any film sensitivity.
- **Stochastic per-frame grain** → every frame has unique texture, no "dirty glass" effect.
- Controls for **temporal coherence** (`--coherence`) to make grain evolve smoothly across time.
- Supports **PNG sequence export** or **direct video export** via OpenCV.
- Optional **FFmpeg encoding** to ProRes 422 HQ, ProRes 4444, or H.264 from PNGs.

---

## Installation

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate    # Windows
   ```

2. Install dependencies:
   ```bash
   pip install -U pip
   pip install numpy opencv-python
   ```

3. (Optional) Install ffmpeg if you want automatic ProRes/H.264 export.
   - macOS: `brew install ffmpeg`
   - Linux: `sudo apt install ffmpeg`
   - Windows: download from https://ffmpeg.org

---

## Usage

### Generate PNG sequence (recommended for ProRes)
```bash
python3 film_grain_generator_cli.py --export png --outdir grain_png --fps 24 \
    --loop-seconds 10 --loop-count 3 --film-sensitivity ASA200
```
This command creates a **10-second seamless grain loop** and repeats it 3 times
in the PNG sequence (30 seconds total, 720 frames at 24 fps) without any
visible seams between loops.

### Encode PNGs to ProRes 422 HQ
```bash
python3 film_grain_generator_cli.py --export png --outdir grain_png --fps 24 \
    --loop-seconds 8 --loop-count 2 --film-sensitivity ASA200 --ffmpeg prores422hq
```

### Direct video export (MP4/H.264)
```bash
python3 film_grain_generator_cli.py --export video --video-path film_grain_4k.mp4 \
    --fps 24 --loop-seconds 6 --loop-count 4 --film-sensitivity ASA200 --codec mp4v
```

---

## Parameters

- `--width` / `--height`: Output resolution (default 3840x2160).
- `--fps`: Frames per second (default 24).
- `--loop-seconds`: Length of a single seamless loop (seconds, default 10s if not set).
- `--loop-count`: How many times the loop is repeated in the output (default 1).
- `--seconds`: Deprecated alias for `--loop-seconds` to maintain backwards compatibility.
- `--film-sensitivity`: ASA stock (`ASA100`, `ASA125`, `ASA200`, `ASA400`, `ASA500`, `ASA800`).
- `--look`: Era styling (`LOOK80S` default, `LOOK70S` for warmer, chunkier 1970s feel).
- `--coherence`: Temporal coherence (0=every frame very different, 0.3=natural, 0.6=smoother evolution).
- `--regen-every`: Regenerate a fresh grain base every N frames (default 1).
- `--export`: `png` (sequence) or `video` (direct mp4).
- `--ffmpeg`: If `png`, encode with `prores422hq`, `prores4444`, or `h264`.

---

## How the profile system works

The CLI separates the creative controls into two layers:

1. **Film sensitivity / stock (`--film-sensitivity`)** – defines the *physical* behavior of the grain.
   - Each ASA preset sets the silver-halide grain diameter range, grain density, and base RGB layer balance.
   - Special Kodak presets (e.g. `KODAK_EASTMAN_5294`) start from the closest ASA profile and then tweak the
     technical values to match the historical stock.
2. **Look (`--look`)** – defines the *stylistic* finishing pass.
   - Presets such as `LOOK80S` or `LOOK70S` control warmth, tint strength, contrast curve, flicker intensity,
     and how strongly shadows carry grain.

When you pick one of the Kodak stocks you still choose a look as the creative base. The generator first loads the
look preset you requested and then applies the film-stock overrides on top of it. This means that
`KODAK_EASTMAN_5294 + LOOK80S` is valid: you get the physical grain characteristics of 5294 combined with the
general 80s palette, and the Kodak profile supplies additional adjustments (contrast, tint, density) so that the
result matches the real stock.

If you want the Kodak stock “as photographed” without extra era stylisation, select the stock and keep the look at
`LOOK80S` (neutral) which already contains the manufacturer's overrides. Switching the look to `LOOK70S` layers the
warmer/flickery 70s grade over the same physical grain behaviour, which can be useful when you want the 5294 grain
but with a different era tint.

---

## Workflow in FCPX / Resolve / Premiere

1. Import the generated ProRes/H.264 clip (or PNG sequence).
2. Place the grain **above your footage** in the timeline.
3. Set the **Blend Mode** to `Overlay` or `Soft Light` (sometimes `Screen`).
4. Adjust **Opacity** (typical 20–40%).

---

## Examples

- Clean daylight film (low grain):
  ```--film-sensitivity ASA100 --coherence 0.5```

- Classic 80s drama/comedy look:
  ```--film-sensitivity ASA200 --coherence 0.3```

- Mid-speed daylight negative (ASA125):
  ```--film-sensitivity ASA125 --coherence 0.35```

- Tungsten stage/night interiors (ASA200):
  ```--film-sensitivity ASA200 --coherence 0.28```

- Gritty 80s night exteriors (ASA400):
  ```--film-sensitivity ASA400 --coherence 0.25```

- Modern high-speed negative (ASA500):
  ```--film-sensitivity ASA500 --coherence 0.22```

- Vintage 70s cinema warmth and flicker:
  ```--film-sensitivity ASA200 --look LOOK70S --coherence 0.25```

- 70s look on a faster, grainier stock:
  ```--film-sensitivity ASA400 --look LOOK70S --coherence 0.2```

- Dark noir / night shots (grainy):
  ```--film-sensitivity ASA800 --coherence 0.2```

---

## Notes
- Each run produces **unique grain** unless you fix `--seed`.
- PNG+ProRes workflow gives the **highest quality** (10-bit, editing friendly).
- H.264 export is lighter but less faithful for fine grain.

---

## License

This project is released into the public domain via [The Unlicense](LICENSE), allowing unrestricted use, modification, and distribution.
