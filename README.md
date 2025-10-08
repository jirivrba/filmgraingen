
# Film Grain Generator (35mm 80s Look)

This Python tool generates **4K film grain overlays** that emulate photochemical film stock from the 1980s. 
The grain is **different in every frame** (no static texture), creating a realistic organic look suitable for overlay in editing software like Final Cut Pro X, DaVinci Resolve, Premiere Pro, etc.

---

## Features
- Emulates **35mm film grain** with independent ASA100 / ASA200 / ASA400 / ASA800 film sensitivities.
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
python3 film_grain_generator_cli.py --export png --outdir grain_png --fps 24 --seconds 10 --film-sensitivity ASA200
```

### Encode PNGs to ProRes 422 HQ
```bash
python3 film_grain_generator_cli.py --export png --outdir grain_png --fps 24 --seconds 10 \
    --film-sensitivity ASA200 --ffmpeg prores422hq
```

### Direct video export (MP4/H.264)
```bash
python3 film_grain_generator_cli.py --export video --video-path film_grain_4k.mp4 \
    --fps 24 --seconds 10 --film-sensitivity ASA200 --codec mp4v
```

---

## Parameters

- `--width` / `--height`: Output resolution (default 3840x2160).
- `--fps`: Frames per second (default 24).
- `--seconds`: Length of the sequence (default 10s).
- `--film-sensitivity`: ASA stock (`ASA100`, `ASA200`, `ASA400`, `ASA800`). Alias: `--preset`.
- `--look`: Era styling (`LOOK80S` default, `LOOK70S` for warmer, chunkier 1970s feel).
- `--coherence`: Temporal coherence (0=every frame very different, 0.3=natural, 0.6=smoother evolution).
- `--regen-every`: Regenerate a fresh grain base every N frames (default 1).
- `--export`: `png` (sequence) or `video` (direct mp4).
- `--ffmpeg`: If `png`, encode with `prores422hq`, `prores4444`, or `h264`.

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
