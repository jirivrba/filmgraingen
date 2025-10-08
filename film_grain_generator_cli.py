#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Film Grain Generator – 35mm 80s look, 4K overlay, photochemical model
CLI with PNG/video export and optional ffmpeg ProRes/H.264 encoding.

This version generates a NEW grain realization for EVERY FRAME (like real film),
with controllable temporal coherence so changes feel organic rather than harsh.

Usage examples:
  python3 film_grain_generator_cli.py --export png --outdir grain_png --fps 24 --seconds 10 --preset ASA250
  python3 film_grain_generator_cli.py --export video --video-path film_grain_4k.mp4 --fps 24 --seconds 10 --codec mp4v
  # Encode PNGs to ProRes 422 HQ via ffmpeg (after generating PNGs into --outdir):
  python3 film_grain_generator_cli.py --export png --outdir grain_png --fps 24 --seconds 10 \
      --ffmpeg prores422hq

Author: prepared for Jirka
"""

import os
import math
import cv2
import sys
import json
import argparse
import subprocess
import shutil
import numpy as np
from pathlib import Path
from typing import Tuple

# -------------------------
# Utils
# -------------------------

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def to_uint8(img: np.ndarray) -> np.ndarray:
    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0 + 0.5).astype(np.uint8)

def gaussian_blob(h, w, cx, cy, sigma_x, sigma_y, amplitude=1.0):
    """Draw a single Gaussian blob (grain clump) into a local ROI for speed."""
    radx = int(3*sigma_x) + 1
    rady = int(3*sigma_y) + 1
    x0 = max(0, int(cx) - radx); x1 = min(w, int(cx) + radx + 1)
    y0 = max(0, int(cy) - rady); y1 = min(h, int(cy) + rady + 1)
    if x0 >= x1 or y0 >= y1:
        return None, None, None
    xs = np.arange(x0, x1) - cx
    ys = np.arange(y0, y1) - cy
    X, Y = np.meshgrid(xs, ys)
    g = amplitude * np.exp(-0.5*((X/sigma_x)**2 + (Y/sigma_y)**2))
    return g, (y0, y1), (x0, x1)

def fbm_fractal_noise(shape, octaves=4, base_freq=1/128, persistence=0.5, seed=None):
    """Fast fBm-like band-limited field used to modulate clumping and tint."""
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    h, w = shape[:2]
    acc = np.zeros((h, w), np.float32)
    amp = 1.0
    freq = base_freq
    for _ in range(octaves):
        sh = max(2, int(h * freq))
        sw = max(2, int(w * freq))
        noise = rng.normal(0.0, 1.0, (sh, sw)).astype(np.float32)
        noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
        k = int(max(3, round(1/(freq+1e-6))))
        if k % 2 == 0: k += 1
        noise = cv2.GaussianBlur(noise, (k, k), sigmaX=k*0.15)
        acc += amp * noise
        amp *= persistence
        freq *= 2.0
    acc = (acc - acc.min()) / (acc.max() - acc.min() + 1e-8)
    return acc

# -------------------------
# Core photochemical model
# -------------------------

class FilmGrainGenerator:
    """
    Photochemical-like color grain with per-frame stochasticity and temporal coherence:
      - Three color layers (B/G/R) with different density/size (B grainer, R cleaner).
      - Log-normal clump size distribution mapped from microns to pixels (35mm gate scale).
      - Each frame gets a fresh grain realization; optional blending with previous frames
        controls temporal coherence (prevents harsh flicker while staying non-static).
      - Flicker, gate weave, vignette, S-curve and shadow weighting.
    """

    GATE_WIDTH_MM = 21.95  # 35mm Academy approx. width

    ASA_PRESETS = {
        "ASA100": {
            "diam_um_range": (8.0, 12.0),
            "density_Mpx": 4500,
            "channel_mul": (1.25, 1.0, 0.85)  # B, G, R
        },
        "ASA250": {
            "diam_um_range": (12.0, 18.0),
            "density_Mpx": 5500,
            "channel_mul": (1.35, 1.0, 0.8)
        },
        "ASA500": {
            "diam_um_range": (18.0, 25.0),
            "density_Mpx": 6500,
            "channel_mul": (1.45, 1.0, 0.75)
        },
    }

    def __init__(self, width=3840, height=2160, fps=24, seconds=10,
                 preset="ASA250", seed=None,
                 coherence=0.3,  # 0=new every frame (chaotic), 0.3=organic, 0.6=smoother
                 regen_every=1    # regenerate master every N frames (1 = per-frame)
                 ):
        self.width = width
        self.height = height
        self.fps = fps
        self.frames = int(round(fps * seconds))
        self.preset = preset
        self.rng = np.random.default_rng(seed)
        self.um_per_px = (self.GATE_WIDTH_MM * 1000.0) / float(self.width)  # ~5.7 µm/px @ 3840
        self.coherence = float(np.clip(coherence, 0.0, 0.95))
        self.regen_every = max(1, int(regen_every))
        self._prepare_static_fields()

    def _prepare_static_fields(self):
        """Static modulation fields shared across frames: clump modulation, tint, vignette."""
        self.clump_mod = fbm_fractal_noise((self.height, self.width),
                                           octaves=4, base_freq=1/256,
                                           persistence=0.55,
                                           seed=self.rng.integers(1<<30))
        self.color_tint = np.stack([
            fbm_fractal_noise((self.height, self.width), octaves=3, base_freq=1/512,
                              persistence=0.6, seed=self.rng.integers(1<<30))
            for _ in range(3)
        ], axis=-1)
        yy, xx = np.mgrid[0:self.height, 0:self.width]
        nx = (xx - self.width/2) / (self.width/2)
        ny = (yy - self.height/2) / (self.height/2)
        r2 = nx*nx + ny*ny
        self.vignette = np.clip(1.0 - 0.15*r2, 0.85, 1.0).astype(np.float32)

    def _sample_size_pixels(self, diam_range_um):
        lo, hi = diam_range_um
        mu = math.log((lo*hi)**0.5)
        sigma = 0.25
        diam_um = np.exp(self.rng.normal(mu, sigma))
        diam_um = float(np.clip(diam_um, lo, hi))
        diam_px = max(0.8, diam_um / self.um_per_px)
        sigma_px = diam_px / 2.355
        return diam_px, sigma_px

    def _generate_grain_tile_once(self, channel_index: int) -> np.ndarray:
        """Generate one stochastic clumpy grain tile for a single color channel (float 0..1)."""
        h, w = self.height, self.width
        base = np.zeros((h, w), np.float32)
        preset = self.ASA_PRESETS[self.preset]
        diam_range_um = preset["diam_um_range"]
        density = preset["density_Mpx"] * preset["channel_mul"][channel_index]
        total_grains = int(density * (w*h/1e6))

        # Stratified positions ~ Poisson-disk-ish using jittered grid
        grid = int(max(64, round(math.sqrt(total_grains))))
        gx = np.linspace(0, w-1, grid, endpoint=True)
        gy = np.linspace(0, h-1, grid, endpoint=True)
        cand = self.rng.choice(grid*grid, size=min(total_grains, grid*grid), replace=False)
        xs_idx = cand % grid
        ys_idx = cand // grid

        for i in range(len(xs_idx)):
            cx = gx[xs_idx[i]] + self.rng.uniform(-0.5, 0.5) * (w/(grid-1))
            cy = gy[ys_idx[i]] + self.rng.uniform(-0.5, 0.5) * (h/(grid-1))
            _, sigma_px = self._sample_size_pixels(diam_range_um)
            aspect = 1.0 + self.rng.uniform(-0.25, 0.25)
            sx = max(0.3, sigma_px * aspect)
            sy = max(0.3, sigma_px / aspect)
            mod = 0.6 + 0.8*(1.0 - self.clump_mod[int(cy)%h, int(cx)%w])
            amp = self.rng.uniform(0.6, 1.2) * mod * 0.9
            g, yr, xr = gaussian_blob(h, w, cx, cy, sx, sy, amplitude=amp)
            if g is not None:
                base[yr[0]:yr[1], xr[0]:xr[1]] += g

        # Band-limited micro noise (silver-halide feel)
        bln = fbm_fractal_noise((h, w), octaves=5, base_freq=1/512,
                                persistence=0.55, seed=self.rng.integers(1<<30))
        base = base + 0.6*bln

        # Optics/scanner slight blur and normalization
        base = cv2.GaussianBlur(base, (3,3), 0.8)
        base = (base - base.min()) / (base.max() - base.min() + 1e-8)
        return base

    def _temporal_params(self, t: int):
        """Return flicker/lift, gate weave and texture drift for frame t."""
        flicker = 1.0 + 0.03 * math.sin(2*math.pi * t / (self.fps*1.7)) + self.rng.normal(0, 0.005)
        lift =  0.0 + 0.015 * math.sin(2*math.pi * t / (self.fps*3.3)) + self.rng.normal(0, 0.003)
        gw_amp = 0.45
        dx = gw_amp * math.sin(2*math.pi * t / (self.fps*2.1)) + self.rng.normal(0, 0.05)
        dy = gw_amp * math.cos(2*math.pi * t / (self.fps*2.6)) + self.rng.normal(0, 0.05)
        rot = 0.05 * math.sin(2*math.pi * t / (self.fps*5.0))
        drift_speed = 0.12
        ux = drift_speed * t
        uy = drift_speed * 0.7 * t
        return flicker, lift, dx, dy, rot, ux, uy

    def _apply_affine(self, img: np.ndarray, dx, dy, rot_deg):
        M = cv2.getRotationMatrix2D((self.width/2, self.height/2), rot_deg, 1.0)
        M[:,2] += (dx, dy)
        return cv2.warpAffine(img, M, (self.width, self.height),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    def render(self, out_dir="grain_out", as_video=False, video_path="film_grain.mp4", fourcc_str="mp4v"):
        ensure_dir(out_dir)

        # Tone curve LUT (kept in float via np.interp)
        x = np.linspace(0, 1, 1024, dtype=np.float32)
        s_curve = 1/(1 + np.exp(-8*(x-0.5)))
        s_curve = (s_curve - s_curve.min()) / (s_curve.max()-s_curve.min())

        writer = None
        if as_video:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            writer = cv2.VideoWriter(video_path, fourcc, self.fps, (self.width, self.height))

        # Initialize previous grain tiles for temporal coherence blending
        prev_tiles = [self._generate_grain_tile_once(ch) for ch in range(3)]

        for t in range(self.frames):
            flicker, lift, dx, dy, rot, ux, uy = self._temporal_params(t)
            frame = np.zeros((self.height, self.width, 3), np.float32)

            # Generate (or regenerate) tiles and blend with previous to control coherence
            if t % self.regen_every == 0:
                new_tiles = [self._generate_grain_tile_once(ch) for ch in range(3)]
            else:
                new_tiles = prev_tiles  # between regen intervals, reuse (still moved by drift)

            for ch in range(3):
                # Temporal coherence mixing: more previous => smoother, less => more stochastic
                tile = (1.0 - self.coherence) * new_tiles[ch] + self.coherence * prev_tiles[ch]
                prev_tiles[ch] = tile  # carry forward blended state (evolves over time)

                # Apply slow drift (wrap) and gentle color tint per channel
                tx = int(ux) % self.width
                ty = int(uy) % self.height
                shifted = np.roll(np.roll(tile, shift=ty, axis=0), shift=tx, axis=1)
                tint = 0.92 + 0.16 * (self.color_tint[..., ch] - 0.5)
                layer = shifted * tint
                frame[..., ch] = layer

            # Vignette (broadcast over channels)
            frame *= self.vignette[..., None]

            # Film-like S-curve, flicker/lift
            frame = np.interp(frame, x, s_curve).astype(np.float32)
            frame = frame * flicker + lift

            # Shadow-weighted grain (stronger in low luma)
            luma = 0.114*frame[...,0] + 0.587*frame[...,1] + 0.299*frame[...,2]
            shadow_boost = 1.0 + 0.35*(1.0 - luma)
            frame *= shadow_boost[..., None]

            # Gate weave (subpixel jitter + micro-rotation per channel)
            for ch in range(3):
                frame[..., ch] = self._apply_affine(frame[..., ch], dx, dy, rot)

            # Subtle per-frame sparkle to avoid banding/over-correlation
            sparkle = np.random.default_rng().normal(0, 0.006, frame.shape).astype(np.float32)
            frame = np.clip(frame + sparkle, 0.0, 1.0)

            # Export
            if writer is not None:
                bgr8 = to_uint8(frame[..., ::-1])  # RGB->BGR
                writer.write(bgr8)
            else:
                rgb8 = to_uint8(frame)
                cv2.imwrite(os.path.join(out_dir, f"grain_{t:04d}.png"),
                            cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR))

        if writer is not None:
            writer.release()

# -------------------------
# ffmpeg helper
# -------------------------

def maybe_encode_with_ffmpeg(out_dir: str, fps: int, mode: str, ffmpeg_bin: str = "ffmpeg"):
    """
    Encode PNG sequence grain_%04d.png into a video in the same folder.
    mode ∈ {"none", "prores422hq", "prores4444", "h264"}
    """
    if mode == "none":
        return None
    if shutil.which(ffmpeg_bin) is None:
        print(f"WARNING: ffmpeg binary '{ffmpeg_bin}' not found in PATH. Skipping encoding.")
        return None

    input_pattern = os.path.join(out_dir, "grain_%04d.png")
    if mode == "prores422hq":
        out_path = os.path.join(out_dir, "film_grain_prores422HQ.mov")
        cmd = [ffmpeg_bin, '-y', '-r', str(fps), '-i', input_pattern,
               '-c:v', 'prores_ks', '-profile:v', '3', '-pix_fmt', 'yuv422p10le',
               '-movflags', '+faststart', out_path]
    elif mode == "prores4444":
        out_path = os.path.join(out_dir, "film_grain_prores4444.mov")
        cmd = [ffmpeg_bin, '-y', '-r', str(fps), '-i', input_pattern,
               '-c:v', 'prores_ks', '-profile:v', '4', '-pix_fmt', 'yuv444p10le',
               '-movflags', '+faststart', out_path]
    elif mode == "h264":
        out_path = os.path.join(out_dir, "film_grain_h264.mp4")
        cmd = [ffmpeg_bin, '-y', '-r', str(fps), '-i', input_pattern,
               '-c:v', 'libx264', '-crf', '12', '-preset', 'slow', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', out_path]
    else:
        print(f"Unknown ffmpeg mode: {mode}")
        return None

    print("Running:", ' '.join(cmd))
    try:
        subprocess.check_call(cmd)
        return out_path
    except subprocess.CalledProcessError as e:
        print("ffmpeg failed:", e)
        return None

# -------------------------
# CLI
# -------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Film Grain Generator – 35mm 80s look (per-frame stochastic)")
    p.add_argument('--width', type=int, default=3840)
    p.add_argument('--height', type=int, default=2160)
    p.add_argument('--fps', type=int, default=24)
    p.add_argument('--seconds', type=float, default=10)
    p.add_argument('--preset', choices=['ASA100','ASA250','ASA500'], default='ASA250')
    p.add_argument('--seed', type=int, default=None, help='Random seed (None => different each run)')

    p.add_argument('--coherence', type=float, default=0.3,
                   help='Temporal coherence 0..0.95 (higher = smoother, lower = more different per frame)')
    p.add_argument('--regen-every', type=int, default=1,
                   help='Regenerate a fresh grain tile every N frames (1=every frame).')

    p.add_argument('--export', choices=['png','video'], default='png', help='PNG sequence or direct video via OpenCV')
    p.add_argument('--outdir', default='grain_out', help='Directory for PNGs (and encoded files)')
    p.add_argument('--video-path', default='film_grain_4k.mp4', help='Path for direct OpenCV video export')
    p.add_argument('--codec', choices=['mp4v','avc1','H264'], default='mp4v', help='OpenCV fourcc for --export video')

    p.add_argument('--ffmpeg', choices=['none','prores422hq','prores4444','h264'], default='none',
                   help='After PNG export, optionally encode to a video via ffmpeg')
    p.add_argument('--ffmpeg-bin', default='ffmpeg', help='Custom ffmpeg binary name/path')
    return p.parse_args()


def main():
    args = parse_args()
    gen = FilmGrainGenerator(width=args.width, height=args.height, fps=args.fps,
                             seconds=args.seconds, preset=args.preset, seed=args.seed,
                             coherence=args.coherence, regen_every=args.regen_every)
    if args.export == 'video':
        print(f"Exporting direct video to {args.video_path} (fourcc={args.codec})…")
        gen.render(out_dir=args.outdir, as_video=True, video_path=args.video_path, fourcc_str=args.codec)
        print("Done.")
    else:
        print(f"Exporting PNG sequence to {args.outdir} …")
        gen.render(out_dir=args.outdir, as_video=False)
        print("PNG export done.")
        if args.ffmpeg != 'none':
            out = maybe_encode_with_ffmpeg(args.outdir, args.fps, args.ffmpeg, ffmpeg_bin=args.ffmpeg_bin)
            if out:
                print(f"Encoded via ffmpeg → {out}")

if __name__ == '__main__':
    main()
