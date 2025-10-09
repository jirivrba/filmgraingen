#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Film Grain Generator – 35mm 80s look, 4K overlay, photochemical model
CLI with PNG/video export and optional ffmpeg ProRes/H.264 encoding.

This version generates a NEW grain realization for EVERY FRAME (like real film),
with controllable temporal coherence so changes feel organic rather than harsh.

The generator can now render **perfectly looping grain segments** using
`--loop-seconds`. When exporting video (directly or via ffmpeg),
`--loop-count` repeats the seamless loop without regenerating extra PNG frames,
making it easy to build long overlays without visible seams.

Usage examples:
  python3 film_grain_generator_cli.py --export png --outdir grain_png --fps 24 \
      --loop-seconds 10 --loop-count 3 --film-sensitivity ASA200
  python3 film_grain_generator_cli.py --export video --video-path film_grain_4k.mp4 \
      --fps 24 --loop-seconds 10 --loop-count 2 --film-sensitivity ASA200 --codec mp4v
  # Encode PNGs to ProRes 422 HQ via ffmpeg (after generating PNGs into --outdir):
  python3 film_grain_generator_cli.py --export png --outdir grain_png --fps 24 --loop-seconds 12 \
      --loop-count 1 --ffmpeg prores422hq

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
from copy import deepcopy
from itertools import count
from pathlib import Path
from typing import Tuple


BASE_FILM_PROFILES = {
    "ASA100": {
        "diam_um_range": (8.0, 11.0),
        "density_Mpx": 4400,
        "channel_mul": (1.22, 1.0, 0.86),
        "asa": 100,
    },
    "ASA125": {
        "diam_um_range": (9.0, 13.5),
        "density_Mpx": 4700,
        "channel_mul": (1.2, 1.0, 0.88),
        "asa": 125,
    },
    "ASA200": {
        "diam_um_range": (11.0, 16.0),
        "density_Mpx": 5200,
        "channel_mul": (1.32, 1.0, 0.82),
        "asa": 200,
    },
    "ASA400": {
        "diam_um_range": (16.0, 22.0),
        "density_Mpx": 6100,
        "channel_mul": (1.4, 1.0, 0.78),
        "asa": 400,
    },
    "ASA500": {
        "diam_um_range": (18.0, 28.0),
        "density_Mpx": 6800,
        "channel_mul": (1.42, 1.0, 0.78),
        "asa": 500,
    },
    "ASA800": {
        "diam_um_range": (22.0, 30.0),
        "density_Mpx": 7100,
        "channel_mul": (1.48, 1.0, 0.74),
        "asa": 800,
    },
}


def build_profile(base_key: str, **overrides) -> dict:
    profile = deepcopy(BASE_FILM_PROFILES[base_key])
    profile.update(overrides)
    return profile

# -------------------------
# Utils
# -------------------------

def prepare_output_directory(p: str, require_unique: bool = True) -> Path:
    """Create an output directory, optionally enforcing uniqueness per run."""
    path = Path(p)

    if not require_unique:
        path.mkdir(parents=True, exist_ok=True)
        return path

    if not path.exists():
        path.mkdir(parents=True, exist_ok=False)
        return path

    base_name = path.name
    parent = path.parent
    for idx in count(1):
        candidate = parent / f"{base_name}_{idx:02d}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate

    return path  # Fallback, though loop should always return

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

    FILM_STOCK_PROFILES = {
        "ASA100": build_profile("ASA100"),
        "ASA125": build_profile("ASA125"),
        "ASA200": build_profile("ASA200"),
        "ASA400": build_profile("ASA400"),
        "ASA500": build_profile("ASA500"),
        "ASA800": build_profile("ASA800"),
        "KODAK_EASTMAN_5247": build_profile(
            "ASA125",
            diam_um_range=(9.0, 13.0),
            density_Mpx=4700,
            channel_mul=(1.18, 1.0, 0.9),
            look_overrides={
                "tint_strength": 1.25,
                "warmth_shift": (1.03, 1.0, 0.95),
                "contrast": 0.94,
                "shadow_boost": 1.08,
                "channel_mul_scale": (1.05, 1.0, 0.96),
            },
        ),
        "KODAK_EASTMAN_5293": build_profile(
            "ASA200",
            diam_um_range=(12.0, 18.0),
            density_Mpx=5500,
            channel_mul=(1.28, 1.0, 0.84),
            look_overrides={
                "tint_strength": 1.35,
                "warmth_shift": (0.99, 1.0, 1.04),
                "contrast": 0.9,
                "shadow_boost": 1.12,
                "density_scale": 1.08,
                "channel_mul_scale": (1.08, 1.0, 0.9),
            },
        ),
        "KODAK_EASTMAN_5294": build_profile(
            "ASA400",
            diam_um_range=(16.0, 23.0),
            density_Mpx=6300,
            channel_mul=(1.36, 1.0, 0.8),
            look_overrides={
                "tint_strength": 1.4,
                "warmth_shift": (0.97, 1.0, 1.06),
                "contrast": 0.88,
                "shadow_boost": 1.18,
                "density_scale": 1.12,
                "channel_mul_scale": (1.12, 1.0, 0.88),
            },
        ),
        "KODAK_VISION_500T": build_profile(
            "ASA500",
            diam_um_range=(20.0, 30.0),
            density_Mpx=7000,
            channel_mul=(1.44, 1.0, 0.76),
            look_overrides={
                "tint_strength": 1.5,
                "warmth_shift": (1.02, 1.0, 0.92),
                "contrast": 0.86,
                "shadow_boost": 1.22,
                "density_scale": 1.15,
                "channel_mul_scale": (1.15, 1.0, 0.85),
            },
        ),
    }

    LOOK_PRESETS = {
        "LOOK80S": {
            "tint_strength": 1.0,
            "warmth_shift": (1.0, 1.0, 1.0),
            "flicker_mult": 1.0,
            "shadow_boost": 1.0,
            "contrast": 1.0,
            "diameter_scale": 1.0,
            "density_scale": 1.0,
            "channel_mul_scale": (1.0, 1.0, 1.0)
        },
        "LOOK70S": {
            "tint_strength": 1.35,
            "warmth_shift": (1.08, 1.0, 0.92),
            "flicker_mult": 1.35,
            "shadow_boost": 1.2,
            "contrast": 0.85,
            "diameter_scale": 1.25,
            "density_scale": 1.12,
            "channel_mul_scale": (1.15, 1.05, 0.9)
        },
    }

    def __init__(self, width=3840, height=2160, fps=24, loop_seconds=10,
                 loop_count=1, seconds=None,
                 film_speed="ASA200", look="LOOK80S", seed=None,
                 coherence=0.3,  # 0=new every frame (chaotic), 0.3=organic, 0.6=smoother
                 regen_every=1,    # regenerate master every N frames (1 = per-frame)
                 debug=False
                 ):
        self.width = width
        self.height = height
        self.fps = fps
        self.debug = bool(debug)
        # backwards compatibility: `seconds` overrides loop_seconds when provided
        if seconds is not None:
            loop_seconds = seconds
        self.loop_seconds = max(0.01, float(loop_seconds))
        self.loop_count = max(1, int(loop_count))
        self.loop_frames = max(1, int(round(self.fps * self.loop_seconds)))
        self.total_frames = self.loop_frames * self.loop_count
        self.film_speed = film_speed
        self.look = look
        self.rng = np.random.default_rng(seed)
        self.um_per_px = (self.GATE_WIDTH_MM * 1000.0) / float(self.width)  # ~5.7 µm/px @ 3840
        self.coherence = float(np.clip(coherence, 0.0, 0.95))
        self.regen_every = max(1, int(regen_every))
        if self.film_speed not in self.FILM_STOCK_PROFILES:
            raise ValueError(f"Unknown film sensitivity preset: {self.film_speed}")
        if self.look not in self.LOOK_PRESETS:
            raise ValueError(f"Unknown look preset: {self.look}")

        self.stock_cfg = self.FILM_STOCK_PROFILES[self.film_speed]
        self.asa_rating = self.stock_cfg.get("asa")
        base_look_cfg = dict(self.LOOK_PRESETS[self.look])
        stock_overrides = self.stock_cfg.get("look_overrides", {})
        if stock_overrides:
            base_look_cfg.update(stock_overrides)
        self.look_cfg = base_look_cfg

        self.tint_strength = float(self.look_cfg.get("tint_strength", 1.0))
        self.warmth_shift = np.array(self.look_cfg.get("warmth_shift", (1.0, 1.0, 1.0)),
                                     dtype=np.float32)
        if self.warmth_shift.shape != (3,):
            raise ValueError("warmth_shift must be a triple of RGB multipliers")
        self.flicker_mult = float(self.look_cfg.get("flicker_mult", 1.0))
        self.shadow_boost_strength = float(self.look_cfg.get("shadow_boost", 1.0))
        self.contrast_strength = float(self.look_cfg.get("contrast", 1.0))
        self.diameter_scale = float(self.look_cfg.get("diameter_scale", 1.0))
        self.density_scale = float(self.look_cfg.get("density_scale", 1.0))
        self.channel_mul_scale = np.array(self.look_cfg.get("channel_mul_scale", (1.0, 1.0, 1.0)),
                                          dtype=np.float32)
        if self.channel_mul_scale.shape != (3,):
            raise ValueError("channel_mul_scale must be a triple of multipliers")
        self._debug(
            "Initialized generator: %dx%d @ %dfps, loop %.2fs x%d (total frames %d), "
            "film %s, look %s, coherence %.2f, regen_every %d"
            % (
                self.width, self.height, self.fps,
                self.loop_seconds, self.loop_count, self.total_frames,
                self.film_speed, self.look, self.coherence, self.regen_every,
            )
        )
        self._prepare_static_fields()

    def _debug(self, message: str):
        if self.debug:
            print(f"[DEBUG] {message}")

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
        self._debug(
            "Prepared static fields: clump_mod[%.3f..%.3f], tint per-channel means %s"
            % (
                float(self.clump_mod.min()),
                float(self.clump_mod.max()),
                np.array2string(self.color_tint.mean(axis=(0, 1)), precision=3),
            )
        )

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
        diam_range_um = tuple(d * self.diameter_scale for d in self.stock_cfg["diam_um_range"])
        density = (self.stock_cfg["density_Mpx"] * self.density_scale *
                   self.stock_cfg["channel_mul"][channel_index] * self.channel_mul_scale[channel_index])
        total_grains = int(density * (w*h/1e6))
        self._debug(
            "Channel %d tile: diam range %.2f-%.2fµm (scale %.2f), target grains %d"
            % (
                channel_index,
                diam_range_um[0],
                diam_range_um[1],
                self.diameter_scale,
                total_grains,
            )
        )

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
        self._debug(
            "Channel %d tile prepared (min %.3f max %.3f)"
            % (channel_index, float(base.min()), float(base.max()))
        )
        return base

    def _temporal_params(self, t: int):
        """Return flicker/lift, gate weave and texture drift for frame t."""
        flicker = 1.0 + (0.03 * self.flicker_mult) * math.sin(2*math.pi * t / (self.fps*1.7))
        flicker += self.rng.normal(0, 0.005 * self.flicker_mult)
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
        out_dir_path = prepare_output_directory(out_dir, require_unique=not as_video)
        out_dir_str = str(out_dir_path)

        if out_dir_str != out_dir:
            print(f"Output directory '{out_dir}' exists. Using '{out_dir_str}' for this run.")

        loops_to_render = self.loop_count if as_video else 1
        frames_expected = self.loop_frames * loops_to_render

        if not as_video and self.loop_count > 1:
            print(
                "PNG export writes a single seamless loop. "
                "Additional repeats will be handled during encoding."
            )

        # Tone curve LUT (kept in float via np.interp)
        x = np.linspace(0, 1, 1024, dtype=np.float32)
        slope = max(1.0, 8 * self.contrast_strength)
        s_curve = 1/(1 + np.exp(-slope*(x-0.5)))
        s_curve = (s_curve - s_curve.min()) / (s_curve.max()-s_curve.min())

        writer = None
        if as_video:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            writer = cv2.VideoWriter(video_path, fourcc, self.fps, (self.width, self.height))

        # Pre-calculate digits for PNG naming when exporting long loops
        digits = max(4, int(math.log10(frames_expected))+1 if frames_expected > 0 else 4)
        target_desc = f"video → {video_path} ({fourcc_str})" if as_video else f"PNGs in {out_dir_str}"
        self._debug(
            "Starting render: %s, frames_to_render=%d, loop_frames=%d, digits=%d"
            % (target_desc, frames_expected, self.loop_frames, digits)
        )

        # Store RNG state so we can re-run the same loop seamlessly
        initial_rng_state = deepcopy(self.rng.bit_generator.state)
        # Cache of the first frames (float16) used for end-of-loop blending
        blend_frames = min(self.loop_frames, max(2, int(round(self.fps * 0.5))))
        self._debug(f"Loop blend frames: {blend_frames}")
        first_loop_cache = [None] * blend_frames

        frame_index = 0
        for loop_idx in range(loops_to_render):
            self._debug(f"Loop {loop_idx+1}/{loops_to_render} starting")
            # Reset random generator to start-of-loop state
            self.rng.bit_generator.state = deepcopy(initial_rng_state)
            # Initialize previous grain tiles for temporal coherence blending
            prev_tiles = [self._generate_grain_tile_once(ch) for ch in range(3)]

            for t in range(self.loop_frames):
                flicker, lift, dx, dy, rot, ux, uy = self._temporal_params(t)
                frame = np.zeros((self.height, self.width, 3), np.float32)

                # Generate (or regenerate) tiles and blend with previous to control coherence
                if t % self.regen_every == 0:
                    self._debug(f"Frame {frame_index}: regenerating grain tiles (t={t})")
                    new_tiles = [self._generate_grain_tile_once(ch) for ch in range(3)]
                else:
                    new_tiles = prev_tiles  # between regen intervals, reuse (still moved by drift)
                    self._debug(f"Frame {frame_index}: reusing previous tiles (t={t})")

                for ch in range(3):
                    # Temporal coherence mixing: more previous => smoother, less => more stochastic
                    tile = (1.0 - self.coherence) * new_tiles[ch] + self.coherence * prev_tiles[ch]
                    prev_tiles[ch] = tile  # carry forward blended state (evolves over time)

                    # Apply slow drift (wrap) and gentle color tint per channel
                    tx = int(round(ux)) % self.width
                    ty = int(round(uy)) % self.height
                    shifted = np.roll(np.roll(tile, shift=ty, axis=0), shift=tx, axis=1)
                    tint = 0.92 + (0.16 * self.tint_strength) * (self.color_tint[..., ch] - 0.5)
                    layer = shifted * tint * self.warmth_shift[ch]
                    frame[..., ch] = layer

                # Vignette (broadcast over channels)
                frame *= self.vignette[..., None]

                # Film-like S-curve, flicker/lift
                frame = np.interp(frame, x, s_curve).astype(np.float32)
                frame = frame * flicker + lift

                # Shadow-weighted grain (stronger in low luma)
                luma = 0.114*frame[...,0] + 0.587*frame[...,1] + 0.299*frame[...,2]
                shadow_boost = 1.0 + (0.35 * self.shadow_boost_strength)*(1.0 - luma)
                frame *= shadow_boost[..., None]

                # Gate weave (subpixel jitter + micro-rotation per channel)
                for ch in range(3):
                    frame[..., ch] = self._apply_affine(frame[..., ch], dx, dy, rot)

                # Subtle per-frame sparkle to avoid banding/over-correlation (deterministic per loop)
                sparkle = self.rng.normal(0, 0.006, frame.shape).astype(np.float32)
                frame = np.clip(frame + sparkle, 0.0, 1.0)

                # Cache first frames for blending at the end of the loop to guarantee seamless looping
                if loop_idx == 0 and t < blend_frames:
                    first_loop_cache[t] = frame.astype(np.float16, copy=True)

                # Blend the tail of the loop back into the cached first frames (perfect loop)
                if blend_frames > 0 and t >= self.loop_frames - blend_frames:
                    idx = t - (self.loop_frames - blend_frames)
                    alpha = (idx + 1) / float(blend_frames)
                    if 0 <= idx < len(first_loop_cache):
                        cached = first_loop_cache[idx]
                        if cached is not None:
                            target = cached.astype(np.float32)
                            frame = frame * (1.0 - alpha) + target * alpha
                        elif loop_idx == 0:
                            # Edge-case: extremely short loops where the cache has not yet been
                            # populated (overlap between head/tail). Store the current frame so
                            # subsequent loops can still blend seamlessly.
                            first_loop_cache[idx] = frame.astype(np.float16, copy=True)

                # Export
                if writer is not None:
                    bgr8 = to_uint8(frame[..., ::-1])  # RGB->BGR
                    writer.write(bgr8)
                else:
                    rgb8 = to_uint8(frame)
                    cv2.imwrite(os.path.join(out_dir_str, f"grain_{frame_index:0{digits}d}.png"),
                                cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR))

                frame_index += 1
                if self.debug and (
                    frame_index % max(1, self.fps) == 0 or frame_index == frames_expected
                ):
                    self._debug(f"Exported frame {frame_index}/{frames_expected}")

        if writer is not None:
            writer.release()

        self._debug(f"Render complete. Frames written: {frame_index} (expected {frames_expected})")
        return out_dir_str

# -------------------------
# ffmpeg helper
# -------------------------

def maybe_encode_with_ffmpeg(out_dir: str, fps: int, mode: str, loop_count: int = 1,
                             ffmpeg_bin: str = "ffmpeg"):
    """
    Encode PNG sequence grain_%04d.png into a video in the same folder.
    mode ∈ {"none", "prores422hq", "prores4444", "h264"}
    loop_count controls how many seamless repeats are baked into the encoded clip.
    """
    if mode == "none":
        return None
    if shutil.which(ffmpeg_bin) is None:
        print(f"WARNING: ffmpeg binary '{ffmpeg_bin}' not found in PATH. Skipping encoding.")
        return None

    input_pattern = os.path.join(out_dir, "grain_%04d.png")
    stream_loop = []
    repeats = max(1, int(loop_count))
    if repeats > 1:
        stream_loop = ['-stream_loop', str(repeats - 1)]
    if mode == "prores422hq":
        out_path = os.path.join(out_dir, "film_grain_prores422HQ.mov")
        cmd = [ffmpeg_bin, '-y', *stream_loop, '-r', str(fps), '-i', input_pattern,
               '-c:v', 'prores_ks', '-profile:v', '3', '-pix_fmt', 'yuv422p10le',
               '-movflags', '+faststart', out_path]
    elif mode == "prores4444":
        out_path = os.path.join(out_dir, "film_grain_prores4444.mov")
        cmd = [ffmpeg_bin, '-y', *stream_loop, '-r', str(fps), '-i', input_pattern,
               '-c:v', 'prores_ks', '-profile:v', '4', '-pix_fmt', 'yuv444p10le',
               '-movflags', '+faststart', out_path]
    elif mode == "h264":
        out_path = os.path.join(out_dir, "film_grain_h264.mp4")
        cmd = [ffmpeg_bin, '-y', *stream_loop, '-r', str(fps), '-i', input_pattern,
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
    p = argparse.ArgumentParser(description="Film Grain Generator – 35mm film stocks with eras")
    p.add_argument('--width', type=int, default=3840)
    p.add_argument('--height', type=int, default=2160)
    p.add_argument('--fps', type=int, default=24)
    p.add_argument('--loop-seconds', type=float, default=None,
                   help='Length of one seamless grain loop (seconds).')
    p.add_argument('--loop-count', type=int, default=1,
                   help='How many times to repeat the seamless loop in the export.')
    p.add_argument('--seconds', type=float, default=None,
                   help='[Deprecated] Alias for --loop-seconds for backwards compatibility.')
    p.add_argument('--film-sensitivity', '--preset', dest='film_speed',
                   choices=sorted(FilmGrainGenerator.FILM_STOCK_PROFILES.keys()),
                   default='ASA200',
                   help='Photochemical stock sensitivity preset (ASA100/125/200/400/500/800 or Kodak Eastman/Vision calibrations).')
    p.add_argument('--look', choices=sorted(FilmGrainGenerator.LOOK_PRESETS.keys()), default='LOOK80S',
                   help='Color/era styling to combine with film sensitivity (LOOK80S, LOOK70S).')
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
    p.add_argument('--debug', action='store_true', help='Enable verbose debug logging to stdout')
    return p.parse_args()


def main():
    args = parse_args()
    loop_seconds = args.loop_seconds if args.loop_seconds is not None else (
        args.seconds if args.seconds is not None else 10.0)
    if args.debug:
        print("Debug logging enabled.")
    gen = FilmGrainGenerator(width=args.width, height=args.height, fps=args.fps,
                             loop_seconds=loop_seconds, loop_count=args.loop_count, seconds=None,
                             film_speed=args.film_speed, look=args.look, seed=args.seed,
                             coherence=args.coherence, regen_every=args.regen_every,
                             debug=args.debug)
    if args.export == 'video':
        print(f"Exporting direct video to {args.video_path} (fourcc={args.codec})…")
        _ = gen.render(out_dir=args.outdir, as_video=True, video_path=args.video_path, fourcc_str=args.codec)
        print("Done.")
    else:
        print(f"Exporting PNG sequence to {args.outdir} …")
        actual_out_dir = gen.render(out_dir=args.outdir, as_video=False)
        print("PNG export done.")
        if args.ffmpeg != 'none':
            out = maybe_encode_with_ffmpeg(
                actual_out_dir, args.fps, args.ffmpeg,
                loop_count=args.loop_count, ffmpeg_bin=args.ffmpeg_bin
            )
            if out:
                print(f"Encoded via ffmpeg → {out}")

if __name__ == '__main__':
    main()
