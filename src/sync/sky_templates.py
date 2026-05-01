#!/usr/bin/env python3
"""
extract_sky_templates.py
Extract Sky Italia watermark templates (MGP logo + PUBBLICITÀ text) from a
source broadcast and save them as grayscale PNGs in fingerprints/.

These PNGs are pre-built fingerprints loaded by sync_sky_it.py at runtime,
avoiding the need to re-extract templates from the source video each run.

Requirements: ffmpeg/ffprobe on PATH, numpy

Usage:
    python extract_sky_templates.py [--pubb-time=S] <sky_file>
"""

import subprocess, sys
import numpy as np
from pathlib import Path

from watermark_detection import build_watermark_template, get_video_dimensions

FP_DIR = Path(__file__).parent.parent.parent / 'fingerprints'

# ── Sky Italia watermark regions (1280×720) ───────────────────────────────────

# MGP logo (bottom-right, present during program, absent during ads)
MGP_WM_X,  MGP_WM_Y  = 1124, 662
MGP_WM_W,  MGP_WM_H  = 115,  43
MGP_OUT_W, MGP_OUT_H = 115,  43

# PUBBLICITÀ text overlay (bottom-right, appears on bumper at break start)
PUBB_X,  PUBB_Y  = 1107, 609
PUBB_W,  PUBB_H  = 100,  35
PUBB_OUT_W, PUBB_OUT_H = 100, 35


def save_template_png(template, out_path, out_w=None, out_h=None):
    """Save a flat float32 watermark template as a grayscale PNG via ffmpeg."""
    if out_w is not None and out_h is not None:
        assert out_w * out_h == len(template), \
            f'Template size {len(template)} does not match {out_w}x{out_h}'
    else:
        out_w = int(np.sqrt(len(template)) + 0.5)
        out_h = len(template) // out_w
        assert out_w * out_h == len(template), \
            f'Template size {len(template)} is not a valid square; pass out_w/out_h'

    # Normalize to 0-255 uint8
    t = template - template.min()
    if t.max() > 0:
        t = t / t.max() * 255.0
    pixels = t.astype(np.uint8).tobytes()

    subprocess.run(
        ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
         '-f', 'rawvideo', '-pix_fmt', 'gray',
         '-s', f'{out_w}x{out_h}',
         '-i', 'pipe:0',
         str(out_path)],
        input=pixels, check=True)


def find_pubblicita_ref_time(src, scan_start=300, scan_end=1800, fps=0.5):
    """
    Auto-detect a frame where the PUBBLICITÀ text overlay is visible.

    Scans from scan_start to scan_end looking for bright white text pixels
    against a relatively dark background in the PUBBLICITÀ crop region.
    Returns the best candidate time, or None.
    """
    scan_dur = max(0.0, scan_end - scan_start)
    if scan_dur <= 0:
        return None

    import time, tempfile
    tmp = os.path.join(tempfile.gettempdir(),
                       f'_tmp_pubb_ref_{int(time.time()*1000)}.raw')

    try:
        subprocess.run(
            ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
             '-ss', f'{scan_start:.1f}', '-t', f'{scan_dur:.0f}',
             '-i', str(src),
             '-vf', (f'fps={fps},'
                     f'format=gray,crop={PUBB_W}:{PUBB_H}:{PUBB_X}:{PUBB_Y},'
                     f'scale={PUBB_OUT_W}:{PUBB_OUT_H}'),
             '-f', 'rawvideo', '-pix_fmt', 'gray', tmp],
            check=True)
        # WSL timing
        for _ in range(10):
            if os.path.exists(tmp):
                break
            time.sleep(0.2)
        if not os.path.exists(tmp):
            return None
        frames = np.fromfile(tmp, dtype=np.uint8).astype(np.float32)
        os.remove(tmp)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        return None

    n_pixels = PUBB_OUT_W * PUBB_OUT_H
    n_frames = len(frames) // n_pixels
    if n_frames == 0:
        return None

    step = 1.0 / fps
    best_t = None
    best_score = -1.0

    for i in range(n_frames):
        frame = frames[i * n_pixels:(i + 1) * n_pixels]
        mean_val = frame.mean()
        # PUBBLICITÀ text: white text (high brightness) on dark bumper clip
        # Background mean should be moderate (not sky-bright, not black)
        if 30 < mean_val < 160:
            # Check top half (where text is) vs bottom half
            top = frame[:n_pixels // 2].mean()
            bot = frame[n_pixels // 2:].mean()
            if bot > 1:  # avoid division by zero
                score = top / bot
                if score > best_score:
                    best_score = score
                    best_t = scan_start + i * step

    if best_t is not None:
        print(f'  PUBBLICITÀ reference found at {best_t:.1f}s (score={best_score:.2f})')
    else:
        print('  WARNING: Could not auto-detect PUBBLICITÀ reference time')

    return best_t


def find_mgp_ref_time(src, scan_start=120, scan_end=600, fps=0.5):
    """
    Auto-detect a frame where the MGP logo is visible (program content).
    Looks for moderate brightness with the MGP region showing the
    characteristic logo pattern (not sky, not ad content).
    """
    scan_dur = max(0.0, scan_end - scan_start)
    if scan_dur <= 0:
        return None

    import time, tempfile
    tmp = os.path.join(tempfile.gettempdir(),
                       f'_tmp_mgp_ref_{int(time.time()*1000)}.raw')

    try:
        subprocess.run(
            ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
             '-ss', f'{scan_start:.1f}', '-t', f'{scan_dur:.0f}',
             '-i', str(src),
             '-vf', (f'fps={fps},'
                     f'format=gray,crop={MGP_WM_W}:{MGP_WM_H}:{MGP_WM_X}:{MGP_WM_Y},'
                     f'scale={MGP_OUT_W}:{MGP_OUT_H}'),
             '-f', 'rawvideo', '-pix_fmt', 'gray', tmp],
            check=True)
        for _ in range(10):
            if os.path.exists(tmp):
                break
            time.sleep(0.2)
        if not os.path.exists(tmp):
            return None
        frames = np.fromfile(tmp, dtype=np.uint8).astype(np.float32)
        os.remove(tmp)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        return None

    n_pixels = MGP_OUT_W * MGP_OUT_H
    n_frames = len(frames) // n_pixels
    if n_frames == 0:
        return None

    step = 1.0 / fps
    best_t = None
    best_score = -1.0

    for i in range(n_frames):
        frame = frames[i * n_pixels:(i + 1) * n_pixels]
        mean_val = frame.mean()
        # MGP logo on dark gray background: moderate brightness (40–140)
        if 40 < mean_val < 140:
            # Logo has some structure (not flat gray) — check std
            std = frame.std()
            score = std  # more structure = better logo visibility
            if score > best_score:
                best_score = score
                best_t = scan_start + i * step

    if best_t is not None:
        print(f'  MGP reference found at {best_t:.1f}s (std={best_score:.2f})')
    else:
        print('  WARNING: Could not auto-detect MGP reference time')

    return best_t


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Extract Sky Italia watermark templates and save as PNG fingerprints.')
    parser.add_argument('--pubb-time', type=float, default=None,
                        help='Reference time for PUBBLICITÀ template (auto-detected if omitted)')
    parser.add_argument('--mgp-time', type=float, default=None,
                        help='Reference time for MGP template (auto-detected if omitted)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Detect reference times but do not extract/save templates')
    parser.add_argument('sky_file', help='Sky Italia broadcast file')
    args = parser.parse_args()

    sky_file = Path(args.sky_file)
    if not sky_file.exists():
        sys.exit(f'ERROR: File not found: {sky_file}')

    print(f'Sky file: {sky_file.name}')

    FP_DIR.mkdir(parents=True, exist_ok=True)

    # ── MGP template ──────────────────────────────────────────────────────────
    print('\n-- MGP watermark template --')
    mgp_time = args.mgp_time
    if mgp_time is None:
        mgp_time = find_mgp_ref_time(sky_file)
    if mgp_time is None:
        mgp_time = 300.0
        print(f'  Using fallback MGP reference time: {mgp_time:.1f}s')

    mgp_template = build_watermark_template(
        sky_file, mgp_time, MGP_WM_X, MGP_WM_Y, MGP_WM_W, MGP_WM_H,
        out_w=MGP_OUT_W, out_h=MGP_OUT_H)
    if mgp_template is None:
        sys.exit('ERROR: Failed to extract MGP template')

    mgp_path = FP_DIR / 'sky_it_mgp.png'
    if not args.dry_run:
        save_template_png(mgp_template, mgp_path, out_w=MGP_OUT_W, out_h=MGP_OUT_H)
        print(f'  Saved: {mgp_path}')
    else:
        print(f'  [DRY RUN] Would save: {mgp_path}')

    # ── PUBBLICITÀ template ───────────────────────────────────────────────────
    print('\n-- PUBBLICITÀ text template --')
    pubb_time = args.pubb_time
    if pubb_time is None:
        pubb_time = find_pubblicita_ref_time(sky_file)
    if pubb_time is None:
        pubb_time = 600.0
        print(f'  Using fallback PUBBLICITÀ reference time: {pubb_time:.1f}s')

    pubb_template = build_watermark_template(
        sky_file, pubb_time, PUBB_X, PUBB_Y, PUBB_W, PUBB_H,
        out_w=PUBB_OUT_W, out_h=PUBB_OUT_H)
    if pubb_template is None:
        sys.exit('ERROR: Failed to extract PUBBLICITÀ template')

    pubb_path = FP_DIR / 'sky_it_pubb.png'
    if not args.dry_run:
        save_template_png(pubb_template, pubb_path, out_w=PUBB_OUT_W, out_h=PUBB_OUT_H)
        print(f'  Saved: {pubb_path}')
    else:
        print(f'  [DRY RUN] Would save: {pubb_path}')

    print('\nDone.')


if __name__ == '__main__':
    import os
    main()
