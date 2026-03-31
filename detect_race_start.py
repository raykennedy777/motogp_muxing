#!/usr/bin/env python3
"""
detect_race_start.py
Detect prerace sting, race start (via starting lights), and first camera change.

Pipeline:
1. Sting detection — find prerace_sting_motogp.wav in the video
2. Race start detection — track 5 red lights in PiP overlay going out
3. Camera change detection — frame-to-frame correlation at 50 fps

Usage:
    python detect_race_start.py [video_file]
"""

import sys, os, subprocess, numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
os.chdir(SCRIPT_DIR)
sys.path.insert(0, str(SCRIPT_DIR))

from audio_utils import SR, get_duration, extract_wav, load_fp_wav, _peak, fmt

SRC = "/mnt/c/Users/raisi/Downloads/MotoGP.2026.Round03.USA.Race.WEB-DL.1080p.H264.English-MWR/MotoGP.2026.Round03.USA.Race.WEB-DL.1080p.H264.English-MWR.mkv"
FP_PATH = str(SCRIPT_DIR / "fingerprints" / "prerace_sting_motogp.wav")

TMP_DIR = Path("/tmp/motogp_detect")
TMP_DIR.mkdir(exist_ok=True)

def tmp_path(name):
    return str(TMP_DIR / name)


# --- Step 1: Sting detection ---

def find_prerace_sting(src, fp_path):
    """Find the prerace sting. Search from 28 min to 34 min."""
    print("=== Step 1: Finding prerace sting ===")
    
    tmp = tmp_path('_tmp_sting_detect.wav')
    if os.path.exists(tmp):
        os.remove(tmp)
    
    search_start = 1680  # 28 min
    search_dur = 360
    actual_dur = min(search_dur, get_duration(src) - search_start)
    
    print(f"  Extracting {actual_dur:.1f}s from {fmt(search_start)}...")
    extract_wav(src, tmp, "0:a:0", start=search_start, duration=actual_dur)
    
    if not os.path.exists(tmp):
        print("  ERROR: Failed to extract audio")
        return search_start, 0.0
    
    haystack = load_fp_wav(tmp)
    if os.path.exists(tmp):
        os.remove(tmp)
    
    needle = load_fp_wav(fp_path)
    
    idx, conf = _peak(haystack, needle)
    t = search_start + idx / SR
    
    print(f"  Sting found at {fmt(t)} (conf={conf:.4f})")
    return t, conf


# --- Step 2: Race start via starting lights detection ---

def detect_race_start_by_lights(src, sting_end, min_delay=150, max_delay=330, fps=10):
    """
    Detect race start by tracking the 5 red starting lights in the PiP overlay.
    
    The lights are in the top-left graphical overlay:
    - y: 74-90 (in 640x360 scaled frame)
    - x: 5 light positions at approximately 26-75
    
    Race start = moment when all 5 red lights go out.
    
    min_delay/max_delay: race start expected between these seconds after sting end.
                         Default: 2.5 to 5.5 minutes (150-330s).
    """
    print(f"\n=== Step 2: Detecting race start via starting lights ===")
    print(f"  Searching from {fmt(sting_end + min_delay)} to {fmt(sting_end + max_delay)}")
    
    search_start = sting_end + min_delay
    search_dur = max_delay - min_delay
    
    tmp = tmp_path("_tmp_lights.raw")
    if os.path.exists(tmp):
        os.remove(tmp)
    
    # Extract frames at fps, focusing on top portion of frame
    subprocess.run(
        ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
         '-ss', f'{search_start:.3f}', '-t', f'{search_dur:.1f}',
         '-i', src,
         '-vf', f'fps={fps},scale=640:360',
         '-f', 'rawvideo', '-pix_fmt', 'rgb24', tmp],
        check=True)
    
    if not os.path.exists(tmp):
        print("  ERROR: Failed to extract frames")
        return sting_end + 10
    
    frames = np.fromfile(tmp, dtype=np.uint8)
    os.remove(tmp)
    
    h, w, c = 360, 640, 3
    n = len(frames) // (h * w * c)
    frames = frames[:n * h * w * c].reshape(n, h, w, c)
    
    print(f"  Extracted {n} frames at {fps}fps")
    
    # 5 light positions in PiP overlay (y=74-90, x ranges)
    lights_x_ranges = [
        (26, 31),
        (39, 43),
        (50, 54),
        (62, 65),
        (72, 75)
    ]
    y_range = (74, 90)
    
    # Track lights sequence
    race_start = None
    consecutive_5_lights = 0
    
    for i in range(n):
        t = search_start + i / fps
        frame = frames[i]
        
        # Count how many lights are currently red
        lights_on = 0
        for x_lo, x_hi in lights_x_ranges:
            region = frame[y_range[0]:y_range[1], x_lo:x_hi+1]
            red_mask = (region[:,:,0] > 150) & (region[:,:,1] < 100) & (region[:,:,2] < 100)
            if np.sum(red_mask) > 5:
                lights_on += 1
        
        # Track consecutive frames with all 5 lights on
        if lights_on == 5:
            consecutive_5_lights += 1
        else:
            # Check if we had sustained 5 lights and now they're out
            if consecutive_5_lights >= 3:  # At least 3 consecutive frames (0.3s at 10fps)
                if lights_on <= 1:  # Lights went out
                    race_start = t
                    print(f"  Sustained {consecutive_5_lights} frames of 5 lights, then out at {fmt(t)}")
                    break
            consecutive_5_lights = 0
    
    if race_start is None:
        print("  WARNING: Lights out not detected properly")
        race_start = sting_end + 270  # Default: 4.5 min after sting (middle of range)
    
    return race_start


# --- Step 3: Camera change detection ---

def detect_first_camera_change(src, race_start, search_window=15, fps=50):
    """
    Detect the first camera cut after race start using frame correlation.
    
    At 50fps:
    - Normal motion: corr > 0.7
    - Camera cut: corr < 0.3
    
    Returns frame number (at 50fps) and timestamp.
    """
    print(f"\n=== Step 3: Detecting first camera change ===")
    print(f"  Scanning from {fmt(race_start)} for {search_window}s at {fps} fps")
    
    out_w, out_h = 320, 180
    n_pixels = out_w * out_h
    
    tmp = tmp_path("_tmp_frames.raw")
    if os.path.exists(tmp):
        os.remove(tmp)
    
    subprocess.run(
        ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
         '-ss', f'{race_start:.3f}', '-t', f'{search_window:.1f}',
         '-i', src,
         '-vf', f'scale={out_w}:{out_h}',
         '-f', 'rawvideo', '-pix_fmt', 'gray', tmp],
        check=True)
    
    frames = np.fromfile(tmp, dtype=np.uint8).astype(np.float32)
    if os.path.exists(tmp):
        os.remove(tmp)
    
    n_frames = len(frames) // n_pixels
    print(f"  Extracted {n_frames} frames")
    
    if n_frames < 2:
        print("  ERROR: Not enough frames")
        return None
    
    frames = frames[:n_frames * n_pixels].reshape(n_frames, n_pixels)
    
    # Compute frame-to-frame correlation
    correlations = np.zeros(n_frames - 1)
    for i in range(n_frames - 1):
        f0 = frames[i] - frames[i].mean()
        f1 = frames[i + 1] - frames[i + 1].mean()
        norm0 = np.linalg.norm(f0)
        norm1 = np.linalg.norm(f1)
        if norm0 > 1e-10 and norm1 > 1e-10:
            correlations[i] = np.dot(f0, f1) / (norm0 * norm1)
        else:
            correlations[i] = 0.0
    
    # Find first camera cut (corr < 0.3, skip first 10 frames)
    THRESH = 0.3
    MIN_FRAME = 10
    
    first_cut_idx = None
    for i in range(MIN_FRAME, len(correlations)):
        if correlations[i] < THRESH:
            first_cut_idx = i
            break
    
    # Find most dramatic cut
    min_corr_idx = np.argmin(correlations[MIN_FRAME:]) + MIN_FRAME
    
    # Convert to 50fps frame numbers
    race_start_frame = int(race_start * fps)
    
    print(f"\n  Correlation statistics:")
    print(f"    Mean: {correlations.mean():.4f}")
    print(f"    Min:  {correlations.min():.4f}")
    print(f"    Max:  {correlations.max():.4f}")
    
    if first_cut_idx is not None:
        first_cut_frame = race_start_frame + first_cut_idx + 1
        first_cut_time = race_start + (first_cut_idx + 1) / fps
        print(f"\n  First camera change (corr < {THRESH}):")
        print(f"    Frame: {first_cut_frame} at {fmt(first_cut_time)}")
        print(f"    Correlation: {correlations[first_cut_idx]:.4f}")
    else:
        print(f"\n  No camera change found with correlation < {THRESH}")
        first_cut_frame = None
        first_cut_time = None
    
    min_corr_frame = race_start_frame + min_corr_idx + 1
    min_corr_time = race_start + (min_corr_idx + 1) / fps
    print(f"\n  Most dramatic camera change:")
    print(f"    Frame: {min_corr_frame} at {fmt(min_corr_time)}")
    print(f"    Correlation: {correlations[min_corr_idx]:.4f}")
    
    return {
        'first_cut_frame': first_cut_frame,
        'first_cut_time': first_cut_time,
        'first_cut_corr': correlations[first_cut_idx] if first_cut_idx is not None else None,
        'min_corr_frame': min_corr_frame,
        'min_corr_time': min_corr_time,
        'min_corr': correlations[min_corr_idx],
    }


# --- Main ---

def main():
    src = SRC
    if len(sys.argv) > 1:
        src = sys.argv[1]
    
    if not os.path.exists(src):
        print(f"ERROR: File not found: {src}")
        sys.exit(1)
    
    print(f"Processing: {os.path.basename(src)}")
    print(f"Duration: {fmt(get_duration(src))}")
    print()
    
    # Step 1: Find prerace sting
    sting_time, sting_conf = find_prerace_sting(src, FP_PATH)
    sting_end = sting_time + 65.3
    
    # Step 2: Detect race start via starting lights
    race_start = detect_race_start_by_lights(src, sting_end)
    
    # Step 3: Detect first camera change
    result = detect_first_camera_change(src, race_start)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Prerace sting:     {fmt(sting_time)} (conf={sting_conf:.4f})")
    print(f"  Sting ends:        {fmt(sting_end)}")
    print(f"  Race start:        {fmt(race_start)}")
    if result:
        if result['first_cut_frame']:
            print(f"  First camera cut:  Frame {result['first_cut_frame']} at {fmt(result['first_cut_time'])}")
            print(f"  Cut correlation:   {result['first_cut_corr']:.4f}")
        print(f"  Most dramatic cut: Frame {result['min_corr_frame']} at {fmt(result['min_corr_time'])}")
        print(f"  Min correlation:   {result['min_corr']:.4f}")


if __name__ == "__main__":
    main()
