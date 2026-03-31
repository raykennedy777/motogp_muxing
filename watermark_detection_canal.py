#!/usr/bin/env python3
"""
watermark_detection_canal.py
Detect ad breaks in Canal+ broadcasts by monitoring the presence/absence of
the Canal+ logo (top-right) and MotoGP logo (bottom-right).

Break starts when the Canal+ logo disappears.
Break ends when either the Canal+ logo reappears or the MGP logo appears.

Resolution-aware: works with 1920x1080 (MotoGP UHD) and 1280x720 (Sport 360).
"""

import subprocess, os, struct, sys
import numpy as np
from pathlib import Path

# ── Resolution profiles ────────────────────────────────────────────────────────

PROFILES = {
    '1080': {
        'width': 1920, 'height': 1080,
        # Canal+ logo ROI (top-right): black box with white "CANAL+" text
        'canal_roi': (1630, 20, 250, 80),    # x, y, w, h
        # MGP logo ROI (bottom-right): white "MGP" stylized text
        'mgp_roi':   (1650, 940, 250, 120),
    },
    '720': {
        'width': 1280, 'height': 720,
        # Scaled ROI for 720p (approximate, will need verification)
        'canal_roi': (1087, 13, 167, 53),
        'mgp_roi':   (1100, 627, 167, 80),
    },
}


def extract_frame(src, time_sec, stream_spec='0:v:0'):
    """Extract a single frame as raw RGB24 bytes at the source resolution."""
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-ss', f'{time_sec:.3f}',
        '-i', str(src),
        '-frames:v', '1',
        '-map', stream_spec,
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-'
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.stdout


def parse_frame(raw_bytes, width, height):
    """Parse raw RGB24 bytes into a numpy array (H, W, 3)."""
    expected = width * height * 3
    if len(raw_bytes) != expected:
        raise ValueError(f'Expected {expected} bytes, got {len(raw_bytes)}')
    return np.frombuffer(raw_bytes, dtype=np.uint8).reshape((height, width, 3))


def mean_brightness(frame_rgb, roi):
    """Return mean luminance in the ROI region."""
    x, y, w, h = roi
    crop = frame_rgb[y:y+h, x:x+w]
    # Simple luminance: mean of all RGB values
    return float(np.mean(crop))


def detect_logos(frame_rgb, profile):
    """Check if Canal+ and MGP logos are present. Returns (canal_present, mgp_present)."""
    canal_roi = profile['canal_roi']
    mgp_roi = profile['mgp_roi']

    canal_brightness = mean_brightness(frame_rgb, canal_roi)
    mgp_brightness = mean_brightness(frame_rgb, mgp_roi)

    # Canal+ logo: white text on black box -> brightness ~80-120 with logo, ~10-30 without
    # MGP logo: white text on content -> brightness varies but >30 with logo
    # Threshold 50 gives clean separation: logo present = 57-250, absent = 0-45
    canal_present = canal_brightness > 50
    mgp_present = mgp_brightness > 30

    return canal_present, mgp_present, canal_brightness, mgp_brightness


def scan_video_fast(src, start_frame, end_frame, fps=50, profile_key='1080',
                    step=10):
    """
    Fast scan: extract frames in a single ffmpeg pass using select filter.
    Much faster than individual frame extractions.
    """
    profile = PROFILES[profile_key]
    width = profile['width']
    height = profile['height']
    canal_roi = profile['canal_roi']
    mgp_roi = profile['mgp_roi']

    start_time = start_frame / fps
    duration = (end_frame - start_frame) / fps

    print(f'Fast-scanning {src}')
    print(f'  Frames: {start_frame}-{end_frame}, step={step}, fps={fps}')
    print(f'  Profile: {profile_key} ({width}x{height})')
    print(f'  Time range: {start_time:.1f}s - {start_time + duration:.1f}s')
    print()

    # Use ffmpeg to extract every Nth frame as raw video
    cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error',
        '-ss', f'{start_time:.3f}',
        '-t', f'{duration:.3f}',
        '-i', str(src),
        '-vf', f'select=not(mod(n\\,{step}))',
        '-vsync', 'vfr',
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-s', f'{width}x{height}',
        '-'
    ]

    result = subprocess.run(cmd, capture_output=True)
    raw = result.stdout

    frame_size = width * height * 3
    n_frames = len(raw) // frame_size
    print(f'  Extracted {n_frames} frames ({len(raw)} bytes)')

    results = []
    prev_canal = None

    for i in range(n_frames):
        offset = i * frame_size
        frame_data = raw[offset:offset + frame_size]
        frame_rgb = np.frombuffer(frame_data, dtype=np.uint8).reshape((height, width, 3))

        frame_num = start_frame + i * step
        time_sec = frame_num / fps

        canal_b = float(np.mean(frame_rgb[canal_roi[1]:canal_roi[1]+canal_roi[3],
                                           canal_roi[0]:canal_roi[0]+canal_roi[2]]))
        mgp_b = float(np.mean(frame_rgb[mgp_roi[1]:mgp_roi[1]+mgp_roi[3],
                                         mgp_roi[0]:mgp_roi[0]+mgp_roi[2]]))
        canal_p = canal_b > 50
        mgp_p = mgp_b > 30

        results.append((frame_num, time_sec, canal_p, mgp_p, canal_b, mgp_b))

        if canal_p != prev_canal:
            state = 'PRESENT' if canal_p else 'ABSENT'
            print(f'  Frame {frame_num} ({fmt_time(time_sec)}): Canal+ {state} '
                  f'(b={canal_b:.0f}) MGP={mgp_p} (b={mgp_b:.0f})')
            prev_canal = canal_p

    print(f'\n  Done. {len(results)} frames scanned.')
    return results


def find_breaks(results, min_break_frames=10):
    """
    Analyze scan results to find ad breaks.
    Returns list of (break_start_frame, break_end_frame) tuples.
    """
    breaks = []
    in_break = False
    break_start = None

    for i, (frame, time_sec, canal_p, mgp_p, canal_b, mgp_b) in enumerate(results):
        if not canal_p and not in_break:
            # Canal+ logo disappeared -> potential break start
            in_break = True
            break_start = frame
        elif canal_p and in_break:
            # Canal+ logo reappeared -> break end
            break_end = frame
            duration_frames = break_end - break_start
            if duration_frames >= min_break_frames:
                breaks.append((break_start, break_end))
            in_break = False

    # Handle break still open at end
    if in_break and break_start is not None:
        break_end = results[-1][0]
        duration_frames = break_end - break_start
        if duration_frames >= min_break_frames:
            breaks.append((break_start, break_end))

    return breaks


def fmt_time(seconds):
    """Format seconds as hh:mm:ss.xx"""
    neg = seconds < 0
    s = abs(seconds)
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f'{"-" if neg else ""}{h:02d}:{m:02d}:{sec:05.2f}'


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Detect ad breaks in Canal+ broadcasts via watermark detection.')
    parser.add_argument('src', help='Video file to scan')
    parser.add_argument('--start-frame', type=int, default=0,
                        help='First frame to scan')
    parser.add_argument('--end-frame', type=int, default=None,
                        help='Last frame to scan (default: end of file)')
    parser.add_argument('--fps', type=float, default=50,
                        help='Frames per second (default: 50)')
    parser.add_argument('--profile', choices=['1080', '720'], default='1080',
                        help='Resolution profile')
    parser.add_argument('--step', type=int, default=5,
                        help='Frame step (check every Nth frame)')
    parser.add_argument('--min-break-frames', type=int, default=50,
                        help='Minimum break duration in frames')
    parser.add_argument('--dry-run', action='store_true',
                        help='Test on a small segment around a known break')
    args = parser.parse_args()

    from audio_utils import get_duration
    d = get_duration(args.src)
    total_frames = int(d * args.fps)
    end_frame = args.end_frame or total_frames

    if args.dry_run:
        # Scan around the known break: 222000-225700
        args.start_frame = 222000
        end_frame = 225700
        args.step = 2  # finer granularity for validation
        print('=== DRY RUN: scanning known break region ===')

    results = scan_video_fast(args.src, args.start_frame, end_frame,
                              fps=args.fps, profile_key=args.profile,
                              step=args.step)

    print(f'\n=== Results ({len(results)} frames scanned) ===\n')

    breaks = find_breaks(results, min_break_frames=args.min_break_frames)
    if breaks:
        print(f'Found {len(breaks)} break(s):\n')
        for i, (bs, be) in enumerate(breaks):
            bt_start = bs / args.fps
            bt_end = be / args.fps
            dur = (be - bs) / args.fps
            print(f'  Break {i+1}:')
            print(f'    Start: frame {bs} ({fmt_time(bt_start)})')
            print(f'    End:   frame {be} ({fmt_time(bt_end)})')
            print(f'    Dur:   {fmt_time(dur)} ({dur:.1f}s, {be-bs} frames)')
            print()
    else:
        print('No breaks found.')

    # Print per-frame results for known break validation
    if args.dry_run:
        print('\n=== Per-frame detail (transitions only) ===\n')
        prev_canal = None
        for frame, ts, canal_p, mgp_p, canal_b, mgp_b in results:
            if canal_p != prev_canal:
                state = 'PRESENT' if canal_p else 'ABSENT'
                print(f'  Frame {frame} ({fmt_time(ts)}): Canal+ {state} '
                      f'(b={canal_b:.0f}) MGP={mgp_p} (b={mgp_b:.0f})')
                prev_canal = canal_p


if __name__ == '__main__':
    main()
