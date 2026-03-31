#!/usr/bin/env python3
"""
sync_rtbf.py
Add RTBF TIPIK broadcast audio to a MotoGP master file.

RTBF recordings are individual race files (not combined like most other sources).
Ad breaks are detected via:
  1. Sting pair detection: lead-in + lead-out stings at break boundaries
  2. Watermark detection: RTBF TIPIK (top-right) and MGP logo (bottom-right)

Sync anchors (from Notes.md, 50fps):
  Moto3:  RTBF frame 5186  = 103.72s  | Master frame 52,362  = 1047.24s
  Moto2:  RTBF frame 35296 = 705.92s  | Master frame 52,185  = 1043.70s
  MotoGP: RTBF frame 72679 = 1453.58s | Master frame 122,100  = 2442.00s

Requirements: ffmpeg/ffprobe on PATH, numpy, scipy

Usage:
    python sync_rtbf.py [--dry-run] [--anchor-source=S] [--anchor-master=S]
                        [--race={moto3|moto2|motogp}]
                        [--wm-fps=1] [--wm-min-break=15]
                        [--watermark-ref-time=S]
                        <rtbf_file> <master_file> <output_dir>
"""

import sys, os, subprocess, argparse
import numpy as np
from pathlib import Path

from audio_utils import SR, get_duration, get_audio_stream_count, extract_wav, extract_seg, concat_segments_to_mka
from sting_detection import find_sting, find_all_transitions
from watermark_detection import build_watermark_template

# Fingerprints directory
FP_DIR = Path(__file__).parent / 'fingerprints'

# RTBF FPS
RTBF_FPS = 50

# Watermark positions (confirmed)
# TIPIK logo: top-right corner
RTBF_WM_X, RTBF_WM_Y = 1780, 20
RTBF_WM_W, RTBF_WM_H = 120, 40

# MGP logo: bottom-right corner
MGP_WM_X, MGP_WM_Y = 1800, 1020
MGP_WM_W, MGP_WM_H = 100, 60

# Watermark lag: watermark appears 0s after content resumes (RTBF)
RTBF_WM_LAG_SECS = 0.0

# Default watermark reference time (frame 91586 at 50fps = 1831.72s)
DEFAULT_WM_REF_TIME = 1831.72


# ── Optimized watermark scanning (vectorized, fast seeking) ──────────────

def find_breaks_via_watermark_fast(src, wm_template, wm_x, wm_y, wm_w, wm_h,
                                    out_w, out_h, scan_start, scan_end,
                                    fps=1, thresh=0.44, min_break_secs=45,
                                    tmp_suffix=''):
    """
    Optimized watermark scanning with:
    - Lower FPS (default 1 instead of 2)
    - Vectorized correlation (no Python loop)
    - Fast seeking in ffmpeg
    
    Returns list of (break_start, break_end) tuples.
    """
    scan_dur = max(0.0, scan_end - scan_start)
    if scan_dur <= 0:
        return []

    n_pixels = out_w * out_h
    tmp = f'_tmp_wm_fast{tmp_suffix}.raw'
    
    try:
        # Use -ss before -i for faster seeking (input seeking)
        subprocess.run(
            ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
             '-ss', f'{scan_start:.3f}', '-t', f'{scan_dur:.0f}',
             '-i', str(src),
             '-vf', (f'fps={fps},'
                     f'crop={wm_w}:{wm_h}:{wm_x}:{wm_y},'
                     f'scale={out_w}:{out_h}'),
             '-f', 'rawvideo', '-pix_fmt', 'gray', tmp],
            check=True)
        frames = np.fromfile(tmp, dtype=np.uint8).astype(np.float32)
        os.remove(tmp)
    except Exception as e:
        print(f'  WARNING: Watermark scan failed: {e}')
        if os.path.exists(tmp):
            os.remove(tmp)
        return []

    n_frames = len(frames) // n_pixels
    if n_frames == 0:
        print('  WARNING: Watermark scan produced no frames.')
        return []

    step = 1.0 / fps
    t0 = wm_template - wm_template.mean()
    t0_norm = np.linalg.norm(t0)
    
    # Vectorized correlation: reshape frames and compute all at once
    frames_2d = frames.reshape(n_frames, n_pixels)
    frames_centered = frames_2d - frames_2d.mean(axis=1, keepdims=True)
    
    # Compute norms for all frames
    frame_norms = np.linalg.norm(frames_centered, axis=1)
    
    # Vectorized Pearson correlation
    dot_products = np.dot(frames_centered, t0)
    corrs = dot_products / (frame_norms * t0_norm + 1e-10)
    
    # Find contiguous below-threshold regions (breaks)
    breaks = []
    in_break = False
    break_start = None
    min_break_frames = int(min_break_secs * fps)
    i = 0
    
    while i < n_frames:
        if not in_break:
            if corrs[i] < thresh:
                in_break = True
                break_start = scan_start + i * step
        else:
            if corrs[i] >= thresh:
                region_frames = i - int((break_start - scan_start) / step)
                if region_frames >= min_break_frames:
                    break_end = scan_start + i * step
                    breaks.append((break_start, break_end))
                    print(f'    Break: {fmt(break_start)} - {fmt(break_end)}  '
                          f'dur={fmt(break_end - break_start)}')
                in_break = False
        i += 1
    
    # Handle break extending to end
    if in_break:
        region_frames = n_frames - int((break_start - scan_start) / step)
        if region_frames >= min_break_frames:
            break_end = scan_end
            breaks.append((break_start, break_end))
            print(f'    Break (to end): {fmt(break_start)} - {fmt(break_end)}')

    return breaks


# ── Watermark-based break detection ──────────────────────────────────────

def detect_watermark_ref_time(src, search_start=300, search_dur=600,
                              stream_spec='v:0', tmp_suffix=''):
    """
    Find a suitable watermark reference time in the video.
    Default: frame 91586 (1831.72s) confirmed as having both logos visible.
    Returns time in seconds.
    """
    d = get_duration(src)
    ref_time = DEFAULT_WM_REF_TIME
    if ref_time > d - 60:
        ref_time = min(search_start + 180, d - 60)
    print(f'  Watermark reference time: {ref_time:.1f}s (frame {int(ref_time * RTBF_FPS)})')
    return ref_time


def detect_rtbf_watermark_breaks(src, d_src, scan_start=0, scan_end=None,
                                  wm_ref_time=None, wm_fps=2,
                                  min_break=15, tmp_suffix=''):
    """
    Detect ad breaks via RTBF TIPIK watermark absence.
    Falls back to MGP logo if TIPIK template fails.
    Returns list of (break_start, break_end) tuples.
    """
    if scan_end is None:
        scan_end = d_src - 10.0

    if wm_ref_time is None:
        wm_ref_time = detect_watermark_ref_time(src, tmp_suffix=tmp_suffix)

    # Build watermark template from RTBF TIPIK logo
    print(f'\n  Building RTBF TIPIK watermark template at {wm_ref_time:.1f}s...')
    rtbf_template = build_watermark_template(
        src, wm_ref_time, RTBF_WM_X, RTBF_WM_Y, RTBF_WM_W, RTBF_WM_H,
        out_w=32, out_h=32, tmp_suffix=f'{tmp_suffix}_rtbf')

    # Also build MGP logo template as fallback
    print(f'  Building MGP logo watermark template at {wm_ref_time:.1f}s...')
    mgp_template = build_watermark_template(
        src, wm_ref_time, MGP_WM_X, MGP_WM_Y, MGP_WM_W, MGP_WM_H,
        out_w=32, out_h=32, tmp_suffix=f'{tmp_suffix}_mgp')

    if rtbf_template is None and mgp_template is None:
        print('  WARNING: Could not build any watermark template')
        return []

    # Prefer RTBF template, fall back to MGP
    template = rtbf_template if rtbf_template is not None else mgp_template
    wm_x = RTBF_WM_X if rtbf_template is not None else MGP_WM_X
    wm_y = RTBF_WM_Y if rtbf_template is not None else MGP_WM_Y
    wm_w = RTBF_WM_W if rtbf_template is not None else MGP_WM_W
    wm_h = RTBF_WM_H if rtbf_template is not None else MGP_WM_H
    wm_name = "RTBF TIPIK" if rtbf_template is not None else "MGP logo"

    print(f'\n  Scanning for breaks via {wm_name} watermark (optimized)...')
    print(f'  Scan: {scan_start:.1f}s - {scan_end:.1f}s  FPS: {wm_fps}')

    breaks = find_breaks_via_watermark_fast(
        src, template,
        wm_x, wm_y, wm_w, wm_h,
        32, 32,
        scan_start=scan_start, scan_end=scan_end,
        fps=wm_fps, thresh=0.44, min_break_secs=min_break,
        tmp_suffix=f'{tmp_suffix}_scan')

    if not breaks:
        print(f'  WARNING: No breaks detected via {wm_name} watermark')
    else:
        print(f'\n  {len(breaks)} break(s) detected via {wm_name} watermark:')
        for i, (s, e) in enumerate(breaks):
            print(f'    Break {i+1}: {fmt(s)} - {fmt(e)}  dur={fmt(e-s)}')

    return breaks


# ── Sting-based break detection ──────────────────────────────────────────

def detect_rtbf_sting_breaks(src, d_src, search_start=0, tmp_suffix=''):
    """
    Detect ad breaks via lead-in/lead-out sting pairs.
    Returns list of (break_start, break_end) tuples.
    """
    fp_leadin = str(FP_DIR / 'rtbf_tipik_leadin.wav')
    fp_leadout = str(FP_DIR / 'rtbf_tipik_leadout.wav')

    if not Path(fp_leadin).exists() or not Path(fp_leadout).exists():
        print('  WARNING: Missing RTBF fingerprint files')
        print(f'    Lead-in:  {fp_leadin}')
        print(f'    Lead-out: {fp_leadout}')
        return []

    print('\n── Sting-based break detection ──')
    print(f'  Scanning from {search_start:.1f}s to end of file...')

    # Find all lead-in stings (ad break starts)
    print('\n  Finding lead-in stings (ad break starts)...')
    leadin_hits = find_all_transitions(
        src, fp_leadin, stream_spec='0:a:0',
        tmp_suffix=f'{tmp_suffix}_leadin',
        conf_thresh=0.3, suppress_secs=120, min_event_secs=search_start)

    # Find all lead-out stings (ad break ends)
    print('\n  Finding lead-out stings (ad break ends)...')
    leadout_hits = find_all_transitions(
        src, fp_leadout, stream_spec='0:a:0',
        tmp_suffix=f'{tmp_suffix}_leadout',
        conf_thresh=0.3, suppress_secs=120, min_event_secs=search_start)

    print(f'\n  Found {len(leadin_hits)} lead-in hits, {len(leadout_hits)} lead-out hits')

    # Pair lead-in with lead-out stings
    breaks = []
    clip_dur = 5.0  # approximate duration of fingerprint clips

    for i, (leadin_time, leadin_conf, _) in enumerate(leadin_hits):
        # Find next lead-out after this lead-in
        for leadout_time, leadout_conf, _ in leadout_hits:
            if leadout_time > leadin_time + 30:  # minimum 30s gap
                break_dur = (leadout_time + clip_dur) - leadin_time
                if 60 < break_dur < 600:  # reasonable break duration
                    break_end = leadout_time + clip_dur
                    breaks.append((leadin_time, break_end))
                    print(f'    Break {len(breaks)}: {fmt(leadin_time)} - {fmt(break_end)}  '
                          f'dur={fmt(break_dur)}  (leadin_conf={leadin_conf:.3f}, '
                          f'leadout_conf={leadout_conf:.3f})')
                    break

    if not breaks:
        print('  WARNING: No break pairs detected via stings')

    return breaks


# ── Break detection (combined) ───────────────────────────────────────────

def detect_rtbf_breaks(src, d_src, use_watermark=False, wm_ref_time=None,
                       wm_fps=2, min_break=15, search_start=0, tmp_suffix=''):
    """
    Detect ad breaks using sting pairs and/or watermark detection.
    Returns list of (break_start, break_end) tuples.
    """
    breaks = []

    # Sting-based detection (primary for RTBF)
    sting_breaks = detect_rtbf_sting_breaks(
        src, d_src, search_start=search_start, tmp_suffix=tmp_suffix)

    if use_watermark:
        wm_breaks = detect_rtbf_watermark_breaks(
            src, d_src, scan_start=search_start, wm_ref_time=wm_ref_time,
            wm_fps=wm_fps, min_break=min_break, tmp_suffix=tmp_suffix)

        # Merge breaks from both methods
        breaks = sorted(set(sting_breaks + wm_breaks))
    else:
        breaks = sting_breaks

    return breaks


# ── Break handling ────────────────────────────────────────────────────────

def breaks_to_content_windows(breaks, d_src, opening_dur=0.0):
    """
    Convert list of breaks to content windows.
    Returns [(start, end), ...] covering all non-break content.
    """
    if not breaks:
        return [(opening_dur, d_src)]

    windows = []
    prev_end = opening_dur

    for s, e in breaks:
        if s > prev_end + 1.0:
            windows.append((prev_end, s))
        prev_end = e

    if prev_end < d_src - 1.0:
        windows.append((prev_end, d_src))

    return windows


# ── Segment building and concatenation ────────────────────────────────────

def build_and_concat(src_file, master_file, src_stream, ns_stream,
                     offset, d_src, d_master, content_windows,
                     output_mka, dry_run=False, tmp_prefix='rtbf'):
    """
    Build output from RTBF content windows and NS from master.

    content_windows: [(c_start, c_end), ...] in RTBF time (seconds)
    offset: rtbf_time + offset = master_time

    Output: NS head | rtbf[w0] | NS break1 | rtbf[w1] | ... | NS tail
    """
    print(f'\n  Offset: {offset:.3f}s  (rtbf t=0 -> master t={offset:.3f}s)')
    for i, (c_s, c_e) in enumerate(content_windows):
        print(f'  Content {i+1}: rtbf {c_s:.1f}-{c_e:.1f}s  '
              f'-> master {c_s+offset:.1f}-{c_e+offset:.1f}s')

    tmp_dir = Path(f'_tmp_{tmp_prefix}_segs')
    if not dry_run:
        tmp_dir.mkdir(exist_ok=True)
    segs = []
    counter = [0]

    def add_ns(m_start, m_end, desc):
        m_start = max(0.0, m_start)
        m_end = min(d_master, m_end)
        dur = m_end - m_start
        if dur <= 0:
            return
        counter[0] += 1
        p = str(tmp_dir / f'seg_{counter[0]:04d}.wav')
        print(f'  {desc}  master {m_start:.1f}-{m_end:.1f}s  dur={dur:.1f}s')
        if not dry_run:
            extract_seg(master_file, p, ns_stream, m_start, dur)
        segs.append(p)

    def add_rtbf(c_start, c_end, desc):
        c_start = max(0.0, c_start)
        c_end = min(d_src, c_end, d_master - offset)
        dur = c_end - c_start
        if dur <= 0:
            return
        counter[0] += 1
        p = str(tmp_dir / f'seg_{counter[0]:04d}.wav')
        print(f'  {desc}  rtbf {c_start:.1f}-{c_end:.1f}s  dur={dur:.1f}s')
        if not dry_run:
            extract_seg(src_file, p, src_stream, c_start, dur)
        segs.append(p)

    if not content_windows:
        add_ns(0.0, d_master, '[NS]  full')
    else:
        first_m = content_windows[0][0] + offset
        if first_m > 0:
            add_ns(0.0, first_m, '[NS]  head')
        elif first_m < 0:
            print(f'  NOTE: RTBF content starts {-first_m:.1f}s before master')

        for i, (c_start, c_end) in enumerate(content_windows):
            m_c_start = c_start + offset
            if m_c_start < 0:
                c_start = c_start + (-m_c_start)
            add_rtbf(c_start, c_end, f'[RTBF] content {i+1}')

            ns_m_start = c_end + offset
            if i + 1 < len(content_windows):
                ns_m_end = content_windows[i+1][0] + offset
                add_ns(ns_m_start, ns_m_end, f'[NS]  break {i+1}')
            else:
                add_ns(ns_m_start, d_master, '[NS]  tail')

    if dry_run:
        print(f'\n[DRY RUN] Would concatenate {len(segs)} segments -> {output_mka}')
        return

    concat_segments_to_mka(segs, output_mka, list_path_prefix=f'_tmp_{tmp_prefix}')
    for f in tmp_dir.iterdir():
        f.unlink()
    tmp_dir.rmdir()


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Sync RTBF TIPIK broadcast audio to a MotoGP master file.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Detect and plan only; do not encode.')
    parser.add_argument('--race', choices=['moto3', 'moto2', 'motogp'],
                        help='Race type for automatic anchor lookup.')
    parser.add_argument('--anchor-source', type=float, default=None,
                        help='Frame-based anchor time in RTBF source (seconds).')
    parser.add_argument('--anchor-master', type=float, default=None,
                        help='Frame-based anchor time in master file (seconds).')
    parser.add_argument('--watermark', action='store_true',
                        help='Use watermark detection for breaks.')
    parser.add_argument('--wm-fps', type=float, default=1,
                        help='Frames per second for watermark scanning (default: 1).')
    parser.add_argument('--wm-min-break', type=float, default=15,
                        help='Minimum break duration for watermark detection (default: 15s).')
    parser.add_argument('--wm-ref-time', type=float, default=None,
                        help='Time in seconds to extract watermark reference template.')
    parser.add_argument('--sting-only', action='store_true',
                        help='Use only sting detection, skip watermark.')
    parser.add_argument('--program-start', type=float, default=None,
                        help='RTBF time (seconds) where program content begins. '
                             'NS fills before this; any breaks entirely within the '
                             'pre-program section are discarded.')
    parser.add_argument('--program-end', type=float, default=None,
                        help='RTBF time (seconds) where program content ends. '
                             'NS fills everything after this point.')
    parser.add_argument('rtbf_file')
    parser.add_argument('master_file')
    parser.add_argument('output_dir')
    args = parser.parse_args()

    if args.dry_run:
        print('[DRY RUN] Detection and planning only - no audio will be encoded.')

    # Validate fingerprint files
    fp_leadin = FP_DIR / 'rtbf_tipik_leadin.wav'
    fp_leadout = FP_DIR / 'rtbf_tipik_leadout.wav'
    if not fp_leadin.exists():
        sys.exit(f'ERROR: Missing fingerprint: {fp_leadin}')
    if not fp_leadout.exists():
        sys.exit(f'ERROR: Missing fingerprint: {fp_leadout}')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_mka = Path(args.output_dir) / (Path(args.rtbf_file).stem + '_rtbf_synced.mka')

    # ── Durations ──
    d_src = get_duration(args.rtbf_file)
    d_master = get_duration(args.master_file)
    print(f'RTBF:   {d_src:.1f}s ({d_src/3600:.2f}h)')
    print(f'Master: {d_master:.1f}s ({d_master/3600:.2f}h)')

    # ── Detect Natural Sounds stream ──
    n_audio = get_audio_stream_count(args.master_file)
    ns_stream = f'0:a:{n_audio - 1}'
    print(f'Master audio tracks: {n_audio}  ->  Natural Sounds on {ns_stream}')

    # ── Sync anchor ──
    if args.anchor_source is not None and args.anchor_master is not None:
        offset = args.anchor_master - args.anchor_source
        print(f'\nFrame-based anchor: RTBF {args.anchor_source:.3f}s = master {args.anchor_master:.3f}s')
        print(f'  Offset: {offset:.3f}s')
    elif args.race is not None:
        # Automatic anchor lookup from Notes.md
        anchors = {
            'moto3':  (5186 / RTBF_FPS,  52362 / 50),   # 103.72s, 1047.24s
            'moto2':  (35296 / RTBF_FPS, 52185 / 50),   # 705.92s, 1043.70s
            'motogp': (72679 / RTBF_FPS, 122100 / 50),  # 1453.58s, 2442.00s
        }
        src_anchor, master_anchor = anchors[args.race]
        offset = master_anchor - src_anchor
        print(f'\nAutomatic anchor for {args.race}:')
        print(f'  RTBF:   frame {int(src_anchor * RTBF_FPS)} = {src_anchor:.3f}s')
        print(f'  Master: frame {int(master_anchor * 50)} = {master_anchor:.3f}s')
        print(f'  Offset: {offset:.3f}s')
    else:
        sys.exit('ERROR: Either --race or --anchor-source/--anchor-master must be specified.')

    # ── Break detection ──
    print('\n── Ad break detection ──')
    if args.sting_only:
        breaks = detect_rtbf_sting_breaks(args.rtbf_file, d_src, tmp_suffix='_main')
    else:
        breaks = detect_rtbf_breaks(
            args.rtbf_file, d_src,
            use_watermark=args.watermark,
            wm_ref_time=args.wm_ref_time,
            wm_fps=args.wm_fps,
            min_break=args.wm_min_break,
            tmp_suffix='_main')

    # Filter breaks to master duration
    master_end = d_master - offset
    breaks = [(s, e) for s, e in breaks if s + offset < d_master and e + offset > 0]
    if breaks:
        print(f'\n  {len(breaks)} break(s) within master duration:')
        for i, (s, e) in enumerate(breaks):
            ms = s + offset
            me = e + offset
            print(f'    Break {i+1}: RTBF {fmt(s)}-{fmt(e)} -> Master {fmt(ms)}-{fmt(me)}')
    else:
        print('  No breaks detected or all breaks outside master duration.')

    content_windows = breaks_to_content_windows(breaks, d_src)

    # Apply program-start: discard content windows that end before program_start,
    # and clip the first surviving window's start to program_start.
    if args.program_start is not None and args.program_start > 0:
        content_windows = [
            (max(c_s, args.program_start), c_e)
            for c_s, c_e in content_windows
            if c_e > args.program_start
        ]
        print(f'\n  Program start at {fmt(args.program_start)}: '
              f'{len(content_windows)} content window(s) after filtering')

    # Apply program-end: drop windows that start at or after program_end,
    # and clip the last surviving window to program_end.
    if args.program_end is not None:
        content_windows = [
            (c_s, min(c_e, args.program_end))
            for c_s, c_e in content_windows
            if c_s < args.program_end
        ]
        print(f'\n  Program end at {fmt(args.program_end)}: '
              f'{len(content_windows)} content window(s) after filtering')

    # ── Build output ──
    print('\nBuilding output segments...')
    build_and_concat(
        args.rtbf_file, args.master_file,
        '0:a:0', ns_stream,
        offset, d_src, d_master,
        content_windows, output_mka, dry_run=args.dry_run)

    print(f'\nDone -> {output_mka}')


def fmt(secs):
    """Format seconds as hh:mm:ss.xx"""
    neg = secs < 0
    s = abs(secs)
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f'{"-" if neg else ""}{h:02d}:{m:02d}:{sec:05.2f}'


if __name__ == '__main__':
    main()
