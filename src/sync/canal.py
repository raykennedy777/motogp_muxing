#!/usr/bin/env python3
"""
sync_canal.py
Add Canal+ French audio to MotoGP race files.

Each race type has a different break structure:

  sprint     -- Canal+Sport 360 Sprint: opening trim + ad breaks
  moto3      -- Moto3 race: opening trim, optional ad breaks
  moto2      -- Canal+Sport 360 Moto2: opening trim + ad breaks + silence tail
  motogp     -- Canal+ UHD Moto GP: opening trim + ad breaks
  moto3moto2 -- Combined Moto3+Moto2 file: opening trim + ad breaks
  sunday     -- Combined Sunday file (Moto3+Moto2+MotoGP): watermark break detection

Break detection methods:
  (default): use audio sting correlation + silence detection
  --watermark: use Canal+ logo watermark absence detection (more robust)

Sync anchor: frame-based (--anchor-source / --anchor-master) or sting-based.

Usage:
    python sync_canal.py --race {sprint|moto3|moto2|motogp|moto3moto2|sunday}
                         [--dry-run] [--watermark] [--wm-fps=2]
                         [--wm-min-break=15] [--wm-min-content=60]
                         [--wm-scan-end=S] [--breaks=S:E,S:E,...]
                         [--anchor-source=S] [--anchor-master=S]
                         [--content-end=S] [--opening-dur=S]
                         <canal_file> <master_file> <output_dir>

Requirements: ffmpeg/ffprobe on PATH, numpy, scipy
"""

import sys, os, argparse
import numpy as np
from pathlib import Path
from scipy.io import wavfile

from src.utils.audio_utils import SR, get_duration, get_audio_stream_count, extract_wav, extract_seg, concat_segments_to_mka
from src.utils.sting_detection import find_sting, find_all_transitions
from src.utils.canal_watermark import detect_canal_breaks, fmt as wm_fmt

FP_DIR = Path(__file__).parent / 'fingerprints'


# ── Silence detection ─────────────────────────────────────────────────────────

def find_silence_start(src, search_start, search_dur, stream_spec='0:a:0',
                       silence_threshold=0.02, min_silence_sec=1.8, label='',
                       tmp_suffix=''):
    """
    Find the first position where audio is silent for >= min_silence_sec.
    Returns absolute time in seconds, or None if not found.
    """
    tmp = f'_tmp_canal_silence{tmp_suffix}.wav'
    actual_dur = min(search_dur, get_duration(src) - search_start)
    if actual_dur <= 0:
        return None
    extract_wav(src, tmp, stream_spec, start=search_start, duration=actual_dur)
    sr, data = wavfile.read(tmp)
    os.remove(tmp)

    data = data.astype(np.float32)
    peak = np.abs(data).max()
    if peak > 0:
        data /= peak

    win = max(1, int(sr * 0.1))           # 0.1s analysis window
    min_wins = max(1, int(min_silence_sec / 0.1))
    n_wins = len(data) // win
    rms = np.array([np.sqrt(np.mean(data[i*win:(i+1)*win]**2))
                    for i in range(n_wins)])

    for i in range(n_wins - min_wins + 1):
        if np.all(rms[i:i+min_wins] < silence_threshold):
            t = search_start + i * 0.1
            if label:
                print(f'  {label}: {t:.3f}s ({t/60:.2f} min)')
            return t

    if label:
        print(f'  {label}: NOT FOUND in search window '
              f'({search_start:.0f}-{search_start+actual_dur:.0f}s)')
    return None


# ── Watermark-based break detection ──────────────────────────────────────────

def detect_breaks_watermark(canal_file, d_canal, scan_start=0, scan_end=None,
                            min_break=15, wm_fps=2, min_content=60,
                            tmp_suffix=''):
    """
    Detect all Canal+ ad breaks via logo watermark absence.
    Returns list of (break_start, break_end) tuples, filtered to:
      - After opening trim (12s)
      - Before end-of-file margin (10s)
    """
    if scan_end is None:
        scan_end = d_canal - 10.0  # don't scan to very end

    print(f'\n  Watermark scan: {wm_fmt(scan_start)} - {wm_fmt(scan_end)}')
    print(f'  Min break: {min_break}s  Min content: {min_content}s  FPS: {wm_fps}')

    breaks = detect_canal_breaks(
        canal_file, scan_start=scan_start, scan_end=scan_end,
        method='brightness', fps=wm_fps, min_break_secs=min_break,
        min_content_sec=min_content,
        wm_lag_secs=0.0, tmp_suffix=tmp_suffix)

    # Filter: keep only breaks after 12s opening trim
    breaks = [(s, e) for s, e in breaks if s > 12.0]

    if not breaks:
        print('  WARNING: No breaks detected via watermark')
    else:
        print(f'\n  {len(breaks)} break(s) detected:')
        for i, (s, e) in enumerate(breaks):
            print(f'    Break {i+1}: {wm_fmt(s)} - {wm_fmt(e)}  dur={wm_fmt(e-s)}')

    return breaks


def breaks_to_content_windows(breaks, d_canal, opening_dur=12.0):
    """
    Convert list of breaks to content windows.
    Returns [(start, end), ...] covering all non-break content.
    """
    if not breaks:
        return [(opening_dur, d_canal)]

    windows = []
    prev_end = opening_dur
    for s, e in breaks:
        if s > prev_end + 1.0:  # skip if break starts immediately after previous end
            windows.append((prev_end, s))
        prev_end = e

    # Final window to end of file
    if prev_end < d_canal - 1.0:
        windows.append((prev_end, d_canal))

    return windows


# ── Segment building and concatenation ────────────────────────────────────────

def build_and_concat(canal_file, master_file, canal_stream, ns_stream,
                     offset, d_canal, d_master, content_windows,
                     output_mka, dry_run=False, tmp_prefix='canal'):
    """
    Build output from Canal+ content windows and NS from master.

    content_windows: [(c_start, c_end), ...] in canal time (seconds)
    offset: canal_time + offset = master_time

    Output: NS head | canal[w0] | NS break1 | canal[w1] | ... | NS tail

    TS source files: pre-extracted to WAV before segmenting to avoid PTS
    discontinuities (ad break clock jumps) causing short/mis-timed extractions.
    """
    print(f'\n  Offset: {offset:.3f}s  (canal t=0 -> master t={offset:.3f}s)')
    for i, (c_s, c_e) in enumerate(content_windows):
        print(f'  Content {i+1}: canal {c_s:.1f}-{c_e:.1f}s  '
              f'-> master {c_s+offset:.1f}-{c_e+offset:.1f}s')

    tmp_dir = Path(f'_tmp_{tmp_prefix}_segs')
    if not dry_run:
        tmp_dir.mkdir(exist_ok=True)
    segs    = []
    counter = [0]

    # Pre-extract canal audio for TS files to avoid PTS discontinuity issues.
    # TS ad breaks often contain PTS clock jumps; seeked extraction stops at
    # the jump boundary, producing segments shorter than requested.
    # Extracting the full audio first merges past discontinuities correctly.
    _preextracted_canal_wav = None
    canal_src = canal_file
    if not dry_run and Path(canal_file).suffix.lower() in ('.ts', '.mts', '.m2ts'):
        _preextracted_canal_wav = str(tmp_dir / '_canal_preextract.wav')
        print(f'\n  Pre-extracting canal audio (TS source)...')
        from src.utils.audio_utils import extract_wav as _exwav
        _exwav(canal_file, _preextracted_canal_wav, canal_stream,
               sr=48000, channels=2)
        canal_src = _preextracted_canal_wav
        print(f'  Pre-extraction done.')

    def add_ns(m_start, m_end, desc):
        m_start = max(0.0, m_start)
        m_end   = min(d_master, m_end)
        dur = m_end - m_start
        if dur <= 0:
            return
        counter[0] += 1
        p = str(tmp_dir / f'seg_{counter[0]:04d}.wav')
        print(f'  {desc}  master {m_start:.1f}-{m_end:.1f}s  dur={dur:.1f}s')
        if not dry_run:
            extract_seg(master_file, p, ns_stream, m_start, dur)
        segs.append(p)

    def add_canal(c_start, c_end, desc):
        c_start = max(0.0, c_start)
        c_end   = min(d_canal, c_end, d_master - offset)
        dur = c_end - c_start
        if dur <= 0:
            return
        counter[0] += 1
        p = str(tmp_dir / f'seg_{counter[0]:04d}.wav')
        print(f'  {desc}  canal {c_start:.1f}-{c_end:.1f}s  dur={dur:.1f}s')
        if not dry_run:
            extract_seg(canal_src, p, canal_stream, c_start, dur)
        segs.append(p)

    if not content_windows:
        add_ns(0.0, d_master, '[NS]  full')
    else:
        # NS head: master[0 to first canal content start mapped to master time]
        first_m = content_windows[0][0] + offset
        if first_m > 0:
            add_ns(0.0, first_m, '[NS]  head')
        elif first_m < 0:
            print(f'  NOTE: Canal content starts {-first_m:.1f}s before master')

        for i, (c_start, c_end) in enumerate(content_windows):
            # Trim c_start if it maps before master start
            m_c_start = c_start + offset
            if m_c_start < 0:
                c_start = c_start + (-m_c_start)
            add_canal(c_start, c_end, f'[CANAL] content {i+1}')

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
    if _preextracted_canal_wav and Path(_preextracted_canal_wav).exists():
        Path(_preextracted_canal_wav).unlink()


# ── Unified Canal+ processing ──────────────────────────────────────────────

def process_canal(canal_file, master_file, output_dir, dry_run,
                  anchor_source=None, anchor_master=None,
                  content_end_secs=None, opening_dur_secs=None):
    """
    Unified Canal+ audio sync with consistent ad break detection.

    Lead-in detection (any of):
      - canal_grid_ending.wav (grid ending sting)
      - canal_zarco_ad.wav (Zarco ad)
      - >2s silence

    Lead-out detection (any of):
      - canal_opening.wav
      - preshow_intro_m2m3.wav
      - prerace_sting_motogp.wav
      - canal_m3m2_opening.wav

    Program start: canal_opening sting if --anchor-source not supplied
    """
    CANAL_STREAM = '0:a:0'

    d_canal  = get_duration(canal_file)
    d_master = get_duration(master_file)
    n_audio  = get_audio_stream_count(master_file)
    ns_stream = f'0:a:{n_audio-1}'
    print(f'Canal:  {d_canal:.1f}s  ({d_canal/3600:.2f}h)')
    print(f'Master: {d_master:.1f}s  ({d_master/3600:.2f}h)  NS: {ns_stream}')

    # ── Sync anchor + program start detection ────────────────────────────────
    if anchor_source is not None and anchor_master is not None:
        offset = anchor_master - anchor_source
        print(f'\nFrame-based anchor: canal {anchor_source:.3f}s = master {anchor_master:.3f}s')
        print(f'  Offset: {offset:.3f}s')
    else:
        print('\nLocating sync anchor...')
        
        # Try opening sting
        fp_opening = str(FP_DIR / 'canal_opening.wav')
        canal_opening, c_conf = find_sting(
            canal_file, fp_opening, 600, 900, CANAL_STREAM, 'Opening sting (canal)',
            tmp_suffix='_canal')

        if c_conf < 0.1:
            sys.exit(f'ERROR: Opening sting not found (conf={c_conf:.4f})')

        anchor_source = canal_opening
        print(f'  Using opening sting: {anchor_source:.1f}s (conf={c_conf:.4f})')

        # Find corresponding position in master
        master_opening, m_conf = find_sting(
            master_file, fp_opening, 600, 900, ns_stream, 'Opening sting (master)',
            tmp_suffix='_canal')
        if m_conf < 0.05:
            sys.exit(f'ERROR: Opening sting not found in master (conf={m_conf:.4f})')
        offset = master_opening - anchor_source

    if opening_dur_secs is not None:
        OPENING_DUR = opening_dur_secs
    else:
        OPENING_DUR = anchor_source + 12.0 if anchor_source else 12.0

    # ── Break detection ────────────────────────────────────────────────────
    print('\n-- Sting/silence break detection --')

    LEADIN_FPS = [
        str(FP_DIR / 'canal_grid_ending.wav'),
        str(FP_DIR / 'canal_zarco_ad.wav'),
    ]
    LEADOUT_FPS = [
        str(FP_DIR / 'canal_opening.wav'),
        str(FP_DIR / 'preshow_intro_m2m3.wav'),
        str(FP_DIR / 'prerace_sting_motogp.wav'),
        str(FP_DIR / 'canal_m3m2_opening.wav'),
    ]

    SEARCH_START = OPENING_DUR + 300
    SEARCH_END = d_canal - 60

    print('\nScanning for lead-out stings...')
    leadouts = find_all_transitions(
        canal_file, LEADOUT_FPS, CANAL_STREAM,
        tmp_suffix='_canal',
        conf_thresh=0.3,
        suppress_secs=30,
        min_event_secs=int(SEARCH_START),
    )
    for t, c, d in leadouts:
        print(f'  Lead-out at {t:.1f}s  conf={c:.4f}  sting_dur={d:.1f}s')

    if not leadouts:
        print('WARNING: No lead-out stings found - using full file as content')
        content_windows = [(OPENING_DUR, d_canal)]
    else:
        breaks = []
        SILENCE_WINDOW = 600.0
        MIN_SILENCE_SEC = 2.0

        for t_out, conf, sting_dur in leadouts:
            search_start = max(SEARCH_START, t_out - SILENCE_WINDOW)
            search_dur = t_out - search_start

            print(f'\nSearching for lead-in before lead-out at {t_out:.1f}s...')

            t_grid, grid_conf = find_sting(
                canal_file, str(FP_DIR / 'canal_grid_ending.wav'),
                search_start, search_dur, CANAL_STREAM, 'Grid ending', tmp_suffix='_canal')

            t_zarco, zarco_conf = find_sting(
                canal_file, str(FP_DIR / 'canal_zarco_ad.wav'),
                search_start, search_dur, CANAL_STREAM, 'Zarco ad', tmp_suffix='_canal')

            t_silence = find_silence_start(
                canal_file, search_start, search_dur, CANAL_STREAM,
                min_silence_sec=MIN_SILENCE_SEC, label='Silence', tmp_suffix='_canal')

            # Prefer sting-based lead-ins over silence — silence fires on light
            # background noise and tends to trigger too early.
            sting_candidates = []
            if grid_conf >= 0.3:
                sting_candidates.append(('grid', t_grid))
            if zarco_conf >= 0.3:
                sting_candidates.append(('zarco', t_zarco))

            if sting_candidates:
                sting_candidates.sort(key=lambda x: x[1])
                leadin_type, t_in = sting_candidates[0]
            elif t_silence is not None:
                leadin_type, t_in = 'silence', t_silence
            else:
                leadin_type, t_in = None, None

            if leadin_type is not None:
                t_end = t_out + sting_dur
                print(f'  Break: {t_in:.1f}s - {t_end:.1f}s  (lead-in: {leadin_type}, dur={t_end-t_in:.1f}s)')
                breaks.append((t_in, t_end))
            else:
                print(f'  No lead-in found - skipping lead-out at {t_out:.1f}s')
                continue

        # Merge overlapping breaks
        breaks.sort(key=lambda x: x[0])
        merged = []
        for s, e in breaks:
            if merged and s < merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        breaks = merged

        print(f'\n{len(breaks)} break(s) detected:')
        for i, (s, e) in enumerate(breaks):
            print(f'  Break {i+1}: {s:.1f}s - {e:.1f}s  dur={e-s:.1f}s')

        content_windows = breaks_to_content_windows(breaks, d_canal, OPENING_DUR)

    # ── Content end truncation ─────────────────────────────────────────────
    if content_end_secs is not None:
        print(f'\n  Truncating content at {content_end_secs:.1f}s')
        content_windows = [(s, min(e, content_end_secs)) for s, e in content_windows]
        content_windows = [(s, e) for s, e in content_windows if e > s + 1.0]

    print('\nBuilding output segments...')
    output_mka = Path(output_dir) / (Path(canal_file).stem + '_canal_synced.mka')
    build_and_concat(canal_file, master_file, CANAL_STREAM, ns_stream,
                     offset, d_canal, d_master, content_windows,
                     output_mka, dry_run, tmp_prefix='canal')
    print(f'\nDone -> {output_mka}')


def moto3_mode(canal_file, master_file, output_dir, dry_run,
               anchor_source=None, anchor_master=None,
               use_watermark=False, wm_fps=2, wm_min_break=15, wm_min_content=60,
               opening_dur_secs=None):
    """
    Canal+ Moto3: opening trim + optional ad breaks.
    """
    CANAL_STREAM = '0:a:0'
    OPENING_DUR  = opening_dur_secs if opening_dur_secs is not None else 12.0

    d_canal  = get_duration(canal_file)
    d_master = get_duration(master_file)
    n_audio  = get_audio_stream_count(master_file)
    ns_stream = f'0:a:{n_audio-1}'
    print(f'Canal:  {d_canal:.1f}s  ({d_canal/3600:.2f}h)')
    print(f'Master: {d_master:.1f}s  ({d_master/3600:.2f}h)  NS: {ns_stream}')

    # ── Sync anchor ────────────────────────────────────────────────────────
    if anchor_source is not None and anchor_master is not None:
        offset = anchor_master - anchor_source
        print(f'\nFrame-based anchor: canal {anchor_source:.3f}s = master {anchor_master:.3f}s')
        print(f'  Offset: {offset:.3f}s')
    else:
        print('\nLocating prerace sting (sync anchor)...')
        fp_sting = str(FP_DIR / 'prerace_sting.wav')
        canal_sting, c_conf = find_sting(
            canal_file, fp_sting, 0, min(3600, d_canal), CANAL_STREAM,
            'Prerace sting (canal)', tmp_suffix='_moto3')
        master_sting, m_conf = find_sting(
            master_file, fp_sting, 0, min(3600, d_master), ns_stream,
            'Prerace sting (master)', tmp_suffix='_moto3')
        if c_conf < 0.1 or m_conf < 0.1:
            sys.exit(f'ERROR: Prerace sting not found '
                     f'(canal={c_conf:.4f}, master={m_conf:.4f})')
        offset = master_sting - canal_sting

    # ── Break detection ────────────────────────────────────────────────────
    if use_watermark:
        print('\n-- Watermark break detection --')
        breaks = detect_breaks_watermark(
            canal_file, d_canal, scan_start=0, scan_end=d_canal,
            min_break=wm_min_break, wm_fps=wm_fps, min_content=wm_min_content, tmp_suffix='_moto3wm')
        content_windows = breaks_to_content_windows(breaks, d_canal, OPENING_DUR)
    else:
        content_windows = [(OPENING_DUR, d_canal)]

    print('\nBuilding output segments...')
    output_mka = Path(output_dir) / (Path(canal_file).stem + '_canal_synced.mka')
    build_and_concat(canal_file, master_file, CANAL_STREAM, ns_stream,
                     offset, d_canal, d_master, content_windows,
                     output_mka, dry_run, tmp_prefix='moto3')
    print(f'\nDone -> {output_mka}')


def moto2_mode(canal_file, master_file, output_dir, dry_run,
               anchor_source=None, anchor_master=None,
               use_watermark=False, wm_fps=2, wm_min_break=15, wm_min_content=60):
    """
    Canal+ Moto2: opening trim + ad breaks + silence tail.
    """
    CANAL_STREAM = '0:a:0'
    OPENING_DUR  = 12.0

    d_canal  = get_duration(canal_file)
    d_master = get_duration(master_file)
    n_audio  = get_audio_stream_count(master_file)
    ns_stream = f'0:a:{n_audio-1}'
    print(f'Canal:  {d_canal:.1f}s  ({d_canal/3600:.2f}h)')
    print(f'Master: {d_master:.1f}s  ({d_master/3600:.2f}h)  NS: {ns_stream}')

    # ── Sync anchor ────────────────────────────────────────────────────────
    if anchor_source is not None and anchor_master is not None:
        offset = anchor_master - anchor_source
        print(f'\nFrame-based anchor: canal {anchor_source:.3f}s = master {anchor_master:.3f}s')
        print(f'  Offset: {offset:.3f}s')
    else:
        print('\nLocating prerace sting (sync anchor)...')
        fp_sting = str(FP_DIR / 'prerace_sting.wav')
        canal_sting, c_conf = find_sting(
            canal_file, fp_sting, 0, min(3600, d_canal), CANAL_STREAM,
            'Prerace sting (canal)', tmp_suffix='_moto2')
        master_sting, m_conf = find_sting(
            master_file, fp_sting, 0, min(3600, d_master), ns_stream,
            'Prerace sting (master)', tmp_suffix='_moto2')
        if c_conf < 0.1 or m_conf < 0.1:
            sys.exit(f'ERROR: Prerace sting not found '
                     f'(canal={c_conf:.4f}, master={m_conf:.4f})')
        offset = master_sting - canal_sting

    # ── Break detection ────────────────────────────────────────────────────
    if use_watermark:
        print('\n-- Watermark break detection --')
        breaks = detect_breaks_watermark(
            canal_file, d_canal, scan_start=0, scan_end=d_canal,
            min_break=wm_min_break, wm_fps=wm_fps, min_content=wm_min_content, tmp_suffix='_moto2wm')
        content_windows = breaks_to_content_windows(breaks, d_canal, OPENING_DUR)
    else:
        print('\n-- Sting/silence break detection --')
        print('\nLocating break 1...')
        fp_moto2_grid = str(FP_DIR / 'canal_moto2_grid_ending.wav')
        fp_opening    = str(FP_DIR / 'canal_opening.wav')
        t_b1_start, _ = find_sting(
            canal_file, fp_moto2_grid, 300, 900, CANAL_STREAM,
            'Moto2 grid ending sting', tmp_suffix='_moto2')
        t_opening1, _ = find_sting(
            canal_file, fp_opening, t_b1_start + 60, 1800, CANAL_STREAM,
            'Opening sting (break 1 end)', tmp_suffix='_moto2')
        t_b1_end = t_opening1 + 12.0
        print(f'  Break 1: {t_b1_start:.1f}s - {t_b1_end:.1f}s  '
              f'(dur={t_b1_end-t_b1_start:.1f}s)')

        print('\nLocating end-of-content silence...')
        scan_start = max(t_b1_end + 2400, d_canal * 0.70)
        scan_dur   = d_canal - scan_start
        t_silence  = None
        if scan_dur > 0:
            t_silence = find_silence_start(
                canal_file, scan_start, scan_dur, CANAL_STREAM,
                label='End-of-content silence', tmp_suffix='_moto2')
        if t_silence is None:
            print('  NOTE: No silence found - using end of file as content end')
            t_silence = d_canal

        content_windows = [
            (OPENING_DUR, t_b1_start),
            (t_b1_end,    t_silence),
        ]

    print('\nBuilding output segments...')
    output_mka = Path(output_dir) / (Path(canal_file).stem + '_canal_synced.mka')
    build_and_concat(canal_file, master_file, CANAL_STREAM, ns_stream,
                     offset, d_canal, d_master, content_windows,
                     output_mka, dry_run, tmp_prefix='moto2')
    print(f'\nDone -> {output_mka}')


def motogp_mode(canal_file, master_file, output_dir, dry_run,
                anchor_source=None, anchor_master=None,
                use_watermark=False, wm_fps=2, wm_min_break=15, wm_min_content=60,
                content_end_secs=None):
    """
    Canal+ MotoGP UHD: opening trim + ad breaks (first audio track only).

    Break detection (sting path):
    - Lead-outs: preshow_intro_m2m3.wav (17s) or prerace_sting_motogp.wav (65s)
    - Lead-ins: silence detection (backward search from each lead-out)
    - Special: canal_zarco_ad.wav → canal_m3m2_opening.wav pair
    - content_end_secs: truncate at program end
    """
    CANAL_STREAM = '0:a:0'   # only first audio track
    OPENING_DUR  = 12.0

    d_canal  = get_duration(canal_file)
    d_master = get_duration(master_file)
    n_audio  = get_audio_stream_count(master_file)
    ns_stream = f'0:a:{n_audio-1}'
    print(f'Canal:  {d_canal:.1f}s  ({d_canal/3600:.2f}h)')
    print(f'Master: {d_master:.1f}s  ({d_master/3600:.2f}h)  NS: {ns_stream}')

    # ── Sync anchor ────────────────────────────────────────────────────────
    if anchor_source is not None and anchor_master is not None:
        offset = anchor_master - anchor_source
        print(f'\nFrame-based anchor: canal {anchor_source:.3f}s = master {anchor_master:.3f}s')
        print(f'  Offset: {offset:.3f}s')
    else:
        print('\nLocating podium anthem (sync anchor)...')
        fp_anthem = str(FP_DIR / 'canal_motogp_anthem.wav')
        canal_anthem, c_conf = find_sting(
            canal_file, fp_anthem, 6253, 240, CANAL_STREAM, 'Anthem (canal)',
            tmp_suffix='_motogp')
        master_anthem, m_conf = find_sting(
            master_file, fp_anthem, 5817, 300, ns_stream, 'Anthem (master)',
            tmp_suffix='_motogp')
        if c_conf < 0.05 or m_conf < 0.05:
            print(f'  WARNING: Low anthem confidence '
                  f'(canal={c_conf:.4f}, master={m_conf:.4f}) - check output')
        offset = master_anthem - canal_anthem

    # ── Break detection ────────────────────────────────────────────────────
    if use_watermark:
        print('\n-- Watermark break detection --')
        breaks = detect_breaks_watermark(
            canal_file, d_canal, scan_start=0, scan_end=d_canal,
            min_break=wm_min_break, wm_fps=wm_fps, min_content=wm_min_content, tmp_suffix='_motogpwm')
        content_windows = breaks_to_content_windows(breaks, d_canal, OPENING_DUR)
    else:
        print('\n-- Sting-based break detection --')
        M3M2_OPENING_DUR = 13.0
        fp_moto3_sting  = str(FP_DIR / 'preshow_intro_m2m3.wav')
        fp_mgp_sting    = str(FP_DIR / 'prerace_sting_motogp.wav')
        fp_zarco        = str(FP_DIR / 'canal_zarco_ad.wav')
        fp_m3m2_opening = str(FP_DIR / 'canal_m3m2_opening.wav')
        SILENCE_WINDOW  = 600.0
        MIN_SILENCE_SEC = 3.0

        # Scan for all lead-out sting instances in one pass
        print('\nScanning for lead-out stings (preshow_intro_m2m3 + prerace_sting_motogp)...')
        leadouts = find_all_transitions(
            canal_file,
            [fp_moto3_sting, fp_mgp_sting],
            CANAL_STREAM,
            tmp_suffix='_motogp',
            conf_thresh=0.3,
            suppress_secs=30,
            min_event_secs=OPENING_DUR,
        )
        for t, c, d in leadouts:
            print(f'  Lead-out at {t:.1f}s  conf={c:.4f}  sting_dur={d:.1f}s')

        # For each lead-out, search backward (up to SILENCE_WINDOW) for break start
        breaks = []
        for t_out, conf, sting_dur in leadouts:
            search_start = max(OPENING_DUR, t_out - SILENCE_WINDOW)
            search_dur   = t_out - search_start
            print(f'\nSearching for break start before lead-out at {t_out:.1f}s '
                  f'(window {search_start:.0f}-{t_out:.0f}s)...')
            t_in = find_silence_start(
                canal_file, search_start, search_dur, CANAL_STREAM,
                label=f'Silence (break before {t_out:.0f}s)',
                min_silence_sec=MIN_SILENCE_SEC, tmp_suffix='_motogp')
            if t_in is not None:
                t_end = t_out + sting_dur
                print(f'  Break: {t_in:.1f}s - {t_end:.1f}s  (dur={t_end-t_in:.1f}s)')
                breaks.append((t_in, t_end))
            else:
                print(f'  No silence found — skipping lead-out at {t_out:.1f}s')

        # Zarco ad sting: special lead-in with canal_m3m2_opening lead-out
        zarco_path = FP_DIR / 'canal_zarco_ad.wav'
        if zarco_path.exists():
            print('\nSearching for Zarco ad sting (special break lead-in)...')
            zarco_search = max(OPENING_DUR, d_canal - 4000)
            t_zarco, zarco_conf = find_sting(
                canal_file, str(zarco_path),
                zarco_search, 4000,
                CANAL_STREAM, 'Zarco ad sting', tmp_suffix='_motogp')
            if zarco_conf >= 0.1:
                in_break = any(s <= t_zarco <= e for s, e in breaks)
                if not in_break:
                    print(f'  Not within existing break — searching for opening sting lead-out...')
                    t_opening, op_conf = find_sting(
                        canal_file, fp_m3m2_opening,
                        t_zarco + 10, 300,
                        CANAL_STREAM, 'Opening sting (Zarco break end)',
                        tmp_suffix='_motogp')
                    if op_conf >= 0.05:
                        t_zarco_end = t_opening + M3M2_OPENING_DUR
                        print(f'  Zarco break: {t_zarco:.1f}s - {t_zarco_end:.1f}s  '
                              f'(dur={t_zarco_end-t_zarco:.1f}s)')
                        breaks.append((t_zarco, t_zarco_end))
                    else:
                        print(f'  WARNING: No opening sting after Zarco ad '
                              f'(conf={op_conf:.4f})')
                else:
                    print(f'  Zarco at {t_zarco:.1f}s already within a detected break')
            else:
                print(f'  Zarco ad not found (conf={zarco_conf:.4f})')

        # Sort and merge overlapping breaks
        breaks.sort(key=lambda x: x[0])
        merged = []
        for s, e in breaks:
            if merged and s < merged[-1][1]:
                print(f'  Overlap: merging ({s:.1f},{e:.1f}) into existing break')
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        breaks = merged

        print(f'\n{len(breaks)} break(s) detected:')
        for i, (s, e) in enumerate(breaks):
            h = int(s//3600); m2 = int((s%3600)//60); sec = s%60
            print(f'  Break {i+1}: {h:02d}:{m2:02d}:{sec:05.2f} - '
                  f'{int(e//3600):02d}:{int((e%3600)//60):02d}:{e%60:05.2f}  '
                  f'dur={e-s:.1f}s')

        content_windows = breaks_to_content_windows(breaks, d_canal, OPENING_DUR)

    # ── Apply content-end truncation ─────────────────────────────────────────
    if content_end_secs is not None:
        print(f'\n  Truncating content at {content_end_secs:.1f}s '
              f'({int(content_end_secs//3600):02d}:{int((content_end_secs%3600)//60):02d}:'
              f'{content_end_secs%60:05.2f})')
        content_windows = [(s, min(e, content_end_secs)) for s, e in content_windows]
        content_windows = [(s, e) for s, e in content_windows if e > s + 1.0]

    print('\nBuilding output segments...')
    output_mka = Path(output_dir) / (Path(canal_file).stem + '_canal_synced.mka')
    build_and_concat(canal_file, master_file, CANAL_STREAM, ns_stream,
                     offset, d_canal, d_master, content_windows,
                     output_mka, dry_run, tmp_prefix='motogp')
    print(f'\nDone -> {output_mka}')


def moto3moto2_mode(canal_file, master_file, output_dir, dry_run,
                    anchor_source=None, anchor_master=None,
                    use_watermark=False, wm_fps=2, wm_min_break=15, wm_min_content=60,
                    wm_scan_end=None,
                    breaks_override=None, content_end_secs=None,
                    opening_dur_secs=None):
    """
    Canal+ combined Moto3+Moto2 file: opening trim + ad breaks.
    Uses frame-based anchor (--anchor-source / --anchor-master) for sync.

    breaks_override: list of (start_secs, end_secs) tuples, bypasses all detection.
    content_end_secs: truncate last content window at this time (end-of-program).
    opening_dur_secs: override OPENING_DUR (default 12s). Use when canal file has
                      a long pre-program lead-in before the actual coverage starts.
    wm_scan_end: limit watermark scan to this time (seconds).
    """
    CANAL_STREAM = '0:a:0'
    OPENING_DUR  = opening_dur_secs if opening_dur_secs is not None else 12.0

    d_canal  = get_duration(canal_file)
    d_master = get_duration(master_file)
    n_audio  = get_audio_stream_count(master_file)
    ns_stream = f'0:a:{n_audio-1}'
    print(f'Canal:  {d_canal:.1f}s  ({d_canal/3600:.2f}h)')
    print(f'Master: {d_master:.1f}s  ({d_master/3600:.2f}h)  NS: {ns_stream}')

    # ── Sync anchor (frame-based, required) ──────────────────────────────────
    if anchor_source is None or anchor_master is None:
        sys.exit('ERROR: moto3moto2 mode requires --anchor-source and --anchor-master')
    offset = anchor_master - anchor_source
    print(f'\nFrame-based anchor: canal {anchor_source:.3f}s = master {anchor_master:.3f}s')
    print(f'  Offset: {offset:.3f}s')

    # ── Break detection ──────────────────────────────────────────────────────
    if breaks_override is not None:
        print(f'\n-- Using {len(breaks_override)} pre-specified break(s) --')
        for i, (s, e) in enumerate(breaks_override):
            h = int(s//3600); m2 = int((s%3600)//60); sec = s%60
            print(f'  Break {i+1}: {h:02d}:{m2:02d}:{sec:05.2f} - '
                  f'{int(e//3600):02d}:{int((e%3600)//60):02d}:{e%60:05.2f}  '
                  f'dur={e-s:.1f}s')
        breaks = breaks_override
        content_windows = breaks_to_content_windows(breaks, d_canal, OPENING_DUR)
    elif use_watermark:
        print('\n-- Watermark break detection --')
        scan_end_wm = wm_scan_end if wm_scan_end is not None else d_canal
        breaks = detect_breaks_watermark(
            canal_file, d_canal, scan_start=OPENING_DUR, scan_end=scan_end_wm,
            min_break=wm_min_break, wm_fps=wm_fps, min_content=wm_min_content, tmp_suffix='_m3m2wm')
        content_windows = breaks_to_content_windows(breaks, d_canal, OPENING_DUR)
    else:
        print('\n-- Sting-based break detection --')
        M3M2_OPENING_DUR = 13.0  # duration of program opening sting (frames 21788-22111 = 12.92s)
        fp_m3m2_opening = str(FP_DIR / 'canal_m3m2_opening.wav')
        fp_moto2_grid   = str(FP_DIR / 'canal_moto2_grid_ending.wav')

        # Step 1: locate program opening sting → OPENING_DUR
        if opening_dur_secs is None:
            print('\nLocating program opening sting (program start)...')
            t_os, os_conf = find_sting(
                canal_file, fp_m3m2_opening, 750, 300, CANAL_STREAM,
                'Program opening sting', tmp_suffix='_m3m2')
            if os_conf < 0.1:
                sys.exit(f'ERROR: Program opening sting not found (conf={os_conf:.4f})')
            OPENING_DUR = t_os + M3M2_OPENING_DUR
            print(f'  Program starts at: {OPENING_DUR:.1f}s')

        # Step 2: silence → break 1 start (after Moto3 race, ~1h+ into program)
        print('\nLocating break 1 start (silence)...')
        t_b1_start = find_silence_start(
            canal_file, OPENING_DUR + 4000, 2000, CANAL_STREAM,
            label='Silence (break 1 start)', min_silence_sec=3.0, tmp_suffix='_m3m2')
        if t_b1_start is None:
            sys.exit('ERROR: Could not find silence for break 1 start')

        # Step 3: opening sting after break 1 silence → break 1 end
        print('\nLocating break 1 end (opening sting)...')
        t_os2, os2_conf = find_sting(
            canal_file, fp_m3m2_opening, t_b1_start + 10, 300, CANAL_STREAM,
            'Opening sting (break 1 end)', tmp_suffix='_m3m2')
        if os2_conf < 0.05:
            sys.exit(f'ERROR: Opening sting not found after break 1 silence '
                     f'(conf={os2_conf:.4f})')
        t_b1_end = t_os2 + M3M2_OPENING_DUR
        print(f'  Break 1: {t_b1_start:.1f}s - {t_b1_end:.1f}s  '
              f'(dur={t_b1_end-t_b1_start:.1f}s)')

        # Step 4: Moto2 grid ending sting → break 2 start
        print('\nLocating break 2 start (Moto2 grid ending sting)...')
        t_b2_start, b2_conf = find_sting(
            canal_file, fp_moto2_grid, t_b1_end + 60, 2000, CANAL_STREAM,
            'Moto2 grid ending sting', tmp_suffix='_m3m2')
        if b2_conf < 0.1:
            sys.exit(f'ERROR: Moto2 grid ending sting not found after break 1 '
                     f'(conf={b2_conf:.4f})')

        # Step 5: opening sting after break 2 grid → break 2 end
        print('\nLocating break 2 end (opening sting)...')
        t_os3, os3_conf = find_sting(
            canal_file, fp_m3m2_opening, t_b2_start + 10, 300, CANAL_STREAM,
            'Opening sting (break 2 end)', tmp_suffix='_m3m2')
        if os3_conf < 0.05:
            sys.exit(f'ERROR: Opening sting not found after break 2 grid sting '
                     f'(conf={os3_conf:.4f})')
        t_b2_end = t_os3 + M3M2_OPENING_DUR
        print(f'  Break 2: {t_b2_start:.1f}s - {t_b2_end:.1f}s  '
              f'(dur={t_b2_end-t_b2_start:.1f}s)')

        # Step 6: silence → end-of-program
        print('\nLocating end-of-program (silence)...')
        t_eop = find_silence_start(
            canal_file, t_b2_end + 60, d_canal - t_b2_end - 60, CANAL_STREAM,
            label='End-of-program silence', tmp_suffix='_m3m2')
        if t_eop is None:
            print('  NOTE: No silence found - using end of file')
            t_eop = d_canal

        content_windows = [
            (OPENING_DUR, t_b1_start),
            (t_b1_end,    t_b2_start),
            (t_b2_end,    t_eop),
        ]

    # ── Apply content-end truncation ─────────────────────────────────────────
    if content_end_secs is not None:
        print(f'\n  Truncating content at {content_end_secs:.1f}s '
              f'({int(content_end_secs//3600):02d}:{int((content_end_secs%3600)//60):02d}:'
              f'{content_end_secs%60:05.2f})')
        content_windows = [(s, min(e, content_end_secs)) for s, e in content_windows]
        content_windows = [(s, e) for s, e in content_windows if e > s + 1.0]

    print('\nBuilding output segments...')
    output_mka = Path(output_dir) / (Path(canal_file).stem + '_canal_synced.mka')
    build_and_concat(canal_file, master_file, CANAL_STREAM, ns_stream,
                     offset, d_canal, d_master, content_windows,
                     output_mka, dry_run, tmp_prefix='moto3moto2')
    print(f'\nDone -> {output_mka}')


def sunday_mode(canal_file, master_file, output_dir, dry_run,
                anchor_source=None, anchor_master=None,
                wm_fps=2, wm_min_break=15, wm_min_content=60,
                wm_scan_end=None,
                breaks_override=None, content_end_secs=None,
                opening_dur_secs=None):
    """
    Canal+ combined Sunday file (Moto3+Moto2+MotoGP): opening trim + ad breaks.
    Uses frame-based anchor (--anchor-source / --anchor-master) for sync.
    Break detection via Canal+ logo watermark absence.

    breaks_override: list of (start_secs, end_secs) tuples, bypasses all detection.
    content_end_secs: truncate last content window at this time (end-of-program).
    opening_dur_secs: override OPENING_DUR (default 12s).
    wm_scan_end: limit watermark scan to this time (seconds).
    """
    CANAL_STREAM = '0:a:0'
    OPENING_DUR  = opening_dur_secs if opening_dur_secs is not None else 12.0

    d_canal  = get_duration(canal_file)
    d_master = get_duration(master_file)
    n_audio  = get_audio_stream_count(master_file)
    ns_stream = f'0:a:{n_audio-1}'
    print(f'Canal:  {d_canal:.1f}s  ({d_canal/3600:.2f}h)')
    print(f'Master: {d_master:.1f}s  ({d_master/3600:.2f}h)  NS: {ns_stream}')

    # ── Sync anchor (frame-based, required) ──────────────────────────────────
    if anchor_source is None or anchor_master is None:
        sys.exit('ERROR: sunday mode requires --anchor-source and --anchor-master')
    offset = anchor_master - anchor_source
    print(f'\nFrame-based anchor: canal {anchor_source:.3f}s = master {anchor_master:.3f}s')
    print(f'  Offset: {offset:.3f}s')

    # ── Break detection ──────────────────────────────────────────────────────
    if breaks_override is not None:
        print(f'\n-- Using {len(breaks_override)} pre-specified break(s) --')
        for i, (s, e) in enumerate(breaks_override):
            h = int(s//3600); m2 = int((s%3600)//60); sec = s%60
            print(f'  Break {i+1}: {h:02d}:{m2:02d}:{sec:05.2f} - '
                  f'{int(e//3600):02d}:{int((e%3600)//60):02d}:{e%60:05.2f}  '
                  f'dur={e-s:.1f}s')
        breaks = breaks_override
        content_windows = breaks_to_content_windows(breaks, d_canal, OPENING_DUR)
    else:
        print('\n-- Watermark break detection --')
        scan_end_wm = wm_scan_end if wm_scan_end is not None else d_canal
        breaks = detect_breaks_watermark(
            canal_file, d_canal, scan_start=OPENING_DUR, scan_end=scan_end_wm,
            min_break=wm_min_break, wm_fps=wm_fps, min_content=wm_min_content,
            tmp_suffix='_sundaywm')
        content_windows = breaks_to_content_windows(breaks, d_canal, OPENING_DUR)

    # ── Apply content-end truncation ─────────────────────────────────────────
    if content_end_secs is not None:
        print(f'\n  Truncating content at {content_end_secs:.1f}s '
              f'({int(content_end_secs//3600):02d}:{int((content_end_secs%3600)//60):02d}:'
              f'{content_end_secs%60:05.2f})')
        content_windows = [(s, min(e, content_end_secs)) for s, e in content_windows]
        content_windows = [(s, e) for s, e in content_windows if e > s + 1.0]

    print('\nBuilding output segments...')
    output_mka = Path(output_dir) / (Path(canal_file).stem + '_canal_synced.mka')
    build_and_concat(canal_file, master_file, CANAL_STREAM, ns_stream,
                     offset, d_canal, d_master, content_windows,
                     output_mka, dry_run, tmp_prefix='sunday')
    print(f'\nDone -> {output_mka}')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Sync Canal+ French audio to a MotoGP master file.')
    parser.add_argument('--race', choices=['sprint', 'moto3', 'moto2', 'motogp',
                                           'moto3moto2', 'sunday'],
                        default=None,
                        help='Race mode (default: unified sprint/general mode).')
    parser.add_argument('--dry-run', action='store_true',
                        help='Detect and plan only; do not encode.')
    parser.add_argument('--watermark', action='store_true',
                        help='Use Canal+ logo watermark for break detection.')
    parser.add_argument('--wm-fps', type=int, default=2,
                        help='Watermark scan frame rate (default: 2).')
    parser.add_argument('--wm-min-break', type=float, default=15.0,
                        help='Minimum break duration for watermark detection (default: 15s).')
    parser.add_argument('--wm-min-content', type=float, default=60.0,
                        help='Minimum content duration for watermark detection (default: 60s).')
    parser.add_argument('--wm-scan-end', type=float, default=None,
                        help='Limit watermark scan to this time (seconds).')
    parser.add_argument('--anchor-source', type=float, default=None,
                        help='Frame-based anchor time in source file (seconds).')
    parser.add_argument('--anchor-master', type=float, default=None,
                        help='Frame-based anchor time in master file (seconds).')
    parser.add_argument('--content-end', type=float, default=None,
                        help='Truncate content at this time (seconds).')
    parser.add_argument('--opening-dur', type=float, default=None,
                        help='Override program start time (seconds).')
    parser.add_argument('--breaks', type=str, default=None,
                        help='Override breaks as comma-separated start:end pairs in seconds '
                             '(e.g. 100:200,300:400).')
    parser.add_argument('canal_file')
    parser.add_argument('master_file')
    parser.add_argument('output_dir')
    args = parser.parse_args()

    if args.dry_run:
        print('[DRY RUN] Detection and planning only - no audio will be encoded.')

    breaks_override = None
    if args.breaks:
        try:
            pairs = [p.split(':') for p in args.breaks.split(',')]
            breaks_override = [(float(s), float(e)) for s, e in pairs]
        except Exception:
            sys.exit('ERROR: --breaks must be comma-separated start:end pairs '
                     '(e.g. 100:200,300:400)')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    race = args.race

    if race is None or race == 'sprint':
        needed = [
            'canal_grid_ending.wav', 'preshow_intro_m2m3.wav',
            'canal_opening.wav', 'canal_zarco_ad.wav',
            'prerace_sting_motogp.wav',
        ]
        for fp in needed:
            p = FP_DIR / fp
            if not p.exists():
                sys.exit(f'ERROR: Missing fingerprint: {p}')
        process_canal(args.canal_file, args.master_file, args.output_dir, args.dry_run,
                      anchor_source=args.anchor_source, anchor_master=args.anchor_master,
                      content_end_secs=args.content_end, opening_dur_secs=args.opening_dur)

    elif race == 'moto3':
        moto3_mode(args.canal_file, args.master_file, args.output_dir, args.dry_run,
                   anchor_source=args.anchor_source, anchor_master=args.anchor_master,
                   use_watermark=args.watermark, wm_fps=args.wm_fps,
                   wm_min_break=args.wm_min_break, wm_min_content=args.wm_min_content,
                   opening_dur_secs=args.opening_dur)

    elif race == 'moto2':
        moto2_mode(args.canal_file, args.master_file, args.output_dir, args.dry_run,
                   anchor_source=args.anchor_source, anchor_master=args.anchor_master,
                   use_watermark=args.watermark, wm_fps=args.wm_fps,
                   wm_min_break=args.wm_min_break, wm_min_content=args.wm_min_content)

    elif race == 'motogp':
        motogp_mode(args.canal_file, args.master_file, args.output_dir, args.dry_run,
                    anchor_source=args.anchor_source, anchor_master=args.anchor_master,
                    use_watermark=args.watermark, wm_fps=args.wm_fps,
                    wm_min_break=args.wm_min_break, wm_min_content=args.wm_min_content,
                    content_end_secs=args.content_end)

    elif race == 'moto3moto2':
        moto3moto2_mode(args.canal_file, args.master_file, args.output_dir, args.dry_run,
                        anchor_source=args.anchor_source, anchor_master=args.anchor_master,
                        use_watermark=args.watermark, wm_fps=args.wm_fps,
                        wm_min_break=args.wm_min_break, wm_min_content=args.wm_min_content,
                        wm_scan_end=args.wm_scan_end,
                        breaks_override=breaks_override, content_end_secs=args.content_end,
                        opening_dur_secs=args.opening_dur)

    elif race == 'sunday':
        sunday_mode(args.canal_file, args.master_file, args.output_dir, args.dry_run,
                    anchor_source=args.anchor_source, anchor_master=args.anchor_master,
                    wm_fps=args.wm_fps, wm_min_break=args.wm_min_break,
                    wm_min_content=args.wm_min_content, wm_scan_end=args.wm_scan_end,
                    breaks_override=breaks_override, content_end_secs=args.content_end,
                    opening_dur_secs=args.opening_dur)


if __name__ == '__main__':
    main()
