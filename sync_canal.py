#!/usr/bin/env python3
"""
sync_canal.py
Add Canal+ French audio to MotoGP race files.

Each race type has a different break structure:

  sprint  -- Canal+Sport 360 Sprint: opening trim + 2 ad breaks
             Break 1: canal grid ending sting -> 17s Moto3 intro sting
             Break 2: 2s silence -> 12s canal opening sting
             Sync anchor: podium music at ~1h08m55s in Canal+ file

  moto3   -- Moto3 race (TEEMU WEB): opening trim only, no ad breaks
             Sync anchor: prerace_sting.wav

  moto2   -- Canal+Sport 360 Moto2: opening trim + 1 ad break + silence tail
             Break 1: shorter canal grid ending sting -> 12s canal opening sting
             Sync anchor: prerace_sting.wav

  motogp  -- Canal+ UHD Moto GP: opening trim + 3 ad breaks (first audio track only)
             Break 1: canal grid ending sting -> 65s MotoGP race sting (incl. in break)
             Break 2: 2s silence at ~1h27m41s -> 12s canal opening sting
             Break 3: 2s silence at ~1h47m21s -> 12s canal opening sting
             Sync anchor: podium anthem at ~1h46m13s in Canal+ / ~1h38m57s in master

Usage:
    python sync_canal.py --race {sprint|moto3|moto2|motogp} [--dry-run]
                         <canal_file> <master_file> <output_dir>

Requirements: ffmpeg/ffprobe on PATH, numpy, scipy
"""

import sys, os, argparse
import numpy as np
from pathlib import Path
from scipy.io import wavfile

from audio_utils import SR, get_duration, get_audio_stream_count, extract_wav, extract_seg, concat_segments_to_mka
from sting_detection import find_sting

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


# ── Segment building and concatenation ────────────────────────────────────────

def build_and_concat(canal_file, master_file, canal_stream, ns_stream,
                     offset, d_canal, d_master, content_windows,
                     output_mka, dry_run=False, tmp_prefix='canal'):
    """
    Build output from Canal+ content windows and NS from master.

    content_windows: [(c_start, c_end), ...] in canal time (seconds)
    offset: canal_time + offset = master_time

    Output: NS head | canal[w0] | NS break1 | canal[w1] | ... | NS tail
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
            extract_seg(canal_file, p, canal_stream, c_start, dur)
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


# ── Race modes ────────────────────────────────────────────────────────────────

def sprint_mode(canal_file, master_file, output_dir, dry_run):
    """
    Canal+ Sprint: opening sting (10s) + 2 ad breaks.

    Break 1: canal_grid_ending.wav -> canal_moto3_sting.wav + 17s
    Break 2: 2s silence (~58m40s) -> canal_opening.wav + 12s
    Sync:    canal_sprint_podium.wav at ~1h08m55s in canal, search broadly in master
    """
    CANAL_STREAM = '0:a:0'
    OPENING_DUR  = 10.0   # opening sting duration at start of Sprint file

    d_canal  = get_duration(canal_file)
    d_master = get_duration(master_file)
    n_audio  = get_audio_stream_count(master_file)
    ns_stream = f'0:a:{n_audio-1}'
    print(f'Canal:  {d_canal:.1f}s  ({d_canal/3600:.2f}h)')
    print(f'Master: {d_master:.1f}s  ({d_master/3600:.2f}h)  NS: {ns_stream}')

    # ── Sync anchor: podium music ──────────────────────────────────────────
    print('\nLocating podium music (sync anchor)...')
    fp_podium = str(FP_DIR / 'canal_sprint_podium.wav')
    canal_podium, c_conf = find_sting(
        canal_file, fp_podium, 4015, 240, CANAL_STREAM, 'Podium (canal)',
        tmp_suffix='_sprint')
    master_podium, m_conf = find_sting(
        master_file, fp_podium, 3400, 1800, ns_stream, 'Podium (master)',
        tmp_suffix='_sprint')
    if c_conf < 0.05 or m_conf < 0.05:
        print(f'  WARNING: Low podium confidence '
              f'(canal={c_conf:.4f}, master={m_conf:.4f}) - check output')
    offset = master_podium - canal_podium

    # ── Break 1 ───────────────────────────────────────────────────────────
    print('\nLocating break 1...')
    fp_grid  = str(FP_DIR / 'canal_grid_ending.wav')
    fp_moto3 = str(FP_DIR / 'canal_moto3_sting.wav')
    t_b1_start, _ = find_sting(
        canal_file, fp_grid, 1580, 220, CANAL_STREAM, 'Grid ending sting',
        tmp_suffix='_sprint')
    t_moto3, _ = find_sting(
        canal_file, fp_moto3, t_b1_start + 27, 1800, CANAL_STREAM,
        '17s Moto3 intro sting', tmp_suffix='_sprint')
    t_b1_end = t_moto3 + 17.0
    print(f'  Break 1: {t_b1_start:.1f}s - {t_b1_end:.1f}s  '
          f'(dur={t_b1_end-t_b1_start:.1f}s)')

    # ── Break 2 ───────────────────────────────────────────────────────────
    print('\nLocating break 2...')
    fp_opening = str(FP_DIR / 'canal_opening.wav')
    t_b2_start = find_silence_start(
        canal_file, 3380, 320, CANAL_STREAM, label='Silence (break 2 start)',
        tmp_suffix='_sprint')
    if t_b2_start is None:
        sys.exit('ERROR: Could not find 2s silence for break 2 in Canal+ Sprint')
    t_opening2, _ = find_sting(
        canal_file, fp_opening, t_b2_start + 60, 2400, CANAL_STREAM,
        'Opening sting (break 2 end)', tmp_suffix='_sprint')
    t_b2_end = t_opening2 + 12.0
    print(f'  Break 2: {t_b2_start:.1f}s - {t_b2_end:.1f}s  '
          f'(dur={t_b2_end-t_b2_start:.1f}s)')

    content_windows = [
        (OPENING_DUR, t_b1_start),
        (t_b1_end,    t_b2_start),
        (t_b2_end,    d_canal),
    ]

    print('\nBuilding output segments...')
    output_mka = Path(output_dir) / (Path(canal_file).stem + '_canal_synced.mka')
    build_and_concat(canal_file, master_file, CANAL_STREAM, ns_stream,
                     offset, d_canal, d_master, content_windows,
                     output_mka, dry_run, tmp_prefix='sprint')
    print(f'\nDone -> {output_mka}')


def moto3_mode(canal_file, master_file, output_dir, dry_run):
    """
    Canal+ Moto3 (TEEMU WEB): opening sting (12s) only, no ad breaks.
    Sync: prerace_sting.wav in both files.
    """
    CANAL_STREAM = '0:a:0'
    OPENING_DUR  = 12.0

    d_canal  = get_duration(canal_file)
    d_master = get_duration(master_file)
    n_audio  = get_audio_stream_count(master_file)
    ns_stream = f'0:a:{n_audio-1}'
    print(f'Canal:  {d_canal:.1f}s  ({d_canal/3600:.2f}h)')
    print(f'Master: {d_master:.1f}s  ({d_master/3600:.2f}h)  NS: {ns_stream}')

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

    content_windows = [(OPENING_DUR, d_canal)]

    print('\nBuilding output segments...')
    output_mka = Path(output_dir) / (Path(canal_file).stem + '_canal_synced.mka')
    build_and_concat(canal_file, master_file, CANAL_STREAM, ns_stream,
                     offset, d_canal, d_master, content_windows,
                     output_mka, dry_run, tmp_prefix='moto3')
    print(f'\nDone -> {output_mka}')


def moto2_mode(canal_file, master_file, output_dir, dry_run):
    """
    Canal+ Moto2: opening sting (12s) + 1 ad break + silence-terminated tail.

    Break 1: canal_moto2_grid_ending.wav (~15m44s) -> canal_opening.wav + 12s
    Tail: 2s silence -> NS to master end
    Sync: prerace_sting.wav in both files.
    """
    CANAL_STREAM = '0:a:0'
    OPENING_DUR  = 12.0

    d_canal  = get_duration(canal_file)
    d_master = get_duration(master_file)
    n_audio  = get_audio_stream_count(master_file)
    ns_stream = f'0:a:{n_audio-1}'
    print(f'Canal:  {d_canal:.1f}s  ({d_canal/3600:.2f}h)')
    print(f'Master: {d_master:.1f}s  ({d_master/3600:.2f}h)  NS: {ns_stream}')

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

    # ── Break 1 ───────────────────────────────────────────────────────────
    print('\nLocating break 1...')
    fp_moto2_grid = str(FP_DIR / 'canal_moto2_grid_ending.wav')
    fp_opening    = str(FP_DIR / 'canal_opening.wav')
    t_b1_start, _ = find_sting(
        canal_file, fp_moto2_grid, 880, 180, CANAL_STREAM,
        'Moto2 grid ending sting', tmp_suffix='_moto2')
    t_opening1, _ = find_sting(
        canal_file, fp_opening, t_b1_start + 60, 1800, CANAL_STREAM,
        'Opening sting (break 1 end)', tmp_suffix='_moto2')
    t_b1_end = t_opening1 + 12.0
    print(f'  Break 1: {t_b1_start:.1f}s - {t_b1_end:.1f}s  '
          f'(dur={t_b1_end-t_b1_start:.1f}s)')

    # ── End-of-content silence ────────────────────────────────────────────
    # Start scanning well after the break to skip any brief audio dips
    # during race coverage. The real silence is near the end of the file.
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


def motogp_mode(canal_file, master_file, output_dir, dry_run):
    """
    Canal+ MotoGP UHD: opening sting (12s) + 3 ad breaks (use first audio track only).

    Break 1: canal_grid_ending.wav -> prerace_sting_motogp.wav start + 65s
    Break 2: 2s silence (~1h27m41s) -> canal_opening.wav + 12s
    Break 3: 2s silence (~1h47m21s) -> canal_opening.wav + 12s
    Sync:    canal_motogp_anthem.wav at ~1h46m13s canal / ~1h38m57s master
    """
    CANAL_STREAM = '0:a:0'   # only first audio track
    OPENING_DUR  = 12.0

    d_canal  = get_duration(canal_file)
    d_master = get_duration(master_file)
    n_audio  = get_audio_stream_count(master_file)
    ns_stream = f'0:a:{n_audio-1}'
    print(f'Canal:  {d_canal:.1f}s  ({d_canal/3600:.2f}h)')
    print(f'Master: {d_master:.1f}s  ({d_master/3600:.2f}h)  NS: {ns_stream}')

    # ── Sync anchor: podium anthem ────────────────────────────────────────
    # Canal+: ~6373s (1h46m13s); Master: ~5937s (1h38m57s)
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

    # ── Break 1 ───────────────────────────────────────────────────────────
    print('\nLocating break 1...')
    fp_grid      = str(FP_DIR / 'canal_grid_ending.wav')
    fp_mgp_sting = str(FP_DIR / 'prerace_sting_motogp.wav')
    t_b1_start, _ = find_sting(
        canal_file, fp_grid, 0, 5400, CANAL_STREAM, 'Grid ending sting',
        tmp_suffix='_motogp')
    t_65s, _ = find_sting(
        canal_file, fp_mgp_sting, t_b1_start + 27, 5400, CANAL_STREAM,
        '65s MotoGP race sting', tmp_suffix='_motogp')
    t_b1_end = t_65s + 65.0   # sting is part of break; content resumes after
    print(f'  Break 1: {t_b1_start:.1f}s - {t_b1_end:.1f}s  '
          f'(dur={t_b1_end-t_b1_start:.1f}s)')

    # ── Break 2 ───────────────────────────────────────────────────────────
    print('\nLocating break 2...')
    fp_opening = str(FP_DIR / 'canal_opening.wav')
    t_b2_start = find_silence_start(
        canal_file, 5141, 300, CANAL_STREAM, label='Silence (break 2 start)',
        tmp_suffix='_motogp')
    if t_b2_start is None:
        sys.exit('ERROR: Could not find 2s silence for break 2 in Canal+ MotoGP')
    t_opening2, _ = find_sting(
        canal_file, fp_opening, t_b2_start + 60, 1800, CANAL_STREAM,
        'Opening sting (break 2 end)', tmp_suffix='_motogp')
    t_b2_end = t_opening2 + 12.0
    print(f'  Break 2: {t_b2_start:.1f}s - {t_b2_end:.1f}s  '
          f'(dur={t_b2_end-t_b2_start:.1f}s)')

    # ── Break 3 ───────────────────────────────────────────────────────────
    print('\nLocating break 3...')
    t_b3_start = find_silence_start(
        canal_file, 6321, 300, CANAL_STREAM, label='Silence (break 3 start)',
        tmp_suffix='_motogp')
    if t_b3_start is None:
        sys.exit('ERROR: Could not find 2s silence for break 3 in Canal+ MotoGP')
    t_opening3, _ = find_sting(
        canal_file, fp_opening, t_b3_start + 60, 1800, CANAL_STREAM,
        'Opening sting (break 3 end)', tmp_suffix='_motogp')
    t_b3_end = t_opening3 + 12.0
    print(f'  Break 3: {t_b3_start:.1f}s - {t_b3_end:.1f}s  '
          f'(dur={t_b3_end-t_b3_start:.1f}s)')

    content_windows = [
        (OPENING_DUR, t_b1_start),
        (t_b1_end,    t_b2_start),
        (t_b2_end,    t_b3_start),
        (t_b3_end,    d_canal),
    ]

    print('\nBuilding output segments...')
    output_mka = Path(output_dir) / (Path(canal_file).stem + '_canal_synced.mka')
    build_and_concat(canal_file, master_file, CANAL_STREAM, ns_stream,
                     offset, d_canal, d_master, content_windows,
                     output_mka, dry_run, tmp_prefix='motogp')
    print(f'\nDone -> {output_mka}')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Sync Canal+ French audio to a MotoGP master file.')
    parser.add_argument('--race', required=True,
                        choices=['sprint', 'moto3', 'moto2', 'motogp'],
                        help='Race type / break structure.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Detect and plan only; do not encode.')
    parser.add_argument('canal_file')
    parser.add_argument('master_file')
    parser.add_argument('output_dir')
    args = parser.parse_args()

    if args.dry_run:
        print('[DRY RUN] Detection and planning only - no audio will be encoded.')

    # Validate fingerprints exist
    needed = {
        'sprint':  ['canal_sprint_podium.wav', 'canal_grid_ending.wav',
                    'canal_moto3_sting.wav', 'canal_opening.wav'],
        'moto3':   ['prerace_sting.wav'],
        'moto2':   ['prerace_sting.wav', 'canal_moto2_grid_ending.wav',
                    'canal_opening.wav'],
        'motogp':  ['canal_motogp_anthem.wav', 'canal_grid_ending.wav',
                    'prerace_sting_motogp.wav', 'canal_opening.wav'],
    }
    for fp in needed[args.race]:
        p = FP_DIR / fp
        if not p.exists():
            sys.exit(f'ERROR: Missing fingerprint: {p}')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    modes = {
        'sprint': sprint_mode,
        'moto3':  moto3_mode,
        'moto2':  moto2_mode,
        'motogp': motogp_mode,
    }
    modes[args.race](args.canal_file, args.master_file,
                     args.output_dir, args.dry_run)


if __name__ == '__main__':
    main()
