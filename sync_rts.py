#!/usr/bin/env python3
"""
sync_rts.py
Add RTS broadcast audio to a MotoGP master file.

The RTS source may have opening and/or closing program stings. Use
--trim-start and --trim-end to specify their durations in seconds;
those periods are replaced with Natural Sounds from the master.

Sync anchor: 65s MotoGP pre-race sting (prerace_sting_motogp.wav)

Output structure:
  1. Natural Sounds from master, t=0 to master_time(trim_start)
  2. RTS audio from trim_start to (d_rts - trim_end)
  3. Natural Sounds from master, master_time(d_rts - trim_end) to master end

Requirements: ffmpeg/ffprobe on PATH, numpy, scipy

Usage:
    python sync_rts.py [--dry-run] [--trim-start S] [--trim-end S]
                       <rts_file> <master.mkv> <output_dir>
"""

import subprocess, sys, os, argparse, numpy as np
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import fftconvolve

# ── Tuning ─────────────────────────────────────────────────────────────────────
SR = 8000  # Hz for correlation

# Search window for the 65s pre-race sting (start_sec, duration_sec)
STING_SEARCH_MASTER = (0, 3600)   # first 60 min of master
STING_SEARCH_SRC    = (0, 3600)   # first 60 min of source (covers all RTS files)

# Fingerprints directory (alongside this script)
FP_DIR = Path(__file__).parent / 'fingerprints'


# ── ffprobe ────────────────────────────────────────────────────────────────────

def get_duration(f):
    r = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', str(f)],
        capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def get_audio_stream_count(f):
    r = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'a',
         '-show_entries', 'stream=index',
         '-of', 'default=noprint_wrappers=1:nokey=1', str(f)],
        capture_output=True, text=True, check=True)
    return len([l for l in r.stdout.strip().splitlines() if l.strip()])


# ── Audio extraction ───────────────────────────────────────────────────────────

def extract_wav(src, dst, stream_spec, start=None, duration=None,
                sr=SR, channels=1):
    cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error']
    if start    is not None: cmd += ['-ss', f'{start:.3f}']
    if duration is not None: cmd += ['-t',  f'{duration:.3f}']
    cmd += ['-i', str(src), '-map', stream_spec,
            '-ar', str(sr), '-ac', str(channels), '-f', 'wav', str(dst)]
    subprocess.run(cmd, check=True)


def extract_seg(src, dst, stream_spec, start, duration):
    """Extract a segment at 48000 Hz stereo for final output concatenation."""
    cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
           '-ss', f'{start:.3f}', '-t', f'{duration:.3f}',
           '-i', str(src), '-map', stream_spec,
           '-ar', '48000', '-ac', '2', '-f', 'wav', str(dst)]
    subprocess.run(cmd, check=True)


# ── Correlation ────────────────────────────────────────────────────────────────

def _peak(haystack, needle):
    h = haystack.astype(np.float32)
    n = needle.astype(np.float32)
    if len(n) >= len(h):
        n = n[:max(1, len(h) - 1)]
    corr  = fftconvolve(h, n[::-1], mode='valid')
    idx   = int(np.argmax(np.abs(corr)))
    h_win = h[idx:idx + len(n)]
    conf  = float(np.abs(corr[idx])) / (
            np.linalg.norm(h_win) * np.linalg.norm(n) + 1e-10)
    return idx, conf


def find_sting(src, fp_path, search_start, search_dur, stream_spec='0:a:0',
               label=''):
    """Find a sting fingerprint within a time window. Returns (time_sec, conf)."""
    tmp = '_tmp_rts_sting.wav'
    actual_dur = min(search_dur, get_duration(src) - search_start)
    if actual_dur <= 0:
        return search_start, 0.0
    extract_wav(src, tmp, stream_spec, start=search_start, duration=actual_dur)
    _, needle   = wavfile.read(fp_path)
    _, haystack = wavfile.read(tmp)
    os.remove(tmp)
    idx, conf = _peak(haystack, needle)
    t = search_start + idx / SR
    if label:
        print(f'  {label}: {t:.3f}s ({t/60:.2f} min)  conf={conf:.4f}')
    return t, conf


# ── Segment building and concatenation ────────────────────────────────────────

def build_and_concat(src_file, master_file, src_sting, master_sting,
                     trim_start, trim_end, d_src, d_master,
                     src_stream, ns_stream,
                     output_mka, dry_run=False):
    """
    Build output and concatenate to MKA.

    Alignment: master_time = src_time + offset
               where offset = master_sting - src_sting

    Section 1: NS  master 0                       -> offset + trim_start
    Section 2: SRC src trim_start                 -> d_src - trim_end
    Section 3: NS  master offset + d_src-trim_end -> d_master
    """
    offset      = master_sting - src_sting
    ns1_end     = offset + trim_start          # master time where RTS content starts
    src_content_end = d_src - trim_end         # RTS time where content ends
    ns3_start   = offset + src_content_end     # master time where NS tail starts

    print(f'\n  Offset: {offset:.3f}s')
    print(f'  RTS content: src {trim_start:.1f}s - {src_content_end:.1f}s '
          f'(dur={src_content_end - trim_start:.1f}s)')
    print(f'  Master coverage: {ns1_end:.1f}s - {ns3_start:.1f}s')

    tmp_dir = Path('_tmp_rts_segs')
    if not dry_run:
        tmp_dir.mkdir(exist_ok=True)
    segs    = []
    counter = [0]

    def new_seg(src, stream, start, duration, desc):
        if duration <= 0:
            return
        counter[0] += 1
        p = str(tmp_dir / f'seg_{counter[0]:04d}.wav')
        print(f'  {desc}  start={start:.1f}s  dur={duration:.1f}s')
        if not dry_run:
            extract_seg(src, p, stream, start=start, duration=duration)
        segs.append(p)

    # ── Section 1: NS from master 0 to where RTS content begins ──
    if ns1_end > 0:
        new_seg(master_file, ns_stream, 0.0, min(ns1_end, d_master),
                '[NS]  pre-RTS  (master 0)')
    elif ns1_end < 0:
        print(f'  NOTE: RTS content starts {-ns1_end:.1f}s before master start '
              '- trimming RTS accordingly.')

    # ── Section 2: RTS content ──
    rts_in  = trim_start + max(0.0, -ns1_end)   # extra trim if RTS starts before master
    rts_dur = src_content_end - rts_in
    if ns3_start > d_master:
        excess = ns3_start - d_master
        rts_dur -= excess
        print(f'  NOTE: RTS extends {excess:.1f}s past master end - trimming.')
    new_seg(src_file, src_stream, rts_in, rts_dur, '[RTS] commentary')

    # ── Section 3: NS from where RTS content ends through master end ──
    ns3_dur = d_master - max(ns3_start, 0)
    if ns3_dur > 0:
        new_seg(master_file, ns_stream, max(ns3_start, 0), ns3_dur,
                f'[NS]  post-RTS (master {max(ns3_start, 0):.1f}s)')

    # ── Concatenate ──
    if dry_run:
        print(f'\n[DRY RUN] Would concatenate {len(segs)} segments -> {output_mka}')
        return

    list_path = '_tmp_rts_concat.txt'
    with open(list_path, 'w') as f:
        for s in segs:
            f.write(f"file '{Path(s).resolve()}'\n")
    print(f'\nConcatenating {len(segs)} segments -> {output_mka}')
    subprocess.run(
        ['ffmpeg', '-y', '-hide_banner',
         '-f', 'concat', '-safe', '0', '-i', list_path,
         '-map', '0', '-c:a', 'aac', '-b:a', '192k', str(output_mka)],
        check=True)
    os.remove(list_path)

    for f in tmp_dir.iterdir():
        f.unlink()
    tmp_dir.rmdir()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Sync RTS broadcast audio to a MotoGP master file.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Detect and plan only; do not encode.')
    parser.add_argument('--trim-start', type=float, default=0.0, metavar='S',
                        help='Opening sting duration to replace with NS (seconds).')
    parser.add_argument('--trim-end',   type=float, default=0.0, metavar='S',
                        help='Closing sting duration to replace with NS (seconds).')
    parser.add_argument('rts_file')
    parser.add_argument('master_file')
    parser.add_argument('output_dir')
    args = parser.parse_args()

    if args.dry_run:
        print('[DRY RUN] Detection and segment planning only - no audio will be encoded.')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_mka = Path(args.output_dir) / (Path(args.rts_file).stem + '_rts_synced.mka')

    fp_sting = str(FP_DIR / 'prerace_sting_motogp.wav')
    if not Path(fp_sting).exists():
        sys.exit(f'ERROR: Missing fingerprint file: {fp_sting}')

    # ── Durations ──
    d_src    = get_duration(args.rts_file)
    d_master = get_duration(args.master_file)
    print(f'RTS:    {d_src:.1f}s ({d_src/3600:.2f}h)')
    print(f'Master: {d_master:.1f}s ({d_master/3600:.2f}h)')
    if args.trim_start or args.trim_end:
        print(f'Trim:   start={args.trim_start:.1f}s  end={args.trim_end:.1f}s')

    # ── Detect Natural Sounds stream (last audio track in master) ──
    n_audio   = get_audio_stream_count(args.master_file)
    ns_stream = f'0:a:{n_audio - 1}'
    print(f'Master audio tracks: {n_audio}  ->  Natural Sounds on {ns_stream}')

    # ── Find pre-race sting in master ──
    print('\nLocating 65s sting in master...')
    master_sting, master_conf = find_sting(
        args.master_file, fp_sting,
        *STING_SEARCH_MASTER, stream_spec=ns_stream,
        label='  Sting (master)')
    if master_conf < 0.1:
        sys.exit('ERROR: Could not find pre-race sting in master. '
                 'Check fingerprint or STING_SEARCH_MASTER window.')

    # ── Find pre-race sting in RTS source ──
    print('\nLocating 65s sting in RTS source...')
    src_sting, src_conf = find_sting(
        args.rts_file, fp_sting,
        *STING_SEARCH_SRC, stream_spec='0:a:0',
        label='  Sting (RTS)')
    if src_conf < 0.1:
        sys.exit('ERROR: Could not find pre-race sting in RTS file. '
                 'Check fingerprint or STING_SEARCH_SRC window.')

    # ── Build output ──
    print('\nBuilding output segments...')
    build_and_concat(
        args.rts_file, args.master_file,
        src_sting, master_sting,
        args.trim_start, args.trim_end,
        d_src, d_master,
        '0:a:0', ns_stream,
        output_mka, dry_run=args.dry_run)

    print(f'\nDone -> {output_mka}')


if __name__ == '__main__':
    main()
