#!/usr/bin/env python3
"""
sync_sky.py
Add Sky Sport MotoGP Italian audio to a MotoGP master file.

The Sky file contains no ad breaks. We find the 65s pre-race sting in both
files to determine the offset, then build a single audio track:

  1. Natural Sounds from master   t = 0              → master_sting - sky_sting
  2. Sky Italian audio            t = sky_start       → sky_end
  3. Natural Sounds from master   t = sky_end_master  → master end

The output is a single MKA file suitable for manual sync checking.

Requirements: ffmpeg/ffprobe on PATH, numpy, scipy

Usage:
    python sync_sky.py [--dry-run] <sky_file> <master.mkv> <output_dir>
"""

import subprocess, sys, os, numpy as np
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import fftconvolve

# ── Tuning ─────────────────────────────────────────────────────────────────────
SR = 8000  # Hz for correlation

# Search window for the 65s pre-race sting in each file
STING_SEARCH_MASTER = (0, 3600)   # first 60 min of master
STING_SEARCH_SKY    = (0, 180)    # first 3 min of Sky file

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
    """Extract a segment at 48 kHz stereo for final output concatenation."""
    extract_wav(src, dst, stream_spec,
                start=start, duration=duration, sr=48000, channels=2)


# ── Correlation ────────────────────────────────────────────────────────────────

def _peak(haystack, needle):
    """Return (sample_index, confidence) for best match of needle in haystack."""
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
    """
    Find a sting fingerprint within a time window of src.
    Returns (absolute_time_sec, confidence).
    """
    tmp = '_tmp_sky_sting.wav'
    extract_wav(src, tmp, stream_spec, start=search_start, duration=search_dur)
    _, needle   = wavfile.read(fp_path)
    _, haystack = wavfile.read(tmp)
    os.remove(tmp)
    idx, conf = _peak(haystack, needle)
    t = search_start + idx / SR
    if label:
        print(f'  {label}: {t:.3f}s ({t/60:.2f} min)  conf={conf:.4f}')
    return t, conf


# ── Segment building and concatenation ────────────────────────────────────────

def build_and_concat(sky_file, master_file, sky_sting, master_sting,
                     d_sky, d_master, ns_stream, output_mka, dry_run=False):
    """
    Build three sections and concatenate to a single MKA file.

    Alignment: sky_time + offset = master_time
               where offset = master_sting - sky_sting

    Section 1: Natural Sounds  master 0         → master_sting - sky_sting
    Section 2: Sky Italian     sky 0             → d_sky
    Section 3: Natural Sounds  master sky_end_m  → d_master
    """
    offset       = master_sting - sky_sting   # master_time = sky_time + offset
    sky_start_m  = offset                     # master time when Sky starts
    sky_end_m    = offset + d_sky             # master time when Sky ends

    print(f'\n  Offset: {offset:.3f}s  (Sky t=0 = master t={offset:.3f}s)')
    print(f'  Sky content covers master {sky_start_m:.1f}s - {sky_end_m:.1f}s')

    tmp_dir = Path('_tmp_sky_segs')
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

    # ── Section 1: Natural Sounds before Sky content ──
    if sky_start_m > 0:
        new_seg(master_file, ns_stream, 0.0, sky_start_m,
                '[NS]  pre-Sky  (master 0)')
    elif sky_start_m < 0:
        print(f'  NOTE: Sky file starts {-sky_start_m:.1f}s before master '
              f'— trimming Sky start by that amount.')

    # ── Section 2: Sky Italian audio ──
    sky_in  = max(0.0, -sky_start_m)   # trim if Sky starts before master
    sky_dur = d_sky - sky_in
    if sky_end_m > d_master:
        sky_dur -= (sky_end_m - d_master)
        print(f'  NOTE: Sky extends {sky_end_m - d_master:.1f}s past master end '
              f'— trimming Sky end by that amount.')
    new_seg(sky_file, '0:a:0', sky_in, sky_dur, '[SKY] Italian')

    # ── Section 3: Natural Sounds after Sky content ──
    ns_start = min(sky_end_m, d_master)
    ns_dur   = d_master - ns_start
    if ns_dur > 0:
        new_seg(master_file, ns_stream, ns_start, ns_dur,
                f'[NS]  post-Sky (master {ns_start:.1f}s)')

    # ── Concatenate all segments to MKA ──
    if dry_run:
        print(f'\n[DRY RUN] Would concatenate {len(segs)} segments -> {output_mka}')
        return

    list_path = '_tmp_sky_concat.txt'
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
    dry_run = '--dry-run' in sys.argv
    if dry_run:
        sys.argv.remove('--dry-run')
        print('[DRY RUN] Detection and segment planning only — no audio will be encoded.')

    if len(sys.argv) != 4:
        sys.exit('Usage: sync_sky.py [--dry-run] <sky_file> <master.mkv> <output_dir>')

    sky_file, master_file, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    output_mka = Path(out_dir) / (Path(sky_file).stem + '_sky_synced.mka')

    fp_sting = str(FP_DIR / 'prerace_sting_motogp.wav')
    if not Path(fp_sting).exists():
        sys.exit(f'ERROR: Missing fingerprint file: {fp_sting}')

    # ── Durations ──
    d_sky    = get_duration(sky_file)
    d_master = get_duration(master_file)
    print(f'Sky:    {d_sky:.1f}s ({d_sky/3600:.2f}h)')
    print(f'Master: {d_master:.1f}s ({d_master/3600:.2f}h)')

    # ── Detect Natural Sounds stream (last audio track in master) ──
    n_audio  = get_audio_stream_count(master_file)
    ns_idx   = n_audio - 1
    ns_stream = f'0:a:{ns_idx}'
    print(f'Master audio tracks: {n_audio}  ->  Natural Sounds on {ns_stream}')

    # ── Find pre-race sting in master (Natural Sounds track) ──
    print('\nLocating 65s sting in master...')
    master_sting, master_conf = find_sting(
        master_file, fp_sting,
        *STING_SEARCH_MASTER, stream_spec=ns_stream,
        label='  Sting (master)')
    if master_conf < 0.1:
        sys.exit('ERROR: Could not find pre-race sting in master. '
                 'Check fingerprint or STING_SEARCH_MASTER window.')

    # ── Find pre-race sting in Sky file ──
    print('\nLocating 65s sting in Sky file...')
    sky_sting, sky_conf = find_sting(
        sky_file, fp_sting,
        *STING_SEARCH_SKY, stream_spec='0:a:0',
        label='  Sting (Sky)')
    if sky_conf < 0.1:
        sys.exit('ERROR: Could not find pre-race sting in Sky file. '
                 'Check fingerprint or STING_SEARCH_SKY window.')

    # ── Build output ──
    print('\nBuilding output segments...')
    build_and_concat(sky_file, master_file, sky_sting, master_sting,
                     d_sky, d_master, ns_stream, output_mka, dry_run=dry_run)

    print(f'\nDone -> {output_mka}')


if __name__ == '__main__':
    main()
