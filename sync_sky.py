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
    (via audio_utils / sting_detection modules)

Usage:
    python sync_sky.py [--dry-run] <sky_file> <master.mkv> <output_dir>
"""

import sys
from pathlib import Path

from audio_utils import get_duration, get_audio_stream_count, extract_seg, concat_segments_to_mka
from sting_detection import find_sting

# Search window for the 65s pre-race sting in each file
STING_SEARCH_MASTER = (0, 3600)   # first 60 min of master
STING_SEARCH_SKY    = (0, 180)    # first 3 min of Sky file

# Fingerprints directory (alongside this script)
FP_DIR = Path(__file__).parent / 'fingerprints'


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

    concat_segments_to_mka(segs, output_mka, list_path_prefix='_tmp_sky')

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
