#!/usr/bin/env python3
"""
sync_ziggo.py
Add Ziggo Sport Dutch audio to a MotoGP master file.

No ad breaks. Uses a frame-based sync anchor (first camera change after
race start) to align the audio.

Output structure:
  1. Natural Sounds from master   t = 0              -> ziggo_start_master
  2. Ziggo audio                  t = ziggo_start     -> ziggo_end
  3. Natural Sounds from master   ziggo_end_master    -> master end

Requirements: ffmpeg/ffprobe on PATH
    (via audio_utils module)

Usage:
    python sync_ziggo.py [--dry-run]
        --anchor-source=S --anchor-master=S
        <ziggo_file> <master.mkv> <output_dir>
"""

import sys
from pathlib import Path

from audio_utils import fmt, get_duration, get_audio_stream_count, \
                        extract_seg, concat_segments_to_mka


# ── Segment building and concatenation ────────────────────────────────────────

def build_and_concat(ziggo_file, master_file, offset,
                     d_ziggo, d_master, ns_stream,
                     output_mka, dry_run=False):
    """
    Build three sections and concatenate to a single MKA file.

    offset = anchor_source - anchor_master
    master_time(ziggo_t) = ziggo_t - offset

    Section 1: Natural Sounds  master 0            -> ziggo_start_master
    Section 2: Ziggo audio     ziggo 0              -> ziggo_end
    Section 3: Natural Sounds  ziggo_end_master     -> master end
    """
    ziggo_start_m = -offset       # master time when Ziggo t=0 occurs
    ziggo_end_m   = d_ziggo - offset

    print(f'\n  Offset: {offset:.3f}s  (Ziggo t=0 = master {fmt(ziggo_start_m)})')
    print(f'  Ziggo covers master {fmt(max(0.0, ziggo_start_m))} - {fmt(min(ziggo_end_m, d_master))}')

    tmp_dir = Path('_tmp_ziggo_segs')
    if not dry_run:
        tmp_dir.mkdir(exist_ok=True)
    segs    = []
    counter = [0]

    def new_seg(src, stream, start, duration, desc):
        if duration <= 0:
            return
        counter[0] += 1
        p = str(tmp_dir / f'seg_{counter[0]:04d}.wav')
        print(f'  {desc}  start={fmt(start)}  dur={duration:.1f}s')
        if not dry_run:
            extract_seg(src, p, stream, start=start, duration=duration)
        segs.append(p)

    # ── Section 1: NS before Ziggo ──
    if ziggo_start_m > 0:
        new_seg(master_file, ns_stream, 0.0, ziggo_start_m,
                '[NS]  pre-Ziggo  (master 0)')
    elif ziggo_start_m < 0:
        print(f'  NOTE: Ziggo starts {fmt(-ziggo_start_m)} before master t=0; '
              f'trimming by that amount.')

    # ── Section 2: Ziggo audio ──
    ziggo_in  = max(0.0, -ziggo_start_m)   # trim if Ziggo starts before master
    ziggo_dur = d_ziggo - ziggo_in
    if ziggo_end_m > d_master:
        ziggo_dur -= (ziggo_end_m - d_master)
        print(f'  NOTE: Ziggo extends {ziggo_end_m - d_master:.1f}s past master end '
              f'-- trimming Ziggo end.')
    new_seg(ziggo_file, '0:a:0', ziggo_in, ziggo_dur, '[ZIGGO] Dutch')

    # ── Section 3: NS tail ──
    ns_start = min(ziggo_end_m, d_master)
    ns_dur   = d_master - ns_start
    if ns_dur > 0:
        new_seg(master_file, ns_stream, ns_start, ns_dur,
                f'[NS]  post-Ziggo  (master {fmt(ns_start)})')

    # ── Concatenate ──
    if dry_run:
        print(f'\n[DRY RUN] Would concatenate {len(segs)} segments -> {output_mka}')
        return

    concat_segments_to_mka(segs, output_mka, list_path_prefix='_tmp_ziggo')
    for f in tmp_dir.iterdir():
        f.unlink()
    tmp_dir.rmdir()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    dry_run = '--dry-run' in sys.argv
    if dry_run:
        sys.argv.remove('--dry-run')
        print('[DRY RUN] Detection and segment planning only -- no audio will be encoded.')

    anchor_source = None
    anchor_master = None
    for arg in list(sys.argv[1:]):
        if arg.startswith('--anchor-source='):
            anchor_source = float(arg.split('=', 1)[1])
            sys.argv.remove(arg)
        elif arg.startswith('--anchor-master='):
            anchor_master = float(arg.split('=', 1)[1])
            sys.argv.remove(arg)

    if anchor_source is None or anchor_master is None or len(sys.argv) != 4:
        sys.exit('Usage: sync_ziggo.py [--dry-run] '
                 '--anchor-source=S --anchor-master=S '
                 '<ziggo_file> <master.mkv> <output_dir>')

    ziggo_file, master_file, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    output_mka = Path(out_dir) / (Path(ziggo_file).stem + '_ziggo_synced.mka')

    # ── Offset ──
    offset = anchor_source - anchor_master   # master_time = ziggo_t - offset
    print(f'Anchor: Ziggo {fmt(anchor_source)} = master {fmt(anchor_master)}')
    print(f'Offset: {offset:.3f}s  (master_time = ziggo_time - {offset:.3f})')

    # ── Durations ──
    d_ziggo  = get_duration(ziggo_file)
    d_master = get_duration(master_file)
    print(f'\nZiggo:  {d_ziggo:.1f}s  ({fmt(d_ziggo)})')
    print(f'Master: {d_master:.1f}s  ({fmt(d_master)})')

    # ── Natural Sounds stream (last audio track in master) ──
    n_audio   = get_audio_stream_count(master_file)
    ns_stream = f'0:a:{n_audio - 1}'
    print(f'Master audio tracks: {n_audio}  ->  NS on {ns_stream}')

    print('\nBuilding output segments...')
    build_and_concat(ziggo_file, master_file, offset,
                     d_ziggo, d_master, ns_stream,
                     output_mka, dry_run=dry_run)

    print(f'\nDone -> {output_mka}')


if __name__ == '__main__':
    main()
