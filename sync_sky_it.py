#!/usr/bin/env python3
"""
sync_sky_it.py
Add Sky Sport MotoGP Italian audio to a MotoGP master file.

Uses a frame-based sync anchor (first camera change after race start)
and sting-pair break detection: the same fingerprint marks both the
lead-in (ads start) and lead-out (program resumes) of each ad break.

Output structure:
  1. Natural Sounds from master   t = 0             → sky_start_master
  2. Sky Italian audio            sky show_start     → sky_end
     (breaks replaced by Natural Sounds at matching master times)
  3. Natural Sounds from master   sky_end_master     → master end

Requirements: ffmpeg/ffprobe on PATH, numpy, scipy
    (via audio_utils / sting_detection modules)

Usage:
    python sync_sky_it.py [--dry-run]
        --anchor-source=S --anchor-master=S
        <sky_it_file> <master.mkv> <output_dir>
"""

import sys
from pathlib import Path

from audio_utils import fmt, get_duration, get_audio_stream_count, \
                        extract_seg, concat_segments_to_mka
from sting_detection import find_all_transitions

FP_DIR       = Path(__file__).parent / 'fingerprints'
MAX_BREAK_SECS = 420   # 7 min — discard pairings longer than this


# ── Break pairing ─────────────────────────────────────────────────────────────

def pair_breaks(events):
    """
    Pair consecutive transition events as (break_start, break_end).
    Sky IT uses the SAME fingerprint for lead-in and lead-out, so events
    alternate: lead-in, lead-out, lead-in, lead-out, ...
    Pairs > MAX_BREAK_SECS are discarded (likely a missed event).
    """
    breaks = []
    for i in range(0, len(events) - 1, 2):
        start = events[i][0]
        end   = events[i + 1][0] + events[i + 1][2]   # time + clip_dur
        dur   = end - start
        if dur > MAX_BREAK_SECS:
            print(f'  WARNING: pair at {fmt(start)}-{fmt(end)} is {dur:.0f}s '
                  f'> {MAX_BREAK_SECS}s -- skipping (likely missed event)')
        else:
            breaks.append((start, end))
    return breaks


# ── Segment building and concatenation ────────────────────────────────────────

def build_and_concat(sky_file, master_file, breaks, show_start,
                     offset, d_sky, d_master, ns_stream,
                     output_mka, dry_run=False):
    """
    Build segments and concatenate to a single MKA file.

    offset = anchor_source - anchor_master
    master_time(sky_t) = sky_t - offset

    Section 1: Natural Sounds  master 0        -> sky_start_master
    Section 2: Sky Italian     show_start      -> sky_end  (breaks -> NS)
    Section 3: Natural Sounds  sky_end_master  -> master end
    """
    def mtime(sky_t):
        return sky_t - offset

    sky_start_m = mtime(show_start)
    print(f'\n  Offset: {offset:.3f}s  (Sky IT t=0 = master {fmt(-offset)})')
    print(f'  Sky IT show starts at sky {fmt(show_start)} = master {fmt(sky_start_m)}')

    tmp_dir = Path('_tmp_sky_it_segs')
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

    # ── Section 1: NS before Sky IT starts ──
    if sky_start_m > 0:
        new_seg(master_file, ns_stream, 0.0, sky_start_m,
                '[NS]  pre-Sky IT  (master 0)')
    elif sky_start_m < 0:
        print(f'  NOTE: Sky IT show_start maps {fmt(-sky_start_m)} before master t=0; '
              f'trimming sky start by that amount.')

    # ── Section 2: Sky IT with breaks replaced by NS ──
    sky_trim = max(0.0, -sky_start_m)   # trim if show_start maps before master
    sky_cur  = show_start + sky_trim

    inner = [(s, e) for s, e in breaks if s >= sky_cur]

    for brk_s, brk_e in inner:
        sky_dur    = brk_s - sky_cur
        master_end = mtime(brk_s)
        if master_end > d_master:
            sky_dur = min(sky_dur, d_master - mtime(sky_cur))
            new_seg(sky_file, '0:a:0', sky_cur, sky_dur,
                    '[SKY] cap at master end  (sky)')
            sky_cur += sky_dur
            break
        new_seg(sky_file, '0:a:0', sky_cur, sky_dur,
                f'[SKY] Italian  (sky)')

        brk_dur  = brk_e - brk_s
        ms_start = mtime(brk_s)
        ns_dur   = min(brk_dur, d_master - ms_start)   # don't overshoot master end
        if ns_dur > 0:
            new_seg(master_file, ns_stream, ms_start, ns_dur,
                    f'[NS]  break  (master)')
        sky_cur = brk_e

    # Final Sky IT segment capped at master end
    final_dur = min(d_sky - sky_cur, d_master - mtime(sky_cur))
    if final_dur > 0:
        new_seg(sky_file, '0:a:0', sky_cur, final_dur,
                f'[SKY] final  (sky)')
        sky_cur += final_dur

    # ── Section 3: NS tail ──
    sky_end_m = mtime(sky_cur)
    ns_tail   = d_master - sky_end_m
    if ns_tail > 0:
        new_seg(master_file, ns_stream, sky_end_m, ns_tail,
                f'[NS]  post-Sky IT  (master)')

    # ── Concatenate ──
    if dry_run:
        print(f'\n[DRY RUN] Would concatenate {len(segs)} segments -> {output_mka}')
        return

    concat_segments_to_mka(segs, output_mka, list_path_prefix='_tmp_sky_it')
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
        sys.exit('Usage: sync_sky_it.py [--dry-run] '
                 '--anchor-source=S --anchor-master=S '
                 '<sky_it_file> <master.mkv> <output_dir>')

    sky_file, master_file, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    output_mka = Path(out_dir) / (Path(sky_file).stem + '_sky_it_synced.mka')

    fp_leadin = str(FP_DIR / 'sky_it_leadin.wav')
    if not Path(fp_leadin).exists():
        sys.exit(f'ERROR: Missing fingerprint: {fp_leadin}')

    # ── Offset ──
    offset = anchor_source - anchor_master   # master_time = sky_t - offset
    print(f'Anchor: sky {fmt(anchor_source)} = master {fmt(anchor_master)}')
    print(f'Offset: {offset:.3f}s  (master_time = sky_time - {offset:.3f})')

    # ── Durations ──
    d_sky    = get_duration(sky_file)
    d_master = get_duration(master_file)
    print(f'\nSky IT: {d_sky:.1f}s  ({fmt(d_sky)})')
    print(f'Master: {d_master:.1f}s  ({fmt(d_master)})')

    # ── Natural Sounds stream (last audio track in master) ──
    n_audio   = get_audio_stream_count(master_file)
    ns_stream = f'0:a:{n_audio - 1}'
    print(f'Master audio tracks: {n_audio}  ->  NS on {ns_stream}')

    # ── Detect ad breaks via sting-pair detection ──
    print('\nScanning Sky IT for ad break stings...')
    events = find_all_transitions(sky_file, [fp_leadin], stream_spec='0:a:0')

    if not events:
        print('  No transition events found -- treating as no-break source.')
        breaks = []
    else:
        print(f'  {len(events)} sting events:')
        for t, c, d in events:
            print(f'    {fmt(t)}  conf={c:.4f}  clip_dur={d:.1f}s')
        breaks = pair_breaks(events)
        if breaks:
            print(f'\n  {len(breaks)} ad breaks:')
            for i, (s, e) in enumerate(breaks):
                ms, me = s - offset, e - offset
                print(f'    Break {i+1}: sky {fmt(s)}-{fmt(e)}  '
                      f'dur={e-s:.1f}s  master {fmt(ms)}-{fmt(me)}')
        else:
            print('  No valid break pairs formed.')

    # Show start: beginning of file (no pre-show assumed for Sky IT Sprint)
    show_start = 0.0

    print('\nBuilding output segments...')
    build_and_concat(sky_file, master_file, breaks, show_start,
                     offset, d_sky, d_master, ns_stream,
                     output_mka, dry_run=dry_run)

    print(f'\nDone -> {output_mka}')


if __name__ == '__main__':
    main()
