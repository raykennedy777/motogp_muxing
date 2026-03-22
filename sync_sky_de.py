#!/usr/bin/env python3
"""
sync_sky_de.py
Add Sky Sport DE German audio to a MotoGP master file.

No lead-in/lead-out stings for ad breaks. Uses:
  - Frame-based sync anchor (first camera change after race start)
  - Watermark full-video scan to detect all ad breaks

Watermark: ~180x34 pixels at position (1040, 35) in 1280x720 video.
           Reappears ~3.44s after actual program resumption.

--program-start=S   Sky DE time (seconds) where actual coverage begins
                    (after the program starting sting). Default: 0.

Output structure:
  1. Natural Sounds from master   t = 0                  -> program_start_master
  2. Sky DE audio                 program_start           -> sky_de_end
     (breaks replaced by Natural Sounds at matching master times)
  3. Natural Sounds from master   sky_de_end_master       -> master end

Requirements: ffmpeg/ffprobe on PATH, numpy, scipy
    (via audio_utils / watermark_detection modules)

Usage:
    python sync_sky_de.py [--dry-run]
        --anchor-source=S --anchor-master=S
        [--program-start=S]
        <sky_de_file> <master.mkv> <output_dir>
"""

import sys
from pathlib import Path

from audio_utils import fmt, get_duration, get_audio_stream_count, \
                        extract_seg, concat_segments_to_mka
from watermark_detection import build_watermark_template, find_all_breaks_via_watermark

# Sky DE watermark parameters (1280x720 source)
WM_X     = 1040   # left edge
WM_Y     = 35     # top edge
WM_W     = 180    # width
WM_H     = 34     # height
WM_OUT_W = 64     # downscale width for template matching
WM_OUT_H = 12     # downscale height (approx 5.3:1 aspect)

# Watermark lags actual program resumption by ~3.44s
WM_LAG_SECS = 3.44

WM_THRESH      = 0.44
WM_FPS         = 2
MIN_BREAK_SECS = 45


# ── Segment building ──────────────────────────────────────────────────────────

def build_and_concat(sky_de_file, master_file, breaks, show_start,
                     offset, d_sky_de, d_master, ns_stream,
                     output_mka, dry_run=False):
    """
    offset = anchor_source - anchor_master
    master_time(sky_de_t) = sky_de_t - offset

    Section 1: NS           master 0        -> show_start_master
    Section 2: Sky DE       show_start      -> end  (breaks -> NS)
    Section 3: NS           sky_de_end_m    -> master end
    """
    def mtime(t):
        return t - offset

    show_start_m = mtime(show_start)
    print(f'\n  Offset: {offset:.3f}s  (Sky DE t=0 = master {fmt(-offset)})')
    print(f'  Show starts at sky_de {fmt(show_start)} = master {fmt(show_start_m)}')

    tmp_dir = Path('_tmp_sky_de_segs')
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

    # ── Section 1: NS head ──
    if show_start_m > 0:
        new_seg(master_file, ns_stream, 0.0, show_start_m,
                '[NS]  pre-Sky DE  (master 0)')
    elif show_start_m < 0:
        print(f'  NOTE: Sky DE show_start maps {fmt(-show_start_m)} before master; trimming.')

    # ── Section 2: Sky DE with breaks replaced by NS ──
    sky_trim = max(0.0, -show_start_m)
    sky_cur  = show_start + sky_trim

    inner = [(s, e) for s, e in breaks if s >= sky_cur]

    for brk_s, brk_e in inner:
        sky_dur    = brk_s - sky_cur
        master_end = mtime(brk_s)
        if master_end > d_master:
            sky_dur = min(sky_dur, d_master - mtime(sky_cur))
            new_seg(sky_de_file, '0:a:0', sky_cur, sky_dur,
                    '[SKY DE] cap at master end  (sky_de)')
            sky_cur += sky_dur
            break
        new_seg(sky_de_file, '0:a:0', sky_cur, sky_dur,
                '[SKY DE] German  (sky_de)')

        brk_dur  = brk_e - brk_s
        ms_start = mtime(brk_s)
        ns_dur   = min(brk_dur, d_master - ms_start)   # don't overshoot master end
        if ns_dur > 0:
            new_seg(master_file, ns_stream, ms_start, ns_dur,
                    f'[NS]  break  (master)')
        sky_cur = brk_e

    # Final Sky DE segment
    final_dur = min(d_sky_de - sky_cur, d_master - mtime(sky_cur))
    if final_dur > 0:
        new_seg(sky_de_file, '0:a:0', sky_cur, final_dur,
                '[SKY DE] final  (sky_de)')
        sky_cur += final_dur

    # ── Section 3: NS tail ──
    sky_end_m = mtime(sky_cur)
    ns_tail   = d_master - sky_end_m
    if ns_tail > 0:
        new_seg(master_file, ns_stream, sky_end_m, ns_tail,
                f'[NS]  post-Sky DE  (master {fmt(sky_end_m)})')

    # ── Concatenate ──
    if dry_run:
        print(f'\n[DRY RUN] Would concatenate {len(segs)} segments -> {output_mka}')
        return

    concat_segments_to_mka(segs, output_mka, list_path_prefix='_tmp_sky_de')
    for f in tmp_dir.iterdir():
        f.unlink()
    tmp_dir.rmdir()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    dry_run = '--dry-run' in sys.argv
    if dry_run:
        sys.argv.remove('--dry-run')
        print('[DRY RUN] Detection and segment planning only -- no audio will be encoded.')

    anchor_source  = None
    anchor_master  = None
    program_start  = 0.0
    for arg in list(sys.argv[1:]):
        if arg.startswith('--anchor-source='):
            anchor_source = float(arg.split('=', 1)[1])
            sys.argv.remove(arg)
        elif arg.startswith('--anchor-master='):
            anchor_master = float(arg.split('=', 1)[1])
            sys.argv.remove(arg)
        elif arg.startswith('--program-start='):
            program_start = float(arg.split('=', 1)[1])
            sys.argv.remove(arg)

    if anchor_source is None or anchor_master is None or len(sys.argv) != 4:
        sys.exit('Usage: sync_sky_de.py [--dry-run] '
                 '--anchor-source=S --anchor-master=S [--program-start=S] '
                 '<sky_de_file> <master.mkv> <output_dir>')

    sky_de_file, master_file, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    output_mka = Path(out_dir) / (Path(sky_de_file).stem + '_sky_de_synced.mka')

    # ── Offset ──
    offset = anchor_source - anchor_master   # master_time = sky_de_t - offset
    print(f'Anchor: Sky DE {fmt(anchor_source)} = master {fmt(anchor_master)}')
    print(f'Offset: {offset:.3f}s  (master_time = sky_de_time - {offset:.3f})')
    if program_start > 0:
        print(f'Program start: sky_de {fmt(program_start)} = master {fmt(program_start - offset)}')

    # ── Durations ──
    d_sky_de = get_duration(sky_de_file)
    d_master = get_duration(master_file)
    print(f'\nSky DE: {d_sky_de:.1f}s  ({fmt(d_sky_de)})')
    print(f'Master: {d_master:.1f}s  ({fmt(d_master)})')

    # ── Natural Sounds stream ──
    n_audio   = get_audio_stream_count(master_file)
    ns_stream = f'0:a:{n_audio - 1}'
    print(f'Master audio tracks: {n_audio}  ->  NS on {ns_stream}')

    # ── Build watermark template from anchor time (confirmed live) ──
    print(f'\nBuilding watermark template at sky_de {fmt(anchor_source)} (race start)...')
    wm_template = build_watermark_template(
        sky_de_file, anchor_source, WM_X, WM_Y, WM_W, WM_H, WM_OUT_W, WM_OUT_H)
    if wm_template is None:
        sys.exit('ERROR: Could not build watermark template. Check file and coordinates.')
    print(f'  Template built from crop={WM_W}x{WM_H}@({WM_X},{WM_Y}) -> {WM_OUT_W}x{WM_OUT_H}')

    # ── Scan for ad breaks via watermark ──
    scan_start = program_start
    scan_end   = min(d_sky_de, offset + d_master)   # don't scan past master overlap
    print(f'\nScanning Sky DE for ad breaks via watermark ({fmt(scan_start)} to {fmt(scan_end)})...')
    print(f'  Watermark params: {WM_W}x{WM_H} at ({WM_X},{WM_Y})  lag={WM_LAG_SECS}s  thresh={WM_THRESH}')

    breaks = find_all_breaks_via_watermark(
        sky_de_file, wm_template, WM_X, WM_Y, WM_W, WM_H,
        WM_OUT_W, WM_OUT_H,
        scan_start=scan_start, scan_end=scan_end,
        fps=WM_FPS, thresh=WM_THRESH,
        min_break_secs=MIN_BREAK_SECS, wm_lag_secs=WM_LAG_SECS)

    print(f'\n  {len(breaks)} ad break(s) found:')
    for i, (s, e) in enumerate(breaks):
        ms, me = s - offset, e - offset
        print(f'  Break {i+1}: sky_de {fmt(s)}-{fmt(e)}  '
              f'dur={e-s:.1f}s  master {fmt(ms)}-{fmt(me)}')

    print('\nBuilding output segments...')
    build_and_concat(sky_de_file, master_file, breaks, program_start,
                     offset, d_sky_de, d_master, ns_stream,
                     output_mka, dry_run=dry_run)

    print(f'\nDone -> {output_mka}')


if __name__ == '__main__':
    main()
