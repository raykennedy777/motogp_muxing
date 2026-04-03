#!/usr/bin/env python3
"""
sync_servustv.py
Add ServusTV German audio to a MotoGP master file.

No sting-based break detection. Uses:
  - Frame-based sync anchor (provided as ServusTV time + master time)
  - Watermark full-scan to detect all ad breaks
  - Two-threshold filter: a below-threshold region must have min correlation
    < MIN_AD_CONF to be accepted as a real ad break. Graphic/banner
    obscurations (where background is still visible) are rejected because
    their minimum correlation stays well above ad-content noise levels.

Watermark: ~150x70 pixels at position (1653, 74) in 1920x1080 video.
           At break end, a coloured variant appears ~4s before the normal
           watermark returns (wm_lag_secs=4.0 compensates).

Template extraction (choose one method):
  Default: Extract from 25:00 (during live race coverage).
  --template-time=S   Extract from a specific time point.
  --template-file=F   Load a pre-saved template PNG from fingerprints/.

--anchor-source=S   ServusTV time (seconds) of the sync anchor frame.
--anchor-master=S   Corresponding master time (seconds) of the same frame.
--template-time=S   ServusTV time to extract the watermark template from.
                    Must be during confirmed live coverage with a clear,
                    steady watermark.  Defaults to --anchor-source.
--program-start=S   ServusTV time (seconds) where actual coverage begins
                    (after any pre-show). Default: 0.
--min-break-secs=S  Minimum duration (seconds) for a region to be classified
                    as an ad break.  Default: 60.  Raise to ~90 for Moto2
                    to suppress residual false positives from in-race graphics.
--min-gap=S         Drop any break whose start falls within S seconds of the
                    previous break's end.  Use 180 for MotoGP to suppress the
                    race-start graphics false positive that immediately follows
                    the formation-lap break.
--max-std=S         Maximum standard deviation of correlation for a region to be
                    classified as an ad break.  Default: 0.08.  True ad breaks
                    have stable correlations (std < 0.07), while graphics/overlays
                    cause variable correlations (std > 0.1).
--min-stable-pct=S  Minimum percentage of frames with stable correlation around
                    -0.265 (the "no watermark" pattern).  Default: 50.  True ad
                    breaks have >90% stable negative correlations.

Output structure:
  1. Natural Sounds from master   t = 0              -> program_start_master
  2. ServusTV audio               program_start      -> stv_end
     (breaks replaced by Natural Sounds at matching master times)
  3. Natural Sounds from master   stv_end_master     -> master end

Requirements: ffmpeg/ffprobe on PATH, numpy
    (via audio_utils / watermark_detection modules)

Usage:
    python sync_servustv.py [--dry-run]
        --anchor-source=S --anchor-master=S
        [--template-time=S] [--program-start=S] [--min-break-secs=S]
        <servustv_file> <master.mkv> <output_dir>
"""

import sys
from pathlib import Path

from audio_utils import fmt, get_duration, get_audio_stream_count, \
                        extract_seg, concat_segments_to_mka
from watermark_detection import (build_watermark_template,
                                  build_watermark_template_averaged,
                                  find_all_breaks_via_watermark)

# ServusTV watermark parameters (1920x1080 source)
WM_X     = 1653   # left edge
WM_Y     = 74     # top edge
WM_W     = 150    # width
WM_H     = 70     # height
WM_OUT_W = 64     # downscale width
WM_OUT_H = 32     # downscale height

# At break end, a coloured watermark variant appears ~4s before the normal one.
# Subtract this lag so break_end aligns with actual program resumption.
WM_LAG_SECS = 4.0

WM_THRESH   = 0.44
WM_FPS      = 2
MIN_AD_CONF = 0.15  # reject regions where min_conf >= this (graphic obscuration)

# max_std / min_stable_pct: when set, reject regions whose correlation std exceeds
# the threshold and require a stable-negative cluster.  Set to None to rely only
# on min_break_secs and min_ad_conf — appropriate when full-screen ad breaks produce
# variable (but consistently low) correlation due to diverse ad content.
MAX_STD = None
MIN_STABLE_PCT = None


# ── Segment building ───────────────────────────────────────────────────────────

def build_and_concat(stv_file, master_file, breaks, show_start,
                     offset, d_stv, d_master, ns_stream,
                     output_mka, dry_run=False):
    """
    offset = anchor_source - anchor_master
    master_time(stv_t) = stv_t - offset

    Section 1: NS           master 0          -> show_start_master
    Section 2: ServusTV     show_start        -> end  (breaks -> NS)
    Section 3: NS           stv_end_master    -> master end
    """
    def mtime(t):
        return t - offset

    show_start_m = mtime(show_start)
    print(f'\n  Offset: {offset:.3f}s  (ServusTV t=0 = master {fmt(-offset)})')
    print(f'  Show starts at stv {fmt(show_start)} = master {fmt(show_start_m)}')

    tmp_dir = Path('_tmp_stv_segs')
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
                '[NS]  pre-ServusTV  (master 0)')
    elif show_start_m < 0:
        print(f'  NOTE: ServusTV show_start maps {fmt(-show_start_m)} '
              f'before master start; trimming.')

    # ── Section 2: ServusTV with breaks replaced by NS ──
    stv_trim = max(0.0, -show_start_m)
    stv_cur  = show_start + stv_trim

    inner = [(s, e) for s, e in breaks if s >= stv_cur]

    for brk_s, brk_e in inner:
        stv_dur    = brk_s - stv_cur
        master_end = mtime(brk_s)
        if master_end > d_master:
            stv_dur = min(stv_dur, d_master - mtime(stv_cur))
            new_seg(stv_file, '0:a:0', stv_cur, stv_dur,
                    '[STV] cap at master end  (stv)')
            stv_cur += stv_dur
            break
        new_seg(stv_file, '0:a:0', stv_cur, stv_dur,
                '[STV] German  (stv)')

        brk_dur  = brk_e - brk_s
        ms_start = mtime(brk_s)
        ns_dur   = min(brk_dur, d_master - ms_start)
        if ns_dur > 0:
            new_seg(master_file, ns_stream, ms_start, ns_dur,
                    f'[NS]  break  (master)')
        stv_cur = brk_e

    # Final ServusTV segment
    final_dur = min(d_stv - stv_cur, d_master - mtime(stv_cur))
    if final_dur > 0:
        new_seg(stv_file, '0:a:0', stv_cur, final_dur,
                '[STV] final  (stv)')
        stv_cur += final_dur

    # ── Section 3: NS tail ──
    stv_end_m = mtime(stv_cur)
    ns_tail   = d_master - stv_end_m
    if ns_tail > 0:
        new_seg(master_file, ns_stream, stv_end_m, ns_tail,
                f'[NS]  post-ServusTV  (master {fmt(stv_end_m)})')

    # ── Concatenate ──
    if dry_run:
        print(f'\n[DRY RUN] Would concatenate {len(segs)} segments -> {output_mka}')
        return

    concat_segments_to_mka(segs, output_mka, list_path_prefix='_tmp_stv')
    for f in tmp_dir.iterdir():
        f.unlink()
    tmp_dir.rmdir()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    dry_run = '--dry-run' in sys.argv
    if dry_run:
        sys.argv.remove('--dry-run')
        print('[DRY RUN] Detection and segment planning only -- no audio will be encoded.')

    anchor_source    = None
    anchor_master    = None
    template_time    = None
    template_file    = None
    template_averaged = False
    program_start    = 0.0
    min_break_secs   = 55
    min_gap          = 0
    max_std          = MAX_STD
    min_stable_pct   = MIN_STABLE_PCT
    for arg in list(sys.argv[1:]):
        if arg.startswith('--anchor-source='):
            anchor_source = float(arg.split('=', 1)[1])
            sys.argv.remove(arg)
        elif arg.startswith('--anchor-master='):
            anchor_master = float(arg.split('=', 1)[1])
            sys.argv.remove(arg)
        elif arg.startswith('--template-time='):
            template_time = float(arg.split('=', 1)[1])
            sys.argv.remove(arg)
        elif arg.startswith('--template-file='):
            template_file = arg.split('=', 1)[1]
            sys.argv.remove(arg)
        elif arg == '--template-averaged':
            template_averaged = True
            sys.argv.remove(arg)
        elif arg.startswith('--program-start='):
            program_start = float(arg.split('=', 1)[1])
            sys.argv.remove(arg)
        elif arg.startswith('--min-break-secs='):
            min_break_secs = int(arg.split('=', 1)[1])
            sys.argv.remove(arg)
        elif arg.startswith('--min-gap='):
            min_gap = float(arg.split('=', 1)[1])
            sys.argv.remove(arg)
        elif arg.startswith('--max-std='):
            max_std = float(arg.split('=', 1)[1])
            sys.argv.remove(arg)
        elif arg.startswith('--min-stable-pct='):
            min_stable_pct = float(arg.split('=', 1)[1])
            sys.argv.remove(arg)

    if anchor_source is None or anchor_master is None or len(sys.argv) != 4:
        sys.exit('Usage: sync_servustv.py [--dry-run] '
                 '--anchor-source=S --anchor-master=S '
                 '[--template-time=S] [--template-file=F] [--template-averaged] '
                 '[--program-start=S] [--min-break-secs=S] [--min-gap=S] '
                 '[--max-std=S] [--min-stable-pct=S] '
                 '<servustv_file> <master.mkv> <output_dir>')

    stv_file, master_file, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    output_mka = Path(out_dir) / (Path(stv_file).stem + '_stv_synced.mka')

    # ── Offset ──
    offset = anchor_source - anchor_master   # master_time = stv_t - offset
    print(f'Anchor: ServusTV {fmt(anchor_source)} = master {fmt(anchor_master)}')
    print(f'Offset: {offset:.3f}s  (master_time = stv_time - {offset:.3f})')
    if program_start > 0:
        print(f'Program start: stv {fmt(program_start)} '
              f'= master {fmt(program_start - offset)}')

    # ── Durations ──
    d_stv    = get_duration(stv_file)
    d_master = get_duration(master_file)
    print(f'\nServusTV: {d_stv:.1f}s  ({fmt(d_stv)})')
    print(f'Master:   {d_master:.1f}s  ({fmt(d_master)})')

    # ── Natural Sounds stream ──
    n_audio   = get_audio_stream_count(master_file)
    ns_stream = f'0:a:{n_audio - 1}'
    print(f'Master audio tracks: {n_audio}  ->  NS on {ns_stream}')

    # ── Build watermark template ──
    print(f'\nBuilding watermark template...')
    if template_file:
        # Load from pre-saved file
        print(f'  Loading from {template_file}...')
        # TODO: Implement PNG loading if needed
        sys.exit('ERROR: --template-file not yet implemented')
    elif template_averaged:
        # Average 8 frames from anchor+3min through anchor+10min (1 per minute).
        # Starting 3 minutes after anchor avoids pre-race sting / unusual visual moments
        # that can occur right at the sync point and corrupt the template.
        avg_start = anchor_source + 3 * 60
        avg_end   = anchor_source + 10 * 60
        print(f'  Averaging frames {fmt(avg_start)}-{fmt(avg_end)} (every 60s, 8 frames)...')
        wm_template = build_watermark_template_averaged(
            stv_file, avg_start, avg_end, 60.0,
            WM_X, WM_Y, WM_W, WM_H, WM_OUT_W, WM_OUT_H)
    elif template_time is not None:
        # User-specified time point
        print(f'  Extracting from stv {fmt(template_time)}...')
        wm_template = build_watermark_template(
            stv_file, template_time, WM_X, WM_Y, WM_W, WM_H, WM_OUT_W, WM_OUT_H)
    else:
        # Default: use anchor_source (known to be during live race coverage)
        print(f'  Extracting from stv {fmt(anchor_source)} (anchor, during race)...')
        print(f'  (use --template-averaged to average frames around anchor, '
              f'or --template-time=S for a specific frame)')
        wm_template = build_watermark_template(
            stv_file, anchor_source, WM_X, WM_Y, WM_W, WM_H, WM_OUT_W, WM_OUT_H)
    if wm_template is None:
        sys.exit('ERROR: Could not build watermark template. Check file and coordinates.')
    print(f'  Template ready: crop={WM_W}x{WM_H}@({WM_X},{WM_Y}) -> {WM_OUT_W}x{WM_OUT_H}')

    # ── Scan for ad breaks via watermark ──
    scan_start = program_start
    scan_end   = min(d_stv, offset + d_master)
    print(f'\nScanning ServusTV for ad breaks ({fmt(scan_start)} to {fmt(scan_end)})...')
    gap_note = f'  min_gap={min_gap:.0f}s' if min_gap > 0 else ''
    std_note = f'  max_std={max_std:.3f}' if max_std is not None else ''
    stable_note = f'  min_stable={min_stable_pct:.0f}%' if min_stable_pct is not None else ''
    print(f'  Watermark: {WM_W}x{WM_H} at ({WM_X},{WM_Y})  '
          f'thresh={WM_THRESH}  lag={WM_LAG_SECS}s  '
          f'min_break={min_break_secs}s  min_ad_conf={MIN_AD_CONF}{gap_note}{std_note}{stable_note}')

    breaks = find_all_breaks_via_watermark(
        stv_file, wm_template, WM_X, WM_Y, WM_W, WM_H,
        WM_OUT_W, WM_OUT_H,
        scan_start=scan_start, scan_end=scan_end,
        fps=WM_FPS, thresh=WM_THRESH,
        min_break_secs=min_break_secs, wm_lag_secs=WM_LAG_SECS,
        min_ad_conf=MIN_AD_CONF,
        max_std=MAX_STD, min_stable_pct=MIN_STABLE_PCT)

    # ── Drop breaks that start too soon after the previous break ──
    if min_gap > 0 and breaks:
        filtered = [breaks[0]]
        for brk in breaks[1:]:
            gap = brk[0] - filtered[-1][1]
            if gap < min_gap:
                print(f'  Skip: gap {gap:.1f}s < min_gap={min_gap:.0f}s  '
                      f'(break at {fmt(brk[0])} dropped)')
            else:
                filtered.append(brk)
        breaks = filtered

    print(f'\n  {len(breaks)} ad break(s) found:')
    for i, (s, e) in enumerate(breaks):
        ms, me = s - offset, e - offset
        print(f'  Break {i+1}: stv {fmt(s)}-{fmt(e)}  '
              f'dur={e-s:.1f}s  master {fmt(ms)}-{fmt(me)}')

    print('\nBuilding output segments...')
    build_and_concat(stv_file, master_file, breaks, program_start,
                     offset, d_stv, d_master, ns_stream,
                     output_mka, dry_run=dry_run)

    print(f'\nDone -> {output_mka}')


if __name__ == '__main__':
    main()
