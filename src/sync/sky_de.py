#!/usr/bin/env python3
"""
sync_sky_de.py
Add Sky Sport DE German audio to a MotoGP master file.

Break detection strategy:
  1. Watermark full-video scan to detect ad break STARTS
     (watermark disappears when ads begin)
  2. If correlation scan finds 0 breaks, falls back to std-based detection:
     - AD frames: wm_std < 8 (solid graphics, black/white bars)
     - Break content: wm_std > 85 (bumpers, promos during breaks)
     - Merges regions with 30s gap tolerance
  3. Lead-out sting search to detect break ENDS (primary)
     sky_de_leadout.wav — the jingle that plays as the broadcast returns
     from ads; break end = sting_start + sting_duration.
  4. Watermark reappearance (secondary fallback for break end)
     Used when the sting is not found with sufficient confidence.

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
    (via audio_utils / sting_detection / watermark_detection modules)

Usage:
    python sync_sky_de.py [--dry-run]
        --anchor-source=S --anchor-master=S
        [--program-start=S]
        <sky_de_file> <master.mkv> <output_dir>
"""

import sys
from pathlib import Path

from src.utils.audio_utils import fmt, get_duration, get_audio_stream_count, \
                        extract_seg, concat_segments_to_mka, load_fp_wav, SR
from src.utils.sting_detection import find_sting
from src.utils.watermark_detection import build_watermark_template, find_all_breaks_via_watermark
import subprocess
import numpy as np

FP_DIR = Path(__file__).parent.parent.parent / 'fingerprints'
MAX_BREAK_SECS = 420    # 7 min — sting search window upper bound
STING_CONF_THRESH = 0.3 # minimum confidence to accept a lead-out sting hit

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

# Std-based detection thresholds (primary detector)
STD_AD_THRESH            = 8    # wm_std < 8 → ad frame (solid graphics/bars)
STD_BREAK_CONTENT_THRESH = 85   # wm_std > 85 → break content (bumpers/promos)
STD_MERGE_GAP            = 30   # max gap between ad/break_content frames to merge
STD_SCAN_FPS             = 2    # scan fps for std-based detection

# Validation thresholds
# Breaks are accepted if: sting confirmed (conf >= STING_CONF_THRESH) and
# std_end is within STING_END_TOLERANCE of sting_end, OR no sting and
# duration >= STD_CONFIRMED_MIN_SECS.
STD_CONFIRMED_MIN_SECS = 150  # min duration to accept a break without sting
STING_END_TOLERANCE    = 60   # max gap (s) between std end and sting end


# ── Std-based break detection (fallback) ───────────────────────────────────────

def find_breaks_via_std(src, scan_start, scan_end, wm_x, wm_y, wm_w, wm_h,
                         fps=STD_SCAN_FPS, min_break_secs=MIN_BREAK_SECS,
                         merge_gap_secs=STD_MERGE_GAP, tmp_suffix=''):
    """
    Std-based break detection fallback when correlation-based scan fails.

    Classifies frames:
      - AD: wm_std < STD_AD_THRESH (solid graphics, black/white bars)
      - BREAK_CONTENT: wm_std > STD_BREAK_CONTENT_THRESH (bumpers, promos)
      - CONTENT: everything else (normal programming)

    Merges AD and BREAK_CONTENT regions with merge_gap_secs tolerance.
    Returns list of (break_start, break_end) tuples.
    """
    scan_dur = max(0.0, scan_end - scan_start)
    if scan_dur <= 0:
        return []

    n_pixels = WM_OUT_W * WM_OUT_H  # scaled size, not original crop size
    tmp = f'_tmp_std_scan{tmp_suffix}.raw'

    print(f'\n  Std-based scan: {fps}fps, AD std<{STD_AD_THRESH}, '
          f'break_content std>{STD_BREAK_CONTENT_THRESH}')

    try:
        subprocess.run(
            ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
             '-ss', f'{scan_start:.3f}', '-t', f'{scan_dur:.0f}',
             '-i', str(src),
             '-vf', (f'fps={fps},'
                     f'crop={wm_w}:{wm_h}:{wm_x}:{wm_y},'
                     f'scale={WM_OUT_W}:{WM_OUT_H}'),
             '-f', 'rawvideo', '-pix_fmt', 'gray', tmp],
            check=True)
        frames = np.fromfile(tmp, dtype=np.uint8).astype(np.float32)
        if Path(tmp).exists():
            Path(tmp).unlink()
    except Exception as e:
        print(f'  WARNING: Std-based scan failed: {e}')
        if Path(tmp).exists():
            Path(tmp).unlink()
        return []

    n_frames = len(frames) // n_pixels
    if n_frames == 0:
        print('  WARNING: Std-based scan produced no frames.')
        return []

    step = 1.0 / fps

    # Classify each frame
    classifications = []  # list of (time, type) where type is 'ad', 'break_content', or 'content'
    for i in range(n_frames):
        frame = frames[i * n_pixels:(i + 1) * n_pixels]
        std = frame.std()
        t_i = scan_start + i * step

        if std < STD_AD_THRESH:
            cls = 'ad'
        elif std > STD_BREAK_CONTENT_THRESH:
            cls = 'break_content'
        else:
            cls = 'content'
        classifications.append((t_i, cls))

    # Find contiguous regions of ad/break_content
    # Allow merging across content gaps <= merge_gap_secs
    breaks = []
    in_break = False
    break_start = None
    last_inbreak_time = None
    gap_start = None

    for t_i, cls in classifications:
        is_inbreak = cls in ('ad', 'break_content')

        if is_inbreak:
            if not in_break:
                # Start of new break
                in_break = True
                break_start = t_i
                last_inbreak_time = t_i
                gap_start = None
            else:
                # Continuing break
                if gap_start is not None:
                    # We were in a gap, check if it's small enough to bridge
                    gap_dur = t_i - gap_start
                    if gap_dur > merge_gap_secs:
                        # Gap too large, end the break and start new one
                        break_end = last_inbreak_time
                        if break_end - break_start >= min_break_secs:
                            breaks.append((break_start, break_end))
                        break_start = t_i
                    gap_start = None
                last_inbreak_time = t_i
        else:
            if in_break:
                if gap_start is None:
                    gap_start = t_i

    # Handle break extending to end of scan
    if in_break:
        break_end = last_inbreak_time
        if break_end - break_start >= min_break_secs:
            breaks.append((break_start, break_end))

    # Merge breaks that are close together (final pass)
    merged = []
    for brk_s, brk_e in breaks:
        if merged and brk_s - merged[-1][1] <= merge_gap_secs:
            # Merge with previous
            merged[-1] = (merged[-1][0], brk_e)
        else:
            merged.append((brk_s, brk_e))

    # Filter by minimum duration
    merged = [(s, e) for s, e in merged if e - s >= min_break_secs]

    print(f'  Std-based scan found {len(merged)} break(s)')
    for i, (brk_s, brk_e) in enumerate(merged):
        print(f'    Break {i+1}: {fmt(brk_s)}-{fmt(brk_e)}  dur={brk_e-brk_s:.1f}s')

    return merged


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

    anchor_source   = None
    anchor_master   = None
    program_start   = 0.0
    template_time   = None
    max_mean_conf   = 0.20   # reject below-thresh regions with mean corr > this
    breaks_override = None
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
        elif arg.startswith('--template-time='):
            template_time = float(arg.split('=', 1)[1])
            sys.argv.remove(arg)
        elif arg.startswith('--max-mean-conf='):
            max_mean_conf = float(arg.split('=', 1)[1])
            sys.argv.remove(arg)
        elif arg.startswith('--breaks='):
            pairs = arg.split('=', 1)[1].split(',')
            breaks_override = [tuple(float(x) for x in p.split(':')) for p in pairs]
            sys.argv.remove(arg)

    if anchor_source is None or anchor_master is None or len(sys.argv) != 4:
        sys.exit('Usage: sync_sky_de.py [--dry-run] '
                 '--anchor-source=S --anchor-master=S [--program-start=S] '
                 '[--template-time=S] [--max-mean-conf=F] '
                 '[--breaks=S:E,S:E,...] '
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

    # ── Break override: bypass all detection ──────────────────────────────────
    if breaks_override is not None:
        print(f'\n-- Using {len(breaks_override)} pre-specified break(s) --')
        for i, (s, e) in enumerate(breaks_override):
            ms, me = s - offset, e - offset
            print(f'  Break {i+1}: sky_de {fmt(s)}-{fmt(e)}  dur={e-s:.1f}s  '
                  f'master {fmt(ms)}-{fmt(me)}')
        print('\nBuilding output segments...')
        build_and_concat(sky_de_file, master_file, breaks_override, program_start,
                         offset, d_sky_de, d_master, ns_stream,
                         output_mka, dry_run=dry_run)
        print(f'\nDone -> {output_mka}')
        return

    scan_start = program_start
    scan_end   = min(d_sky_de, offset + d_master)

    # ── Build watermark template ──────────────────────────────────────────────
    # Default: 5 minutes after anchor (mid-race, stable coverage).
    # Override with --template-time if the default frame has graphics/overlays.
    if template_time is None:
        template_time = anchor_source + 300
    print(f'\nBuilding watermark template from sky_de {fmt(template_time)}...')
    wm_template = build_watermark_template(
        sky_de_file, template_time, WM_X, WM_Y, WM_W, WM_H, WM_OUT_W, WM_OUT_H)
    if wm_template is None:
        sys.exit('ERROR: Could not extract watermark template. '
                 'Try a different --template-time during stable live coverage.')
    print(f'  Template ready: crop={WM_W}x{WM_H}@({WM_X},{WM_Y}) -> {WM_OUT_W}x{WM_OUT_H}')

    # ── Correlation-based watermark scan (primary detector) ───────────────────
    print(f'\nScanning Sky DE for ad breaks via watermark ({fmt(scan_start)} to {fmt(scan_end)})...')
    print(f'  Watermark: {WM_W}x{WM_H} at ({WM_X},{WM_Y})  '
          f'thresh={WM_THRESH}  lag={WM_LAG_SECS}s  min_break={MIN_BREAK_SECS}s  fps={WM_FPS}')

    breaks = find_all_breaks_via_watermark(
        sky_de_file, wm_template, WM_X, WM_Y, WM_W, WM_H, WM_OUT_W, WM_OUT_H,
        scan_start, scan_end,
        fps=WM_FPS, thresh=WM_THRESH, min_break_secs=MIN_BREAK_SECS,
        wm_lag_secs=WM_LAG_SECS, max_mean_conf=max_mean_conf,
        min_mean_conf=None, min_min_conf=None)

    # ── Std-based fallback if correlation scan finds nothing ──────────────────
    if not breaks:
        print('\n  Watermark correlation found 0 breaks; falling back to std-based scan...')
        std_raw    = find_breaks_via_std(
            sky_de_file, scan_start, scan_end,
            WM_X, WM_Y, WM_W, WM_H,
            fps=STD_SCAN_FPS, min_break_secs=MIN_BREAK_SECS,
            merge_gap_secs=STD_MERGE_GAP)
        std_breaks = [(s, e + WM_LAG_SECS) for s, e in std_raw]
        breaks     = [(s, e) for s, e in std_breaks if e - s >= STD_CONFIRMED_MIN_SECS]

    print(f'\n  {len(breaks)} break(s) detected:')
    for i, (s, e) in enumerate(breaks):
        ms, me = s - offset, e - offset
        print(f'    Break {i+1}: sky_de {fmt(s)}-{fmt(e)}  dur={e-s:.1f}s  '
              f'master {fmt(ms)}-{fmt(me)}')

    # ── Refine break ends via lead-out sting (where available) ────────────────
    fp_leadout = str(FP_DIR / 'sky_de_leadout.wav')
    if breaks and Path(fp_leadout).exists():
        sting_needle = load_fp_wav(fp_leadout)
        sting_dur    = len(sting_needle) / SR
        print(f'\nRefining break ends via lead-out sting (conf >= {STING_CONF_THRESH})...')
        refined = []
        for i, (brk_s, brk_e) in enumerate(breaks):
            # Search ±60s around the already-detected watermark-based break end
            search_start = max(brk_s + MIN_BREAK_SECS, brk_e - 60)
            search_end   = min(brk_e + 60, scan_end)
            search_dur   = search_end - search_start
            sting_t, sting_conf = find_sting(
                sky_de_file, fp_leadout, search_start, search_dur,
                stream_spec='0:a:0',
                label=f'  Break {i+1} sting')
            if sting_conf >= STING_CONF_THRESH:
                new_end = sting_t + sting_dur
                ms, me  = brk_s - offset, new_end - offset
                print(f'    Break {i+1}: end refined via sting '
                      f'sky_de {fmt(brk_s)}-{fmt(new_end)}  master {fmt(ms)}-{fmt(me)}')
                refined.append((brk_s, new_end))
            else:
                refined.append((brk_s, brk_e))
        breaks = refined

    # ── MotoGP 65s pre-race sting extension ──
    fp_sting_gp = str(FP_DIR / 'prerace_sting_motogp.wav')
    if breaks and Path(fp_sting_gp).exists():
        print(f'\nSearching for 65s MotoGP pre-race sting...')
        # Restrict search: MotoGP sting in a Combined feed arrives roughly
        # anchor+9000s into the file (Combined = all 3 classes, ~2:30 before MotoGP).
        # Cap at 2400s window so fftconvolve doesn't OOM on long audio.
        mgp_search_start = max(scan_start, anchor_source + 9000)
        mgp_search_end   = min(scan_end, mgp_search_start + 2400)
        # Search in 1200s chunks to stay within fftconvolve memory limits
        CHUNK = 1200
        mgp_t, mgp_conf = mgp_search_start, 0.0
        pos = mgp_search_start
        while pos < mgp_search_end:
            chunk_dur = min(CHUNK, mgp_search_end - pos)
            t, c = find_sting(
                sky_de_file, fp_sting_gp, pos, chunk_dur,
                stream_spec='0:a:0', label=f'  65s MotoGP sting [{fmt(pos)}-{fmt(pos+chunk_dur)}]')
            if c > mgp_conf:
                mgp_t, mgp_conf = t, c
            pos += CHUNK - 70  # overlap by sting length
        if mgp_conf >= 0.3:
            # Find the last break whose start is at or before the sting
            last_idx = max((i for i, (s, _) in enumerate(breaks) if s <= mgp_t),
                           default=-1)
            if last_idx >= 0:
                brk_s, brk_e = breaks[last_idx]
                new_end = mgp_t + 65.0
                gap = mgp_t - brk_e
                if gap > 90.0:
                    print(f'  Sting at {fmt(mgp_t)} is {gap:.1f}s after break {last_idx+1} end; skipping extension.')
                elif brk_e <= new_end:
                    breaks[last_idx] = (brk_s, new_end)
                    ms, me = brk_s - offset, new_end - offset
                    print(f'  Extended break {last_idx + 1}: sky_de {fmt(brk_s)}-{fmt(new_end)}  '
                          f'dur={new_end-brk_s:.1f}s  master {fmt(ms)}-{fmt(me)}')
                else:
                    print(f'  Break {last_idx + 1} already covers sting end; no extension needed.')
        else:
            print(f'  Not found (conf={mgp_conf:.4f}); no extension applied.')

    print(f'\n{len(breaks)} ad break(s) detected.')
    print('\nBuilding output segments...')
    build_and_concat(sky_de_file, master_file, breaks, program_start,
                     offset, d_sky_de, d_master, ns_stream,
                     output_mka, dry_run=dry_run)

    print(f'\nDone -> {output_mka}')


if __name__ == '__main__':
    main()
