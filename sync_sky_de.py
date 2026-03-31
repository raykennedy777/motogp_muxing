#!/usr/bin/env python3
"""
sync_sky_de.py
Add Sky Sport DE German audio to a MotoGP master file.

Break detection strategy:
  1. Watermark full-video scan to detect ad break STARTS
     (watermark disappears when ads begin)
  2. Lead-out sting search to detect break ENDS (primary)
     sky_de_leadout.wav — the jingle that plays as the broadcast returns
     from ads; break end = sting_start + sting_duration.
  3. Watermark reappearance (secondary fallback for break end)
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

from audio_utils import fmt, get_duration, get_audio_stream_count, \
                        extract_seg, concat_segments_to_mka, load_fp_wav, SR
from sting_detection import find_sting
from watermark_detection import build_watermark_template, find_all_breaks_via_watermark

FP_DIR         = Path(__file__).parent / 'fingerprints'
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

    # ── Scan for ad break starts via watermark ──
    scan_start = program_start
    scan_end   = min(d_sky_de, offset + d_master)   # don't scan past master overlap
    print(f'\nScanning Sky DE for ad breaks via watermark ({fmt(scan_start)} to {fmt(scan_end)})...')
    print(f'  Watermark params: {WM_W}x{WM_H} at ({WM_X},{WM_Y})  lag={WM_LAG_SECS}s  thresh={WM_THRESH}')

    wm_breaks = find_all_breaks_via_watermark(
        sky_de_file, wm_template, WM_X, WM_Y, WM_W, WM_H,
        WM_OUT_W, WM_OUT_H,
        scan_start=scan_start, scan_end=scan_end,
        fps=WM_FPS, thresh=WM_THRESH,
        min_break_secs=MIN_BREAK_SECS, wm_lag_secs=WM_LAG_SECS)

    # ── Refine break ends via lead-out sting (primary) ──
    fp_leadout = str(FP_DIR / 'sky_de_leadout.wav')
    leadout_exists = Path(fp_leadout).exists()
    if leadout_exists:
        sting_needle = load_fp_wav(fp_leadout)
        sting_dur    = len(sting_needle) / SR
        print(f'\nRefining break ends via lead-out sting ({fp_leadout}, {sting_dur:.2f}s)...')
    else:
        print(f'\nWARNING: Lead-out sting fingerprint not found ({fp_leadout}); '
              f'using watermark for all break ends.')

    breaks = []
    for i, (brk_s, wm_brk_e) in enumerate(wm_breaks):
        if leadout_exists:
            search_start = brk_s + MIN_BREAK_SECS
            search_end   = min(brk_s + MAX_BREAK_SECS, scan_end)
            search_dur   = search_end - search_start
            sting_t, sting_conf = find_sting(
                sky_de_file, fp_leadout, search_start, search_dur,
                stream_spec='0:a:0',
                label=f'  Break {i+1} lead-out sting')
            if sting_conf >= STING_CONF_THRESH:
                brk_e = sting_t + sting_dur
                method = f'sting (conf={sting_conf:.4f})'
            else:
                brk_e  = wm_brk_e
                method = f'watermark fallback (sting conf={sting_conf:.4f} < {STING_CONF_THRESH})'
        else:
            brk_e  = wm_brk_e
            method = 'watermark (no sting fingerprint)'
        breaks.append((brk_s, brk_e))
        ms, me = brk_s - offset, brk_e - offset
        print(f'  Break {i+1}: sky_de {fmt(brk_s)}-{fmt(brk_e)}  '
              f'dur={brk_e-brk_s:.1f}s  master {fmt(ms)}-{fmt(me)}  [{method}]')

    # ── MotoGP 65s pre-race sting extension ──
    fp_sting_gp = str(FP_DIR / 'prerace_sting_motogp.wav')
    if breaks and Path(fp_sting_gp).exists():
        print(f'\nSearching for 65s MotoGP pre-race sting...')
        mgp_t, mgp_conf = find_sting(
            sky_de_file, fp_sting_gp, scan_start, scan_end - scan_start,
            stream_spec='0:a:0', label='  65s MotoGP sting')
        if mgp_conf >= 0.3:
            # Find the last break whose start is at or before the sting
            last_idx = max((i for i, (s, _) in enumerate(breaks) if s <= mgp_t),
                           default=-1)
            if last_idx >= 0:
                brk_s, brk_e = breaks[last_idx]
                new_end = mgp_t + 65.0
                if brk_e <= new_end:
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
