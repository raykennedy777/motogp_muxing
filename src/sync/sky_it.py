#!/usr/bin/env python3
"""
sync_sky_it.py
Add Sky Sport MotoGP Italian audio to a MotoGP master file.

Uses a frame-based sync anchor (first camera change after race start)
and a three-tier break detection strategy:

  1. Sting-pair detection (primary)
     The same fingerprint (sky_it_leadin.wav) marks both the lead-in
     (ads start) and lead-out (program resumes) of each ad break.
     Consecutive events are paired; pairs > MAX_BREAK_SECS discarded.
     If the broadcast starts mid-break (first sting is a lead-out), the
     pairing tries both offset-0 and offset-1 and uses whichever yields
     more valid pairs.

  2. PUBBLICITÀ text detection (secondary — break START fallback)
     A white "PUBBLICITÀ" overlay appears in the bottom-right at the
     start of the bumper clip that precedes each ad block. Template-matched
     against a pre-built fingerprint (fingerprints/sky_it_pubb.png).
     Used to detect starts not caught by stings.

  3. MGP logo watermark reappearance (secondary — break END fallback)
     The MGP logo (bottom-right) disappears during ads and reappears when
     program resumes. Uses a pre-built fingerprint (fingerprints/sky_it_mgp.png).
     Used to find the END of a break whose start was found via PUBBLICITÀ
     when no lead-out sting is present.

Pre-built templates:
  Run extract_sky_templates.py once per broadcast season to generate
  fingerprints/sky_it_mgp.png and fingerprints/sky_it_pubb.png.

Output structure:
  1. Natural Sounds from master   t = 0             → sky_start_master
  2. Sky Italian audio            sky show_start     → sky_end
     (breaks replaced by Natural Sounds at matching master times)
  3. Natural Sounds from master   sky_end_master     → master end

Requirements: ffmpeg/ffprobe on PATH, numpy, scipy
    (via audio_utils / sting_detection / watermark_detection modules)

Usage:
    python sync_sky_it.py [--dry-run]
        --anchor-source=S --anchor-master=S
        <sky_it_file> <master.mkv> <output_dir>
"""

import subprocess, os, sys
import numpy as np
from pathlib import Path

from src.utils.audio_utils import fmt, get_duration, get_audio_stream_count, \
                        extract_seg, concat_segments_to_mka
from src.utils.sting_detection import find_all_transitions, find_sting
from src.utils.watermark_detection import load_watermark_png, find_break_end_via_watermark, \
                                find_break_start_via_watermark

FP_DIR = Path(__file__).parent.parent.parent / 'fingerprints'
STING_PAIR_GAP_SECS = 480  # 8 min — max sting-start gap to form a pair
MAX_BREAK_SECS      = 420  # 7 min — max break duration for sanity checks
WM_FALLBACK_SECS    = 60.0 # break-end fallback when watermark search fails

# ── PUBBLICITÀ text region (bottom-right, 1280×720) — break START indicator ──
SKY_PUBB_X,     SKY_PUBB_Y     = 1107, 609
SKY_PUBB_W,     SKY_PUBB_H     = 100,  35
SKY_PUBB_OUT_W, SKY_PUBB_OUT_H = 100,  35
SKY_PUBB_THRESH       = 0.70   # text is very distinctive; high threshold
SKY_PUBB_SUPPRESS_SECS = 120   # min gap between detections (same break)

# ── MGP logo watermark region (bottom-right, 1280×720) — break END indicator ──
SKY_MGP_WM_X,     SKY_MGP_WM_Y     = 1124, 662
SKY_MGP_WM_W,     SKY_MGP_WM_H     = 115,  43
SKY_MGP_OUT_W,    SKY_MGP_OUT_H    = 115,  43
SKY_MGP_WM_THRESH    = 0.28    # gradient mode: program ~0.31–0.44; ad spikes max ~0.37; no ad frames reach 0.28 in search windows
SKY_MGP_WM_MIN_OFFSET = 20     # don't look for return within first 20s


# ── Break pairing ─────────────────────────────────────────────────────────────

def pair_breaks(events):
    """
    Pair sting events by proximity: consecutive stings whose start times are
    within STING_PAIR_GAP_SECS of each other are treated as (lead-in, lead-out)
    pairs. Any sting that cannot be paired is returned as orphaned.

    Returns (pairs, orphaned_events).
    """
    if not events:
        return [], []
    pairs    = []
    orphaned = []
    i = 0
    while i < len(events):
        if i + 1 < len(events):
            gap = events[i + 1][0] - events[i][0]
            if gap <= STING_PAIR_GAP_SECS:
                s = events[i][0]
                e = events[i + 1][0] + events[i + 1][2]
                print(f'  Pair: {fmt(s)}-{fmt(e)}  gap={gap:.1f}s  dur={e-s:.1f}s')
                pairs.append((s, e))
                i += 2
            else:
                print(f'  Orphaned (next sting {gap:.1f}s away > {STING_PAIR_GAP_SECS}s): '
                      f'{fmt(events[i][0])}  conf={events[i][1]:.4f}')
                orphaned.append(events[i])
                i += 1
        else:
            print(f'  Orphaned (last sting): {fmt(events[i][0])}  conf={events[i][1]:.4f}')
            orphaned.append(events[i])
            i += 1
    return pairs, orphaned


# ── PUBBLICITÀ detection ───────────────────────────────────────────────────────

def find_pubblicita_starts(src, template, scan_start=0.0, scan_end=0.0,
                            fps=1, thresh=SKY_PUBB_THRESH,
                            suppress_secs=SKY_PUBB_SUPPRESS_SECS):
    """
    Scan for frames where the PUBBLICITÀ text overlay is visible.
    Returns list of (time, conf) — one entry per distinct appearance,
    suppressed by suppress_secs to avoid multiple hits per break.
    """
    scan_dur = max(0.0, scan_end - scan_start)
    if scan_dur <= 0 or template is None:
        return []

    n_pixels = SKY_PUBB_OUT_W * SKY_PUBB_OUT_H
    import time, tempfile
    tmp = os.path.join(tempfile.gettempdir(), f'_tmp_pubb_scan_{int(time.time()*1000)}.raw')
    try:
        # Extract full frames to avoid YUV420 chroma rounding
        subprocess.run(
            ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
             '-ss', f'{scan_start:.3f}', '-t', f'{scan_dur:.0f}', '-i', str(src),
             '-vf', f'fps={fps},scale=1280:720',
             '-f', 'rawvideo', '-pix_fmt', 'gray', tmp],
            check=True)
        # WSL interop timing
        for _ in range(10):
            if os.path.exists(tmp):
                break
            time.sleep(0.2)
        if not os.path.exists(tmp):
            print(f'  WARNING: PUBBLICITÀ scan: file not created (WSL timing issue)')
            return []
        frames = np.fromfile(tmp, dtype=np.uint8).astype(np.float32)
        os.remove(tmp)
    except Exception as e:
        print(f'  WARNING: PUBBLICITÀ scan failed: {e}')
        if os.path.exists(tmp):
            os.remove(tmp)
        return []

    frame_px = 1280 * 720
    n_frames = len(frames) // frame_px
    step = 1.0 / fps
    t0 = template - template.mean()
    t0_norm = np.linalg.norm(t0)
    row_offs = [(SKY_PUBB_Y + r) * 1280 + SKY_PUBB_X for r in range(SKY_PUBB_H)]

    detections  = []
    last_detect = -suppress_secs

    for i in range(n_frames):
        frame = frames[i * frame_px:(i + 1) * frame_px]
        crop = np.concatenate([frame[off:off + SKY_PUBB_W] for off in row_offs])
        if SKY_PUBB_W != SKY_PUBB_OUT_W or SKY_PUBB_H != SKY_PUBB_OUT_H:
            import subprocess as _sp
            _tmp = os.path.join(tempfile.gettempdir(), f'_tmp_pubb_s_{i}.raw')
            _sp.run(['ffmpeg','-y','-hide_banner','-loglevel','error',
                     '-f','rawvideo','-pix_fmt','gray','-s',f'{SKY_PUBB_W}:{SKY_PUBB_H}',
                     '-i','pipe:0','-vf',f'scale={SKY_PUBB_OUT_W}:{SKY_PUBB_OUT_H}',
                     '-f','rawvideo','-pix_fmt','gray',_tmp],
                    input=crop.astype(np.uint8).tobytes(), check=True)
            crop = np.fromfile(_tmp, dtype=np.uint8).astype(np.float32)
            if os.path.exists(_tmp): os.remove(_tmp)
        f0    = crop - crop.mean()
        conf  = np.dot(f0, t0) / (np.linalg.norm(f0) * t0_norm + 1e-10)
        t_i   = scan_start + i * step
        if conf >= thresh and (t_i - last_detect) >= suppress_secs:
            detections.append((t_i, conf))
            last_detect = t_i

    return detections


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
    sky_trim = max(0.0, -sky_start_m)
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
        ns_dur   = min(brk_dur, d_master - ms_start)
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
    dry_run    = '--dry-run'    in sys.argv
    sting_only = '--sting-only' in sys.argv
    if dry_run:
        sys.argv.remove('--dry-run')
        print('[DRY RUN] Detection and segment planning only -- no audio will be encoded.')
    if sting_only:
        sys.argv.remove('--sting-only')
        print('[sting-only] Skipping PUBBLICITÀ and watermark fallback detection.')

    anchor_source       = None
    anchor_master       = None
    orphan_break_starts = []
    for arg in list(sys.argv[1:]):
        if arg.startswith('--anchor-source='):
            anchor_source = float(arg.split('=', 1)[1])
            sys.argv.remove(arg)
        elif arg.startswith('--anchor-master='):
            anchor_master = float(arg.split('=', 1)[1])
            sys.argv.remove(arg)
        elif arg.startswith('--orphan-break-starts='):
            orphan_break_starts = [float(t) for t in arg.split('=', 1)[1].split(',')]
            sys.argv.remove(arg)

    if anchor_source is None or anchor_master is None or len(sys.argv) != 4:
        sys.exit('Usage: sync_sky_it.py [--dry-run] [--sting-only] '
                 '--anchor-source=S --anchor-master=S '
                 '[--orphan-break-starts=T1,T2,...] '
                 '<sky_it_file> <master.mkv> <output_dir>')

    sky_file, master_file, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    output_mka = Path(out_dir) / (Path(sky_file).stem + '_sky_it_synced.mka')

    fp_leadin = str(FP_DIR / 'sky_it_leadin.wav')
    if not Path(fp_leadin).exists():
        sys.exit(f'ERROR: Missing fingerprint: {fp_leadin}')
    fp_leadin_dur = get_duration(fp_leadin)

    # ── Offset ──
    offset = anchor_source - anchor_master
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

    # ══════════════════════════════════════════════════════════════════════════
    # TIER 1: Sting-pair detection (primary)
    # ══════════════════════════════════════════════════════════════════════════
    print('\nScanning Sky IT for ad break stings...')
    events = find_all_transitions(sky_file, [fp_leadin], stream_spec='0:a:0')

    sting_breaks    = []
    orphaned_stings = []

    if not events:
        print('  No sting events found.')
    else:
        print(f'  {len(events)} sting events:')
        for t, c, d in events:
            print(f'    {fmt(t)}  conf={c:.4f}  clip_dur={d:.1f}s')
        sting_breaks, orphaned_stings = pair_breaks(events)

        if sting_breaks:
            print(f'\n  {len(sting_breaks)} sting-detected breaks:')
            for i, (s, e) in enumerate(sting_breaks):
                ms, me = s - offset, e - offset
                print(f'    Break {i+1}: sky {fmt(s)}-{fmt(e)}  '
                      f'dur={e-s:.1f}s  master {fmt(ms)}-{fmt(me)}')
        else:
            print('  No valid sting pairs formed.')

        if orphaned_stings:
            print(f'\n  {len(orphaned_stings)} orphaned sting(s):')
            for t, c, d in orphaned_stings:
                print(f'    {fmt(t)}  conf={c:.4f}  clip_dur={d:.1f}s')

    # ══════════════════════════════════════════════════════════════════════════
    # TIER 2 + 3: Orphaned sting role determination
    #
    # For each orphaned sting, determine its role:
    #   Lead-in:  PUBBLICITÀ appears within 30s of sting end
    #             break_start = sting_start
    #             break_end   = lead-out sting (from sting_end) or MGP watermark
    #   Lead-out: no PUBBLICITÀ within 30s of sting end
    #             break_end   = sting_end
    #             break_start = PUBBLICITÀ found in 7-min look-back window
    # ══════════════════════════════════════════════════════════════════════════
    fallback_breaks = []
    if not sting_only:
        print('\nLoading watermark templates from fingerprints...')
        mgp_path      = FP_DIR / 'sky_it_mgp.png'
        pubb_path     = FP_DIR / 'sky_it_pubb.png'
        mgp_template  = load_watermark_png(mgp_path)
        pubb_template = load_watermark_png(pubb_path)

        if pubb_template is not None and mgp_template is not None:
            pubb_used_times = set()
            for orphan_idx, (t_sting, sting_conf, clip_dur) in enumerate(orphaned_stings):
                sting_end = t_sting + clip_dur

                print(f'\n  Checking role of orphaned sting at {fmt(t_sting)} '
                      f'(end={fmt(sting_end)}, conf={sting_conf:.4f})...')

                # ── Step A: PUBBLICITÀ within 30s of sting end? ───────────
                pubb_hits = find_pubblicita_starts(
                    sky_file, pubb_template,
                    scan_start=sting_end,
                    scan_end=sting_end + 30.0,
                    fps=1, thresh=SKY_PUBB_THRESH,
                    suppress_secs=SKY_PUBB_SUPPRESS_SECS)
                pubb_hits = [(t, c) for t, c in pubb_hits
                             if not any(abs(t - u) < 30 for u in pubb_used_times)]

                if pubb_hits:
                    # ── Lead-in ───────────────────────────────────────────
                    pubb_t, pubb_conf = pubb_hits[0]
                    pubb_used_times.add(pubb_t)
                    print(f'  PUBBLICITÀ at {fmt(pubb_t)} (conf={pubb_conf:.4f}) '
                          f'within 30s of sting end -> LEAD-IN')
                    break_start = t_sting

                    # Find break end: lead-out sting from sting_end (primary)
                    end_t, end_conf = find_sting(
                        sky_file, fp_leadin,
                        sting_end, MAX_BREAK_SECS,
                        stream_spec='0:a:0')
                    if end_conf >= 0.3:
                        break_end = end_t + fp_leadin_dur
                        print(f'  Lead-out sting at {fmt(end_t)} (conf={end_conf:.4f}) '
                              f'-> break_end={fmt(break_end)}')
                        fallback_breaks.append((break_start, break_end))
                    else:
                        # MGP watermark reappearance fallback
                        print(f'  No lead-out sting (best conf={end_conf:.4f}); '
                              f'searching for MGP watermark return...')
                        wm_end, found = find_break_end_via_watermark(
                            sky_file, t_sting, clip_dur, mgp_template,
                            SKY_MGP_WM_X, SKY_MGP_WM_Y, SKY_MGP_WM_W, SKY_MGP_WM_H,
                            SKY_MGP_OUT_W, SKY_MGP_OUT_H,
                            search_secs=MAX_BREAK_SECS, fps=2,
                            thresh=SKY_MGP_WM_THRESH, min_offset_secs=SKY_MGP_WM_MIN_OFFSET,
                            wm_lag_secs=0.0, tmp_suffix=f'_li{t_sting:.0f}',
                            use_gradient=True)
                        if found:
                            print(f'  MGP watermark return at {fmt(wm_end)} -> break_end')
                            fallback_breaks.append((break_start, wm_end))
                        else:
                            print(f'  No watermark found either; skipping this break')

                else:
                    # ── Lead-out: scan 7 min before sting for PUBBLICITÀ ──
                    print(f'  No PUBBLICITÀ within 30s of sting end -> LEAD-OUT  '
                          f'break_end={fmt(sting_end)}')
                    scan_start = max(0.0, t_sting - MAX_BREAK_SECS)
                    print(f'  Scanning for PUBBLICITÀ in {fmt(scan_start)}-{fmt(t_sting)}...')
                    pubb_hits = find_pubblicita_starts(
                        sky_file, pubb_template,
                        scan_start=scan_start, scan_end=t_sting,
                        fps=1, thresh=SKY_PUBB_THRESH,
                        suppress_secs=SKY_PUBB_SUPPRESS_SECS)
                    pubb_hits = [(t, c) for t, c in pubb_hits
                                 if t >= 60.0
                                 and not any(abs(t - u) < 30 for u in pubb_used_times)]

                    if pubb_hits:
                        pubb_t, pubb_conf = pubb_hits[-1]  # latest = closest to sting
                        pubb_used_times.add(pubb_t)
                        print(f'  PUBBLICITÀ at {fmt(pubb_t)} (conf={pubb_conf:.4f}) '
                              f'-> break_start={fmt(pubb_t)}  break_end={fmt(sting_end)}')
                        fallback_breaks.append((pubb_t, sting_end))
                    else:
                        # No PUBBLICITÀ in either direction — cannot confirm role via PUBBLICITÀ.
                        # Treat as lead-in (break_start = sting_start) and use MGP watermark
                        # to find the break end.
                        print(f'  No PUBBLICITÀ in look-back either; treating as LEAD-IN, '
                              f'searching for MGP watermark return...')
                        if orphan_idx < len(orphan_break_starts):
                            break_start = orphan_break_starts[orphan_idx]
                            print(f'  [override] break_start = {fmt(break_start)} '
                                  f'(from --orphan-break-starts)')
                        else:
                            break_start = t_sting
                        wm_end, found = find_break_end_via_watermark(
                            sky_file, t_sting, clip_dur, mgp_template,
                            SKY_MGP_WM_X, SKY_MGP_WM_Y, SKY_MGP_WM_W, SKY_MGP_WM_H,
                            SKY_MGP_OUT_W, SKY_MGP_OUT_H,
                            search_secs=MAX_BREAK_SECS, fps=2,
                            thresh=SKY_MGP_WM_THRESH, min_offset_secs=SKY_MGP_WM_MIN_OFFSET,
                            wm_lag_secs=0.0, tmp_suffix=f'_lo{t_sting:.0f}',
                            use_gradient=True)
                        if found:
                            print(f'  MGP watermark return at {fmt(wm_end)} -> break_end')
                            fallback_breaks.append((break_start, wm_end))
                        else:
                            print(f'  No watermark found either; skipping this break')
        else:
            print('  WARNING: Could not load watermark templates; skipping secondary detection.')
            print('  Run extract_sky_templates.py to generate '
                  'fingerprints/sky_it_mgp.png and fingerprints/sky_it_pubb.png')

    # ── Merge: sting breaks (primary) + PUBBLICITÀ/sting (fallback) ──
    breaks = sorted(sting_breaks + fallback_breaks)
    if breaks:
        print(f'\nFinal break list ({len(breaks)} breaks):')
        for i, (s, e) in enumerate(breaks):
            src = 'sting' if (s, e) in sting_breaks else 'fallback'
            ms, me = s - offset, e - offset
            print(f'  Break {i+1} [{src}]: sky {fmt(s)}-{fmt(e)}  '
                  f'dur={e-s:.1f}s  master {fmt(ms)}-{fmt(me)}')
    else:
        print('\nNo breaks detected.')

    # Show start: beginning of file (no pre-show trimming for Sky IT)
    show_start = 0.0

    print('\nBuilding output segments...')
    build_and_concat(sky_file, master_file, breaks, show_start,
                     offset, d_sky, d_master, ns_stream,
                     output_mka, dry_run=dry_run)

    print(f'\nDone -> {output_mka}')


if __name__ == '__main__':
    main()
