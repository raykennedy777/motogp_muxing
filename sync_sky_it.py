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
     against a reference frame. Used to detect starts not caught by stings.

  3. MGP logo watermark reappearance (secondary — break END fallback)
     The MGP logo (bottom-right) disappears during ads and reappears when
     program resumes. Used only to find the END of a break whose start was
     found via PUBBLICITÀ. Stings are used for break ends when available.

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

from audio_utils import fmt, get_duration, get_audio_stream_count, \
                        extract_seg, concat_segments_to_mka
from sting_detection import find_all_transitions, find_sting
from watermark_detection import build_watermark_template, find_break_end_via_watermark, \
                                find_break_start_via_watermark

FP_DIR         = Path(__file__).parent / 'fingerprints'
MAX_BREAK_SECS = 420    # 7 min — discard sting pairings longer than this
WM_FALLBACK_SECS = 60.0 # break-end fallback when watermark search fails

# ── PUBBLICITÀ text region (bottom-right, 1280×720) — break START indicator ──
SKY_PUBB_X,     SKY_PUBB_Y     = 1040, 615
SKY_PUBB_W,     SKY_PUBB_H     = 220,  50
SKY_PUBB_OUT_W, SKY_PUBB_OUT_H = 48,   12
SKY_PUBB_THRESH       = 0.70   # text is very distinctive; high threshold
SKY_PUBB_SUPPRESS_SECS = 120   # min gap between detections (same break)

# ── MGP logo watermark region (bottom-right, 1280×720) — break END indicator ──
SKY_MGP_WM_X,     SKY_MGP_WM_Y     = 1100, 645
SKY_MGP_WM_W,     SKY_MGP_WM_H     = 120,  50
SKY_MGP_OUT_W,    SKY_MGP_OUT_H    = 32,   16
SKY_MGP_WM_THRESH    = 0.55    # program 0.76–0.86; ad spikes max ~0.51
SKY_MGP_WM_MIN_OFFSET = 20     # don't look for return within first 20s


# ── Break pairing ─────────────────────────────────────────────────────────────

def _pair_from(events, start_idx, verbose=True):
    """Strictly pair events starting at start_idx: (start_idx, start_idx+1), ..."""
    pairs = []
    for i in range(start_idx, len(events) - 1, 2):
        s   = events[i][0]
        e   = events[i + 1][0] + events[i + 1][2]   # time + clip_dur
        dur = e - s
        if dur > MAX_BREAK_SECS:
            if verbose:
                print(f'  (offset={start_idx}) pair {fmt(s)}-{fmt(e)} '
                      f'is {dur:.0f}s > {MAX_BREAK_SECS}s -- skipping')
        else:
            pairs.append((s, e))
    return pairs


def pair_breaks(events):
    """
    Pair sting events as (lead-in, lead-out) breaks.
    Tries pairing from index 0 and index 1 (skipping a possible orphaned
    lead-out at the start of the broadcast), and uses whichever yields
    more valid pairs. Returns (pairs, used_offset_1).
    """
    if not events:
        return [], False
    p0 = _pair_from(events, 0)
    p1 = _pair_from(events, 1, verbose=False)
    if len(p1) > len(p0):
        print(f'  NOTE: offset-1 pairing gives more valid pairs ({len(p1)} vs {len(p0)}); '
              f'first sting treated as orphaned lead-out.')
        return p1, True
    return p0, False


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
    import time
    tmp = f'/tmp/_tmp_pubb_scan_{int(time.time()*1000)}.raw'
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
             '-ss', f'{scan_start:.3f}', '-t', f'{scan_dur:.0f}', '-i', str(src),
             '-vf', (f'fps={fps},'
                     f'crop={SKY_PUBB_W}:{SKY_PUBB_H}:{SKY_PUBB_X}:{SKY_PUBB_Y},'
                     f'scale={SKY_PUBB_OUT_W}:{SKY_PUBB_OUT_H}'),
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

    n_frames = len(frames) // n_pixels
    step = 1.0 / fps
    t0 = template - template.mean()
    t0_norm = np.linalg.norm(t0)

    detections  = []
    last_detect = -suppress_secs

    for i in range(n_frames):
        frame = frames[i * n_pixels:(i + 1) * n_pixels]
        f0    = frame - frame.mean()
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
        sys.exit('Usage: sync_sky_it.py [--dry-run] [--sting-only] '
                 '--anchor-source=S --anchor-master=S '
                 '<sky_it_file> <master.mkv> <output_dir>')

    sky_file, master_file, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    output_mka = Path(out_dir) / (Path(sky_file).stem + '_sky_it_synced.mka')

    fp_leadin = str(FP_DIR / 'sky_it_leadin.wav')
    if not Path(fp_leadin).exists():
        sys.exit(f'ERROR: Missing fingerprint: {fp_leadin}')

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

    sting_breaks = []
    unpaired_stings = []

    if not events:
        print('  No sting events found.')
    else:
        print(f'  {len(events)} sting events:')
        for t, c, d in events:
            print(f'    {fmt(t)}  conf={c:.4f}  clip_dur={d:.1f}s')
        sting_breaks, used_offset_1 = pair_breaks(events)

        # When the first sting is an orphaned lead-out, the preceding break's
        # lead-in may have been missed due to the suppression window
        # (lead-in and lead-out too close together to both survive dedup).
        # Re-scan with find_sting (no suppression) in the window before it.
        if used_offset_1:
            orphaned_t   = events[0][0]
            orphaned_dur = events[0][2]
            t_break_end  = orphaned_t + orphaned_dur
            rescan_start = max(0.0, orphaned_t - MAX_BREAK_SECS)
            rescan_dur   = orphaned_t - rescan_start - 10  # stop 10s before lead-out
            if rescan_dur > 5:
                print(f'\n  Rescanning before orphaned lead-out ({fmt(orphaned_t)}) '
                      f'for missed lead-in...')
                t_li, conf_li = find_sting(sky_file, fp_leadin,
                                           rescan_start, rescan_dur,
                                           stream_spec='0:a:0')
                if conf_li >= 0.3:
                    print(f'  Lead-in found at {fmt(t_li)} (conf={conf_li:.4f}); '
                          f'adding break {fmt(t_li)}-{fmt(t_break_end)}')
                    sting_breaks.insert(0, (t_li, t_break_end))
                else:
                    print(f'  No lead-in found before orphaned lead-out '
                          f'(best conf={conf_li:.4f}).')

        if sting_breaks:
            print(f'\n  {len(sting_breaks)} sting-detected breaks:')
            for i, (s, e) in enumerate(sting_breaks):
                ms, me = s - offset, e - offset
                print(f'    Break {i+1}: sky {fmt(s)}-{fmt(e)}  '
                      f'dur={e-s:.1f}s  master {fmt(ms)}-{fmt(me)}')
        else:
            print('  No valid sting pairs formed.')

        # Identify unpaired stings for watermark-based break start detection
        paired_times = set()
        for s, e in sting_breaks:
            paired_times.add(s)
            paired_times.add(e)
        unpaired_stings = [(t, c, d) for t, c, d in events if t not in paired_times]
        if unpaired_stings:
            print(f'\n  {len(unpaired_stings)} unpaired sting(s):')
            for t, c, d in unpaired_stings:
                print(f'    {fmt(t)}  conf={c:.4f}  clip_dur={d:.1f}s')

    # ══════════════════════════════════════════════════════════════════════════
    # TIER 2 + 3: PUBBLICITÀ (break start) + MGP watermark (break end)
    # Only scan near unpaired stings (300s window before each)
    # ══════════════════════════════════════════════════════════════════════════
    fallback_breaks = []
    if not sting_only:
      print('\nBuilding templates for secondary detection...')
      mgp_template  = build_watermark_template(
          sky_file, ref_time=anchor_source,
          wm_x=SKY_MGP_WM_X,  wm_y=SKY_MGP_WM_Y,
          wm_w=SKY_MGP_WM_W,  wm_h=SKY_MGP_WM_H,
          out_w=SKY_MGP_OUT_W, out_h=SKY_MGP_OUT_H)

      # PUBBLICITÀ template: extracted from a frame where the text is visible.
      # The text appears on the bumper clip at the start of every ad break.
      pubb_ref = 1179.5   # seconds into this broadcast — adjust if clip changes
      pubb_template = build_watermark_template(
          sky_file, ref_time=pubb_ref,
          wm_x=SKY_PUBB_X,     wm_y=SKY_PUBB_Y,
          wm_w=SKY_PUBB_W,     wm_h=SKY_PUBB_H,
          out_w=SKY_PUBB_OUT_W, out_h=SKY_PUBB_OUT_H)

      if pubb_template is not None and mgp_template is not None:
        # Only scan for PUBBLICITÀ near unpaired stings
        pubb_used_times = set()  # Track PUBBLICITÀ times already used
        for t_sting, conf, clip_dur in unpaired_stings:
          # Skip if already covered by a sting break
          if any(s - 60 <= t_sting <= e + 30 for s, e in sting_breaks):
            continue

          # Scan 300s before the unpaired sting for PUBBLICITÀ
          scan_start = max(0.0, t_sting - 300.0)
          scan_end = t_sting
          print(f'\n  Scanning for PUBBLICITÀ before unpaired sting at {fmt(t_sting)}...')
          pubb_hits = find_pubblicita_starts(
              sky_file, pubb_template,
              scan_start=scan_start, scan_end=scan_end,
              fps=1, thresh=SKY_PUBB_THRESH,
              suppress_secs=SKY_PUBB_SUPPRESS_SECS)

          # Filter out PUBBLICITÀ times already used
          pubb_hits = [(t, c) for t, c in pubb_hits
                       if not any(abs(t - used) < 30 for used in pubb_used_times)]
          # Filter out PUBBLICITÀ too close to file start (pre-show bumpers)
          pubb_hits = [(t, c) for t, c in pubb_hits if t >= 60.0]
          # Filter out PUBBLICITÀ too close to file start (pre-show bumpers)
          pubb_hits = [(t, c) for t, c in pubb_hits if t >= 60.0]

          if not pubb_hits:
            print(f'    No PUBBLICITÀ found in {fmt(scan_start)}-{fmt(scan_end)}')
            continue

          # Use the latest PUBBLICITÀ hit (closest to the sting)
          pubb_t, pubb_conf = pubb_hits[-1]
          pubb_used_times.add(pubb_t)
          print(f'    PUBBLICITÀ at {fmt(pubb_t)} (conf={pubb_conf:.4f})')

          # Find break end: search for sting AFTER PUBBLICITÀ (primary)
          print(f'    Searching for sting after PUBBLICITÀ (within {MAX_BREAK_SECS}s)...')
          end_t, end_conf = find_sting(
              sky_file, fp_leadin,
              search_start=pubb_t + 5.0,  # skip 5s after PUBBLICITÀ
              search_dur=MAX_BREAK_SECS,
              stream_spec='0:a:0',
              label=f'    Sting after PUBBLICITÀ')

          if end_conf >= 0.3:
            print(f'    Sting found at {fmt(end_t)} (conf={end_conf:.4f}) — break end')
            fallback_breaks.append((pubb_t, end_t))
          else:
            print(f'    No sting found after PUBBLICITÀ; skipping this break')
      else:
          print('  WARNING: Could not build templates; skipping secondary detection.')

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
