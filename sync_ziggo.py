#!/usr/bin/env python3
"""
sync_ziggo.py
Add Ziggo Sport Dutch audio to a MotoGP master file.

Sync anchor: frame-based (first camera change at Moto3 race start).
Ad breaks detected via:
  1. Lead-in/lead-out sting pairs (fingerprints/ziggo_leadin.wav) — primary
  2. Watermark absence scan as fallback (Ziggo Sport logo, top-right)
     Requires --wm-ref-time (TS absolute seconds during live content).

Output structure:
  [NS head] [Ziggo] [NS break] [Ziggo] ... [NS tail]

Requirements: ffmpeg/ffprobe on PATH, numpy, scipy

Usage:
    python sync_ziggo.py [--dry-run]
        --anchor-source=S --anchor-master=S
        [--wm-ref-time=S]
        <ziggo_file> <master.mkv> <output_dir>
"""

import sys, subprocess
from pathlib import Path

from audio_utils import fmt, get_duration, get_audio_stream_count, \
                        extract_seg, concat_segments_to_mka
from sting_detection import find_all_transitions
from watermark_detection import (build_watermark_template,
                                  find_all_breaks_via_watermark)

# ── Watermark constants (Ziggo Sport logo, top-right, 1920×1080) ───────────────
WM_X, WM_Y, WM_W, WM_H = 1620, 50, 260, 35
WM_OUT_W, WM_OUT_H      = 64, 16
WM_LAG_SECS             = 7.0    # watermark returns ~7s after lead-out clip ends
WM_THRESH               = 0.44

FP_DIR = Path(__file__).parent / 'fingerprints'


def get_pts_offset(src):
    """Return video stream start_time (PTS offset) in seconds, or 0.0."""
    r = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=start_time',
         '-of', 'default=noprint_wrappers=1:nokey=1', str(src)],
        capture_output=True, text=True, check=True)
    lines = [l for l in r.stdout.strip().splitlines()
             if l.strip() and l.strip() != 'N/A']
    try:
        return float(lines[0]) if lines else 0.0
    except ValueError:
        return 0.0


STING_CONF_THRESH  = 0.3    # lower than default to catch weaker lead-out hits
STING_MIN_PAIR_GAP = 60     # minimum seconds between lead-in and lead-out
STING_MAX_PAIR_GAP = 400    # maximum seconds between lead-in and lead-out


def detect_breaks_sting(ziggo_file, fp_path, pts_offset):
    """
    Find ad breaks via lead-in/lead-out sting pairs.

    find_all_transitions returns WAV-relative times (t=0 = first audio sample
    at TS PTS pts_offset); add pts_offset to get TS absolute times.

    Uses temporal pairing: each hit is paired with the next hit within
    [STING_MIN_PAIR_GAP, STING_MAX_PAIR_GAP] seconds. More robust than
    even/odd pairing when some lead-outs have lower confidence.

    Returns list of (break_start, break_end) in TS absolute seconds.
    Both start and end include the sting clips themselves.
    """
    print('\nSting-pair detection...')
    hits = find_all_transitions(ziggo_file, fp_path, stream_spec='0:a:0',
                                conf_thresh=STING_CONF_THRESH)
    if not hits:
        print('  No sting hits found.')
        return []

    print(f'  {len(hits)} hit(s):')
    for t, conf, dur in hits:
        print(f'    t={fmt(t + pts_offset)}  conf={conf:.4f}  sting_dur={dur:.2f}s')

    # Temporal pairing: pair each hit with the next within the gap window
    pairs = []
    used  = set()
    for i, (t_in, conf_in, dur_in) in enumerate(hits):
        if i in used:
            continue
        for j, (t_out, conf_out, dur_out) in enumerate(hits[i + 1:], i + 1):
            if j in used:
                continue
            gap = t_out - t_in
            if gap < STING_MIN_PAIR_GAP:
                continue
            if gap > STING_MAX_PAIR_GAP:
                break
            used.add(i)
            used.add(j)
            b_start = t_in  + pts_offset
            b_end   = t_out + pts_offset + dur_out
            pairs.append((b_start, b_end))
            print(f'  Break {len(pairs)}: {fmt(b_start)} - {fmt(b_end)}  '
                  f'dur={b_end - b_start:.1f}s  '
                  f'(lead-in conf={conf_in:.4f}, lead-out conf={conf_out:.4f})')
            break

    unpaired = [i for i in range(len(hits)) if i not in used]
    if unpaired:
        print(f'  {len(unpaired)} unpaired hit(s) (no partner within '
              f'{STING_MIN_PAIR_GAP}-{STING_MAX_PAIR_GAP}s):')
        for i in unpaired:
            t, conf, _ = hits[i]
            print(f'    t={fmt(t + pts_offset)}  conf={conf:.4f}  (ignored)')

    return pairs


def detect_breaks_watermark(ziggo_file, wm_ref_time, scan_start, scan_end):
    """
    Find ad breaks via watermark absence scan (fallback).
    All times are TS absolute seconds.
    Returns list of (break_start, break_end) in TS absolute seconds.
    """
    print(f'\nWatermark fallback scan ({fmt(scan_start)} – {fmt(scan_end)})...')
    template = build_watermark_template(
        ziggo_file, wm_ref_time,
        WM_X, WM_Y, WM_W, WM_H, WM_OUT_W, WM_OUT_H)
    if template is None:
        print('  WARNING: Could not build watermark template — skipping.')
        return []

    return find_all_breaks_via_watermark(
        ziggo_file, template,
        WM_X, WM_Y, WM_W, WM_H, WM_OUT_W, WM_OUT_H,
        scan_start, scan_end,
        fps=2, thresh=WM_THRESH,
        min_break_secs=45,
        wm_lag_secs=WM_LAG_SECS)


def merge_breaks(sting_breaks, wm_breaks):
    """Add any watermark-detected breaks not already covered by a sting pair."""
    if not wm_breaks:
        return list(sting_breaks)
    result = list(sting_breaks)
    for ws, we in wm_breaks:
        covered = any(max(ws, ss) < min(we, se) for ss, se in sting_breaks)
        if not covered:
            print(f'  Watermark fallback adding break {fmt(ws)} – {fmt(we)}  '
                  f'dur={we - ws:.1f}s')
            result.append((ws, we))
    result.sort()
    return result


def build_and_concat(ziggo_file, master_file, offset,
                     breaks, d_ziggo, d_master, ns_stream,
                     output_mka, dry_run=False):
    """
    Build interleaved Ziggo/NS segments and concatenate to MKA.

    offset = anchor_source - anchor_master
    master_time(ziggo_ts) = ziggo_ts - offset

    All break times are TS absolute seconds.
    """
    ziggo_start_m = -offset   # master time when Ziggo TS t=0 occurs

    print(f'\n  Offset: {offset:.3f}s  (Ziggo t=0 = master {fmt(ziggo_start_m)})')

    # TS absolute time bounds for the Ziggo content we will use
    ziggo_in  = max(0.0, -ziggo_start_m)     # trim if Ziggo starts before master
    ziggo_end = min(d_ziggo, d_master + offset)  # trim if Ziggo runs past master end

    print(f'  Ziggo covers master {fmt(max(0.0, ziggo_start_m))} – '
          f'{fmt(min(ziggo_end - offset, d_master))}')

    if ziggo_start_m < 0:
        print(f'  NOTE: Ziggo starts {fmt(-ziggo_start_m)} before master t=0 — '
              f'trimming Ziggo by that amount.')
    if d_ziggo > ziggo_end:
        print(f'  NOTE: Ziggo extends {d_ziggo - ziggo_end:.1f}s past master end — '
              f'trimming Ziggo end.')

    # Clip break list to the usable Ziggo window
    valid_breaks = []
    for bs, be in breaks:
        bs = max(bs, ziggo_in)
        be = min(be, ziggo_end)
        if be > bs:
            valid_breaks.append((bs, be))

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

    # ── NS head (before Ziggo content reaches master t=0) ──
    if ziggo_start_m > 0:
        new_seg(master_file, ns_stream, 0.0, ziggo_start_m,
                '[NS]    pre-Ziggo   (master 0)')

    # ── Interleaved Ziggo / NS-break segments ──
    cursor = ziggo_in   # current position in TS absolute seconds

    for bs, be in valid_breaks:
        new_seg(ziggo_file, '0:a:0', cursor, bs - cursor, '[ZIGGO]')
        ns_start = max(0.0, bs - offset)
        ns_dur   = min(be - bs, d_master - ns_start)
        new_seg(master_file, ns_stream, ns_start, ns_dur,
                f'[NS]    break       (master {fmt(ns_start)})')
        cursor = be

    # ── Final Ziggo segment ──
    new_seg(ziggo_file, '0:a:0', cursor, ziggo_end - cursor, '[ZIGGO] final')

    # ── NS tail (after Ziggo ends before master end) ──
    ns_tail_start = ziggo_end - offset
    ns_tail_dur   = d_master - ns_tail_start
    if ns_tail_dur > 0:
        new_seg(master_file, ns_stream, ns_tail_start, ns_tail_dur,
                f'[NS]    post-Ziggo  (master {fmt(ns_tail_start)})')

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
        print('[DRY RUN] Detection and segment planning only — no audio will be encoded.')

    anchor_source = None
    anchor_master = None
    wm_ref_time   = None
    for arg in list(sys.argv[1:]):
        if arg.startswith('--anchor-source='):
            anchor_source = float(arg.split('=', 1)[1])
            sys.argv.remove(arg)
        elif arg.startswith('--anchor-master='):
            anchor_master = float(arg.split('=', 1)[1])
            sys.argv.remove(arg)
        elif arg.startswith('--wm-ref-time='):
            wm_ref_time = float(arg.split('=', 1)[1])
            sys.argv.remove(arg)

    if anchor_source is None or anchor_master is None or len(sys.argv) != 4:
        sys.exit('Usage: sync_ziggo.py [--dry-run] '
                 '--anchor-source=S --anchor-master=S '
                 '[--wm-ref-time=S] '
                 '<ziggo_file> <master.mkv> <output_dir>')

    ziggo_file, master_file, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    output_mka = Path(out_dir) / (Path(ziggo_file).stem + '_ziggo_synced.mka')

    fp_path = FP_DIR / 'ziggo_leadin.wav'
    if not fp_path.exists():
        sys.exit(f'ERROR: Missing fingerprint: {fp_path}')

    # ── File info ──
    pts_offset = get_pts_offset(ziggo_file)
    d_ziggo    = get_duration(ziggo_file)
    d_master   = get_duration(master_file)
    print(f'Ziggo:      {d_ziggo:.1f}s  ({fmt(d_ziggo)})')
    print(f'Master:     {d_master:.1f}s  ({fmt(d_master)})')
    print(f'PTS offset: {pts_offset:.4f}s')

    n_audio   = get_audio_stream_count(master_file)
    ns_stream = f'0:a:{n_audio - 1}'
    print(f'Master audio tracks: {n_audio}  ->  NS on {ns_stream}')

    # ── Offset ──
    offset = anchor_source - anchor_master
    print(f'\nAnchor: Ziggo {fmt(anchor_source)} = master {fmt(anchor_master)}')
    print(f'Offset: {offset:.3f}s  (master_time = ziggo_ts - {offset:.3f})')

    # ── Break detection ──
    sting_breaks = detect_breaks_sting(ziggo_file, str(fp_path), pts_offset)

    if wm_ref_time is not None:
        wm_breaks = detect_breaks_watermark(
            ziggo_file, wm_ref_time,
            scan_start=pts_offset, scan_end=d_ziggo)
    else:
        print('\nNo --wm-ref-time provided — skipping watermark fallback.')
        wm_breaks = []

    breaks = merge_breaks(sting_breaks, wm_breaks)

    print(f'\nFinal break list ({len(breaks)} break(s)):')
    for i, (bs, be) in enumerate(breaks, 1):
        print(f'  {i}. Ziggo {fmt(bs)} – {fmt(be)}  ({be - bs:.1f}s)'
              f'  -> master {fmt(bs - offset)} – {fmt(be - offset)}')

    # ── Build output ──
    print('\nBuilding output segments...')
    build_and_concat(ziggo_file, master_file, offset,
                     breaks, d_ziggo, d_master, ns_stream,
                     output_mka, dry_run=dry_run)

    print(f'\nDone -> {output_mka}')


if __name__ == '__main__':
    main()
