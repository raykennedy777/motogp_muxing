#!/usr/bin/env python3
"""
sync_tnt.py
Process a TNT Sports race broadcast into a synced audio track for the MotoGP master.

Designed for individual per-race files (Moto3, Moto2, or MotoGP).

Output structure (three sections):
  1. Natural Sounds from web master, t=0 until after the last pre-race ad break
  2. TNT commentary (ad breaks replaced by Natural Sounds at matching master positions)
  3. Natural Sounds from web master until master ends

Sync anchor: frame-based via --anchor-source / --anchor-master (primary).
             Falls back to sting detection if no anchor is provided.

Pre-race boundary: the last ad break ending before the pre-race sting.
  Moto3/Moto2: fingerprints/prerace_sting.wav (5s)
  MotoGP:      fingerprints/prerace_sting_motogp.wav (65s) — use --motogp

Requirements: ffmpeg/ffprobe on PATH, numpy, scipy
    (via audio_utils / sting_detection modules)

Usage:
    python sync_tnt.py [--dry-run] [--motogp]
        [--anchor-source=S --anchor-master=S]
        [--ns-track=STREAM] [--sting-tnt=S] [--break=S:E]
        <tnt_file> <web_master.mkv> <output_dir>
"""

import sys
from pathlib import Path

from src.utils.audio_utils import fmt, get_duration, extract_seg, concat_segments_to_mka
from src.utils.sting_detection import find_sting, find_all_transitions

# Fallback sting search windows (used when no anchor is provided)
FALLBACK_MASTER_SEARCH = (600, 1200)   # 10–30 min into web master
FALLBACK_TNT_SEARCH    = (1200, 5400)  # 20 min – 1h 50m into TNT file

# When anchor IS provided, search for the pre-race sting this many seconds before it
STING_PRE_ANCHOR_SECS = 300    # generous window — covers 65s MotoGP sting + gaps

MAX_BREAK_SECS = 420  # 7 minutes - discard pairings longer than this

# Fingerprints directory (alongside this script)
FP_DIR = Path(__file__).parent.parent.parent / 'fingerprints'


def pair_breaks(events):
    """
    Pair consecutive transition events as (break_start, break_end).
    Starts not-in-break; each event toggles the state.
    Pairs longer than MAX_BREAK_SECS are discarded (likely a missed event).
    """
    breaks = []
    for i in range(0, len(events) - 1, 2):
        start = events[i][0]
        end   = events[i + 1][0] + events[i + 1][2]   # time + clip_dur
        dur   = end - start
        if dur > MAX_BREAK_SECS:
            print(f'  WARNING: pair at {start:.1f}s-{end:.1f}s is {dur:.0f}s '
                  f'> {MAX_BREAK_SECS}s - skipping (likely missed event)')
        else:
            breaks.append((start, end))
    return breaks


# ── Segment building and concatenation ───────────────────────────────────────

def build_and_concat(tnt, web_master, breaks,
                     pre_break_end_tnt, post_gp_break_start_tnt,
                     offset, d_web, output_mka, dry_run=False, ns_track='0:a:1'):
    """
    Build all output segments and concatenate directly to the output MKA.

    offset = T3_tnt - moto3_sting_master
    master_time(tnt_t) = tnt_t - offset
    """
    tmp_dir = Path('_tmp_tnt_segs')
    if not dry_run:
        tmp_dir.mkdir(exist_ok=True)
    segs    = []
    counter = [0]

    def mtime(tnt_t):
        return tnt_t - offset

    def new_seg(src, stream, start, duration, desc):
        if duration <= 0:
            return
        counter[0] += 1
        p = str(tmp_dir / f'seg_{counter[0]:04d}.wav')
        print(f'  {desc}  start={start:.1f}s  dur={duration:.1f}s')
        if not dry_run:
            extract_seg(src, p, stream, start=start, duration=duration)
        segs.append(p)

    # ── Section 1: Natural Sounds from master t=0 to pre_break_end ──
    pre_end_master = max(0.0, mtime(pre_break_end_tnt))
    if pre_end_master > 0:
        new_seg(web_master, ns_track, 0.0, pre_end_master,
                '[NS]  pre-Moto3 (master 0)')

    # ── Section 2: TNT bulk with breaks replaced by NS ──
    # If TNT starts before master t=0 (offset > pre_break_end_tnt), skip the preshow
    # so the output length matches the master and the NS tail stays in sync.
    tnt_start = max(pre_break_end_tnt, offset)
    if tnt_start > pre_break_end_tnt:
        print(f'  [Skipping {fmt(tnt_start - pre_break_end_tnt)} of TNT preshow before master t=0]')
    inner = [(s, e) for s, e in breaks
             if s >= tnt_start and s < post_gp_break_start_tnt]

    tnt_cur = tnt_start

    for brk_s, brk_e in inner:
        # TNT segment before this break
        tnt_dur = brk_s - tnt_cur
        master_end = mtime(brk_s)
        if master_end > d_web:
            # Master has ended before this break; cap TNT here
            tnt_dur = min(tnt_dur, d_web - mtime(tnt_cur))
            new_seg(tnt, '0:a:0', tnt_cur, tnt_dur,
                    f'[TNT] cap at master end')
            tnt_cur = tnt_cur + tnt_dur
            break
        new_seg(tnt, '0:a:0', tnt_cur, tnt_dur,
                f'[TNT] -> master {mtime(tnt_cur):.1f}s')

        # NS replacing the break
        brk_dur  = brk_e - brk_s
        ms_start = max(0.0, min(mtime(brk_s), d_web - brk_dur))
        new_seg(web_master, ns_track, ms_start, brk_dur,
                f'[NS]  break  (master {ms_start:.1f}s)')
        tnt_cur = brk_e

    # Final TNT segment up to post-GP break (capped at master end)
    tnt_to_pgp = post_gp_break_start_tnt - tnt_cur
    master_avail = d_web - mtime(tnt_cur)
    final_tnt_dur = min(tnt_to_pgp, master_avail)

    if final_tnt_dur > 0:
        new_seg(tnt, '0:a:0', tnt_cur, final_tnt_dur,
                f'[TNT] final -> master {mtime(tnt_cur):.1f}s')

    # ── Section 3: NS from post-GP break start through master end ──
    post_gp_master = mtime(post_gp_break_start_tnt)
    ns_tail = d_web - post_gp_master
    if 0 < ns_tail:
        new_seg(web_master, ns_track, post_gp_master, ns_tail,
                f'[NS]  post-MotoGP (master {post_gp_master:.1f}s)')

    # ── Concatenate all segments to MKA ──
    if dry_run:
        print(f'\n[DRY RUN] Would concatenate {len(segs)} segments -> {output_mka}')
        return

    concat_segments_to_mka(segs, output_mka, list_path_prefix='_tmp_tnt')

    for f in tmp_dir.iterdir():
        f.unlink()
    tmp_dir.rmdir()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    dry_run = '--dry-run' in sys.argv
    if dry_run:
        sys.argv.remove('--dry-run')
        print('[DRY RUN] Detection and segment planning only — no audio will be encoded.')

    motogp_mode     = '--motogp' in sys.argv
    if motogp_mode:
        sys.argv.remove('--motogp')

    offset_override    = None
    anchor_source   = None
    anchor_master   = None
    sting_tnt_override = None
    ns_track        = '0:a:1'
    manual_breaks   = []
    file_label      = ''
    for arg in list(sys.argv[1:]):
        if arg.startswith('--offset='):
            offset_override = float(arg.split('=', 1)[1])
            sys.argv.remove(arg)
        elif arg.startswith('--anchor-source='):
            anchor_source = float(arg.split('=', 1)[1])
            sys.argv.remove(arg)
        elif arg.startswith('--anchor-master='):
            anchor_master = float(arg.split('=', 1)[1])
            sys.argv.remove(arg)
        elif arg.startswith('--sting-tnt='):
            sting_tnt_override = float(arg.split('=', 1)[1])
            sys.argv.remove(arg)
        elif arg.startswith('--ns-track='):
            ns_track = arg.split('=', 1)[1]
            sys.argv.remove(arg)
        elif arg.startswith('--break='):
            s, e = arg.split('=', 1)[1].split(':')
            manual_breaks.append((float(s), float(e)))
            sys.argv.remove(arg)
        elif arg.startswith('--label='):
            file_label = arg.split('=', 1)[1]
            sys.argv.remove(arg)

    if len(sys.argv) != 4:
        sys.exit('Usage: sync_tnt.py [--dry-run] [--motogp] '
                 '[--offset=S | --anchor-source=S --anchor-master=S] '
                 '[--ns-track=STREAM] [--sting-tnt=S] [--break=S:E] [--label=LABEL] '
                 '<tnt_file> <web_master.mkv> <output_dir>')

    tnt_file, web_master, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    label_part = f'_{file_label}' if file_label else ''
    output_mka = Path(out_dir) / (Path(tnt_file).stem + label_part + '_synced.mka')

    fp_sting      = str(FP_DIR / 'prerace_sting.wav')
    fp_sting_gp   = str(FP_DIR / 'prerace_sting_motogp.wav')
    fp_leadin     = str(FP_DIR / 'tnt_leadin.wav')
    fp_leadin_alt = str(FP_DIR / 'tnt_leadin_alt.wav')
    fp_race_sting = fp_sting_gp if motogp_mode else fp_sting

    for fp in [fp_sting, fp_sting_gp, fp_leadin]:
        if not Path(fp).exists():
            sys.exit(f'ERROR: Missing fingerprint file: {fp}')

    fp_list = [fp_leadin]
    if Path(fp_leadin_alt).exists():
        fp_list.append(fp_leadin_alt)
        print(f'  Using alt transition fingerprint: {fp_leadin_alt}')
    fp_program_intro = str(FP_DIR / 'tnt_program_intro_2025.wav')
    if Path(fp_program_intro).exists():
        fp_list.append(fp_program_intro)
        print(f'  Using 2025 program intro fingerprint: {fp_program_intro}')

    # ── Durations ──
    d_tnt = get_duration(tnt_file)
    d_web = get_duration(web_master)
    print(f'TNT: {fmt(d_tnt)}  |  Web master: {fmt(d_web)}')

    # ── Sync offset ──
    if offset_override is not None:
        offset = offset_override
        print(f'\nManual offset: {offset:.3f}s  (master_time = tnt_time - {offset:.3f})')
    elif anchor_source is not None and anchor_master is not None:
        offset = anchor_source - anchor_master
        print(f'\nAnchor: TNT {fmt(anchor_source)} = master {fmt(anchor_master)}')
        print(f'  Offset: {offset:.3f}s  (master_time = tnt_time - {offset:.3f})')
    else:
        sting_label = 'MotoGP' if motogp_mode else 'Moto3/2'
        print(f'\nNo anchor provided — using {sting_label} sting for sync (fallback)...')
        m_master, _ = find_sting(web_master, fp_race_sting,
                                  *FALLBACK_MASTER_SEARCH, stream_spec='0:a:1',
                                  label='  Sting (master)')
        m_tnt, m_conf = find_sting(tnt_file, fp_race_sting,
                                    *FALLBACK_TNT_SEARCH, label='  Sting (TNT)')
        if m_conf < 0.1:
            sys.exit('ERROR: Could not find pre-race sting in TNT. '
                     'Provide --anchor-source and --anchor-master.')
        offset = m_tnt - m_master
        print(f'  Offset: {offset:.3f}s  (master_time = tnt_time - {offset:.3f})')

    # ── Pre-race sting in TNT (determines where TNT commentary starts) ──
    print('\nLocating pre-race sting in TNT for break boundary...')
    if sting_tnt_override is not None:
        race_sting_tnt = sting_tnt_override
        print(f'  Sting (TNT): {fmt(race_sting_tnt)}  [manual override]')
    else:
        if anchor_source is not None:
            s_start = max(0.0, anchor_source - STING_PRE_ANCHOR_SECS)
            s_dur   = anchor_source - s_start
        else:
            s_start, s_dur = FALLBACK_TNT_SEARCH
        race_sting_tnt, conf = find_sting(tnt_file, fp_race_sting,
                                           s_start, s_dur, label='  Sting (TNT)')
        if conf < 0.05:
            print('  WARNING: Pre-race sting not found — TNT commentary starts at t=0.')
            race_sting_tnt = 0.0

    # ── Ad break detection ──
    print('\nScanning TNT for ad break transitions...')
    events = find_all_transitions(tnt_file, fp_list)

    if events:
        print(f'  {len(events)} transition events:')
        for t, c, d in events:
            print(f'    {fmt(t)}  conf={c:.4f}  clip={d:.1f}s')
    else:
        print('  WARNING: No transition events found.')

    breaks = pair_breaks(events) if len(events) >= 2 else []

    print(f'\n  {len(breaks)} ad breaks:')
    for i, (s, e) in enumerate(breaks):
        ms, me = s - offset, e - offset
        print(f'    Break {i+1}: TNT {fmt(s)}-{fmt(e)}  '
              f'dur={e-s:.1f}s  master {fmt(ms)}-{fmt(me)}')

    if manual_breaks:
        breaks = sorted(breaks + manual_breaks, key=lambda x: x[0])
        print(f'  {len(manual_breaks)} manual break(s) added; total {len(breaks)} break(s).')

    # ── Pre-race break boundary ──
    pre_breaks = [(s, e) for s, e in breaks if s < race_sting_tnt]
    if pre_breaks:
        pre_break = pre_breaks[-1]
        pre_break_end_tnt = pre_break[1]
        print(f'\nPre-race break: TNT {fmt(pre_break[0])}-{fmt(pre_break_end_tnt)}')
        print(f'  TNT commentary starts at {fmt(pre_break_end_tnt)} '
              f'= master {fmt(pre_break_end_tnt - offset)}')
    else:
        pre_break_end_tnt = 0.0
        print(f'\nNo pre-race break found; TNT commentary starts at t=0 '
              f'= master {fmt(-offset)}')

    # ── Post-race boundary ──
    # If the last event is unpaired (orphaned lead-in = file ended mid-break),
    # treat that as the end of commentary and fill the remainder with NS.
    if events and len(events) % 2 == 1:
        orphaned_tnt = events[-1][0]
        post_gp_break_start_tnt = orphaned_tnt
        print(f'\nOrphaned lead-in at {fmt(orphaned_tnt)} '
              f'(unpaired) — NS from master {fmt(orphaned_tnt - offset)} to end')
    else:
        post_gp_break_start_tnt = offset + d_web

    # ── Build output ──
    print('\nBuilding output segments...')
    build_and_concat(tnt_file, web_master, breaks,
                     pre_break_end_tnt, post_gp_break_start_tnt,
                     offset, d_web, output_mka, dry_run=dry_run, ns_track=ns_track)

    print(f'\nDone -> {output_mka}')


if __name__ == '__main__':
    main()
