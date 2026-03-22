#!/usr/bin/env python3
"""
sync_tnt.py
Process a TNT Sports Sunday broadcast into a synced audio track for the MotoGP master.

Output structure (three sections):
  1. Natural Sounds from web master, t=0 until after the last pre-Moto3 ad break
  2. TNT commentary (ad breaks replaced by Natural Sounds at matching master positions)
  3. Natural Sounds from web master until master ends
     (if post-MotoGP podium break falls after master end, section 2 ends at master end)

Sync anchor: 5s pre-race sting found in both TNT and web master.
             Moto2/MotoGP sting positions are reported but not used for time-warping.

Requirements: ffmpeg/ffprobe on PATH, numpy, scipy
    (via audio_utils / sting_detection modules)

Usage:
    python sync_tnt.py [--dry-run] <tnt_file> <web_master.mkv> <output_dir>
"""

import sys
from pathlib import Path

from audio_utils import fmt, get_duration, extract_seg, concat_segments_to_mka
from sting_detection import find_sting, find_all_transitions

# Search window for Moto3 sting in web master (absolute)
MOTO3_STING_SEARCH  = (600,  1200)   # (start, duration) - 10-30 min into web master

# Typical intervals between race stings (used for both web master and TNT searches)
MOTO3_TO_MOTO2_SECS  = 4500   # 1h 15m
MOTO2_TO_MOTOGP_SECS = 6240   # 1h 44m
STING_SEARCH_MARGIN  = 60     # search +-60s around the expected time

MAX_BREAK_SECS = 420  # 7 minutes - discard pairings longer than this

# Fingerprints directory (alongside this script)
FP_DIR = Path(__file__).parent / 'fingerprints'


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
                     offset, d_web, output_mka, dry_run=False):
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
        new_seg(web_master, '0:a:1', 0.0, pre_end_master,
                '[NS]  pre-Moto3 (master 0)')

    # ── Section 2: TNT bulk with breaks replaced by NS ──
    inner = [(s, e) for s, e in breaks
             if s >= pre_break_end_tnt and s < post_gp_break_start_tnt]

    tnt_cur = pre_break_end_tnt

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
        new_seg(web_master, '0:a:1', ms_start, brk_dur,
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
        new_seg(web_master, '0:a:1', post_gp_master, ns_tail,
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

    sprint_mode   = '--sprint' in sys.argv
    anchor_source = None
    anchor_master = None
    manual_breaks = []
    if sprint_mode:
        sys.argv.remove('--sprint')
    for arg in list(sys.argv[1:]):
        if arg.startswith('--anchor-source='):
            anchor_source = float(arg.split('=', 1)[1])
            sys.argv.remove(arg)
        elif arg.startswith('--anchor-master='):
            anchor_master = float(arg.split('=', 1)[1])
            sys.argv.remove(arg)
        elif arg.startswith('--break='):
            s, e = arg.split('=', 1)[1].split(':')
            manual_breaks.append((float(s), float(e)))
            sys.argv.remove(arg)
    if sprint_mode and (anchor_source is None or anchor_master is None):
        sys.exit('ERROR: --sprint requires --anchor-source=S and --anchor-master=S')

    if len(sys.argv) != 4:
        sys.exit('Usage: sync_tnt.py [--dry-run] [--sprint --anchor-source=S --anchor-master=S] '
                 '[--break=S:E] <tnt_file> <web_master.mkv> <output_dir>')

    tnt_file, web_master, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    output_mka = Path(out_dir) / (Path(tnt_file).stem + '_synced.mka')

    fp_sting      = str(FP_DIR / 'prerace_sting.wav')
    fp_sting_gp   = str(FP_DIR / 'prerace_sting_motogp.wav')
    fp_leadin     = str(FP_DIR / 'tnt_leadin.wav')
    fp_leadin_alt = str(FP_DIR / 'tnt_leadin_alt.wav')

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
    print(f'TNT: {d_tnt:.1f}s ({d_tnt/3600:.2f}h)  |  '
          f'Web master: {d_web:.1f}s ({d_web/3600:.2f}h)')

    if sprint_mode:
        # ── Sprint mode: use frame-based anchor directly ──
        offset = anchor_source - anchor_master   # tnt_time - offset = master_time
        print(f'\nSprint mode — anchor: TNT {fmt(anchor_source)} = master {fmt(anchor_master)}')
        print(f'  Offset: {offset:.3f}s  (master_time = tnt_time - {offset:.3f})')

    else:
        # ── Sunday mode: find Moto3 sting in web master and TNT ──
        print('\nLocating pre-race stings in web master...')
        m3_master, _ = find_sting(web_master, fp_sting,
                                   *MOTO3_STING_SEARCH, stream_spec='0:a:1',
                                   label='  Moto3 sting (master)')
        m2_master, _ = find_sting(web_master, fp_sting,
                                   m3_master + MOTO3_TO_MOTO2_SECS - STING_SEARCH_MARGIN,
                                   STING_SEARCH_MARGIN * 2, stream_spec='0:a:1',
                                   label='  Moto2 sting (master)')
        mgp_master, _= find_sting(web_master, fp_sting_gp,
                                   m2_master + MOTO2_TO_MOTOGP_SECS - STING_SEARCH_MARGIN,
                                   STING_SEARCH_MARGIN * 2, stream_spec='0:a:1',
                                   label='  MotoGP sting (master)')

        print('\nSearching for Moto3 pre-race sting in TNT...')
        m3_tnt, m3_conf = find_sting(tnt_file, fp_sting,
                                      1200, 5400, label='  Moto3 sting (TNT)')
        if m3_conf < 0.1:
            sys.exit('ERROR: Could not find Moto3 pre-race sting in TNT. '
                     'Check fingerprint or search window.')

        offset = m3_tnt - m3_master
        print(f'  TNT offset: {offset:.3f}s  (master_time = tnt_time - {offset:.3f})')

        print('\nSearching for Moto2/MotoGP stings in TNT (informational)...')
        m2_tnt,  m2_conf  = find_sting(tnt_file, fp_sting,
                                        max(0, m3_tnt + MOTO3_TO_MOTO2_SECS - STING_SEARCH_MARGIN),
                                        STING_SEARCH_MARGIN * 2,
                                        label='  Moto2 sting (TNT)')
        mgp_tnt, mgp_conf = find_sting(tnt_file, fp_sting_gp,
                                        max(0, m2_tnt + MOTO2_TO_MOTOGP_SECS - STING_SEARCH_MARGIN),
                                        STING_SEARCH_MARGIN * 2,
                                        label='  MotoGP sting (TNT)')
        print(f'  Moto2 drift:  {m2_tnt - (m3_tnt + MOTO3_TO_MOTO2_SECS):+.1f}s')
        print(f'  MotoGP drift: {mgp_tnt - (m2_tnt + MOTO2_TO_MOTOGP_SECS):+.1f}s')

    # ── Find all break transitions ──
    print('\nScanning TNT for ad break transitions...')
    events = find_all_transitions(tnt_file, fp_list)

    if len(events) < 2:
        if sprint_mode:
            print('  WARNING: Fewer than 2 transition events found.')
            breaks = []
        else:
            sys.exit('ERROR: Fewer than 2 transition events found. '
                     'Check fingerprints or CONF_THRESH.')
    else:
        breaks = []

    if events:
        print(f'  {len(events)} transition events:')
        for t, c, d in events:
            print(f'    {fmt(t)}  conf={c:.4f}  clip={d:.1f}s')

    if len(events) >= 2:
        breaks = pair_breaks(events)
    print(f'\n  {len(breaks)} ad breaks:')
    for i, (s, e) in enumerate(breaks):
        ms, me = s - offset, e - offset
        print(f'    Break {i+1}: TNT {fmt(s)}-{fmt(e)}  '
              f'dur={e-s:.1f}s  master {fmt(ms)}-{fmt(me)}')

    if manual_breaks:
        breaks = sorted(breaks + manual_breaks, key=lambda x: x[0])
        print(f'  {len(manual_breaks)} manual break(s) added; total {len(breaks)} break(s).')

    if sprint_mode:
        # Show start: last break ending before anchor (= last pre-race break end)
        pre_breaks = [(s, e) for s, e in breaks if e < anchor_source]
        if pre_breaks:
            pre_break_end_tnt = pre_breaks[-1][1]
            print(f'\nSprint show start (last pre-anchor break end): '
                  f'TNT {fmt(pre_break_end_tnt)} = master {fmt(pre_break_end_tnt - offset)}')
        else:
            pre_break_end_tnt = 0.0
            print(f'\nNo pre-anchor break found; TNT commentary starts at t=0 '
                  f'= master {fmt(-offset)}')
        # Show end: full overlap with master (no post-race boundary)
        post_gp_break_start_tnt = offset + d_web
        print(f'Sprint show end: TNT section runs to master end (no NS tail).')

    else:
        # ── Identify pre-Moto3 break ──
        pre_breaks = [(s, e) for s, e in breaks if s < m3_tnt]
        if not pre_breaks:
            sys.exit('ERROR: No break found before Moto3 sting in TNT.')
        pre_break = pre_breaks[-1]
        pre_break_end_tnt = pre_break[1]
        print(f'\nPre-Moto3 break : TNT {pre_break[0]:.1f}s - {pre_break_end_tnt:.1f}s')
        print(f'  TNT commentary starts at TNT {pre_break_end_tnt:.1f}s '
              f'= master {pre_break_end_tnt - offset:.1f}s')

        # ── Identify post-MotoGP podium break ──
        if mgp_conf >= 0.1:
            pgp_threshold = mgp_tnt + 3600
        else:
            pgp_threshold = m3_tnt + (mgp_master - m3_master) + 3600
        master_end_tnt = offset + d_web
        post_breaks = [(s, e) for s, e in breaks
                       if s > pgp_threshold and s < master_end_tnt]
        if not post_breaks:
            post_gp_break_start_tnt = master_end_tnt
            print('  No post-MotoGP break within master duration; '
                  'TNT section runs to master end, no NS tail.')
        else:
            post_break = post_breaks[0]
            post_gp_break_start_tnt = post_break[0]
            pgp_master = post_gp_break_start_tnt - offset
            print(f'Post-MotoGP break: TNT {post_break[0]:.1f}s - {post_break[1]:.1f}s  '
                  f'= master {pgp_master:.1f}s')

    # ── Build output ──
    print('\nBuilding output segments...')
    build_and_concat(tnt_file, web_master, breaks,
                     pre_break_end_tnt, post_gp_break_start_tnt,
                     offset, d_web, output_mka, dry_run=dry_run)

    print(f'\nDone -> {output_mka}')


if __name__ == '__main__':
    main()
