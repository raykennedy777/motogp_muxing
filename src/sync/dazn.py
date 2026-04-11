#!/usr/bin/env python3
"""
sync_dazn.py
Process a DAZN Sunday broadcast into a synced audio track for the MotoGP master.

Output structure (three sections):
  1. Natural Sounds from web master, t=0 until after the last pre-Moto3 ad break
  2. DAZN commentary (ad breaks replaced by Natural Sounds at matching master positions)
  3. Natural Sounds from web master from end program sting until master ends

Sync anchor: 5s pre-race sting found in both DAZN and web master.
             Moto2/MotoGP sting positions are reported but not used for time-warping.

DAZN-specific:
  - No lead-out sting: break ends detected via show intro sting, then watermark reappearance (-4s), with 73s fallback
  - 65s MotoGP intro sting at ~4h23m05s appears out-of-sync after a break (treated as break extension)
  - 7s end-program sting marks handoff from DAZN to Natural Sounds tail

Requirements: ffmpeg/ffprobe on PATH, numpy, scipy
    (via audio_utils / sting_detection / watermark_detection modules)

Usage:
    python sync_dazn.py [--dry-run] <dazn_file> <web_master.mkv> <output_dir>
"""

import sys, os
from pathlib import Path

from src.utils.audio_utils import SR, fmt, get_duration, extract_wav, extract_seg, load_fp_wav, _peak, concat_segments_to_mka
from src.utils.sting_detection import CONF_THRESH, find_sting, find_all_transitions
from watermark_detection import build_watermark_template, find_break_end_via_watermark

# Search window for Moto3 sting in web master (absolute)
MOTO3_STING_SEARCH  = (600,  1200)  # 10–30 min into web master

# Search window for Moto3 sting in DAZN (absolute)
DAZN_MOTO3_SEARCH   = (4500, 1800)  # 75–105 min into DAZN broadcast

# Typical intervals between race stings
MOTO3_TO_MOTO2_SECS  = 4500   # ~1h 15m
MOTO2_TO_MOTOGP_SECS = 6240   # ~1h 44m
STING_SEARCH_MARGIN  = 120    # search ±120s around expected times

# DAZN-specific sting positions
DAZN_MGP_STING_EXPECTED = 15785   # ~4h23m05s; center of 65s MotoGP sting search
DAZN_END_STING_SEARCH   = (18600, 2400)  # end program sting: ~5h10m, 40 min window

# Break-end detection parameters
SHOWINTRO_SEARCH_SECS    = 900  # how far after lead-in to search for show intro sting (15 min)
DAZN_FALLBACK_BREAK_SECS = 73   # assumed break length when no other detection succeeds

# Watermark video detection (fallback between show-intro sting and 73s, for all breaks)
WM_SEARCH_SECS      = 300   # seconds to probe video for watermark after break lead-in
WM_MIN_OFFSET_SECS  = 30    # ignore first N seconds of search (ad content won't end this fast)
WM_X                = 1730  # pixels from left edge of frame
WM_Y                = 50    # pixels from top of frame
WM_W                = 100   # crop width in pixels
WM_H                = 100   # crop height in pixels
WM_OUT_W            = 64    # downscaled width for template matching
WM_OUT_H            = 32    # downscaled height for template matching
WATERMARK_THRESH    = 0.44  # Pearson correlation threshold
WATERMARK_FPS       = 2     # frames per second to sample during break-end scan
# DAZN watermark lag: the DAZN watermark reappears ~58.3s BEFORE actual content resumes
# (the logo is shown over the final seconds of the ad reel).
# break_end = watermark_time - DAZN_WM_LAG_SECS  →  negative lag pushes break_end later.
DAZN_WM_LAG_SECS    = -58.3

# Fingerprints directory (alongside this script)
FP_DIR = Path(__file__).parent / 'fingerprints'


# ── DAZN break detection ──────────────────────────────────────────────────────

def find_break_end_via_showintro(dazn, fp_showintro_list, showintro_dur, break_start, clip_dur):
    """
    Find break end by locating the show intro sting DAZN plays when returning from ads.
    Extracts the search window once and correlates against all provided fingerprints;
    takes the earliest high-confidence hit across all variants.
    Searches up to SHOWINTRO_SEARCH_SECS after the lead-in clip ends.
    Returns (break_end_sec, found_bool).
    Fallback: break_start + DAZN_FALLBACK_BREAK_SECS.
    """
    search_start = break_start + clip_dur
    tmp = '_tmp_showintro_search.wav'
    extract_wav(dazn, tmp, '0:a:0', start=search_start, duration=SHOWINTRO_SEARCH_SECS)
    h = load_fp_wav(tmp)
    os.remove(tmp)

    best_t = None
    for fp in fp_showintro_list:
        needle = load_fp_wav(fp)
        idx, conf = _peak(h, needle)
        if conf >= CONF_THRESH:
            t = search_start + idx / SR
            if best_t is None or t < best_t:
                best_t = t

    if best_t is not None:
        return best_t + showintro_dur, True
    print(f'  Show intro sting not found after break at {break_start:.1f}s; '
          f'using {DAZN_FALLBACK_BREAK_SECS}s fallback.')
    return break_start + DAZN_FALLBACK_BREAK_SECS, False


def detect_breaks_dazn(dazn, fp_list, fp_showintro_list, showintro_dur,
                        wm_template=None, wm_x=0, wm_y=0, wm_w=0, wm_h=0):
    """
    Detect ad breaks in DAZN broadcast.
    Break-end detection hierarchy:
      1. Show intro sting fingerprint
      2. Watermark reappearance in video (all breaks; break end = watermark time - DAZN_WM_LAG_SECS)
      3. DAZN_FALLBACK_BREAK_SECS (73s) last resort
    Returns list of (break_start, break_end) in DAZN time.
    """
    events = find_all_transitions(dazn, fp_list)
    if not events:
        print('  WARNING: No lead-in events found.')
        return []

    print(f'  {len(events)} lead-in events:')
    for t, c, d in events:
        print(f'    {t:.1f}s ({t/60:.1f} min)  conf={c:.4f}  clip={d:.1f}s')

    breaks = []
    for t, conf, clip_dur in events:
        end, found = find_break_end_via_showintro(
            dazn, fp_showintro_list, showintro_dur, t, clip_dur)

        if not found and wm_template is not None:
            wm_end, wm_found = find_break_end_via_watermark(
                dazn, t, clip_dur, wm_template,
                wm_x, wm_y, wm_w, wm_h,
                WM_OUT_W, WM_OUT_H,
                search_secs=WM_SEARCH_SECS, fps=WATERMARK_FPS,
                thresh=WATERMARK_THRESH, min_offset_secs=WM_MIN_OFFSET_SECS,
                wm_lag_secs=DAZN_WM_LAG_SECS)
            if wm_found:
                end   = wm_end
                found = True
                print(f'    Watermark detected: break at {t:.1f}s ends at {end:.1f}s')

        if not found:
            print(f'  WARNING: Break at {t:.1f}s: no detection; '
                  f'fallback end {end:.1f}s')
        breaks.append((t, end))

    return breaks


def apply_mgp_sting_extension(breaks, mgp_sting_dazn, mgp_sting_dur=65.0):
    """
    Post-process breaks: extend the LAST break before the 65s MotoGP intro sting
    so that it covers the full sting duration (if it doesn't already).
    Only the immediately-preceding break is extended; earlier breaks are untouched.
    """
    # Find index of the last break whose start is at or before the sting
    last_idx = -1
    for i, (brk_s, _) in enumerate(breaks):
        if brk_s <= mgp_sting_dazn:
            last_idx = i

    result = list(breaks)
    if last_idx >= 0:
        brk_s, brk_e = result[last_idx]
        if brk_e <= mgp_sting_dazn + mgp_sting_dur:
            new_end = mgp_sting_dazn + mgp_sting_dur
            print(f'  MotoGP sting extension: break {brk_s:.1f}s-{brk_e:.1f}s '
                  f'-> {brk_s:.1f}s-{new_end:.1f}s')
            result[last_idx] = (brk_s, new_end)
        else:
            print(f'  MotoGP sting extension: break at {brk_s:.1f}s already covers sting (no change)')
    return result


# ── Segment building and concatenation ───────────────────────────────────────

def build_and_concat(dazn, web_master, breaks,
                     pre_break_end_dazn, end_sting_dazn,
                     offset, d_web, output_mka, dry_run=False):
    """
    Build all output segments and concatenate to the output MKA.

    offset = m3_dazn - m3_master
    master_time(dazn_t) = dazn_t - offset

    Section 1: NS from master, t=0 -> pre_break_end_dazn - offset
    Section 2: DAZN with inner breaks replaced by NS from master
    Section 3: NS from master, end_sting_dazn - offset -> d_web
    """
    tmp_dir = Path('_tmp_dazn_segs')
    if not dry_run:
        tmp_dir.mkdir(exist_ok=True)
    segs    = []
    counter = [0]

    def mtime(dazn_t):
        return dazn_t - offset

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
    pre_end_master = max(0.0, mtime(pre_break_end_dazn))
    if pre_end_master > 0:
        new_seg(web_master, '0:a:1', 0.0, pre_end_master,
                '[NS]  pre-Moto3 (master 0)')

    # ── Section 2: DAZN bulk with breaks replaced by NS ──
    inner = [(s, e) for s, e in breaks
             if s >= pre_break_end_dazn and s < end_sting_dazn]

    dazn_cur = pre_break_end_dazn

    for brk_s, brk_e in inner:
        if brk_s < dazn_cur:
            print(f'  Skipping overlapping break {brk_s:.1f}s-{brk_e:.1f}s '
                  f'(starts before current pos {dazn_cur:.1f}s)')
            continue
        # DAZN segment before this break
        dazn_dur   = brk_s - dazn_cur
        master_end = mtime(brk_s)
        if master_end > d_web:
            # Master has ended before this break; cap DAZN here
            dazn_dur = min(dazn_dur, d_web - mtime(dazn_cur))
            new_seg(dazn, '0:a:0', dazn_cur, dazn_dur,
                    '[DAZN] cap at master end')
            dazn_cur = dazn_cur + dazn_dur
            break
        new_seg(dazn, '0:a:0', dazn_cur, dazn_dur,
                f'[DAZN] -> master {mtime(dazn_cur):.1f}s')

        # NS replacing the break
        brk_dur  = brk_e - brk_s
        ms_start = max(0.0, min(mtime(brk_s), d_web - brk_dur))
        new_seg(web_master, '0:a:1', ms_start, brk_dur,
                f'[NS]  break  (master {ms_start:.1f}s)')
        dazn_cur = brk_e

    # Final DAZN segment up to end sting (capped at master end)
    dazn_to_end    = end_sting_dazn - dazn_cur
    master_avail   = d_web - mtime(dazn_cur)
    final_dazn_dur = min(dazn_to_end, master_avail)

    if final_dazn_dur > 0:
        new_seg(dazn, '0:a:0', dazn_cur, final_dazn_dur,
                f'[DAZN] final -> master {mtime(dazn_cur):.1f}s')

    # ── Section 3: NS from end sting through master end ──
    end_sting_master = mtime(end_sting_dazn)
    ns_tail = d_web - end_sting_master
    if 0 < ns_tail:
        new_seg(web_master, '0:a:1', end_sting_master, ns_tail,
                f'[NS]  post-race tail (master {end_sting_master:.1f}s)')

    # ── Concatenate all segments to MKA ──
    if dry_run:
        print(f'\n[DRY RUN] Would concatenate {len(segs)} segments -> {output_mka}')
        return

    concat_segments_to_mka(segs, output_mka, list_path_prefix='_tmp_dazn')

    for f in tmp_dir.iterdir():
        f.unlink()
    tmp_dir.rmdir()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    dry_run = '--dry-run' in sys.argv
    if dry_run:
        sys.argv.remove('--dry-run')
        print('[DRY RUN] Detection and segment planning only — no audio will be encoded.')

    mgp_sting_override = None
    m3_dazn_override   = None
    sprint_mode        = '--sprint' in sys.argv
    anchor_source      = None
    anchor_master      = None
    program_start      = None
    if sprint_mode:
        sys.argv.remove('--sprint')
    for arg in list(sys.argv[1:]):
        if arg.startswith('--mgp-sting-dazn='):
            mgp_sting_override = float(arg.split('=', 1)[1])
            sys.argv.remove(arg)
        elif arg.startswith('--m3-dazn='):
            m3_dazn_override = float(arg.split('=', 1)[1])
            sys.argv.remove(arg)
        elif arg.startswith('--anchor-source='):
            anchor_source = float(arg.split('=', 1)[1])
            sys.argv.remove(arg)
        elif arg.startswith('--anchor-master='):
            anchor_master = float(arg.split('=', 1)[1])
            sys.argv.remove(arg)
        elif arg.startswith('--program-start='):
            program_start = float(arg.split('=', 1)[1])
            sys.argv.remove(arg)
    if sprint_mode and (anchor_source is None or anchor_master is None):
        sys.exit('ERROR: --sprint requires --anchor-source=S and --anchor-master=S')

    if len(sys.argv) != 4:
        sys.exit('Usage: sync_dazn.py [--dry-run] [--mgp-sting-dazn=S] [--m3-dazn=S] '
                 '[--sprint --anchor-source=S --anchor-master=S [--program-start=S]] '
                 '<dazn_file> <web_master.mkv> <output_dir>')

    dazn_file, web_master, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    output_mka = Path(out_dir) / (Path(dazn_file).stem + '_synced.mka')

    fp_sting      = str(FP_DIR / 'prerace_sting.wav')
    fp_sting_gp   = str(FP_DIR / 'prerace_sting_motogp.wav')
    fp_leadin     = str(FP_DIR / 'dazn_leadin.wav')
    fp_leadin_alt = str(FP_DIR / 'dazn_leadin_alt.wav')
    fp_end_sting  = str(FP_DIR / 'dazn_end_sting.wav')
    fp_showintro     = str(FP_DIR / 'dazn_showintro.wav')
    fp_showintro_alt = str(FP_DIR / 'dazn_showintro_alt.wav')

    for fp in [fp_sting, fp_sting_gp, fp_leadin, fp_end_sting, fp_showintro]:
        if not Path(fp).exists():
            sys.exit(f'ERROR: Missing fingerprint file: {fp}')

    # Read show intro sting duration from the WAV file itself
    showintro_needle = load_fp_wav(fp_showintro)
    showintro_dur = len(showintro_needle) / SR
    print(f'  Show intro sting duration: {showintro_dur:.2f}s')

    fp_showintro_list = [fp_showintro]
    if Path(fp_showintro_alt).exists():
        fp_showintro_list.append(fp_showintro_alt)
        print(f'  Using alt show intro fingerprint: {fp_showintro_alt}')

    fp_list = [fp_leadin]
    if Path(fp_leadin_alt).exists():
        fp_list.append(fp_leadin_alt)
        print(f'  Using alt transition fingerprint: {fp_leadin_alt}')

    # ── Durations ──
    d_dazn = get_duration(dazn_file)
    d_web  = get_duration(web_master)
    print(f'DAZN: {d_dazn:.1f}s ({d_dazn/3600:.2f}h)  |  '
          f'Web master: {d_web:.1f}s ({d_web/3600:.2f}h)')

    if sprint_mode:
        # ── Sprint mode: use frame-based anchor directly ──
        offset = anchor_source - anchor_master   # dazn_time - offset = master_time
        print(f'\nSprint mode — anchor: DAZN {fmt(anchor_source)} = master {fmt(anchor_master)}')
        print(f'  Offset: {offset:.3f}s  (master_time = dazn_time - {offset:.3f})')

        # Build watermark template from anchor time (race start = confirmed live)
        print('\nBuilding watermark template...')
        wm_template = None
        try:
            wm_ref = anchor_source + 300   # 5 min after race start
            wm_template = build_watermark_template(
                dazn_file, wm_ref, WM_X, WM_Y, WM_W, WM_H, WM_OUT_W, WM_OUT_H)
            if wm_template is not None:
                print(f'  Template at DAZN {fmt(wm_ref)}  crop={WM_W}x{WM_H}@({WM_X},{WM_Y})')
            else:
                print('  Watermark detection disabled (template extraction failed).')
        except Exception as e:
            print(f'  Watermark detection disabled: {e}')

        # Detect breaks (same method as Sunday)
        print('\nScanning DAZN for ad break lead-ins...')
        breaks = detect_breaks_dazn(dazn_file, fp_list, fp_showintro_list, showintro_dur,
                                     wm_template, WM_X, WM_Y, WM_W, WM_H)

        # In sprint mode, search for 65s sting between break start and anchor
        # (DAZN sprint pattern: ads end, then 65s sting, then content)
        if breaks:
            ss_start = breaks[0][0]
            ss_dur   = anchor_source - ss_start + 60
            print(f'\nSearching for 65s sting in DAZN sprint (window {fmt(ss_start)}-{fmt(ss_start+ss_dur)})...')
            sprint_sting_pos, sprint_sting_conf = find_sting(
                dazn_file, fp_sting_gp, ss_start, ss_dur,
                label='  MotoGP 65s sting (sprint)')
            if sprint_sting_conf >= 0.1:
                new_end = sprint_sting_pos + 65.0
                print(f'  Setting show-start break end to sting end: {fmt(new_end)}')
                breaks = [(brk_s, new_end) if brk_s <= sprint_sting_pos else (brk_s, brk_e)
                          for brk_s, brk_e in breaks]
            else:
                print('  65s sting not found in sprint window; break end unchanged.')

        print(f'\n  {len(breaks)} ad breaks:')
        for i, (s, e) in enumerate(breaks):
            ms, me = s - offset, e - offset
            print(f'    Break {i+1}: DAZN {fmt(s)}-{fmt(e)}  '
                  f'dur={e-s:.1f}s  master {fmt(ms)}-{fmt(me)}')

        # Show start: first break ending before anchor (= show intro break)
        pre_breaks = [(s, e) for s, e in breaks if e < anchor_source]
        if pre_breaks:
            pre_break_end_dazn = pre_breaks[0][1]
            print(f'\nShow-start break: DAZN {fmt(pre_breaks[0][0])}-{fmt(pre_break_end_dazn)}')
            print(f'  DAZN commentary starts at DAZN {fmt(pre_break_end_dazn)} '
                  f'= master {fmt(pre_break_end_dazn - offset)}')
        elif program_start is not None:
            pre_break_end_dazn = program_start
            print(f'\nNo pre-anchor break found; using --program-start: '
                  f'DAZN {fmt(program_start)} = master {fmt(program_start - offset)}')
        else:
            pre_break_end_dazn = offset   # start at master t=0
            print(f'\nNo pre-anchor break found; DAZN starts concurrently with master.')

        # Sprint: cap at DAZN file end so NS fills any gap to master end
        end_sting_dazn = min(offset + d_web, d_dazn)

    else:
        # ── Sunday mode: full multi-race detection ──
        print('\nLocating pre-race stings in web master...')
        m3_master, _ = find_sting(web_master, fp_sting,
                                   *MOTO3_STING_SEARCH, stream_spec='0:a:1',
                                   label='  Moto3 sting (master)')
        m2_master, _ = find_sting(web_master, fp_sting,
                                   m3_master + MOTO3_TO_MOTO2_SECS - STING_SEARCH_MARGIN,
                                   STING_SEARCH_MARGIN * 2, stream_spec='0:a:1',
                                   label='  Moto2 sting (master)')
        mgp_master, _ = find_sting(web_master, fp_sting_gp,
                                    m2_master + MOTO2_TO_MOTOGP_SECS - STING_SEARCH_MARGIN,
                                    STING_SEARCH_MARGIN * 2, stream_spec='0:a:1',
                                    label='  MotoGP sting (master)')

        print('\nSearching for Moto3 pre-race sting in DAZN...')
        if m3_dazn_override is not None:
            m3_dazn, m3_conf = m3_dazn_override, 1.0
            print(f'  Moto3 sting (DAZN): {m3_dazn:.3f}s  [manual override]')
        else:
            m3_dazn, m3_conf = find_sting(dazn_file, fp_sting,
                                           *DAZN_MOTO3_SEARCH,
                                           label='  Moto3 sting (DAZN)')
            if m3_conf < 0.1:
                sys.exit('ERROR: Could not find Moto3 pre-race sting in DAZN. '
                         'Check fingerprint or search window.')

        offset = m3_dazn - m3_master
        print(f'  DAZN offset: {offset:.3f}s  '
              f'(master_time = dazn_time - {offset:.3f})')

        print('\nSearching for Moto2/MotoGP stings in DAZN (informational)...')
        m2_dazn, _ = find_sting(dazn_file, fp_sting,
                                  max(0, m3_dazn + MOTO3_TO_MOTO2_SECS - STING_SEARCH_MARGIN),
                                  STING_SEARCH_MARGIN * 2,
                                  label='  Moto2 sting (DAZN)')
        mgp_dazn_info, _ = find_sting(dazn_file, fp_sting,
                                       max(0, m2_dazn + MOTO2_TO_MOTOGP_SECS - STING_SEARCH_MARGIN),
                                       STING_SEARCH_MARGIN * 2,
                                       label='  MotoGP sting (DAZN, informational)')
        print(f'  Moto2 drift:  {m2_dazn - (m3_dazn + MOTO3_TO_MOTO2_SECS):+.1f}s')
        print(f'  MotoGP drift: {mgp_dazn_info - (m2_dazn + MOTO2_TO_MOTOGP_SECS):+.1f}s')

        print('\nSearching for 65s MotoGP intro sting in DAZN...')
        if mgp_sting_override is not None:
            mgp_dazn, mgp_conf = mgp_sting_override, 1.0
            print(f'  MotoGP 65s sting (DAZN): {mgp_dazn:.3f}s  [manual override]')
        else:
            mgp_dazn, mgp_conf = find_sting(
                dazn_file, fp_sting_gp,
                max(0, DAZN_MGP_STING_EXPECTED - STING_SEARCH_MARGIN),
                STING_SEARCH_MARGIN * 2,
                label='  MotoGP 65s sting (DAZN)')
            if mgp_conf < 0.1:
                print('  WARNING: 65s MotoGP sting not found in DAZN. '
                      'Sting extension will be skipped.')

        print('\nSearching for end program sting in DAZN...')
        end_sting_dazn, end_conf = find_sting(
            dazn_file, fp_end_sting,
            *DAZN_END_STING_SEARCH,
            label='  End program sting (DAZN)')
        if end_conf < CONF_THRESH:
            print('  WARNING: End program sting not found; '
                  'DAZN section will run to master end (no NS tail).')
            end_sting_dazn = offset + d_web

        print('\nBuilding watermark template...')
        wm_template = None
        try:
            wm_ref = m3_dazn + 300
            wm_template = build_watermark_template(
                dazn_file, wm_ref, WM_X, WM_Y, WM_W, WM_H, WM_OUT_W, WM_OUT_H)
            if wm_template is not None:
                print(f'  Template at {wm_ref:.0f}s  crop={WM_W}x{WM_H}@({WM_X},{WM_Y})')
            else:
                print('  Watermark detection disabled (template extraction failed).')
        except Exception as e:
            print(f'  Watermark detection disabled: {e}')

        print('\nScanning DAZN for ad break lead-ins (watermark fallback active for all breaks)...')
        breaks = detect_breaks_dazn(dazn_file, fp_list, fp_showintro_list, showintro_dur,
                                     wm_template, WM_X, WM_Y, WM_W, WM_H)

        if mgp_conf >= 0.1:
            print('\nApplying MotoGP 65s sting extension to breaks...')
            breaks = apply_mgp_sting_extension(breaks, mgp_dazn)
        else:
            print('\nSkipping MotoGP sting extension (sting not detected).')

        print(f'\n  {len(breaks)} ad breaks:')
        for i, (s, e) in enumerate(breaks):
            ms, me = s - offset, e - offset
            print(f'    Break {i+1}: DAZN {s:.1f}s-{e:.1f}s  '
                  f'dur={e-s:.1f}s  master {ms:.1f}s-{me:.1f}s')

        pre_breaks = [(s, e) for s, e in breaks if e < m3_dazn]
        if not pre_breaks:
            sys.exit('ERROR: No break found ending before Moto3 sting in DAZN.')
        pre_break = pre_breaks[0]
        pre_break_end_dazn = pre_break[1]

        if pre_break_end_dazn - offset < 0:
            pre_break_end_dazn = offset
            print(f'\nShow start is before master t=0; DAZN will start concurrently '
                  f'with master at DAZN {pre_break_end_dazn:.1f}s')
        else:
            print(f'\nShow-start break : DAZN {pre_break[0]:.1f}s - {pre_break_end_dazn:.1f}s')
            print(f'  DAZN commentary starts at DAZN {pre_break_end_dazn:.1f}s '
                  f'= master {pre_break_end_dazn - offset:.1f}s')

    # ── Build output ──
    print('\nBuilding output segments...')
    build_and_concat(dazn_file, web_master, breaks,
                     pre_break_end_dazn, end_sting_dazn,
                     offset, d_web, output_mka, dry_run=dry_run)

    print(f'\nDone -> {output_mka}')


if __name__ == '__main__':
    main()
