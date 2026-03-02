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
  - No lead-out sting: break ends detected via show intro sting fingerprint, with 73s fallback
  - 65s MotoGP intro sting at ~4h23m05s appears out-of-sync after a break (treated as break extension)
  - 7s end-program sting marks handoff from DAZN to Natural Sounds tail

Requirements: ffmpeg/ffprobe on PATH, numpy, scipy

Usage:
    python sync_dazn.py [--dry-run] <dazn_file> <web_master.mkv> <output_dir>
"""

import subprocess, sys, os, numpy as np
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import fftconvolve

# ── Tuning ────────────────────────────────────────────────────────────────────
SR              = 8000   # Hz for correlation
CONF_THRESH     = 0.5    # minimum confidence to accept a transition hit
SUPPRESS_SECS   = 30     # deduplicate transition hits within this window (seconds)
MIN_EVENT_SECS  = 120    # ignore transition hits in the first 2 minutes (broadcast intro)
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

# MGP watermark video detection (fallback between show-intro sting and 73s)
WM_SEARCH_SECS      = 300   # seconds to probe video for watermark after break lead-in
WM_MIN_OFFSET_SECS  = 30    # ignore first N seconds of search (ad content won't end this fast)
WM_CROP_RIGHT_FRAC  = 0.12  # rightmost fraction of frame width for watermark region
WM_CROP_BOTTOM_FRAC = 0.07  # bottom fraction of frame height for watermark region
WM_OUT_W            = 64    # downscaled width for template matching
WM_OUT_H            = 32    # downscaled height for template matching
WATERMARK_THRESH    = 0.44  # Pearson correlation threshold; only applied post-MotoGP-sting
WATERMARK_FPS       = 2     # frames per second to sample during break-end scan

# Fingerprints directory (alongside this script)
FP_DIR = Path(__file__).parent / 'fingerprints'


# ── ffprobe ───────────────────────────────────────────────────────────────────

def get_duration(f):
    r = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', str(f)],
        capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


# ── Audio extraction ──────────────────────────────────────────────────────────

def extract_wav(src, dst, stream_spec, start=None, duration=None,
                sr=SR, channels=1):
    cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error']
    if start    is not None: cmd += ['-ss', f'{start:.3f}']
    if duration is not None: cmd += ['-t',  f'{duration:.3f}']
    cmd += ['-i', str(src), '-map', stream_spec,
            '-ar', str(sr), '-ac', str(channels), '-f', 'wav', str(dst)]
    subprocess.run(cmd, check=True)

def extract_seg(src, dst, stream_spec, start, duration):
    """Extract a segment at 48 kHz stereo for final output concatenation."""
    extract_wav(src, dst, stream_spec,
                start=start, duration=duration, sr=48000, channels=2)


# ── Correlation ───────────────────────────────────────────────────────────────

def _peak(haystack, needle):
    """Return (sample_index, confidence) for best match of needle in haystack."""
    h = haystack.astype(np.float32)
    n = needle.astype(np.float32)
    if len(n) >= len(h):
        n = n[:max(1, len(h) - 1)]
    corr  = fftconvolve(h, n[::-1], mode='valid')
    idx   = int(np.argmax(np.abs(corr)))
    h_win = h[idx:idx + len(n)]
    conf  = float(np.abs(corr[idx])) / (
            np.linalg.norm(h_win) * np.linalg.norm(n) + 1e-10)
    return idx, conf


def find_sting(src, fp_path, search_start, search_dur, stream_spec='0:a:0',
               label=''):
    """
    Find a sting fingerprint within a time window of src.
    Returns (absolute_time_sec, confidence).
    """
    tmp = '_tmp_sting.wav'
    extract_wav(src, tmp, stream_spec, start=search_start, duration=search_dur)
    _, needle   = wavfile.read(fp_path)
    _, haystack = wavfile.read(tmp)
    os.remove(tmp)
    idx, conf = _peak(haystack, needle)
    t = search_start + idx / SR
    if label:
        print(f'  {label}: {t:.3f}s ({t/60:.1f} min)  conf={conf:.4f}')
    return t, conf


def find_all_transitions(src, fp_paths, stream_spec='0:a:0'):
    """
    Scan src for all occurrences of one or more transition fingerprints.
    fp_paths: str or list of str.
    Returns a sorted list of (time_sec, confidence, clip_dur_sec).
    Events before MIN_EVENT_SECS are discarded.
    """
    if isinstance(fp_paths, str):
        fp_paths = [fp_paths]

    tmp = '_tmp_full_scan.wav'
    print('  Extracting full audio for transition scan...')
    extract_wav(src, tmp, stream_spec)
    _, h = wavfile.read(tmp)
    os.remove(tmp)
    h = h.astype(np.float32)

    all_hits = []   # (time, confidence, clip_dur)

    for fp_path in fp_paths:
        _, needle = wavfile.read(fp_path)
        needle    = needle.astype(np.float32)
        clip_dur  = len(needle) / SR

        corr  = fftconvolve(h, needle[::-1], mode='valid')
        abs_c = np.abs(corr)
        n_len = len(needle)
        supp  = int(SUPPRESS_SECS * SR)
        tmp_c = abs_c.copy()

        while True:
            idx  = int(np.argmax(tmp_c))
            h_w  = h[idx:idx + n_len]
            conf = float(abs_c[idx]) / (
                   np.linalg.norm(h_w) * np.linalg.norm(needle) + 1e-10)
            if conf < CONF_THRESH:
                break
            t = idx / SR
            if t >= MIN_EVENT_SECS:
                all_hits.append((t, conf, clip_dur))
            lo = max(0, idx - supp)
            hi = min(len(tmp_c), idx + supp)
            tmp_c[lo:hi] = 0

    # Merge and cross-fingerprint dedup: sort, suppress within SUPPRESS_SECS
    all_hits.sort()
    deduped = []
    for t, c, d in all_hits:
        if not deduped or t - deduped[-1][0] > SUPPRESS_SECS:
            deduped.append((t, c, d))
        elif c > deduped[-1][1]:
            deduped[-1] = (t, c, d)   # keep higher-confidence hit in same window

    return deduped


# ── Video watermark detection ─────────────────────────────────────────────────

def get_video_dimensions(src):
    """Return (width, height) of the first video stream."""
    r = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=width,height', '-of', 'csv=p=0', str(src)],
        capture_output=True, text=True, check=True)
    w, h = r.stdout.strip().split(',')
    return int(w), int(h)


def build_watermark_template(dazn, ref_time, wm_x, wm_y, wm_w, wm_h):
    """
    Extract the MGP watermark reference template from a single frame at ref_time.
    ref_time must be during confirmed live on-track coverage (watermark present).
    Returns float32 array of WM_OUT_W*WM_OUT_H pixels, or None on failure.
    """
    tmp = '_tmp_wm_template.raw'
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
             '-ss', f'{ref_time:.3f}', '-i', str(dazn), '-frames:v', '1',
             '-vf', f'crop={wm_w}:{wm_h}:{wm_x}:{wm_y},scale={WM_OUT_W}:{WM_OUT_H}',
             '-f', 'rawvideo', '-pix_fmt', 'gray', tmp],
            check=True)
        arr = np.fromfile(tmp, dtype=np.uint8).astype(np.float32)
        os.remove(tmp)
    except Exception as e:
        print(f'  WARNING: Could not extract watermark template: {e}')
        if os.path.exists(tmp):
            os.remove(tmp)
        return None
    n = WM_OUT_W * WM_OUT_H
    return arr[:n] if len(arr) >= n else None


def find_break_end_via_watermark(dazn, break_start, clip_dur,
                                  wm_template, wm_x, wm_y, wm_w, wm_h):
    """
    Detect break end by scanning for the MGP watermark returning in live video.
    Extracts WM_SEARCH_SECS of the bottom-right frame region at WATERMARK_FPS fps
    and correlates each frame against the reference template.
    Returns (break_end_sec, found_bool).
    Only fires when DAZN cuts back to live on-track coverage (watermark present);
    studio segments and unrelated content will not trigger it.
    """
    search_start = break_start + clip_dur
    tmp = '_tmp_wm_probe.raw'
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
             '-ss', f'{search_start:.3f}', '-t', f'{WM_SEARCH_SECS:.0f}',
             '-i', str(dazn),
             '-vf', (f'fps={WATERMARK_FPS},'
                     f'crop={wm_w}:{wm_h}:{wm_x}:{wm_y},'
                     f'scale={WM_OUT_W}:{WM_OUT_H}'),
             '-f', 'rawvideo', '-pix_fmt', 'gray', tmp],
            check=True)
        frames = np.fromfile(tmp, dtype=np.uint8).astype(np.float32)
        os.remove(tmp)
    except Exception as e:
        print(f'  WARNING: Watermark probe failed at break {break_start:.1f}s: {e}')
        if os.path.exists(tmp):
            os.remove(tmp)
        return None, False

    n_pixels = WM_OUT_W * WM_OUT_H
    n_frames  = len(frames) // n_pixels
    step      = 1.0 / WATERMARK_FPS

    t0 = wm_template - wm_template.mean()  # mean-subtracted template (precomputed)
    t0_norm = np.linalg.norm(t0)
    min_frame = int(WM_MIN_OFFSET_SECS / step)  # skip early frames (ad content / transient spikes)

    max_conf = 0.0
    max_conf_t = search_start
    for i in range(n_frames):
        frame = frames[i * n_pixels:(i + 1) * n_pixels]
        f0   = frame - frame.mean()
        conf = np.dot(f0, t0) / (np.linalg.norm(f0) * t0_norm + 1e-10)
        t_i  = search_start + i * step
        if conf > max_conf:
            max_conf = conf
            max_conf_t = t_i
        if i >= min_frame and conf > WATERMARK_THRESH:
            print(f'    Watermark conf={conf:.3f} at +{i*step:.1f}s (t={t_i:.1f}s)')
            return t_i, True

    print(f'    Watermark not found (max={max_conf:.3f} at t={max_conf_t:.1f}s, '
          f'+{max_conf_t - search_start:.0f}s from search start)')
    return None, False


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
    _, h = wavfile.read(tmp)
    os.remove(tmp)
    h = h.astype(np.float32)

    best_t = None
    for fp in fp_showintro_list:
        _, needle = wavfile.read(fp)
        idx, conf = _peak(h, needle.astype(np.float32))
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
                        wm_template=None, wm_x=0, wm_y=0, wm_w=0, wm_h=0,
                        wm_after_t=0.0):
    """
    Detect ad breaks in DAZN broadcast.
    Break-end detection hierarchy:
      1. Show intro sting fingerprint
      2. MGP watermark in video (only for breaks at/after wm_after_t, i.e. during live MotoGP race)
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

        if not found and wm_template is not None and t >= wm_after_t:
            wm_end, wm_found = find_break_end_via_watermark(
                dazn, t, clip_dur, wm_template, wm_x, wm_y, wm_w, wm_h)
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

    list_path = '_tmp_dazn_concat.txt'
    with open(list_path, 'w') as f:
        for s in segs:
            f.write(f"file '{Path(s).resolve()}'\n")
    print(f'\nConcatenating {len(segs)} segments -> {output_mka}')
    subprocess.run(
        ['ffmpeg', '-y', '-hide_banner',
         '-f', 'concat', '-safe', '0', '-i', list_path,
         '-map', '0', '-c:a', 'aac', '-b:a', '192k', str(output_mka)],
        check=True)
    os.remove(list_path)

    for f in tmp_dir.iterdir():
        f.unlink()
    tmp_dir.rmdir()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    dry_run = '--dry-run' in sys.argv
    if dry_run:
        sys.argv.remove('--dry-run')
        print('[DRY RUN] Detection and segment planning only — no audio will be encoded.')

    if len(sys.argv) != 4:
        sys.exit('Usage: sync_dazn.py [--dry-run] <dazn_file> <web_master.mkv> <output_dir>')

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
    _, _si_wav = wavfile.read(fp_showintro)
    showintro_dur = len(_si_wav) / SR
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

    # ── Find pre-race sting positions in web master (Natural Sounds track) ──
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

    # ── Find Moto3 sting in DAZN (sync anchor) ──
    print('\nSearching for Moto3 pre-race sting in DAZN...')
    m3_dazn, m3_conf = find_sting(dazn_file, fp_sting,
                                   *DAZN_MOTO3_SEARCH,
                                   label='  Moto3 sting (DAZN)')
    if m3_conf < 0.1:
        sys.exit('ERROR: Could not find Moto3 pre-race sting in DAZN. '
                 'Check fingerprint or search window.')

    offset = m3_dazn - m3_master  # dazn_time - offset = master_time
    print(f'  DAZN offset: {offset:.3f}s  '
          f'(master_time = dazn_time - {offset:.3f})')

    # ── Find Moto2 and MotoGP stings in DAZN (informational — drift report only) ──
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

    # ── Find 65s MotoGP intro sting in DAZN ──
    print('\nSearching for 65s MotoGP intro sting in DAZN...')
    mgp_dazn, mgp_conf = find_sting(
        dazn_file, fp_sting_gp,
        max(0, DAZN_MGP_STING_EXPECTED - STING_SEARCH_MARGIN),
        STING_SEARCH_MARGIN * 2,
        label='  MotoGP 65s sting (DAZN)')
    if mgp_conf < 0.1:
        print('  WARNING: 65s MotoGP sting not found in DAZN. '
              'Sting extension will be skipped.')

    # ── Find end program sting in DAZN ──
    print('\nSearching for end program sting in DAZN...')
    end_sting_dazn, end_conf = find_sting(
        dazn_file, fp_end_sting,
        *DAZN_END_STING_SEARCH,
        label='  End program sting (DAZN)')
    if end_conf < 0.1:
        print('  WARNING: End program sting not found; '
              'DAZN section will run to master end (no NS tail).')
        end_sting_dazn = offset + d_web

    # ── Build MGP watermark template for video-based break-end detection ──
    print('\nBuilding MGP watermark template...')
    wm_template = wm_x = wm_y = wm_w = wm_h = None
    try:
        vid_w, vid_h = get_video_dimensions(dazn_file)
        wm_w  = int(vid_w * WM_CROP_RIGHT_FRAC)
        wm_h  = int(vid_h * WM_CROP_BOTTOM_FRAC)
        wm_x  = vid_w - wm_w
        wm_y  = vid_h - wm_h
        wm_ref = m3_dazn + 300   # 5 min into Moto3 — confirmed live coverage
        wm_template = build_watermark_template(
            dazn_file, wm_ref, wm_x, wm_y, wm_w, wm_h)
        if wm_template is not None:
            print(f'  Template at {wm_ref:.0f}s  '
                  f'video={vid_w}x{vid_h}  '
                  f'crop={wm_w}x{wm_h}@({wm_x},{wm_y})')
        else:
            print('  Watermark detection disabled (template extraction failed).')
            wm_x = wm_y = wm_w = wm_h = 0
    except Exception as e:
        print(f'  Watermark detection disabled: {e}')
        wm_template = None
        wm_x = wm_y = wm_w = wm_h = 0

    # ── Detect ad breaks ──
    # Watermark detection only applies during live MotoGP race (after the 65s intro sting).
    # Inter-race studio content doesn't show the MotoGP watermark; using watermark there
    # would give false positives from ad content.
    wm_after_t = (mgp_dazn + 65.0) if mgp_conf >= 0.1 else float('inf')
    print(f'\nScanning DAZN for ad break lead-ins '
          f'(watermark active for breaks after {wm_after_t:.0f}s)...')
    breaks = detect_breaks_dazn(dazn_file, fp_list, fp_showintro_list, showintro_dur,
                                 wm_template, wm_x, wm_y, wm_w, wm_h,
                                 wm_after_t=wm_after_t)

    # Apply MotoGP sting extension
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

    # ── Identify pre-Moto3 break boundary ──
    pre_breaks = [(s, e) for s, e in breaks if e < m3_dazn]
    if not pre_breaks:
        sys.exit('ERROR: No break found ending before Moto3 sting in DAZN.')
    pre_break = pre_breaks[-1]
    pre_break_end_dazn = pre_break[1]
    print(f'\nPre-Moto3 break : DAZN {pre_break[0]:.1f}s - {pre_break_end_dazn:.1f}s')
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
