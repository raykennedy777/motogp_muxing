#!/usr/bin/env python3
"""
sync_sporttv.py
Process a Sport TV Portuguese Sunday broadcast into a synced audio track for the MotoGP master.

Output structure:
  1. Natural Sounds from web master, t=0 until end of opening preshow intro sting
  2. Sport TV commentary (ad breaks replaced by Natural Sounds at matching master positions)
     (no program end sting; section 2 runs to master end)

Break-end detection hierarchy:
  1. preshow_intro_m2m3 sting (break_end = sting_start + sting_duration)
  2. prerace_sting_motogp sting — if found within BREAKEND_SEARCH_SECS of a break
     lead-in, it is treated as a "back from break" opener and absorbed into the break
     (break_end = sting_start + 65s).  The real pre-race sting (which does not follow
     an ad break) falls outside all search windows, so it is left untouched.
  3. Watermark reappearance in video (break_end = watermark_time - 4s)
  4. SPORTTV_FALLBACK_BREAK_SECS as last resort

Sync anchor: prerace_sting.wav (Moto3 start sting) found in both Sport TV and web master.

Requirements: ffmpeg/ffprobe on PATH, numpy, scipy

Usage:
    python sync_sporttv.py [--dry-run] <sporttv_file> <web_master.mkv> <output_dir>
"""

import subprocess, sys, os, numpy as np
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import fftconvolve

# ── Tuning ────────────────────────────────────────────────────────────────────
SR              = 8000   # Hz for correlation
CONF_THRESH     = 0.5    # minimum confidence to accept a hit
SUPPRESS_SECS   = 30     # deduplicate transition hits within this window
MIN_EVENT_SECS  = 120    # ignore transition hits in the first 2 minutes

# Search window for Moto3 sting in web master (absolute)
MOTO3_STING_SEARCH    = (600, 1200)   # 10-30 min into web master

# Search window for Moto3 sting in Sport TV (broad: catchup broadcast, timing unknown)
SPORTTV_MOTO3_SEARCH  = (0, 10800)   # 0-3 hours

# Search window for opening preshow intro sting (should be in first 30 min of broadcast)
PRESHOW_SEARCH_SECS   = 1800

# Typical intervals between race stings
MOTO3_TO_MOTO2_SECS   = 4500   # ~1h 15m
MOTO2_TO_MOTOGP_SECS  = 6240   # ~1h 44m
STING_SEARCH_MARGIN   = 120    # search ±120s around expected times

# Break-end detection
BREAKEND_SEARCH_SECS        = 900   # how far after break lead-in to search for return sting
SPORTTV_FALLBACK_BREAK_SECS = 120   # last-resort break duration if nothing else detected
MIN_RETURN_STING_OFFSET_SECS = 60   # ignore return stings detected less than this many seconds
                                     # after the break lead-in (prevents false positives on
                                     # sounds at the very start of an ad break)

# Sport TV watermark (sport·tv 4 logo, top-right)
WM_SEARCH_SECS      = 300   # seconds to probe video for watermark after break lead-in
WM_MIN_OFFSET_SECS  = 30    # ignore first N seconds (ad content won't end this fast)
WM_X                = 1545  # pixels from left edge
WM_Y                = 65    # pixels from top
WM_W                = 165   # crop width in pixels
WM_H                = 50    # crop height in pixels
WM_OUT_W            = 64    # downscaled width for template matching
WM_OUT_H            = 16    # downscaled height for template matching
WM_THRESH           = 0.44  # Pearson correlation threshold
WM_FPS              = 2     # frames per second to sample during break-end scan

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


def load_fp_wav(path):
    """
    Load fingerprint audio from any format (WAV, MKA, etc.) at SR Hz mono.
    Returns a float32 numpy array.
    """
    p = Path(path)
    if p.suffix.lower() == '.wav':
        _, data = wavfile.read(str(p))
        return data.astype(np.float32)
    tmp = f'_tmp_sporttv_fp_{p.stem}.wav'
    extract_wav(str(p), tmp, '0:a:0')
    _, data = wavfile.read(tmp)
    os.remove(tmp)
    return data.astype(np.float32)


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
    tmp = '_tmp_sporttv_sting.wav'
    extract_wav(src, tmp, stream_spec, start=search_start, duration=search_dur)
    needle      = load_fp_wav(fp_path)
    _, haystack = wavfile.read(tmp)
    os.remove(tmp)
    idx, conf = _peak(haystack, needle)
    t = search_start + idx / SR
    if label:
        print(f'  {label}: {t:.3f}s ({t/3600:.2f}h)  conf={conf:.4f}')
    return t, conf


def find_all_transitions(src, fp_paths, stream_spec='0:a:0'):
    """
    Scan src for all occurrences of one or more transition fingerprints.
    fp_paths: str or list of str/Path.
    Returns a sorted list of (time_sec, confidence, clip_dur_sec).
    Events before MIN_EVENT_SECS are discarded.
    """
    if isinstance(fp_paths, (str, Path)):
        fp_paths = [fp_paths]

    tmp = '_tmp_sporttv_scan.wav'
    print('  Extracting full audio for transition scan...')
    extract_wav(src, tmp, stream_spec)
    _, h = wavfile.read(tmp)
    os.remove(tmp)
    h = h.astype(np.float32)

    all_hits = []

    for fp_path in fp_paths:
        needle   = load_fp_wav(fp_path)
        clip_dur = len(needle) / SR

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
            deduped[-1] = (t, c, d)

    return deduped


# ── Video watermark detection ─────────────────────────────────────────────────

def build_watermark_template(sporttv, ref_time):
    """
    Extract the Sport TV watermark reference template from a single frame at ref_time.
    ref_time must be during confirmed live on-track coverage (watermark present).
    Returns float32 array of WM_OUT_W*WM_OUT_H pixels, or None on failure.
    """
    tmp = '_tmp_sporttv_wm_template.raw'
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
             '-ss', f'{ref_time:.3f}', '-i', str(sporttv), '-frames:v', '1',
             '-vf', f'crop={WM_W}:{WM_H}:{WM_X}:{WM_Y},scale={WM_OUT_W}:{WM_OUT_H}',
             '-f', 'rawvideo', '-pix_fmt', 'gray', tmp],
            check=True)
        arr = np.fromfile(tmp, dtype=np.uint8).astype(np.float32)
        os.remove(tmp)
    except Exception as e:
        print(f'  WARNING: Could not extract watermark template: {e}')
        if os.path.exists(tmp): os.remove(tmp)
        return None
    n = WM_OUT_W * WM_OUT_H
    return arr[:n] if len(arr) >= n else None


def find_break_end_via_watermark(sporttv, break_start, clip_dur, wm_template):
    """
    Detect break end by scanning for the Sport TV watermark returning in live video.
    Returns (break_end_sec, found_bool).
    break_end = first frame where watermark detected - 4s.
    """
    search_start = break_start + clip_dur
    tmp = '_tmp_sporttv_wm_probe.raw'
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
             '-ss', f'{search_start:.3f}', '-t', f'{WM_SEARCH_SECS:.0f}',
             '-i', str(sporttv),
             '-vf', (f'fps={WM_FPS},'
                     f'crop={WM_W}:{WM_H}:{WM_X}:{WM_Y},'
                     f'scale={WM_OUT_W}:{WM_OUT_H}'),
             '-f', 'rawvideo', '-pix_fmt', 'gray', tmp],
            check=True)
        frames = np.fromfile(tmp, dtype=np.uint8).astype(np.float32)
        os.remove(tmp)
    except Exception as e:
        print(f'  WARNING: Watermark probe failed at break {break_start:.1f}s: {e}')
        if os.path.exists(tmp): os.remove(tmp)
        return None, False

    n_pixels = WM_OUT_W * WM_OUT_H
    n_frames  = len(frames) // n_pixels
    step      = 1.0 / WM_FPS
    t0        = wm_template - wm_template.mean()
    t0_norm   = np.linalg.norm(t0)
    min_frame = int(WM_MIN_OFFSET_SECS / step)

    max_conf   = 0.0
    max_conf_t = search_start
    for i in range(n_frames):
        frame = frames[i * n_pixels:(i + 1) * n_pixels]
        f0    = frame - frame.mean()
        conf  = np.dot(f0, t0) / (np.linalg.norm(f0) * t0_norm + 1e-10)
        t_i   = search_start + i * step
        if conf > max_conf:
            max_conf   = conf
            max_conf_t = t_i
        if i >= min_frame and conf > WM_THRESH:
            print(f'    Watermark conf={conf:.3f} at +{i*step:.1f}s '
                  f'(t={t_i:.1f}s); break end = {t_i - 4.0:.1f}s')
            return t_i - 4.0, True

    print(f'    Watermark not found (max={max_conf:.3f} at t={max_conf_t:.1f}s, '
          f'+{max_conf_t - search_start:.0f}s from search start)')
    return None, False


# ── Show-start and break-end detection ────────────────────────────────────────

def find_show_start(sporttv, fp_preshow):
    """
    Find the opening preshow intro sting in the first PRESHOW_SEARCH_SECS of Sport TV.
    Returns (show_start_sec, sting_dur_sec):
      show_start_sec = sting_start + sting_dur  (NS covers master t=0 through this point)
    Returns (None, sting_dur) if not found with sufficient confidence.
    """
    needle    = load_fp_wav(fp_preshow)
    sting_dur = len(needle) / SR

    tmp = '_tmp_sporttv_preshow.wav'
    extract_wav(sporttv, tmp, '0:a:0', start=0, duration=PRESHOW_SEARCH_SECS)
    _, h = wavfile.read(tmp)
    os.remove(tmp)

    idx, conf = _peak(h, needle)
    if conf < CONF_THRESH:
        print(f'  WARNING: Opening preshow intro not found (conf={conf:.4f}). '
              f'NS section will cover to master t=0.')
        return None, sting_dur

    t          = idx / SR
    show_start = t + sting_dur
    print(f'  Opening preshow intro: {t:.3f}s  conf={conf:.4f}  '
          f'dur={sting_dur:.1f}s  -> show start {show_start:.3f}s')
    return show_start, sting_dur


def find_break_end_via_return_stings(sporttv, fp_return_list, fp_content_list,
                                      break_start, clip_dur):
    """
    Search for return stings after a break lead-in.
      fp_return_list:  stings that are transitional; break_end = sting_start + sting_dur
      fp_content_list: stings that are content;      break_end = sting_start
    Returns the earliest hit across all fingerprints, or (None, False) if none found.
    """
    search_start = break_start + clip_dur
    tmp = '_tmp_sporttv_retsting.wav'
    extract_wav(sporttv, tmp, '0:a:0', start=search_start, duration=BREAKEND_SEARCH_SECS)
    _, h = wavfile.read(tmp)
    os.remove(tmp)
    h = h.astype(np.float32)

    candidates = []
    for fp_path, add_dur in ([(p, True)  for p in fp_return_list] +
                              [(p, False) for p in fp_content_list]):
        needle    = load_fp_wav(fp_path)
        sting_dur = len(needle) / SR
        idx, conf = _peak(h, needle)
        if conf >= CONF_THRESH:
            t = search_start + idx / SR
            if t < break_start + MIN_RETURN_STING_OFFSET_SECS:
                print(f'    {Path(fp_path).name}: sting at {t:.1f}s  conf={conf:.4f}  '
                      f'IGNORED (only {t - break_start:.0f}s from break start, '
                      f'min={MIN_RETURN_STING_OFFSET_SECS}s)')
                continue
            end = (t + sting_dur) if add_dur else t
            candidates.append((t, end, Path(fp_path).name, conf))

    if candidates:
        candidates.sort()
        t, end, name, conf = candidates[0]
        print(f'    {name}: sting at {t:.1f}s  conf={conf:.4f}  -> break end {end:.1f}s')
        return end, True

    return None, False


def detect_breaks_sporttv(sporttv, fp_leadin_list, fp_return_list=None,
                           fp_content_list=None, wm_template=None):
    """
    Detect ad breaks in Sport TV broadcast.
    Break-end detection hierarchy:
      1. Return stings (fp_return_list: break_end = sting_start + sting_dur)
      2. Content stings (fp_content_list: break_end = sting_start)
      3. Watermark reappearance in video (break_end = watermark_time - 4s)
      4. SPORTTV_FALLBACK_BREAK_SECS last resort
    Returns list of (break_start, break_end) in Sport TV time.
    """
    fp_return_list  = fp_return_list  or []
    fp_content_list = fp_content_list or []

    events = find_all_transitions(sporttv, fp_leadin_list)
    if not events:
        print('  WARNING: No lead-in events found.')
        return []

    print(f'  {len(events)} lead-in events:')
    for t, c, d in events:
        print(f'    {t:.1f}s ({t/3600:.2f}h)  conf={c:.4f}  clip={d:.1f}s')

    breaks = []
    for t, conf, clip_dur in events:
        end, found = None, False

        if fp_return_list or fp_content_list:
            end, found = find_break_end_via_return_stings(
                sporttv, fp_return_list, fp_content_list, t, clip_dur)

        if not found and wm_template is not None:
            end, found = find_break_end_via_watermark(
                sporttv, t, clip_dur, wm_template)
            if found:
                print(f'    Watermark detected: break at {t:.1f}s ends at {end:.1f}s')

        if not found:
            end = t + SPORTTV_FALLBACK_BREAK_SECS
            print(f'  WARNING: Break at {t:.1f}s: no end detected; fallback end {end:.1f}s')

        breaks.append((t, end))

    return breaks


def apply_mgp_sting_extension(breaks, mgp_sporttv, mgp_conf):
    """
    If the MotoGP 65s sting immediately follows an ad break, extend that break
    to cover sting_start + 65s (treating the sting as part of the ad break).
    """
    MGP_STING_DUR = 65.0
    if mgp_conf < CONF_THRESH:
        print(f'  MotoGP sting not found with sufficient confidence '
              f'(conf={mgp_conf:.4f}); no extension applied.')
        return breaks

    sting_end = mgp_sporttv + MGP_STING_DUR
    # Find the last break whose start is before the MotoGP sting
    idx = None
    for i, (s, e) in enumerate(breaks):
        if s < mgp_sporttv:
            idx = i

    if idx is None:
        print(f'  No break found before MotoGP sting at {mgp_sporttv:.1f}s; no extension.')
        return breaks

    s, e = breaks[idx]
    if e >= sting_end:
        print(f'  Break {idx+1} already covers MotoGP sting end ({sting_end:.1f}s).')
    else:
        print(f'  MotoGP sting extension: break {s:.1f}s-{e:.1f}s -> {s:.1f}s-{sting_end:.1f}s')
        breaks[idx] = (s, sting_end)

    return breaks


# ── Segment building and concatenation ───────────────────────────────────────

def build_and_concat(sporttv, web_master, breaks, show_start_sporttv,
                     offset, d_web, output_mka, dry_run=False):
    """
    Build all output segments and concatenate to the output MKA.

    offset = m3_sporttv - m3_master
    master_time(sporttv_t) = sporttv_t - offset

    Section 1: NS from master t=0 to mtime(show_start_sporttv)
    Section 2: Sport TV with inner breaks replaced by NS (runs to master end; no end sting)
    """
    tmp_dir = Path('_tmp_sporttv_segs')
    if not dry_run:
        tmp_dir.mkdir(exist_ok=True)
    segs    = []
    counter = [0]

    def mtime(sporttv_t):
        return sporttv_t - offset

    def new_seg(src, stream, start, duration, desc):
        if duration <= 0:
            return
        counter[0] += 1
        p = str(tmp_dir / f'seg_{counter[0]:04d}.wav')
        print(f'  {desc}  start={start:.1f}s  dur={duration:.1f}s')
        if not dry_run:
            extract_seg(src, p, stream, start=start, duration=duration)
        segs.append(p)

    # ── Section 1: NS from master t=0 to show_start ──
    pre_end_master = max(0.0, mtime(show_start_sporttv))
    if pre_end_master > 0:
        new_seg(web_master, '0:a:1', 0.0, pre_end_master,
                '[NS]  pre-show (master 0)')

    # ── Section 2: Sport TV with breaks replaced ──
    end_sporttv = offset + d_web   # Sport TV time corresponding to master end
    inner = [(s, e) for s, e in breaks
             if s >= show_start_sporttv and s < end_sporttv]

    sporttv_cur = show_start_sporttv

    for brk_s, brk_e in inner:
        sporttv_dur = brk_s - sporttv_cur
        master_end  = mtime(brk_s)
        if master_end > d_web:
            sporttv_dur = min(sporttv_dur, d_web - mtime(sporttv_cur))
            new_seg(sporttv, '0:a:0', sporttv_cur, sporttv_dur,
                    '[STV] cap at master end')
            sporttv_cur = sporttv_cur + sporttv_dur
            break
        new_seg(sporttv, '0:a:0', sporttv_cur, sporttv_dur,
                f'[STV] -> master {mtime(sporttv_cur):.1f}s')

        brk_dur  = brk_e - brk_s
        ms_start = max(0.0, min(mtime(brk_s), d_web - brk_dur))
        new_seg(web_master, '0:a:1', ms_start, brk_dur,
                f'[NS]  break  (master {ms_start:.1f}s)')
        sporttv_cur = brk_e

    # Final Sport TV segment to master end
    sporttv_to_end = end_sporttv - sporttv_cur
    master_avail   = d_web - mtime(sporttv_cur)
    final_stv_dur  = min(sporttv_to_end, master_avail)

    if final_stv_dur > 0:
        new_seg(sporttv, '0:a:0', sporttv_cur, final_stv_dur,
                f'[STV] final -> master {mtime(sporttv_cur):.1f}s')

    # ── Concatenate all segments to MKA ──
    if dry_run:
        print(f'\n[DRY RUN] Would concatenate {len(segs)} segments -> {output_mka}')
        return

    list_path = '_tmp_sporttv_concat.txt'
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

    moto3_override = None
    for arg in sys.argv[1:]:
        if arg.startswith('--moto3-time='):
            moto3_override = float(arg.split('=', 1)[1])
            sys.argv.remove(arg)
            break

    if len(sys.argv) != 4:
        sys.exit('Usage: sync_sporttv.py [--dry-run] [--moto3-time=<seconds>] '
                 '<sporttv_file> <web_master.mkv> <output_dir>')

    sporttv_file, web_master, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    output_mka = Path(out_dir) / (Path(sporttv_file).stem + '_synced.mka')

    fp_sting    = str(FP_DIR / 'prerace_sting.wav')
    fp_sting_gp = str(FP_DIR / 'prerace_sting_motogp.wav')
    fp_preshow  = str(FP_DIR / 'preshow_intro_m2m3.mka')
    fp_leadin   = str(FP_DIR / 'sporttv_leadin.mka')

    for fp in [fp_sting, fp_sting_gp, fp_preshow, fp_leadin]:
        if not Path(fp).exists():
            sys.exit(f'ERROR: Missing fingerprint file: {fp}')

    fp_leadin_list = [fp_leadin]

    # ── Durations ──
    d_sporttv = get_duration(sporttv_file)
    d_web     = get_duration(web_master)
    print(f'Sport TV: {d_sporttv:.1f}s ({d_sporttv/3600:.2f}h)  |  '
          f'Web master: {d_web:.1f}s ({d_web/3600:.2f}h)')

    # ── Find pre-race stings in web master (Natural Sounds track) ──
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

    # ── Find opening preshow intro sting (done first to gate Moto3 search) ──
    print('\nLocating opening preshow intro sting...')
    show_start_sporttv, _ = find_show_start(sporttv_file, fp_preshow)

    # ── Find Moto3 sting in Sport TV (sync anchor) ──
    print('\nSearching for Moto3 pre-race sting in Sport TV...')
    if moto3_override is not None:
        m3_sporttv, m3_conf = moto3_override, 1.0
        print(f'  Moto3 sting (Sport TV): {m3_sporttv:.3f}s  [manual override]')
    else:
        if show_start_sporttv is not None:
            m3_search_start = show_start_sporttv + 60
            m3_search_dur   = SPORTTV_MOTO3_SEARCH[1]
            print(f'  (Searching from {m3_search_start:.1f}s, after preshow end + 60s)')
        else:
            m3_search_start, m3_search_dur = SPORTTV_MOTO3_SEARCH
        m3_sporttv, m3_conf = find_sting(sporttv_file, fp_sting,
                                          m3_search_start, m3_search_dur,
                                          label='  Moto3 sting (Sport TV)')
        if m3_conf < 0.1:
            sys.exit('ERROR: Could not find Moto3 pre-race sting in Sport TV. '
                     'Check fingerprint or search window.')

    offset = m3_sporttv - m3_master
    print(f'  Sport TV offset: {offset:.3f}s  '
          f'(master_time = sporttv_time - {offset:.3f})')

    # ── Find Moto2 and MotoGP stings in Sport TV (informational) ──
    print('\nSearching for Moto2/MotoGP stings in Sport TV (informational)...')
    m2_sporttv, _ = find_sting(sporttv_file, fp_sting,
                                max(0, m3_sporttv + MOTO3_TO_MOTO2_SECS - STING_SEARCH_MARGIN),
                                STING_SEARCH_MARGIN * 2,
                                label='  Moto2 sting (Sport TV)')
    mgp_sporttv, mgp_conf = find_sting(sporttv_file, fp_sting_gp,
                                        max(0, m2_sporttv + MOTO2_TO_MOTOGP_SECS - STING_SEARCH_MARGIN),
                                        STING_SEARCH_MARGIN * 2,
                                        label='  MotoGP sting (Sport TV)')
    print(f'  Moto2 drift:  {m2_sporttv - (m3_sporttv + MOTO3_TO_MOTO2_SECS):+.1f}s')
    print(f'  MotoGP drift: {mgp_sporttv - (m2_sporttv + MOTO2_TO_MOTOGP_SECS):+.1f}s')

    if show_start_sporttv is None:
        show_start_sporttv = offset   # mtime(offset) = 0; NS section will be zero length
    print(f'  Show start: Sport TV {show_start_sporttv:.3f}s '
          f'= master {show_start_sporttv - offset:.3f}s')

    # ── Build watermark template ──
    print('\nBuilding watermark template...')
    wm_template = None
    try:
        wm_ref = m3_sporttv + 300   # 5 min into Moto3 — confirmed live coverage
        wm_template = build_watermark_template(sporttv_file, wm_ref)
        if wm_template is not None:
            print(f'  Template at {wm_ref:.0f}s  crop={WM_W}x{WM_H}@({WM_X},{WM_Y})')
        else:
            print('  Watermark detection disabled (template extraction failed).')
    except Exception as e:
        print(f'  Watermark detection disabled: {e}')

    # ── Detect ad breaks ──
    # prerace_sting_motogp is included as a return sting: if it appears at the end of
    # an ad break (within BREAKEND_SEARCH_SECS), the full 65s sting is absorbed into
    # the break (break_end = sting_start + sting_dur).  The real pre-race sting that
    # does NOT follow an ad break will not be found within the search window of any
    # lead-in event, so it is left untouched.
    fp_return_list = [fp_preshow, fp_sting_gp]   # break_end = sting_start + sting_dur

    print('\nScanning Sport TV for ad break lead-ins...')
    breaks = detect_breaks_sporttv(sporttv_file, fp_leadin_list,
                                   fp_return_list=fp_return_list,
                                   fp_content_list=[],
                                   wm_template=wm_template)

    print(f'\n  {len(breaks)} ad breaks:')
    for i, (s, e) in enumerate(breaks):
        ms, me = s - offset, e - offset
        h_s,  m_s,  sec_s  = int(s/3600),   int((s%3600)/60),   s%60
        h_e,  m_e,  sec_e  = int(e/3600),   int((e%3600)/60),   e%60
        h_ms, m_ms, sec_ms = int(ms/3600),  int((ms%3600)/60),  ms%60
        h_me, m_me, sec_me = int(me/3600),  int((me%3600)/60),  me%60
        print(f'    Break {i+1}: STV {h_s:02d}:{m_s:02d}:{sec_s:05.2f}-{h_e:02d}:{m_e:02d}:{sec_e:05.2f}  '
              f'dur={e-s:.1f}s  '
              f'master {h_ms:02d}:{m_ms:02d}:{sec_ms:05.2f}-{h_me:02d}:{m_me:02d}:{sec_me:05.2f}')

    # ── Build output ──
    print('\nBuilding output segments...')
    build_and_concat(sporttv_file, web_master, breaks, show_start_sporttv,
                     offset, d_web, output_mka, dry_run=dry_run)

    print(f'\nDone -> {output_mka}')


if __name__ == '__main__':
    main()
