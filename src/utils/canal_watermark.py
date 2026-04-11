#!/usr/bin/env python3
"""
canal_watermark.py
Canal+ watermark-based ad break detection for MotoGP muxing.

Detects ad breaks by scanning for Canal+ logo absence in video frames.
The Canal+ logo (top-right) disappears when ads begin and reappears
when programming resumes.

Designed to handle transport-stream corruption common in OTA recordings.
"""

import subprocess, os
import numpy as np
from pathlib import Path

# Canal+ logo ROI parameters for different resolutions
CANAL_WM_PROFILES = {
    720:  {'x': 1240, 'y': 0, 'w': 40, 'h': 27, 'out_w': 40, 'out_h': 27},
    1080: {'x': 1860, 'y': 0, 'w': 60, 'h': 40, 'out_w': 40, 'out_h': 40},
}

# Brightness threshold: logo present when mean > threshold
CANAL_BRIGHT_THRESH = 150

# Brightness-based detection threshold (normalized 0-255 mean of ROI)
BRIGHT_THRESH = 150


def _ffprobe_dimensions(src):
    """Return (width, height) of the first video stream."""
    r = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=width,height', '-of', 'csv=p=0', str(src)],
        capture_output=True, text=True, check=True)
    vals = r.stdout.strip().split('\n')[0].split(',')
    return int(vals[0]), int(vals[1])


def _detect_profile(w, h):
    """Select watermark profile based on video height."""
    if h >= 1000:
        return CANAL_WM_PROFILES[1080]
    return CANAL_WM_PROFILES[720]


def build_watermark_template(src, ref_time, wm_x, wm_y, wm_w, wm_h,
                              out_w=32, out_h=32, tmp_suffix=''):
    """
    Extract a watermark reference template from a frame at ref_time.
    Uses slow seek (-ss after -i) and -fflags +discardcorrupt for TS robustness.
    Returns a float32 array of out_w*out_h pixels, or None on failure.
    """
    tmp = f'_tmp_canal_wm_template{tmp_suffix}.raw'
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
             '-fflags', '+discardcorrupt+genpts',
             '-i', str(src),
             '-ss', f'{ref_time:.3f}', '-frames:v', '1',
             '-vf', f'crop={wm_w}:{wm_h}:{wm_x}:{wm_y},scale={out_w}:{out_h}',
             '-f', 'rawvideo', '-pix_fmt', 'gray', tmp],
            check=True, timeout=120)
        arr = np.fromfile(tmp, dtype=np.uint8).astype(np.float32)
        os.remove(tmp)
    except Exception as e:
        print(f'  WARNING: Could not extract watermark template at {ref_time:.1f}s: {e}')
        if os.path.exists(tmp):
            os.remove(tmp)
        return None
    n = out_w * out_h
    return arr[:n] if len(arr) >= n else None


def _extract_chunk(src, chunk_start, chunk_dur, wm_x, wm_y, wm_w, wm_h,
                   out_w, out_h, fps, tmp_path):
    """Extract one chunk of ROI frames. Uses fast seek (-ss before -i)."""
    subprocess.run(
        ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
         '-fflags', '+discardcorrupt+genpts',
         '-ss', f'{chunk_start:.3f}', '-i', str(src),
         '-t', f'{chunk_dur:.0f}',
         '-vf', (f'fps={fps},'
                 f'crop={wm_w}:{wm_h}:{wm_x}:{wm_y},'
                 f'scale={out_w}:{out_h}'),
         '-f', 'rawvideo', '-pix_fmt', 'gray', tmp_path],
        check=True, timeout=120)


def scan_watermark_presence(src, scan_start, scan_end, wm_x, wm_y, wm_w, wm_h,
                             out_w, out_h, fps=2, tmp_suffix='', chunk_secs=1800):
    """
    Scan a video segment and return list of (time, mean_brightness) tuples.
    Splits the scan into chunks using fast seek (-ss before -i) for speed,
    falling back to slow seek on corruption.
    """
    scan_dur = max(0.0, scan_end - scan_start)
    if scan_dur <= 0:
        return []

    n_pixels = out_w * out_h
    step = 1.0 / fps
    all_results = []
    tmp = f'_tmp_canal_wm_scan{tmp_suffix}.raw'

    chunk_idx = 0
    while chunk_idx * chunk_secs < scan_dur:
        c_start = scan_start + chunk_idx * chunk_secs
        c_dur = min(chunk_secs, scan_end - c_start)

        try:
            _extract_chunk(src, c_start, c_dur, wm_x, wm_y, wm_w, wm_h,
                           out_w, out_h, fps, tmp)
        except Exception:
            # Fast seek may land on corrupt GOP — retry with slow seek
            try:
                _extract_chunk_slow(src, c_start, c_dur, wm_x, wm_y, wm_w, wm_h,
                                    out_w, out_h, fps, tmp)
            except Exception as e:
                print(f'    WARNING: chunk {fmt(c_start)}-{fmt(c_start+c_dur)} failed: {e}')
                if os.path.exists(tmp):
                    os.remove(tmp)
                chunk_idx += 1
                continue

        try:
            frames = np.fromfile(tmp, dtype=np.uint8).astype(np.float32)
            os.remove(tmp)
        except Exception:
            chunk_idx += 1
            continue

        n_frames = len(frames) // n_pixels
        for i in range(n_frames):
            frame = frames[i * n_pixels:(i + 1) * n_pixels]
            t_i = c_start + i * step
            if len(frame) == n_pixels:
                all_results.append((t_i, float(frame.mean())))

        chunk_idx += 1

    if os.path.exists(tmp):
        os.remove(tmp)
    return all_results


def _extract_chunk_slow(src, chunk_start, chunk_dur, wm_x, wm_y, wm_w, wm_h,
                        out_w, out_h, fps, tmp_path):
    """Slow seek fallback (-ss after -i) for corrupt regions."""
    subprocess.run(
        ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
         '-fflags', '+discardcorrupt+genpts',
         '-i', str(src),
         '-ss', f'{chunk_start:.3f}', '-t', f'{chunk_dur:.0f}',
         '-vf', (f'fps={fps},'
                 f'crop={wm_w}:{wm_h}:{wm_x}:{wm_y},'
                 f'scale={out_w}:{out_h}'),
         '-f', 'rawvideo', '-pix_fmt', 'gray', tmp_path],
        check=True, timeout=300)


def scan_watermark_correlation(src, wm_template, scan_start, scan_end,
                                wm_x, wm_y, wm_w, wm_h, out_w, out_h,
                                fps=2, tmp_suffix='', chunk_secs=1800):
    """
    Scan a video segment and return list of (time, correlation) tuples
    using template correlation. Chunked with fast seek for speed.
    """
    scan_dur = max(0.0, scan_end - scan_start)
    if scan_dur <= 0:
        return []

    n_pixels = out_w * out_h
    step = 1.0 / fps
    t0 = wm_template - wm_template.mean()
    t0_norm = np.linalg.norm(t0)
    tmp = f'_tmp_canal_wm_corr{tmp_suffix}.raw'
    all_results = []

    chunk_idx = 0
    while chunk_idx * chunk_secs < scan_dur:
        c_start = scan_start + chunk_idx * chunk_secs
        c_dur = min(chunk_secs, scan_end - c_start)

        try:
            _extract_chunk(src, c_start, c_dur, wm_x, wm_y, wm_w, wm_h,
                           out_w, out_h, fps, tmp)
        except Exception:
            try:
                _extract_chunk_slow(src, c_start, c_dur, wm_x, wm_y, wm_w, wm_h,
                                    out_w, out_h, fps, tmp)
            except Exception as e:
                print(f'    WARNING: corr chunk {fmt(c_start)} failed: {e}')
                if os.path.exists(tmp):
                    os.remove(tmp)
                chunk_idx += 1
                continue

        try:
            frames = np.fromfile(tmp, dtype=np.uint8).astype(np.float32)
            os.remove(tmp)
        except Exception:
            chunk_idx += 1
            continue

        n_frames = len(frames) // n_pixels
        for i in range(n_frames):
            frame = frames[i * n_pixels:(i + 1) * n_pixels]
            f0 = frame - frame.mean()
            conf = np.dot(f0, t0) / (np.linalg.norm(f0) * t0_norm + 1e-10)
            all_results.append((c_start + i * step, float(conf)))

        chunk_idx += 1

    if os.path.exists(tmp):
        os.remove(tmp)
    return all_results


def find_breaks_brightness(src, scan_start, scan_end, fps=2,
                            thresh=BRIGHT_THRESH, min_break_secs=45,
                            gap_tolerance_secs=5, wm_lag_secs=0.0,
                            tmp_suffix=''):
    """
    Find all ad breaks by scanning for Canal+ logo brightness dropping below threshold.

    Uses a rolling-window density approach to handle brief logo flickers
    (e.g., when on-screen graphics cover the logo during an ad break).

    gap_tolerance_secs: seconds of logo-present frames to tolerate before
                        ending a break region. Set higher to handle flicker.
    """
    profile = _detect_profile(*_ffprobe_dimensions(src))
    wm_x, wm_y = profile['x'], profile['y']
    wm_w, wm_h = profile['w'], profile['h']
    out_w, out_h = profile['out_w'], profile['out_h']

    print(f'  Watermark ROI: {wm_w}x{wm_h} at ({wm_x},{wm_y})  thresh={thresh}')
    print(f'  Scanning {scan_start:.0f}s to {scan_end:.0f}s at {fps} fps...')

    results = scan_watermark_presence(
        src, scan_start, scan_end, wm_x, wm_y, wm_w, wm_h,
        out_w, out_h, fps=fps, tmp_suffix=tmp_suffix)

    if not results:
        print('  WARNING: No frames extracted during scan.')
        return []

    step = 1.0 / fps
    max_gap_frames = int(gap_tolerance_secs * fps)  # max consecutive present frames within a break

    # First pass: mark each frame as absent (1) or present (0)
    absent = [1 if b < thresh else 0 for _, b in results]
    times = [t for t, _ in results]

    # Second pass: find break regions using gap tolerance.
    # A break starts when we hit an absent frame.
    # A break continues as long as the gap of present frames stays <= max_gap_frames.
    # A break ends when we see more than max_gap_frames consecutive present frames.
    breaks = []
    in_break = False
    break_start = None
    present_streak = 0

    for i, (t, is_absent) in enumerate(zip(times, absent)):
        if not in_break:
            if is_absent:
                in_break = True
                break_start = t
                present_streak = 0
        else:
            if is_absent:
                present_streak = 0
            else:
                present_streak += 1
                if present_streak > max_gap_frames:
                    # End of break
                    break_end = times[i - present_streak] + step
                    region_dur = break_end - break_start
                    if region_dur >= min_break_secs:
                        breaks.append((break_start, break_end))
                        print(f'  Break: {fmt(break_start)} - {fmt(break_end)}  '
                              f'dur={fmt(region_dur)}')
                    else:
                        print(f'  Skip: short region {fmt(break_start)}-{fmt(break_end)} '
                              f'({fmt(region_dur)} < {fmt(min_break_secs)})')
                    in_break = False

    # Handle break extending to end of scan
    if in_break:
        break_end = times[-1] + step
        region_dur = break_end - break_start
        if region_dur >= min_break_secs:
            breaks.append((break_start, break_end))
            print(f'  Break (extends to scan end): {fmt(break_start)} - {fmt(break_end)}')

    return breaks


def find_breaks_correlation(src, wm_template, scan_start, scan_end,
                             wm_x, wm_y, wm_w, wm_h, out_w, out_h,
                             fps=2, thresh=0.44, min_break_secs=45,
                             wm_lag_secs=0.0, min_ad_conf=None,
                             tmp_suffix=''):
    """
    Find all ad breaks by template correlation against a reference watermark.
    Returns list of (break_start, break_end) tuples in absolute seconds.
    """
    print(f'  Watermark ROI: {wm_w}x{wm_h} at ({wm_x},{wm_y})  thresh={thresh}')
    print(f'  Scanning {scan_start:.0f}s to {scan_end:.0f}s at {fps} fps...')

    results = scan_watermark_correlation(
        src, wm_template, scan_start, scan_end,
        wm_x, wm_y, wm_w, wm_h, out_w, out_h,
        fps=fps, tmp_suffix=tmp_suffix)

    if not results:
        return []

    step = 1.0 / fps

    breaks = []
    in_break = False
    break_start = None
    break_min_conf = 1.0

    for t_i, conf in results:
        if not in_break:
            if conf < thresh:
                in_break = True
                break_start = t_i
                break_min_conf = conf
        else:
            if conf < break_min_conf:
                break_min_conf = conf
            if conf >= thresh:
                in_break = False
                region_dur = t_i - break_start
                if region_dur >= min_break_secs:
                    if min_ad_conf is not None and break_min_conf >= min_ad_conf:
                        print(f'  Skip: obscuration {fmt(break_start)}-{fmt(t_i)} '
                              f'({fmt(region_dur)}, min_conf={break_min_conf:.3f})')
                    else:
                        break_end = t_i - wm_lag_secs
                        breaks.append((break_start, break_end))
                        print(f'  Break: {fmt(break_start)} - {fmt(break_end)}  '
                              f'dur={fmt(break_end - break_start)}  '
                              f'(min_conf={break_min_conf:.3f})')
                else:
                    print(f'  Skip: short dip {fmt(break_start)}-{fmt(t_i)} '
                          f'({fmt(region_dur)} < {fmt(min_break_secs)})')

    if in_break:
        region_dur = results[-1][0] - break_start
        if region_dur >= min_break_secs:
            break_end = results[-1][0] + step - wm_lag_secs
            breaks.append((break_start, break_end))
            print(f'  Break (extends to scan end): {fmt(break_start)} - {fmt(break_end)}')

    return breaks


def fmt(seconds):
    """Format seconds as hh:mm:ss.xxx."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f'{h:02d}:{m:02d}:{s:06.3f}'


def filter_breaks(breaks, min_content_sec=60):
    """
    Merge breaks separated by short content windows.

    If two consecutive breaks have less than min_content_sec of content
    between them (gap < min_content_sec), merge them into a single break
    spanning both. This eliminates false positives from on-screen graphics
    that briefly cover the Canal+ logo ROI.
    """
    if len(breaks) <= 1:
        return breaks

    merged = [breaks[0]]
    for b_start, b_end in breaks[1:]:
        gap = b_start - merged[-1][1]  # content between previous break end and this break start
        if gap < min_content_sec:
            # Merge: extend previous break to cover this one
            merged[-1] = (merged[-1][0], b_end)
            print(f'  Merge: gap {fmt(gap)} < {fmt(min_content_sec)} — '
                  f'extended to {fmt(merged[-1][0])} - {fmt(merged[-1][1])}')
        else:
            merged.append((b_start, b_end))

    return merged


def detect_canal_breaks(src, scan_start=0, scan_end=None,
                         method='brightness', fps=2, min_break_secs=45,
                         wm_lag_secs=0.0, min_content_sec=60,
                         tmp_suffix=''):
    """
    Detect all Canal+ ad breaks in a video file.

    method: 'brightness' (simple ROI brightness) or 'correlation' (template matching)
    scan_start/end: time range to scan (seconds)
    min_content_sec: merge breaks separated by less than this many seconds of content
    Returns list of (break_start, break_end) tuples.
    """
    from .audio_utils import get_duration

    if scan_end is None:
        scan_end = get_duration(src)

    profile = _detect_profile(*_ffprobe_dimensions(src))

    if method == 'brightness':
        raw = find_breaks_brightness(
            src, scan_start, scan_end, fps=fps,
            min_break_secs=min_break_secs, wm_lag_secs=wm_lag_secs,
            tmp_suffix=tmp_suffix)
    elif method == 'correlation':
        template_time = max(10.0, scan_start - 300)
        print(f'  Building watermark template at {fmt(template_time)}...')
        template = build_watermark_template(
            src, template_time,
            profile['x'], profile['y'], profile['w'], profile['h'],
            profile['out_w'], profile['out_h'], tmp_suffix=tmp_suffix)
        if template is None:
            print('  WARNING: Template build failed, falling back to brightness method')
            raw = find_breaks_brightness(
                src, scan_start, scan_end, fps=fps,
                min_break_secs=min_break_secs, wm_lag_secs=wm_lag_secs,
                tmp_suffix=tmp_suffix)
        else:
            raw = find_breaks_correlation(
                src, template, scan_start, scan_end,
                profile['x'], profile['y'], profile['w'], profile['h'],
                profile['out_w'], profile['out_h'],
                fps=fps, min_break_secs=min_break_secs, wm_lag_secs=wm_lag_secs,
                tmp_suffix=tmp_suffix)
    else:
        raise ValueError(f'Unknown method: {method}')

    # Filter: merge breaks separated by short content windows
    if raw and min_content_sec > 0:
        print(f'\n  Filtering: merging breaks with < {fmt(min_content_sec)} content gap...')
        filtered = filter_breaks(raw, min_content_sec=min_content_sec)
        print(f'  {len(raw)} raw breaks -> {len(filtered)} after filter')
        return filtered
    return raw
