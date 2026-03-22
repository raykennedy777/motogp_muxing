#!/usr/bin/env python3
"""
watermark_detection.py
Video watermark detection via frame correlation against a reference template.
Used by sync_dazn.py and sync_sporttv.py.
"""

import subprocess, os, numpy as np
from pathlib import Path


def get_video_dimensions(src):
    """Return (width, height) of the first video stream."""
    r = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=width,height', '-of', 'csv=p=0', str(src)],
        capture_output=True, text=True, check=True)
    w, h = r.stdout.strip().split(',')
    return int(w), int(h)


def build_watermark_template(src, ref_time, wm_x, wm_y, wm_w, wm_h,
                              out_w=32, out_h=32, tmp_suffix=''):
    """
    Extract a watermark reference template from a single frame at ref_time.
    ref_time must be during confirmed live on-track coverage (watermark present).
    Returns a flat float32 array of out_w*out_h pixels, or None on failure.
    """
    tmp = f'_tmp_wm_template{tmp_suffix}.raw'
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
             '-ss', f'{ref_time:.3f}', '-i', str(src), '-frames:v', '1',
             '-vf', f'crop={wm_w}:{wm_h}:{wm_x}:{wm_y},scale={out_w}:{out_h}',
             '-f', 'rawvideo', '-pix_fmt', 'gray', tmp],
            check=True)
        arr = np.fromfile(tmp, dtype=np.uint8).astype(np.float32)
        os.remove(tmp)
    except Exception as e:
        print(f'  WARNING: Could not extract watermark template: {e}')
        if os.path.exists(tmp):
            os.remove(tmp)
        return None
    n = out_w * out_h
    return arr[:n] if len(arr) >= n else None


def find_break_end_via_watermark(src, break_start, clip_dur, wm_template,
                                  wm_x, wm_y, wm_w, wm_h,
                                  out_w, out_h,
                                  search_secs=300, fps=2,
                                  thresh=0.44, min_offset_secs=30,
                                  tmp_suffix=''):
    """
    Detect break end by scanning for the watermark returning in live video.
    Extracts search_secs of video at fps frames/sec and correlates each frame
    against the reference template using Pearson correlation.

    Returns (break_end_sec, found_bool).
    break_end = first matching frame time - 4s (watermark appears slightly
    before the cut back to live coverage).
    Only fires when the live watermark is present; ad content will not trigger it.
    """
    search_start = break_start + clip_dur
    n_pixels = out_w * out_h
    tmp = f'_tmp_wm_probe{tmp_suffix}.raw'
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
             '-ss', f'{search_start:.3f}', '-t', f'{search_secs:.0f}',
             '-i', str(src),
             '-vf', (f'fps={fps},'
                     f'crop={wm_w}:{wm_h}:{wm_x}:{wm_y},'
                     f'scale={out_w}:{out_h}'),
             '-f', 'rawvideo', '-pix_fmt', 'gray', tmp],
            check=True)
        frames = np.fromfile(tmp, dtype=np.uint8).astype(np.float32)
        os.remove(tmp)
    except Exception as e:
        print(f'  WARNING: Watermark probe failed at break {break_start:.1f}s: {e}')
        if os.path.exists(tmp):
            os.remove(tmp)
        return None, False

    n_frames  = len(frames) // n_pixels
    step      = 1.0 / fps
    t0        = wm_template - wm_template.mean()
    t0_norm   = np.linalg.norm(t0)
    min_frame = int(min_offset_secs * fps)

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
        if i >= min_frame and conf > thresh:
            print(f'    Watermark conf={conf:.3f} at +{i*step:.1f}s '
                  f'(t={t_i:.1f}s); break end = {t_i - 4.0:.1f}s')
            return t_i - 4.0, True

    print(f'    Watermark not found (max={max_conf:.3f} at t={max_conf_t:.1f}s, '
          f'+{max_conf_t - search_start:.0f}s from search start)')
    return None, False


def find_all_breaks_via_watermark(src, wm_template, wm_x, wm_y, wm_w, wm_h,
                                   out_w, out_h, scan_start, scan_end,
                                   fps=2, thresh=0.44, min_break_secs=45,
                                   wm_lag_secs=0.0, min_ad_conf=None,
                                   tmp_suffix=''):
    """
    Scan a video segment for all periods where the watermark is absent (= ad breaks).
    Returns list of (break_start, break_end) tuples in absolute seconds.

    wm_lag_secs:  seconds the watermark lags behind actual program resumption.
                  break_end = watermark_return_time - wm_lag_secs.
                  (Positive = watermark appears AFTER program resumes.)
    min_ad_conf:  if set, a below-threshold region is only accepted as a real ad break
                  when at least one frame has correlation < min_ad_conf.  Regions where
                  the minimum correlation stays >= min_ad_conf are classified as graphic/
                  banner obscurations (partial cover of the watermark during live content)
                  and are silently skipped.  Use None (default) to disable this filter.
    """
    scan_dur = max(0.0, scan_end - scan_start)
    if scan_dur <= 0:
        return []

    n_pixels = out_w * out_h
    tmp = f'_tmp_wm_allscan{tmp_suffix}.raw'
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
             '-ss', f'{scan_start:.3f}', '-t', f'{scan_dur:.0f}',
             '-i', str(src),
             '-vf', (f'fps={fps},'
                     f'crop={wm_w}:{wm_h}:{wm_x}:{wm_y},'
                     f'scale={out_w}:{out_h}'),
             '-f', 'rawvideo', '-pix_fmt', 'gray', tmp],
            check=True)
        frames = np.fromfile(tmp, dtype=np.uint8).astype(np.float32)
        os.remove(tmp)
    except Exception as e:
        print(f'  WARNING: Watermark full scan failed: {e}')
        if os.path.exists(tmp):
            os.remove(tmp)
        return []

    n_frames = len(frames) // n_pixels
    if n_frames == 0:
        print('  WARNING: Watermark scan produced no frames.')
        return []

    step = 1.0 / fps
    t0 = wm_template - wm_template.mean()
    t0_norm = np.linalg.norm(t0)

    # Compute per-frame correlation
    corrs = []
    for i in range(n_frames):
        frame = frames[i * n_pixels:(i + 1) * n_pixels]
        f0    = frame - frame.mean()
        conf  = np.dot(f0, t0) / (np.linalg.norm(f0) * t0_norm + 1e-10)
        corrs.append((scan_start + i * step, conf))

    # Find contiguous below-threshold regions = breaks
    breaks         = []
    in_break       = False
    break_start    = None
    break_min_conf = 1.0

    for t_i, conf in corrs:
        if not in_break:
            if conf < thresh:
                in_break       = True
                break_start    = t_i
                break_min_conf = conf
        else:
            if conf < break_min_conf:
                break_min_conf = conf
            if conf >= thresh:
                in_break   = False
                region_dur = t_i - break_start
                if region_dur >= min_break_secs:
                    if min_ad_conf is not None and break_min_conf >= min_ad_conf:
                        print(f'  Skip: obscuration {break_start:.1f}s-{t_i:.1f}s '
                              f'({region_dur:.1f}s, min_conf={break_min_conf:.3f} '
                              f'>= min_ad_conf={min_ad_conf})')
                    else:
                        break_end = t_i - wm_lag_secs
                        breaks.append((break_start, break_end))
                        print(f'  Break detected: {break_start:.1f}s - {break_end:.1f}s  '
                              f'dur={break_end-break_start:.1f}s  '
                              f'(watermark returned at {t_i:.1f}s, '
                              f'min_conf={break_min_conf:.3f})')
                else:
                    print(f'  Skip: short dip {break_start:.1f}s-{t_i:.1f}s '
                          f'({region_dur:.1f}s < {min_break_secs}s)')

    # Handle break extending to end of scan
    if in_break:
        region_dur = corrs[-1][0] - break_start
        if region_dur >= min_break_secs:
            if min_ad_conf is not None and break_min_conf >= min_ad_conf:
                print(f'  Skip: obscuration (open-ended) {break_start:.1f}s  '
                      f'({region_dur:.1f}s, min_conf={break_min_conf:.3f} '
                      f'>= min_ad_conf={min_ad_conf})')
            else:
                break_end = corrs[-1][0] + step - wm_lag_secs
                breaks.append((break_start, break_end))
                print(f'  Break (open-ended at scan boundary): {break_start:.1f}s - {break_end:.1f}s  '
                      f'(break may still be ongoing)')

    return breaks
