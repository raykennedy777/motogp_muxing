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
    first_line = r.stdout.strip().split('\n')[0]
    w, h = first_line.split(',')
    return int(w), int(h)


def load_watermark_png(png_path):
    """
    Load a pre-built grayscale watermark template from a PNG file.
    Returns a flat float32 array, or None on failure.
    The PNG is decoded via ffmpeg to raw grayscale, so no PIL/Pillow required.
    """
    try:
        result = subprocess.run(
            ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
             '-i', str(png_path),
             '-f', 'rawvideo', '-pix_fmt', 'gray', 'pipe:1'],
            check=True, capture_output=True)
        arr = np.frombuffer(result.stdout, dtype=np.uint8).astype(np.float32)
    except Exception as e:
        print(f'  WARNING: Could not load watermark PNG {png_path}: {e}')
        return None
    return arr if len(arr) > 0 else None


def build_watermark_template(src, ref_time, wm_x, wm_y, wm_w, wm_h,
                              out_w=32, out_h=32, tmp_suffix=''):
    """
    Extract a watermark reference template from a single frame at ref_time.
    ref_time must be during confirmed live on-track coverage (watermark present).
    Returns a flat float32 array of out_w*out_h pixels, or None on failure.
    Uses full-frame extraction with Python slicing to avoid YUV420 chroma rounding.
    """
    try:
        # Extract full frame, then crop and scale in Python
        result = subprocess.run(
            ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
             '-ss', f'{ref_time:.3f}', '-i', str(src), '-frames:v', '1',
             '-f', 'rawvideo', '-pix_fmt', 'gray', 'pipe:1'],
            check=True, capture_output=True)
        full = np.frombuffer(result.stdout, dtype=np.uint8).astype(np.float32)
        src_w, src_h = get_video_dimensions(src)
        if len(full) != src_w * src_h:
            print(f'  WARNING: Unexpected frame size: {len(full)} (expected {src_w}x{src_h}={src_w*src_h})')
            return None
        # Slice exact window from full frame
        rows = [(wm_y + r) * src_w + wm_x for r in range(wm_h)]
        crop = np.concatenate([full[off:off + wm_w] for off in rows])
        # Resize to output size if needed
        if wm_w != out_w or wm_h != out_h:
            import tempfile as _tf, time as _time
            _tmp = os.path.join(_tf.gettempdir(), f'_tmp_wm_tpl_{int(_time.time()*1000)}.raw')
            subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error',
                     '-f','rawvideo','-pix_fmt','gray','-s',f'{wm_w}:{wm_h}',
                     '-i','pipe:0','-vf',f'scale={out_w}:{out_h}',
                     '-f','rawvideo','-pix_fmt','gray',_tmp],
                    input=crop.astype(np.uint8).tobytes(), check=True)
            arr = np.fromfile(_tmp, dtype=np.uint8).astype(np.float32)
            if os.path.exists(_tmp): os.remove(_tmp)
        else:
            arr = crop
    except Exception as e:
        print(f'  WARNING: Could not extract watermark template: {e}')
        return None
    n = out_w * out_h
    return arr[:n] if len(arr) >= n else None


def build_watermark_template_averaged(src, start_time, end_time, interval,
                                       wm_x, wm_y, wm_w, wm_h,
                                       out_w=32, out_h=32, tmp_suffix=''):
    """
    Extract and average multiple watermark templates from start_time to end_time.
    This provides a more robust template by averaging several frames, reducing
    the impact of any single frame having unusual brightness or artifacts.

    Parameters:
        start_time: First extraction time (seconds)
        end_time: Last extraction time (seconds)
        interval: Seconds between extractions (e.g., 30 for every 30s)

    Returns a flat float32 array of out_w*out_h pixels, or None on failure.
    """
    templates = []
    times = []
    t = start_time
    while t <= end_time:
        times.append(t)
        t += interval

    print(f'  Extracting templates from {len(times)} time points: ', end='')
    for i, ref_time in enumerate(times):
        template = build_watermark_template(
            src, ref_time, wm_x, wm_y, wm_w, wm_h, out_w, out_h,
            tmp_suffix=f'{tmp_suffix}_{i}')
        if template is not None:
            templates.append(template)
            print(f'{ref_time:.0f}s ', end='')
        else:
            print(f'{ref_time:.0f}s(failed) ', end='')

    if not templates:
        print('- all failed')
        return None

    print(f'-> averaging {len(templates)} templates')
    return np.mean(templates, axis=0)


def _grad_mag(arr, w, h):
    """
    Compute gradient magnitude of a flat grayscale array (row-major, w×h).
    Returns a flat float32 array of the same length.
    Used by find_break_end_via_watermark(use_gradient=True) for edge-based
    watermark detection, which is more robust when the watermark overlays
    variable video content (e.g. Sky Italia MGP logo).
    """
    img = arr.reshape(h, w)
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, 1:] = img[:, 1:] - img[:, :-1]
    gy[1:, :] = img[1:, :] - img[:-1, :]
    return np.sqrt(gx ** 2 + gy ** 2).ravel()


def find_break_end_via_watermark(src, break_start, clip_dur, wm_template,
                                  wm_x, wm_y, wm_w, wm_h,
                                  out_w, out_h,
                                  search_secs=300, fps=2,
                                  thresh=0.44, min_offset_secs=30,
                                  wm_lag_secs=4.0, tmp_suffix='',
                                  use_gradient=False):
    """
    Detect break end by scanning for the watermark returning in live video.
    Extracts search_secs of video at fps frames/sec and correlates each frame
    against the reference template using Pearson correlation.

    Returns (break_end_sec, found_bool).
    break_end = first_matching_frame_time - wm_lag_secs.

    wm_lag_secs: seconds by which the watermark leads actual program resumption.
      Positive (default 4.0): watermark appears AFTER content resumes (break_end is before watermark).
      Negative (e.g. -58.3): watermark appears BEFORE content resumes (break_end is after watermark).
    """
    import time, tempfile
    search_start = break_start + clip_dur
    n_pixels = out_w * out_h
    tmp = os.path.join(tempfile.gettempdir(), f'_tmp_wm_probe_{int(time.time()*1000)}{tmp_suffix}.raw')
    try:
        # Use crop+scale filter to extract only the watermark region.
        # Avoids allocating full-resolution frames (which OOM on 1080p+ sources).
        # format=gray first so the subsequent crop operates on luma only,
        # eliminating any YUV420 chroma-alignment constraints.
        vf = f'fps={fps},format=gray,crop={wm_w}:{wm_h}:{wm_x}:{wm_y}'
        if wm_w != out_w or wm_h != out_h:
            vf += f',scale={out_w}:{out_h}'
        subprocess.run(
            ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
             '-ss', f'{search_start:.3f}', '-t', f'{search_secs:.0f}',
             '-i', str(src),
             '-vf', vf,
             '-f', 'rawvideo', '-pix_fmt', 'gray', tmp],
            check=True)
        # WSL interop timing
        for _ in range(10):
            if os.path.exists(tmp):
                break
            time.sleep(0.2)
        if not os.path.exists(tmp):
            print(f'  WARNING: Watermark probe: file not created')
            return None, False
        frames = np.fromfile(tmp, dtype=np.uint8).astype(np.float32)
        os.remove(tmp)
    except Exception as e:
        print(f'  WARNING: Watermark probe failed at break {break_start:.1f}s: {e}')
        if os.path.exists(tmp):
            os.remove(tmp)
        return None, False

    frame_px  = n_pixels
    n_frames  = len(frames) // frame_px
    step      = 1.0 / fps
    # For gradient mode: correlate edge magnitudes rather than raw pixel values.
    # This is robust to variable video backgrounds behind semi-transparent overlays.
    if use_gradient:
        tpl_signal = _grad_mag(wm_template.astype(np.float32), out_w, out_h)
    else:
        tpl_signal = wm_template
    t0        = tpl_signal - tpl_signal.mean()
    t0_norm   = np.linalg.norm(t0)
    min_frame = int(min_offset_secs * fps)

    max_conf   = 0.0
    max_conf_t = search_start
    for i in range(n_frames):
        crop = frames[i * frame_px:(i + 1) * frame_px]
        if use_gradient:
            signal = _grad_mag(crop, out_w, out_h)
        else:
            signal = crop
        f0    = signal - signal.mean()
        conf  = np.dot(f0, t0) / (np.linalg.norm(f0) * t0_norm + 1e-10)
        t_i   = search_start + i * step
        if conf > max_conf:
            max_conf   = conf
            max_conf_t = t_i
        if i >= min_frame and conf > thresh:
            print(f'    Watermark conf={conf:.3f} at +{i*step:.1f}s '
                  f'(t={t_i:.1f}s); break end = {t_i - wm_lag_secs:.1f}s')
            return t_i - wm_lag_secs, True

    print(f'    Watermark not found (max={max_conf:.3f} at t={max_conf_t:.1f}s, '
          f'+{max_conf_t - search_start:.0f}s from search start)')
    return None, False


def find_break_start_via_watermark(src, sting_time, wm_template,
                                   wm_x, wm_y, wm_w, wm_h,
                                   out_w, out_h,
                                   search_secs=300, fps=2,
                                   thresh=0.44, tmp_suffix=''):
    """
    Detect break start by scanning BACKWARD from a sting time to find when
    the watermark disappeared (correlation drops below threshold).

    Returns (break_start_sec, found_bool).
    """
    import time
    search_start = max(0.0, sting_time - search_secs)
    search_dur = sting_time - search_start
    n_pixels = out_w * out_h
    tmp = os.path.join(tempfile.gettempdir(), f'_tmp_wm_break_start_{int(time.time()*1000)}{tmp_suffix}.raw')
    try:
        # Extract full frames to avoid YUV420 chroma rounding
        subprocess.run(
            ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
             '-ss', f'{search_start:.3f}', '-t', f'{search_dur:.1f}',
             '-i', str(src),
             '-vf', f'fps={fps}',
             '-f', 'rawvideo', '-pix_fmt', 'gray', tmp],
            check=True)
        for _ in range(10):
            if os.path.exists(tmp):
                break
            time.sleep(0.2)
        if not os.path.exists(tmp):
            print(f'  WARNING: break start scan: file not created')
            return None, False
        frames = np.fromfile(tmp, dtype=np.uint8).astype(np.float32)
        os.remove(tmp)
    except Exception as e:
        print(f'  WARNING: break start scan failed: {e}')
        if os.path.exists(tmp):
            os.remove(tmp)
        return None, False

    frame_px = 1280 * 720
    n_frames = len(frames) // frame_px
    if n_frames == 0:
        return None, False

    step = 1.0 / fps
    t0 = wm_template - wm_template.mean()
    t0_norm = np.linalg.norm(t0)
    row_offs = [(wm_y + r) * 1280 + wm_x for r in range(wm_h)]

    last_visible_t = None
    in_break = False

    for i in range(n_frames):
        frame = frames[i * frame_px:(i + 1) * frame_px]
        crop = np.concatenate([frame[off:off + wm_w] for off in row_offs])
        if wm_w != out_w or wm_h != out_h:
            import subprocess as _sp, tempfile as _tf
            _tmp = os.path.join(_tf.gettempdir(), f'_tmp_wm_bs_{i}.raw')
            _sp.run(['ffmpeg','-y','-hide_banner','-loglevel','error',
                     '-f','rawvideo','-pix_fmt','gray','-s',f'{wm_w}:{wm_h}',
                     '-i','pipe:0','-vf',f'scale={out_w}:{out_h}',
                     '-f','rawvideo','-pix_fmt','gray',_tmp],
                    input=crop.astype(np.uint8).tobytes(), check=True)
            crop = np.fromfile(_tmp, dtype=np.uint8).astype(np.float32)
            if os.path.exists(_tmp): os.remove(_tmp)
        f0 = crop - crop.mean()
        conf = np.dot(f0, t0) / (np.linalg.norm(f0) * t0_norm + 1e-10)
        t_i = search_start + i * step

        if not in_break:
            if conf >= thresh:
                last_visible_t = t_i
            else:
                in_break = True

    if last_visible_t is not None:
        for i in range(n_frames - 1, -1, -1):
            frame = frames[i * n_pixels:(i + 1) * n_pixels]
            f0 = frame - frame.mean()
            conf = np.dot(f0, t0) / (np.linalg.norm(f0) * t0_norm + 1e-10)
            t_i = search_start + i * step
            if conf >= thresh and t_i <= last_visible_t:
                break_start = t_i + step
                print(f'    Break start (WM disappeared): {break_start:.1f}s '
                      f'(last visible at {t_i:.1f}s, conf={conf:.3f})')
                return break_start, True

        break_start = last_visible_t + 1.0
        print(f'    Break start (fallback): {break_start:.1f}s')
        return break_start, True

    print(f'    Watermark never visible in scan window')
    return None, False


def find_all_breaks_via_watermark(src, wm_template, wm_x, wm_y, wm_w, wm_h,
                                   out_w, out_h, scan_start, scan_end,
                                   fps=2, thresh=0.44, min_break_secs=45,
                                   wm_lag_secs=0.0, min_ad_conf=None,
                                   max_std=None, min_stable_pct=None,
                                   max_mean_conf=0.05, min_mean_conf=None,
                                   min_min_conf=None,
                                   tmp_suffix=''):
    """
    Scan a video segment for all periods where the watermark is absent (= ad breaks).
    Returns list of (break_start, break_end) tuples in absolute seconds.

    wm_lag_secs:    seconds the watermark lags behind actual program resumption.
                    break_end = watermark_return_time - wm_lag_secs.
                    (Positive = watermark appears AFTER program resumes.)
    min_ad_conf:    if set, a below-threshold region is only accepted as a real ad break
                    when at least one frame has correlation < min_ad_conf.  Regions where
                    the minimum correlation stays >= min_ad_conf are classified as graphic/
                    banner obscurations (partial cover of the watermark during live content)
                    and are silently skipped.  Use None (default) to disable this filter.
    max_std:        if set, reject regions where standard deviation of correlation exceeds
                    this value. High std indicates variable correlations (graphics/overlays)
                    rather than stable ad content.  Use None (default) to disable.
    min_stable_pct: if set, require this percentage of frames to have correlation around
                    -0.265 (the "no watermark" pattern). True ad breaks have >90% stable
                    negative correlations.  Use None (default) to disable.
    max_mean_conf:  reject below-threshold regions whose mean correlation exceeds this value.
                    Full-screen ad breaks have near-zero or slightly positive mean correlation.
                    Default 0.05. Raise (e.g. 0.20) if ads produce slightly positive means.
    min_mean_conf:  reject below-threshold regions whose mean correlation is below this value.
                    Regions with very negative mean are content that inverts the template
                    (e.g. different graphics/replays), not genuine ad breaks. Default None (disabled).
    min_min_conf:   reject below-threshold regions where no frame drops below this value.
                    Genuine ad breaks always have at least some frames with strongly negative
                    correlation. Regions where all frames stay above this threshold are
                    borderline content obscurations. Default None (disabled).
    """
    scan_dur = max(0.0, scan_end - scan_start)
    if scan_dur <= 0:
        return []

    n_pixels = out_w * out_h
    tmp = f'_tmp_wm_allscan{tmp_suffix}.raw'
    try:
        # Extract only the watermark crop region, scaled to template size.
        # This avoids loading full frames for a potentially very long scan window.
        # YUV420 chroma rounding is not a concern here since we work in grayscale
        # (luma-only) and the crop filter operates at full luma resolution.
        vf = f'fps={fps},format=gray,crop={wm_w}:{wm_h}:{wm_x}:{wm_y}'
        if wm_w != out_w or wm_h != out_h:
            vf += f',scale={out_w}:{out_h}'
        subprocess.run(
            ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
             '-ss', f'{scan_start:.3f}', '-t', f'{scan_dur:.0f}',
             '-i', str(src),
             '-vf', vf,
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

    # Compute per-frame correlation directly from cropped frames
    corrs = []
    for i in range(n_frames):
        crop  = frames[i * n_pixels:(i + 1) * n_pixels]
        f0    = crop - crop.mean()
        conf  = np.dot(f0, t0) / (np.linalg.norm(f0) * t0_norm + 1e-10)
        corrs.append((scan_start + i * step, conf))

    # Find contiguous below-threshold regions = breaks
    breaks         = []
    in_break       = False
    break_start    = None
    break_min_conf = 1.0
    break_corrs    = []  # Collect correlations for analysis

    def analyze_break(start_t, end_t, min_conf, corr_list):
        """Analyze a break region and return True if it's a real ad break."""
        region_dur = end_t - start_t
        if region_dur < min_break_secs:
            print(f'  Skip: short dip {start_t:.1f}s-{end_t:.1f}s '
                  f'({region_dur:.1f}s < {min_break_secs}s)')
            return False

        # Existing min_ad_conf filter
        if min_ad_conf is not None and min_conf >= min_ad_conf:
            print(f'  Skip: obscuration {start_t:.1f}s-{end_t:.1f}s '
                  f'({region_dur:.1f}s, min_conf={min_conf:.3f} '
                  f'>= min_ad_conf={min_ad_conf})')
            return False

        # New filters based on correlation statistics
        if corr_list:
            corr_array = np.array(corr_list)
            mean_conf = np.mean(corr_array)
            std_conf = np.std(corr_array)
            
            # Check standard deviation filter (reject variable correlations).
            # When max_std is set, also require the trimmed distribution to be
            # tightly clustered and negative (stable-negative pattern).
            # When max_std is None both checks are skipped — rely on min_break_secs
            # and min_ad_conf alone (appropriate for full-screen ad breaks where
            # diverse ad content produces variable but consistently low correlation).
            if max_std is not None:
                if std_conf > max_std:
                    print(f'  Skip: variable correlation {start_t:.1f}s-{end_t:.1f}s '
                          f'({region_dur:.1f}s, std={std_conf:.3f} > max_std={max_std})')
                    return False

                sorted_corrs = np.sort(corr_array)
                n_trim = max(1, len(sorted_corrs) // 20)
                trimmed = sorted_corrs[n_trim:-n_trim] if n_trim > 0 else sorted_corrs
                corr_std   = np.std(trimmed)
                corr_range = np.max(trimmed) - np.min(trimmed)
                is_stable_negative = (corr_std < 0.08 and
                                      np.mean(trimmed) < 0 and
                                      corr_range < 0.3)
                if not is_stable_negative:
                    print(f'  Skip: unstable correlation {start_t:.1f}s-{end_t:.1f}s '
                          f'({region_dur:.1f}s, std={corr_std:.3f}, range={corr_range:.3f})')
                    return False

            # Require mean correlation in acceptable range for ad content.
            if mean_conf > max_mean_conf:
                print(f'  Skip: positive mean correlation {start_t:.1f}s-{end_t:.1f}s '
                      f'({region_dur:.1f}s, mean={mean_conf:.3f} > {max_mean_conf})')
                return False
            if min_mean_conf is not None and mean_conf < min_mean_conf:
                print(f'  Skip: negative mean correlation {start_t:.1f}s-{end_t:.1f}s '
                      f'({region_dur:.1f}s, mean={mean_conf:.3f} < min_mean_conf={min_mean_conf})')
                return False

        # Require at least one deeply-negative frame (genuine watermark absence).
        if min_min_conf is not None and min_conf > min_min_conf:
            print(f'  Skip: no deep-negative frame {start_t:.1f}s-{end_t:.1f}s '
                  f'({region_dur:.1f}s, min_conf={min_conf:.3f} > min_min_conf={min_min_conf})')
            return False

        return True

    for t_i, conf in corrs:
        if not in_break:
            if conf < thresh:
                in_break       = True
                break_start    = t_i
                break_min_conf = conf
                break_corrs    = [conf]
            else:
                # Reset correlation collection when not in break
                break_corrs = []
        else:
            break_corrs.append(conf)
            if conf < break_min_conf:
                break_min_conf = conf
            if conf >= thresh:
                in_break   = False
                break_end  = t_i
                
                brk_mean = float(np.mean(break_corrs)) if break_corrs else 0.0
                if analyze_break(break_start, break_end, break_min_conf, break_corrs):
                    break_end_adjusted = break_end - wm_lag_secs
                    breaks.append((break_start, break_end_adjusted))
                    print(f'  Break detected: {break_start:.1f}s - {break_end_adjusted:.1f}s  '
                          f'dur={break_end_adjusted-break_start:.1f}s  '
                          f'(watermark returned at {break_end:.1f}s, '
                          f'min_conf={break_min_conf:.3f}, mean={brk_mean:.3f})')
                
                # Reset for next region
                break_corrs = []

    # Handle break extending to end of scan
    if in_break:
        break_end = corrs[-1][0] + step
        region_dur = break_end - break_start
        
        brk_mean = float(np.mean(break_corrs)) if break_corrs else 0.0
        if analyze_break(break_start, break_end, break_min_conf, break_corrs):
            break_end_adjusted = break_end - wm_lag_secs
            breaks.append((break_start, break_end_adjusted))
            print(f'  Break (open-ended at scan boundary): {break_start:.1f}s - {break_end_adjusted:.1f}s  '
                  f'(break may still be ongoing, mean={brk_mean:.3f})')

    return breaks
