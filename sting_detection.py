#!/usr/bin/env python3
"""
sting_detection.py
Fingerprint-based sting and transition detection for MotoGP sync scripts.
"""

import os, numpy as np
from pathlib import Path
from scipy.signal import fftconvolve

from audio_utils import SR, get_duration, extract_wav, load_fp_wav, _peak

CONF_THRESH    = 0.25   # minimum confidence to accept a transition hit
SUPPRESS_SECS  = 10     # deduplicate transition hits within this window (seconds)
MIN_EVENT_SECS = 120    # ignore transition hits in the first 2 minutes (broadcast intro)


def find_sting(src, fp_path, search_start, search_dur, stream_spec='0:a:0',
               label='', tmp_suffix=''):
    """
    Find a sting fingerprint within a time window of src.
    Returns (absolute_time_sec, confidence).
    Includes a duration guard so the search window never exceeds the file length.
    """
    import time, tempfile
    tmp = os.path.join(tempfile.gettempdir(), f'_tmp_sting_{int(time.time()*1000)}{tmp_suffix}.wav')
    actual_dur = min(search_dur, get_duration(src) - search_start)
    if actual_dur <= 0:
        return search_start, 0.0
    extract_wav(src, tmp, stream_spec, start=search_start, duration=actual_dur)
    # WSL interop timing
    for _ in range(10):
        if os.path.exists(tmp):
            break
        time.sleep(0.2)
    if not os.path.exists(tmp):
        return search_start, 0.0
    haystack = load_fp_wav(tmp)
    try:
        os.remove(tmp)
    except FileNotFoundError:
        pass
    needle = load_fp_wav(fp_path)
    idx, conf = _peak(haystack, needle)
    t = search_start + idx / SR
    if label:
        print(f'  {label}: {t:.3f}s ({t/60:.1f} min)  conf={conf:.4f}')
    return t, conf


def find_all_transitions(src, fp_paths, stream_spec='0:a:0', tmp_suffix='',
                         conf_thresh=CONF_THRESH, suppress_secs=SUPPRESS_SECS,
                         min_event_secs=MIN_EVENT_SECS):
    """
    Scan src for all occurrences of one or more transition fingerprints.
    fp_paths: str, Path, or list of either.
    Returns a sorted list of (time_sec, confidence, clip_dur_sec).
    Events before min_event_secs are discarded.
    """
    if isinstance(fp_paths, (str, Path)):
        fp_paths = [fp_paths]

    tmp = f'_tmp_full_scan{tmp_suffix}.wav'
    print('  Extracting full audio for transition scan...')
    extract_wav(src, tmp, stream_spec)
    h = load_fp_wav(tmp)
    try:
        os.remove(tmp)
    except FileNotFoundError:
        pass

    all_hits = []   # (time, confidence, clip_dur)

    for fp_path in fp_paths:
        needle   = load_fp_wav(fp_path)
        clip_dur = len(needle) / SR

        corr  = fftconvolve(h, needle[::-1], mode='valid')
        abs_c = np.abs(corr)
        n_len = len(needle)
        supp  = int(suppress_secs * SR)
        tmp_c = abs_c.copy()

        while True:
            idx  = int(np.argmax(tmp_c))
            h_w  = h[idx:idx + n_len]
            conf = float(abs_c[idx]) / (
                   np.linalg.norm(h_w) * np.linalg.norm(needle) + 1e-10)
            if conf < conf_thresh:
                break
            t = idx / SR
            if t >= min_event_secs:
                all_hits.append((t, conf, clip_dur))
            lo = max(0, idx - supp)
            hi = min(len(tmp_c), idx + supp)
            tmp_c[lo:hi] = 0

    # Merge and cross-fingerprint dedup: sort, suppress within suppress_secs
    all_hits.sort()
    deduped = []
    for t, c, d in all_hits:
        if not deduped or t - deduped[-1][0] > suppress_secs:
            deduped.append((t, c, d))
        elif c > deduped[-1][1]:
            deduped[-1] = (t, c, d)   # keep higher-confidence hit in same window

    return deduped
