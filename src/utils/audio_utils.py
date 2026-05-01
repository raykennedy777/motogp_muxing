#!/usr/bin/env python3
"""
audio_utils.py
Shared audio extraction, correlation, and concatenation utilities
for MotoGP sync scripts.
"""

import subprocess, os, numpy as np
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import fftconvolve, butter, filtfilt

SR = 8000   # Hz for correlation

# Multi-band correlation bands: (low_hz, high_hz, weight)
MULTIBANDS = [
    (80,   300,  0.2),   # low: engine rumble, bass stings
    (300, 3000,  0.6),   # mid: commentary, most sting energy
    (3000, 4000, 0.2),   # high: sibilance, high-frequency harmonics
]
MULTIBAND_TOLERANCE = 50  # samples (~6ms at 8kHz) for index agreement
MULTIBAND_MIN_AGREE = 2   # need this many bands to agree


def _bandpass(data, low, high, fs, order=4):
    """Zero-phase bandpass filter."""
    nyq = fs / 2.0
    lo  = max(0.1, low / nyq)
    hi  = min(0.99, high / nyq)
    if lo >= hi:
        return data
    b, a = butter(order, [lo, hi], btype='band')
    return filtfilt(b, a, data)


def get_duration(f):
    r = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', str(f)],
        capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def get_fps(path):
    """Probe video stream frame rate as a float (e.g. 50.0)."""
    r = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=r_frame_rate',
         '-of', 'default=noprint_wrappers=1:nokey=1', str(path)],
        capture_output=True, text=True, check=True)
    val = r.stdout.strip()
    if '/' in val:
        num, den = val.split('/')
        return float(num) / float(den)
    return float(val)


def get_video_start_pts(path):
    """Return video stream start_time in seconds (container PTS offset), or 0.0.

    MKV catchup recordings can have a non-zero video stream_start.  Frame N at
    fps F starts at container time  start_pts + N/F  — seek targets must include
    this offset or the cut will land in the wrong place.
    """
    r = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=start_time',
         '-of', 'default=noprint_wrappers=1:nokey=1', str(path)],
        capture_output=True, text=True, check=True)
    lines = [l for l in r.stdout.strip().splitlines()
             if l.strip() and l.strip() != 'N/A']
    try:
        return float(lines[0]) if lines else 0.0
    except ValueError:
        return 0.0


def get_audio_start_time(path, stream='a:0'):
    """Return audio stream start_time in seconds, or 0.0.

    If this differs from get_video_start_pts(), the file has an A/V delay.
    Sting detection runs in audio space; add (audio_start - video_start) to
    any sting time before comparing it against a video-frame-based anchor.
    """
    r = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', stream,
         '-show_entries', 'stream=start_time',
         '-of', 'default=noprint_wrappers=1:nokey=1', str(path)],
        capture_output=True, text=True, check=True)
    lines = [l for l in r.stdout.strip().splitlines()
             if l.strip() and l.strip() != 'N/A']
    try:
        return float(lines[0]) if lines else 0.0
    except ValueError:
        return 0.0


def get_audio_stream_count(f):
    r = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'a',
         '-show_entries', 'stream=index',
         '-of', 'default=noprint_wrappers=1:nokey=1', str(f)],
        capture_output=True, text=True, check=True)
    return len([l for l in r.stdout.strip().splitlines() if l.strip()])


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


def load_fp_wav(path, sr=SR):
    """
    Load fingerprint audio from any format (WAV, MKA, etc.) at sr Hz mono.
    Returns a float32 numpy array.
    """
    p = Path(path)
    if p.suffix.lower() == '.wav':
        _, data = wavfile.read(str(p))
        return data.astype(np.float32)
    tmp = f'_tmp_fp_{p.stem}.wav'
    extract_wav(str(p), tmp, '0:a:0', sr=sr)
    _, data = wavfile.read(tmp)
    os.remove(tmp)
    return data.astype(np.float32)


def _peak(haystack, needle):
    """Return (sample_index, confidence) for best match of needle in haystack.
    Uses RMS pre-normalization + multi-band correlation for robustness."""
    h = haystack.astype(np.float32)
    n = needle.astype(np.float32)

    # RMS pre-normalize both signals
    h = h / (np.sqrt(np.mean(h**2)) + 1e-10)
    n = n / (np.sqrt(np.mean(n**2)) + 1e-10)

    if len(n) >= len(h):
        n = n[:max(1, len(h) - 1)]

    # Full-band correlation first (for confidence at agreed position)
    corr  = fftconvolve(h, n[::-1], mode='valid')
    abs_c = np.abs(corr)

    # Multi-band correlation to find agreed position
    band_results = []
    for lo, hi, _w in MULTIBANDS:
        h_b = _bandpass(h, lo, hi, SR)
        n_b = _bandpass(n, lo, hi, SR)
        if len(h_b) < len(n_b) + 1:
            continue
        c = fftconvolve(h_b, n_b[::-1], mode='valid')
        idx_b = int(np.argmax(np.abs(c)))
        h_win = h_b[idx_b:idx_b + len(n_b)]
        conf_b = float(np.abs(c[idx_b])) / (
                 np.linalg.norm(h_win) * np.linalg.norm(n_b) + 1e-10)
        band_results.append((idx_b, conf_b, hi - lo))

    if len(band_results) >= MULTIBAND_MIN_AGREE:
        # Pairwise agreement: find the largest agreeing subset
        indices = [r[0] for r in band_results]
        best_subset = None
        best_weight = 0
        for i in range(len(indices)):
            subset = [i]
            for j in range(i + 1, len(indices)):
                if abs(indices[j] - indices[i]) <= MULTIBAND_TOLERANCE:
                    subset.append(j)
            weight = sum(MULTIBANDS[k][2] for k in subset)
            if len(subset) >= MULTIBAND_MIN_AGREE and weight > best_weight:
                best_subset = subset
                best_weight = weight

        if best_subset is not None:
            agreed = int(round(np.mean([band_results[k][0] for k in best_subset])))
            # Search full-band peak near the agreed position (within tolerance)
            lo = max(0, agreed - MULTIBAND_TOLERANCE)
            hi = min(len(abs_c), agreed + MULTIBAND_TOLERANCE)
            idx = lo + int(np.argmax(abs_c[lo:hi]))
            h_win = h[idx:idx + len(n)]
            conf = float(abs_c[idx]) / (
                   np.linalg.norm(h_win) * np.linalg.norm(n) + 1e-10)
            return idx, conf

    # Fallback: full-band peak
    idx   = int(np.argmax(abs_c))
    h_win = h[idx:idx + len(n)]
    conf  = float(abs_c[idx]) / (
            np.linalg.norm(h_win) * np.linalg.norm(n) + 1e-10)
    return idx, conf

    # Fallback: full-band peak
    idx   = int(np.argmax(abs_c))
    h_win = h[idx:idx + len(n)]
    conf  = float(abs_c[idx]) / (
            np.linalg.norm(h_win) * np.linalg.norm(n) + 1e-10)
    return idx, conf


def fmt(secs):
    """Format seconds as hh:mm:ss.hh (hundredths of a second)."""
    neg = secs < 0
    s   = abs(secs)
    h   = int(s // 3600)
    m   = int((s % 3600) // 60)
    sec = s % 60
    return f'{"-" if neg else ""}{h:02d}:{m:02d}:{sec:05.2f}'


def concat_segments_to_mka(seg_paths, output_mka, list_path_prefix='_tmp'):
    """Write a concat list, run ffmpeg AAC encode, then remove the list file."""
    list_path = f'{list_path_prefix}_concat.txt'
    with open(list_path, 'w') as f:
        for s in seg_paths:
            f.write(f"file '{Path(s).resolve()}'\n")
    print(f'\nConcatenating {len(seg_paths)} segments -> {output_mka}')
    subprocess.run(
        ['ffmpeg', '-y', '-hide_banner',
         '-f', 'concat', '-safe', '0', '-i', list_path,
         '-map', '0', '-c:a', 'aac', '-b:a', '192k', str(output_mka)],
        check=True)
    os.remove(list_path)
