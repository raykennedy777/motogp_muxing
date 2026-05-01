#!/usr/bin/env python3
"""
web_combine.py
Sync three separate web race rips (Moto3, Moto2, MotoGP) to the Polsat broadcast
timeline and combine them into a single continuous file, with black+silent gaps
between races matching the actual broadcast gaps.

Sync method: cross-correlate the Natural Sounds track (stream 1) from the start
of each web file against the Polsat main audio.  The 18-second world feed sting
at the top of each race is the anchor.

Requirements: ffmpeg/ffprobe on PATH, numpy, scipy
    pip install numpy scipy

Usage:
    python web_combine.py <polsat.mkv> <moto3.mkv> <moto2.mkv> <motogp.mkv> <output.mkv>
"""

import subprocess, sys, os, json, numpy as np
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import fftconvolve

from src.utils.audio_utils import get_audio_stream_count, extract_wav as _extract_wav

# ── Tuning ────────────────────────────────────────────────────────────────────
SAMPLE_RATE       = 8000   # Hz for correlation audio - fast and more than sufficient
NEEDLE_SECS       = 30     # seconds of needle for Moto2 search (sting is 18s; 30s is sufficient)
MOTO2_SEARCH_SECS = 400    # search window after Moto3 end; gap1 is typically ~3 min


# ── ffprobe helpers ───────────────────────────────────────────────────────────

def get_duration(f):
    r = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', f],
        capture_output=True, text=True, check=True)
    return float(r.stdout.strip())

def get_video_info(f):
    r = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=width,height,r_frame_rate,codec_name',
         '-of', 'json', f],
        capture_output=True, text=True, check=True)
    s = json.loads(r.stdout)['streams'][0]
    return {'width': s['width'], 'height': s['height'],
            'fps': s['r_frame_rate'], 'codec': s['codec_name']}

def get_audio_info(f, stream=0):
    r = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', f'a:{stream}',
         '-show_entries', 'stream=sample_rate,channels,codec_name',
         '-of', 'json', f],
        capture_output=True, text=True, check=True)
    s = json.loads(r.stdout)['streams'][0]
    return {'rate': s['sample_rate'], 'channels': s['channels'], 'codec': s['codec_name']}

def find_offset(polsat, web_file, search_start, search_duration, label, web_ambient_stream=1):
    """
    Find where web_file's sting appears within polsat[search_start : search_start+search_duration].
    Uses the Natural Sounds track from the web file as the needle.

    web_ambient_stream: 0-based audio stream index of the Natural Sounds track in web_file.
                        Defaults to 1 (full web files with World Feed on stream 0).
                        Pass 0 if the web file is a pre-stripped clip with only the ambient track.

    Returns the absolute timestamp in seconds within polsat.
    """
    print(f'  [{label}] Searching Polsat '
          f'{search_start:.0f}s - {search_start + search_duration:.0f}s ...')

    needle_wav   = f'_tmp_{label}_needle.wav'
    haystack_wav = f'_tmp_{label}_haystack.wav'

    # Clamp needle so it never exceeds the haystack (required for mode='valid')
    needle_secs = min(NEEDLE_SECS, search_duration - 0.5)
    _extract_wav(web_file, needle_wav, f'0:a:{web_ambient_stream}', duration=needle_secs)
    _extract_wav(polsat, haystack_wav, '0:a:0',
                 start=search_start, duration=search_duration)

    _, needle   = wavfile.read(needle_wav)
    _, haystack = wavfile.read(haystack_wav)
    needle      = needle.astype(np.float32)
    haystack    = haystack.astype(np.float32)

    corr       = fftconvolve(haystack, needle[::-1], mode='valid')
    peak_idx   = int(np.argmax(np.abs(corr)))
    offset_sec = peak_idx / SAMPLE_RATE

    # Normalised peak as a rough confidence indicator
    n_len      = len(needle)
    h_window   = haystack[peak_idx : peak_idx + n_len]
    confidence = (float(np.abs(corr[peak_idx]))
                  / (np.linalg.norm(h_window) * np.linalg.norm(needle) + 1e-10))

    absolute = search_start + offset_sec
    print(f'  [{label}] -> {absolute:.3f}s in Polsat  (confidence {confidence:.4f})')
    if confidence < 0.05:
        print(f'  [{label}] WARNING: low confidence - check inputs or adjust NEEDLE_SECS')

    for p in [needle_wav, haystack_wav]:
        if os.path.exists(p): os.remove(p)

    return absolute


# ── Gap generation ────────────────────────────────────────────────────────────

def make_gap(duration, ref_file, output):
    """
    Generate a black+silent clip of the given duration whose video/audio specs
    match ref_file so that the concat demuxer can join them with -c copy.
    """
    vi      = get_video_info(ref_file)
    n_audio = get_audio_stream_count(ref_file)

    lavfi_inputs = ['-f', 'lavfi', '-i',
                    f"color=black:size={vi['width']}x{vi['height']}:rate={vi['fps']}"]
    maps         = ['-map', '0:v']

    for i in range(n_audio):
        ai = get_audio_info(ref_file, i)
        cl = 'stereo' if ai['channels'] == 2 else 'mono'
        lavfi_inputs += ['-f', 'lavfi', '-i',
                         f"anullsrc=r={ai['rate']}:cl={cl}"]
        maps += ['-map', f'{i + 1}:a']

    subprocess.run(
        ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error'] +
        lavfi_inputs +
        ['-t', f'{duration:.3f}'] +
        maps +
        ['-c:v', 'libx264', '-crf', '18', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p',
         '-c:a', 'aac', '-b:a', '192k',
         output],
        check=True)
    print(f'  Gap clip: {duration:.3f}s -> {output}')


# ── Concatenation ─────────────────────────────────────────────────────────────

def concat_files(segments, output):
    list_path = '_tmp_concat.txt'
    with open(list_path, 'w') as f:
        for s in segments:
            f.write(f"file '{Path(s).resolve()}'\n")
    subprocess.run(
        ['ffmpeg', '-y', '-hide_banner',
         '-f', 'concat', '-safe', '0', '-i', list_path, '-map', '0', '-c', 'copy', output],
        check=True)
    os.remove(list_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) != 6:
        sys.exit('Usage: web_combine.py polsat.mkv moto3.mkv moto2.mkv motogp.mkv output.mkv')

    polsat, moto3, moto2, motogp, output = sys.argv[1:]

    # ── Durations ──
    print('Getting durations...')
    d_polsat = get_duration(polsat)
    d_moto3  = get_duration(moto3)
    d_moto2  = get_duration(moto2)
    d_motogp = get_duration(motogp)
    print(f'  Polsat: {d_polsat:.1f}s | Moto3: {d_moto3:.1f}s | '
          f'Moto2: {d_moto2:.1f}s | MotoGP: {d_motogp:.1f}s')

    # ── Sync ──
    print('\nFinding sync points...')

    # Moto3 always starts at the very beginning of the Polsat file
    moto3_start = 0.0
    moto3_end   = d_moto3
    print(f'  [Moto3] Fixed at 0.000s - {moto3_end:.3f}s')

    # MotoGP always ends at the very end of the Polsat file
    motogp_start = d_polsat - d_motogp
    motogp_end   = d_polsat
    print(f'  [MotoGP] Fixed at {motogp_start:.3f}s - {motogp_end:.3f}s')

    # Moto2: search for the sting in a window starting right after Moto3 ends.
    # Gap 1 is typically ~3 minutes; MOTO2_SEARCH_SECS gives comfortable headroom.
    moto2_start = find_offset(
        polsat, moto2,
        search_start=moto3_end,
        search_duration=MOTO2_SEARCH_SECS,
        label='Moto2')
    moto2_end = moto2_start + d_moto2

    gap1 = moto2_start  - moto3_end
    gap2 = motogp_start - moto2_end

    print(f'\nResults:')
    print(f'  Moto3  : {moto3_start:.3f}s - {moto3_end:.3f}s')
    print(f'  Gap 1  : {gap1:.3f}s')
    print(f'  Moto2  : {moto2_start:.3f}s - {moto2_end:.3f}s')
    print(f'  Gap 2  : {gap2:.3f}s')
    print(f'  MotoGP : {motogp_start:.3f}s - {motogp_end:.3f}s')
    print(f'  Total  : {motogp_end:.3f}s  (Polsat: {d_polsat:.3f}s)')

    if gap1 < 0 or gap2 < 0:
        sys.exit('\nERROR: negative gap - correlation likely failed. '
                 'Check that Natural Sounds is stream 1 in the web files.')

    # ── Build output ──
    print('\nGenerating gap clips...')
    gap1_file = '_tmp_gap1.mkv'
    gap2_file = '_tmp_gap2.mkv'
    make_gap(gap1, moto3, gap1_file)
    make_gap(gap2, moto2, gap2_file)

    print('\nConcatenating...')
    concat_files([moto3, gap1_file, moto2, gap2_file, motogp], output)

    for f in [gap1_file, gap2_file]:
        if os.path.exists(f): os.remove(f)

    print(f'\nDone -> {output}')


if __name__ == '__main__':
    main()
