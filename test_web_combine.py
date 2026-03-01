#!/usr/bin/env python3
"""
Synthetic test for web_combine.py

Generates fake Polsat and web files using ffmpeg tone generators, with:
  - Polsat: background 200Hz tone + race-specific sting tone at known offsets
            + different-frequency race audio after each sting
  - Web Natural Sounds (stream 1): sting tone for 18s, then different race-audio tone

This mirrors the real scenario where the sting is a distinctive 18s audio event
followed by clearly different race content, giving a sharp correlation peak.

Pass tolerance is 2 frames at 50fps (40ms).
"""

import subprocess, os, sys, numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from web_combine import find_offset, get_duration

TOLERANCE = 0.040   # seconds – 2 frames at 50fps

# ── Ground-truth offsets ──────────────────────────────────────────────────────
MOTO3_OFFSET  =  20.0
MOTO3_DUR     =  50.0
MOTO2_OFFSET  = 100.0   # 30s gap after Moto3
MOTO2_DUR     =  50.0
MOTOGP_OFFSET = 175.0   # 25s gap after Moto2
MOTOGP_DUR    =  50.0
POLSAT_DUR    = MOTOGP_OFFSET + MOTOGP_DUR + 10.0   # 235s total

STING_REAL    = 18.0    # sting length (matches the real 18-second sting)

# ── Synthetic audio frequencies ───────────────────────────────────────────────
# Each race has a unique sting freq and a different race-audio freq.
# Polsat background (simulating Polish commentary) uses 200 Hz throughout.
BG_HZ  = 200
RACES  = {
    'Moto3' : {'sting': 440, 'race': 800,  'off': MOTO3_OFFSET,  'dur': MOTO3_DUR},
    'Moto2' : {'sting': 550, 'race': 900,  'off': MOTO2_OFFSET,  'dur': MOTO2_DUR},
    'MotoGP': {'sting': 660, 'race': 1000, 'off': MOTOGP_OFFSET, 'dur': MOTOGP_DUR},
}


def run(cmd):
    subprocess.run(cmd, check=True, capture_output=True)


def make_polsat(path):
    """
    Single audio stream:
      - 200 Hz background throughout (simulates Polish commentary)
      - Race sting tone (amplitude 1.0) for each race at its known offset
      - Race audio tone (amplitude 0.3, different freq) after each sting
    """
    terms = [f"sin(2*PI*{BG_HZ}*t)*0.3"]
    for r in RACES.values():
        s, e = r['off'], r['off'] + STING_REAL
        re   = r['off'] + r['dur']
        terms.append(f"sin(2*PI*{r['sting']}*t)*between(t,{s},{e})")
        terms.append(f"sin(2*PI*{r['race']}*t)*0.3*between(t,{e},{re})")

    af = "aevalsrc='" + '+'.join(terms) + f"':s=44100:c=mono:d={POLSAT_DUR}"
    run(['ffmpeg', '-y', '-hide_banner',
         '-f', 'lavfi', '-i', f'color=black:size=320x180:rate=25:d={POLSAT_DUR}',
         '-f', 'lavfi', '-i', af,
         '-map', '0:v', '-map', '1:a',
         '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '40',
         '-c:a', 'mp2', '-b:a', '192k', path])


def make_web_race(name, path):
    r = RACES[name]
    dur = r['dur']
    # Stream 0 (commentary placeholder): silence
    af_commentary = f"aevalsrc='0':s=44100:c=mono:d={dur}"
    # Stream 1 (Natural Sounds): sting for STING_REAL seconds, then race audio
    af_natural = (
        f"aevalsrc='sin(2*PI*{r['sting']}*t)*between(t,0,{STING_REAL})"
        f"+sin(2*PI*{r['race']}*t)*0.3*(gt(t,{STING_REAL}))"
        f"':s=44100:c=mono:d={dur}"
    )
    run(['ffmpeg', '-y', '-hide_banner',
         '-f', 'lavfi', '-i', f'color=black:size=320x180:rate=25:d={dur}',
         '-f', 'lavfi', '-i', af_commentary,
         '-f', 'lavfi', '-i', af_natural,
         '-map', '0:v', '-map', '1:a', '-map', '2:a',
         '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '40',
         '-c:a', 'aac', '-b:a', '64k', path])


def strip_to_ambient(src, dst, duration=30):
    """
    Simulates what the user provides as a sample clip:
    video + only the ambient audio track (stream 1 → becomes stream 0 in output).
    """
    run(['ffmpeg', '-y', '-hide_banner',
         '-t', str(duration), '-i', src,
         '-map', '0:v:0', '-map', '0:a:1',
         '-c', 'copy', dst])


def check(label, expected, found):
    err    = abs(found - expected)
    status = 'PASS' if err <= TOLERANCE else 'FAIL'
    print(f'  [{status}] {label}: expected {expected:.3f}s  found {found:.3f}s  '
          f'(error {err * 1000:.1f} ms)')
    return status == 'PASS'


def main():
    files = {
        'polsat': '_test_polsat.mkv',
        'moto3':  '_test_moto3.mkv',
        'moto2':  '_test_moto2.mkv',
        'motogp': '_test_motogp.mkv',
    }

    # Full web race files (two audio tracks each)
    full = {k: f'_test_{k}_full.mkv' for k in ('moto3', 'moto2', 'motogp')}
    # Sample clips as the user would provide them: video + ambient track only (stream 0)
    clips = {k: f'_test_{k}_clip.mkv' for k in ('moto3', 'moto2', 'motogp')}

    print('Building synthetic test files...')
    make_polsat(files['polsat'])
    for name, key in [('Moto3','moto3'), ('Moto2','moto2'), ('MotoGP','motogp')]:
        make_web_race(name, full[key])
        strip_to_ambient(full[key], clips[key])
    print('  Done.\n')

    d_polsat = get_duration(files['polsat'])
    d_moto3  = get_duration(full['moto3'])
    d_moto2  = get_duration(full['moto2'])
    d_motogp = get_duration(full['motogp'])

    SEARCH_BUF = 60  # generous for a short test file

    # Use the stripped clips as needles (ambient is now stream 0)
    print('Finding sync points (using stripped ambient-only clips as needles)...')
    moto3_start = find_offset(files['polsat'], clips['moto3'],
                              search_start=0,
                              search_duration=min(d_moto3 + SEARCH_BUF, d_polsat / 3),
                              label='Moto3', web_ambient_stream=0)
    moto3_end = moto3_start + d_moto3

    mgp_search_start = max(0.0, d_polsat - d_motogp - SEARCH_BUF)
    motogp_start = find_offset(files['polsat'], clips['motogp'],
                               search_start=mgp_search_start,
                               search_duration=d_polsat - mgp_search_start,
                               label='MotoGP', web_ambient_stream=0)

    moto2_start = find_offset(files['polsat'], clips['moto2'],
                              search_start=moto3_end,
                              search_duration=motogp_start - moto3_end,
                              label='Moto2', web_ambient_stream=0)

    print('\nResults vs ground truth:')
    results = [
        check('Moto3',  MOTO3_OFFSET,  moto3_start),
        check('Moto2',  MOTO2_OFFSET,  moto2_start),
        check('MotoGP', MOTOGP_OFFSET, motogp_start),
    ]

    print(f'\n{"All tests passed." if all(results) else f"{results.count(False)} test(s) FAILED."}')

    for f in list(files.values()) + list(full.values()) + list(clips.values()):
        if os.path.exists(f): os.remove(f)


if __name__ == '__main__':
    main()
