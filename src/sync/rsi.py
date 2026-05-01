#!/usr/bin/env python3
"""
sync_rsi.py
Add RSI Motociclismo broadcast audio to a MotoGP master file.

Two modes:
  sunday   -- Combined Sunday file (Moto3+Moto2+MotoGP). Sync anchor:
              prerace_sting.wav (Moto3 pre-race sting), searched 5-20 min
              into the RSI file; master search covers full web_master.mkv.
  saturday -- MotoGP Sprint/race file. Sync anchor:
              prerace_sting_motogp.wav (65s MotoGP pre-race sting), searched
              35-50 min before the end of the RSI file.

No ad breaks: output is NS head + RSI content + NS tail.

Usage:
    python sync_rsi.py --mode {sunday|saturday}
                       [--dry-run] [--trim-start S] [--trim-end S]
                       [--src-sting-time S] [--master-sting-time S]
                       <rsi_file> <master.mkv> <output_dir>

Requirements: ffmpeg/ffprobe on PATH, numpy, scipy
    (via audio_utils / sting_detection modules)
"""

import sys, argparse
from pathlib import Path

from src.utils.audio_utils import get_duration, get_audio_stream_count, extract_seg, concat_segments_to_mka
from src.utils.sting_detection import find_sting

FP_DIR = Path(__file__).parent.parent.parent / 'fingerprints'

# Search windows per mode: (start_sec, duration_sec)
STING_CFG = {
    'sunday': {
        'fp':            'prerace_sting.wav',
        'src_window':    (300, 900),       # 5-20 min into file
        'master_window': (0, 7200),        # up to 2h into combined master
        'conf_thresh':   0.1,
    },
    'saturday': {
        'fp':            'prerace_sting_motogp.wav',
        'src_window':    None,             # computed from d_src at runtime (35-50 min before end)
        'master_window': (0, 3600),        # first 60 min of master
        'conf_thresh':   0.1,
    },
}


# ── Segment building and concatenation ────────────────────────────────────────

def build_and_concat(src_file, master_file, src_sting, master_sting,
                     trim_start, trim_end, d_src, d_master,
                     src_stream, ns_stream,
                     output_mka, dry_run=False):
    """
    Build output and concatenate to MKA.

    Alignment: master_time = src_time + offset
               where offset = master_sting - src_sting

    Section 1: NS  master 0                       -> offset + trim_start
    Section 2: RSI src trim_start                 -> d_src - trim_end
    Section 3: NS  master offset + d_src-trim_end -> d_master
    """
    offset           = master_sting - src_sting
    ns1_end          = offset + trim_start
    src_content_end  = d_src - trim_end
    ns3_start        = offset + src_content_end

    print(f'\n  Offset: {offset:.3f}s')
    print(f'  RSI content: src {trim_start:.1f}s - {src_content_end:.1f}s '
          f'(dur={src_content_end - trim_start:.1f}s)')
    print(f'  Master coverage: {ns1_end:.1f}s - {ns3_start:.1f}s')

    tmp_dir = Path('_tmp_rsi_segs')
    if not dry_run:
        tmp_dir.mkdir(exist_ok=True)
    segs    = []
    counter = [0]

    def new_seg(src, stream, start, duration, desc):
        if duration <= 0:
            return
        counter[0] += 1
        p = str(tmp_dir / f'seg_{counter[0]:04d}.wav')
        print(f'  {desc}  start={start:.1f}s  dur={duration:.1f}s')
        if not dry_run:
            extract_seg(src, p, stream, start=start, duration=duration)
        segs.append(p)

    # Section 1: NS from master 0 to where RSI content begins
    if ns1_end > 0:
        new_seg(master_file, ns_stream, 0.0, min(ns1_end, d_master),
                '[NS]  pre-RSI  (master 0)')
    elif ns1_end < 0:
        print(f'  NOTE: RSI content starts {-ns1_end:.1f}s before master start '
              '- trimming RSI accordingly.')

    # Section 2: RSI content
    rsi_in  = trim_start + max(0.0, -ns1_end)
    rsi_dur = src_content_end - rsi_in
    if ns3_start > d_master:
        excess = ns3_start - d_master
        rsi_dur -= excess
        print(f'  NOTE: RSI extends {excess:.1f}s past master end - trimming.')
    new_seg(src_file, src_stream, rsi_in, rsi_dur, '[RSI] commentary')

    # Section 3: NS from where RSI content ends through master end
    ns3_dur = d_master - max(ns3_start, 0)
    if ns3_dur > 0:
        new_seg(master_file, ns_stream, max(ns3_start, 0), ns3_dur,
                f'[NS]  post-RSI (master {max(ns3_start, 0):.1f}s)')

    if dry_run:
        print(f'\n[DRY RUN] Would concatenate {len(segs)} segments -> {output_mka}')
        return

    concat_segments_to_mka(segs, output_mka, list_path_prefix='_tmp_rsi')

    for f in tmp_dir.iterdir():
        f.unlink()
    tmp_dir.rmdir()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Sync RSI Motociclismo audio to a MotoGP master file.')
    parser.add_argument('--mode', choices=['sunday', 'saturday'], required=True,
                        help='sunday: prerace_sting.wav (5-20 min into file); '
                             'saturday: prerace_sting_motogp.wav (35-50 min before end).')
    parser.add_argument('--dry-run', action='store_true',
                        help='Detect and plan only; do not encode.')
    parser.add_argument('--trim-start', type=float, default=0.0, metavar='S',
                        help='Opening sting duration to replace with NS (seconds).')
    parser.add_argument('--trim-end', type=float, default=0.0, metavar='S',
                        help='Closing sting duration to replace with NS (seconds).')
    parser.add_argument('--offset', type=float, default=None, metavar='S',
                        help='Direct sync offset (master_time = src_time + offset). '
                             'Skips all sting detection.')
    parser.add_argument('--src-sting-time', type=float, default=None, metavar='S',
                        help='Override: known sting/anchor time in source (seconds).')
    parser.add_argument('--master-sting-time', type=float, default=None, metavar='S',
                        help='Override: known sting/anchor time in master (seconds).')
    parser.add_argument('rsi_file')
    parser.add_argument('master_file')
    parser.add_argument('output_dir')
    args = parser.parse_args()

    if args.dry_run:
        print('[DRY RUN] Detection and segment planning only - no audio will be encoded.')

    cfg = STING_CFG[args.mode]
    fp_sting = str(FP_DIR / cfg['fp'])
    if not Path(fp_sting).exists():
        sys.exit(f'ERROR: Missing fingerprint file: {fp_sting}')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_mka = Path(args.output_dir) / (Path(args.rsi_file).stem + '_rsi_synced.mka')

    d_src    = get_duration(args.rsi_file)
    d_master = get_duration(args.master_file)
    print(f'RSI:    {d_src:.1f}s ({d_src/3600:.2f}h)')
    print(f'Master: {d_master:.1f}s ({d_master/3600:.2f}h)')
    if args.trim_start or args.trim_end:
        print(f'Trim:   start={args.trim_start:.1f}s  end={args.trim_end:.1f}s')

    n_audio   = get_audio_stream_count(args.master_file)
    ns_stream = f'0:a:{n_audio - 1}'
    print(f'Master audio tracks: {n_audio}  ->  Natural Sounds on {ns_stream}')

    # Compute source search window
    src_window = cfg['src_window']
    if src_window is None:
        # saturday: 35-50 min before end
        src_start = max(0.0, d_src - 3000)
        src_window = (src_start, 900)

    # ── Sync anchor ──
    if args.offset is not None:
        src_sting    = 0.0
        master_sting = args.offset
        print(f'\nManual offset: {args.offset:.3f}s  (master_time = rsi_time + {args.offset:.3f})')
    else:
        # Find sting in master
        if args.master_sting_time is not None:
            master_sting = args.master_sting_time
            print(f'\nMaster anchor: {master_sting:.3f}s (manual override)')
        else:
            print(f'\nLocating {cfg["fp"]} in master...')
            m_start, m_dur = cfg['master_window']
            master_sting, master_conf = find_sting(
                args.master_file, fp_sting,
                m_start, m_dur, stream_spec=ns_stream,
                label=f'  Sting (master)')
            if master_conf < cfg['conf_thresh']:
                sys.exit(f'ERROR: Sting not found in master (conf={master_conf:.4f}). '
                         f'Use --master-sting-time to override.')

        # Find sting in RSI source
        if args.src_sting_time is not None:
            src_sting = args.src_sting_time
            print(f'Source anchor: {src_sting:.3f}s (manual override)')
        else:
            print(f'\nLocating {cfg["fp"]} in RSI source...')
            src_sting, src_conf = find_sting(
                args.rsi_file, fp_sting,
                src_window[0], src_window[1], stream_spec='0:a:0',
                label='  Sting (RSI)')
            if src_conf < cfg['conf_thresh']:
                sys.exit(f'ERROR: Sting not found in RSI source (conf={src_conf:.4f}). '
                         f'Use --src-sting-time to override.')

    print('\nBuilding output segments...')
    build_and_concat(
        args.rsi_file, args.master_file,
        src_sting, master_sting,
        args.trim_start, args.trim_end,
        d_src, d_master,
        '0:a:0', ns_stream,
        output_mka, dry_run=args.dry_run)

    print(f'\nDone -> {output_mka}')


if __name__ == '__main__':
    main()
