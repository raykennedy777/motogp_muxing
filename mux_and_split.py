#!/usr/bin/env python3
"""
mux_and_split.py
Mux polsat_master video with all synced audio tracks into combined_master.mkv,
then split into 3 race MKVs (Moto3 / Moto2 / MotoGP) whose durations match the
WEB-DL references frame-accurately.

Phase 1 — detect split points via pre-race sting fingerprints
Phase 2 — mkvmerge: polsat video + combined audio tracks
Phase 3 — frame-accurate split (smart-cut: re-encode partial GOP only)
Phase 4 — per-race remux inserting individual tracks + priority reordering

Usage:
    python mux_and_split.py --round N --session Race|Sprint
        --wip-dir DIR [--downloads-dir DIR] [--output-dir DIR]
        [--keep-combined] [--dry-run]
"""

import argparse, gc, json, subprocess, sys
from pathlib import Path
import numpy as np

from audio_utils import get_duration, extract_wav
from sting_detection import CONF_THRESH, find_sting

# ── Constants ─────────────────────────────────────────────────────────────────
MOTO3_STING_SEARCH   = (600, 1200)   # (start_s, dur_s) — 10-30 min
MOTO3_TO_MOTO2_SECS  = 4500          # ~1h15m between race stings
MOTO2_TO_MOTOGP_SECS = 6240          # ~1h44m between race stings
STING_SEARCH_MARGIN  = 60            # ±60 s around expected sting time

# WEB-DL reference durations (fallback when files not available)
MOTO3_WEBDL_DUR      = 4413.800   # seconds
MOTO2_WEBDL_DUR      = 4961.000
MOTOGP_WEBDL_DUR     = 6491.060

FPS                  = 50        # broadcast frame rate
MOTOGP_END_TOLERANCE = 5.0       # cross-check tolerance (seconds)

FP_DIR = Path(__file__).parent / 'fingerprints'

# Video frame matching (for Moto2 / MotoGP start refinement)
VIDEO_ANCHOR_T      = 0.5    # seconds into WEB-DL to sample reference frame
VIDEO_MATCH_W       = 128    # downsample width for comparison
VIDEO_MATCH_H       = 72     # downsample height
VIDEO_SEARCH_MARGIN = 2.0    # ±seconds to search around audio estimate
VIDEO_MIN_QUALITY   = 0.3    # minimum normalised-SSD quality to accept

# Audio track priority order (see CLAUDE.md). Matched against track names
# set by mux_combined / remux_with_individual_tracks.
TRACK_PRIORITY = [
    'World Feed', 'TNT Sports', 'DAZN', 'Sky Sport MotoGP', 'RSI',
    'Canal+', 'RTBF', 'RTS', 'Sky Sport', 'ServusTV', 'SRF',
    'Sport TV', 'Ziggo', 'Polsat Sport', 'evgenymotogp', 'Natural Sounds',
]

CALENDARS = {
    2026: {
         1: 'Thailand',        2: 'Brazil',          3: 'USA',           4: 'Qatar',
         5: 'Spain.Jerez',     6: 'France',           7: 'Spain.Catalunya', 8: 'Italy',
         9: 'Hungary',        10: 'Czechia',          11: 'Netherlands',  12: 'Germany',
        13: 'Britain',        14: 'Spain.Aragon',     15: 'SanMarino',    16: 'Austria',
        17: 'Japan',          18: 'Indonesia',        19: 'Australia',    20: 'Malaysia',
        21: 'Portugal',       22: 'Spain.Valencia',
    },
    # 2025: rounds 16-22 confirmed from broadcast files.
    # Rounds 1-15 are approximate — verify and correct for any round you process.
    2025: {
         1: 'Thailand',        2: 'Argentina',        3: 'USA',           4: 'Spain.Jerez',
         5: 'France',          6: 'Italy',             7: 'Spain.Catalunya', 8: 'Germany',
         9: 'Netherlands',    10: 'Britain',           11: 'Austria',      12: 'Czechia',
        13: 'Spain.Aragon',   14: 'Hungary',           15: 'Emilia-Romagna', 16: 'SanMarino',
        17: 'Japan',          18: 'Indonesia',        19: 'Australia',    20: 'Malaysia',
        21: 'Portugal',       22: 'Spain.Valencia',
    },
}


# ── ffprobe helpers ───────────────────────────────────────────────────────────

def get_frame_count(f):
    """
    Return the video frame count.  Reads stream metadata first (O(1));
    falls back to counting packets (reads headers, no decode) if the
    metadata field is absent.
    """
    r = subprocess.run([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_frames',
        '-of', 'default=noprint_wrappers=1:nokey=1', str(f),
    ], capture_output=True, text=True, check=True)
    val = r.stdout.strip()
    if val and val != 'N/A':
        return int(val)
    # Fallback: count packets (no decode needed)
    r = subprocess.run([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-count_packets', '-show_entries', 'stream=nb_read_packets',
        '-of', 'default=noprint_wrappers=1:nokey=1', str(f),
    ], capture_output=True, text=True, check=True)
    return int(r.stdout.strip())


# ── Video frame matching ─────────────────────────────────────────────────────

def _extract_single_frame(src, t):
    """Extract one frame at time t as a (VIDEO_MATCH_H, VIDEO_MATCH_W) uint8 array."""
    r = subprocess.run([
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-ss', f'{t:.3f}', '-i', str(src),
        '-frames:v', '1',
        '-vf', f'scale={VIDEO_MATCH_W}:{VIDEO_MATCH_H},format=gray',
        '-f', 'rawvideo', 'pipe:',
    ], capture_output=True, check=True)
    data = np.frombuffer(r.stdout, dtype=np.uint8)
    expected = VIDEO_MATCH_W * VIDEO_MATCH_H
    return data.reshape(VIDEO_MATCH_H, VIDEO_MATCH_W) if len(data) == expected else None


def _extract_frame_window(src, t_start, duration):
    """Extract frames for [t_start, t_start+duration) as (N, H, W) uint8 array."""
    r = subprocess.run([
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-ss', f'{max(0.0, t_start):.3f}', '-i', str(src),
        '-t', f'{duration:.3f}',
        '-vf', f'scale={VIDEO_MATCH_W}:{VIDEO_MATCH_H},format=gray',
        '-f', 'rawvideo', 'pipe:',
    ], capture_output=True, check=True)
    data = np.frombuffer(r.stdout, dtype=np.uint8)
    n = len(data) // (VIDEO_MATCH_W * VIDEO_MATCH_H)
    return data[:n * VIDEO_MATCH_W * VIDEO_MATCH_H].reshape(
        n, VIDEO_MATCH_H, VIDEO_MATCH_W) if n > 0 else None


def find_video_anchor(search_src, webdl, t_approx, label=''):
    """
    Refine t_approx (estimated webdl t=0 position in search_src) using video
    frame matching.  Extracts a reference frame at VIDEO_ANCHOR_T seconds into
    the WEB-DL and finds the closest match in search_src within
    ±VIDEO_SEARCH_MARGIN of (t_approx + VIDEO_ANCHOR_T).
    Returns refined start time, or t_approx unchanged on failure.
    """
    ref = _extract_single_frame(webdl, VIDEO_ANCHOR_T)
    if ref is None:
        print(f'  [{label}] WARNING: could not extract reference frame; '
              f'keeping audio estimate')
        return t_approx

    t_search_start = t_approx + VIDEO_ANCHOR_T - VIDEO_SEARCH_MARGIN
    search_dur     = VIDEO_SEARCH_MARGIN * 2 + 1.0 / FPS

    window = _extract_frame_window(search_src, t_search_start, search_dur)
    if window is None or len(window) == 0:
        print(f'  [{label}] WARNING: could not extract search window; '
              f'keeping audio estimate')
        return t_approx

    ref_f   = ref.astype(np.float32)
    ssd     = np.sum((window.astype(np.float32) - ref_f) ** 2, axis=(1, 2))
    best    = int(np.argmin(ssd))
    quality = 1.0 - float(ssd[best]) / (float(np.mean(ssd)) + 1e-6)

    t_anchor = t_search_start + best / FPS
    t_start  = t_anchor - VIDEO_ANCHOR_T
    delta_s  = t_start - t_approx
    print(f'  [{label}] video anchor: quality={quality:.3f}  '
          f'delta={delta_s:+.3f}s ({delta_s * FPS:+.1f} frames)')

    if quality < VIDEO_MIN_QUALITY:
        print(f'  [{label}] WARNING: low quality match; keeping audio estimate')
        return t_approx

    return t_start


# ── mkvmerge track identification ────────────────────────────────────────────

def identify_tracks(path):
    """Return (video_tracks, audio_tracks) from mkvmerge --identify."""
    r = subprocess.run(
        ['mkvmerge', '--identify', '--identification-format', 'json', str(path)],
        capture_output=True, text=True, check=True)
    tracks = json.loads(r.stdout).get('tracks', [])
    return (
        [t for t in tracks if t['type'] == 'video'],
        [t for t in tracks if t['type'] == 'audio'],
    )


# ── File auto-detection ───────────────────────────────────────────────────────

def find_file(directory, *globs):
    """Return first match for any glob pattern in directory, or None."""
    for pattern in globs:
        matches = sorted(Path(directory).glob(pattern))
        if matches:
            return matches[0]
    return None


def find_webdl(downloads_dir, cls, round_num, session):
    """Recursively find WEB-DL MKV for a given class/round/session."""
    nn = f'{round_num:02d}'
    matches = sorted(Path(downloads_dir).glob(f'**/*{cls}*Round{nn}*{session}*.mkv'))
    return matches[0] if matches else None


# ── Phase 1: Detect split points ─────────────────────────────────────────────

def detect_split_points(web_master, polsat_master, moto2_webdl, motogp_webdl):
    """
    Locate split points in the master.
    Moto2  — audio sting gives coarse position; video frame matching refines it.
    MotoGP — end-anchor (master_dur - motogp_dur) is primary; sting detection
             and video frame matching are used as cross-checks / refinements.
    Returns (moto2_start, motogp_start, moto2_dur, motogp_dur).
    """
    fp_sting    = str(FP_DIR / 'prerace_sting.wav')
    fp_sting_gp = str(FP_DIR / 'prerace_sting_motogp.wav')

    # ── Moto3 and Moto2 stings in web master ──
    print('\nLocating pre-race stings in web master (Natural Sounds)...')
    m3_master, _ = find_sting(web_master, fp_sting,
                               *MOTO3_STING_SEARCH, stream_spec='0:a:1',
                               label='Moto3 sting (master)')
    m2_master, _ = find_sting(web_master, fp_sting,
                               m3_master + MOTO3_TO_MOTO2_SECS - STING_SEARCH_MARGIN,
                               STING_SEARCH_MARGIN * 2, stream_spec='0:a:1',
                               label='Moto2 sting (master)')

    # ── Moto2 start: audio sting → coarse, video matching → frame-accurate ──
    if moto2_webdl:
        print('\nLocating pre-race sting in Moto2 WEB-DL...')
        m2_webdl, m2_conf = find_sting(str(moto2_webdl), fp_sting,
                                        *MOTO3_STING_SEARCH,
                                        label='Moto2 sting (WEB-DL)')
        if m2_conf < CONF_THRESH:
            print(f'  WARNING: Low confidence ({m2_conf:.4f}); assuming sting at t=0')
            m2_webdl = 0.0
    else:
        print('  WARNING: No Moto2 WEB-DL found; assuming sting at t=0')
        m2_webdl = 0.0

    moto2_start = m2_master - m2_webdl
    if moto2_start < 0:
        print(f'  WARNING: moto2_start={moto2_start:.3f}s is negative; clamping to 0')
        moto2_start = 0.0

    if moto2_webdl:
        print('\n  Refining Moto2 start with video frame matching...')
        moto2_start = find_video_anchor(str(polsat_master), str(moto2_webdl),
                                        moto2_start, label='Moto2')

    # ── MotoGP start: count backwards from end of polsat_master ──
    master_frames = get_frame_count(polsat_master)
    if motogp_webdl:
        motogp_frames = get_frame_count(str(motogp_webdl))
        motogp_dur    = motogp_frames / FPS
    else:
        motogp_frames = round(MOTOGP_WEBDL_DUR * FPS)
        motogp_dur    = MOTOGP_WEBDL_DUR

    motogp_start_frame = master_frames - motogp_frames
    motogp_start       = motogp_start_frame / FPS
    print(f'\nMotoGP frame-count anchor: '
          f'master={master_frames} frames  webdl={motogp_frames} frames  '
          f'-> start=frame {motogp_start_frame} = {motogp_start:.3f}s '
          f'({motogp_start / 60:.1f} min)')

    if motogp_start_frame < 0:
        sys.exit(f'ERROR: motogp_start_frame={motogp_start_frame} is negative — '
                 f'WEB-DL ({motogp_frames} frames) is longer than the master '
                 f'({master_frames} frames)')

    # Sanity-check with sting detection (informational only)
    mgp_search_start = max(0.0, motogp_start - STING_SEARCH_MARGIN)
    mgp_master, mgp_conf = find_sting(web_master, fp_sting_gp,
                                       mgp_search_start, STING_SEARCH_MARGIN * 2,
                                       stream_spec='0:a:1',
                                       label='MotoGP sting (sanity check)')
    if mgp_conf >= CONF_THRESH:
        drift = abs(mgp_master - motogp_start)
        print(f'  Sting drift from frame-count anchor: {drift:.3f}s '
              f'({drift * FPS:.1f} frames)')
        if drift > MOTOGP_END_TOLERANCE:
            print(f'  WARNING: drift > {MOTOGP_END_TOLERANCE}s — confirm '
                  f'polsat_master ends at the same frame as the WEB-DL')
    else:
        print(f'  Sting not found clearly (conf={mgp_conf:.4f}); '
              f'frame-count anchor stands')

    moto2_frames = get_frame_count(str(moto2_webdl)) if moto2_webdl else round(MOTO2_WEBDL_DUR * FPS)
    moto2_dur    = moto2_frames / FPS
    return moto2_start, motogp_start, moto2_dur, motogp_dur, moto2_frames, motogp_frames


# ── Phase 2: Mux with mkvmerge ────────────────────────────────────────────────

def mux_combined(polsat_master, web_master, combined_tracks, output):
    """
    Combine polsat_master video with arbitrary audio tracks into combined_master.mkv.

    combined_tracks: list of (path, track_name, language) tuples for synced MKA files.
    These tracks are inserted between World Feed and Polsat/Natural Sounds.

    Track order: video | World Feed | [combined_tracks...] | Polsat | Natural Sounds
    Returns (codec_str, resolution_str) detected from polsat_master.
    """
    pol_vids, pol_audios = identify_tracks(polsat_master)
    _,        web_audios = identify_tracks(web_master)

    if not pol_vids:
        sys.exit(f'ERROR: No video track found in {polsat_master}')
    if not pol_audios:
        sys.exit(f'ERROR: No audio track found in {polsat_master}')
    if len(web_audios) < 2:
        sys.exit(f'ERROR: Need ≥2 audio tracks in {web_master} '
                 f'(World Feed + Natural Sounds); found {len(web_audios)}')

    vid_id = pol_vids[0]['id']
    pol_id = pol_audios[0]['id']
    wf_id  = web_audios[0]['id']
    ns_id  = web_audios[1]['id']

    # Detect codec + resolution for output filename
    props  = pol_vids[0].get('properties', {})
    dims   = props.get('pixel_dimensions', '1920x1080')
    try:
        height = int(dims.split('x')[1])
    except (IndexError, ValueError):
        height = 1080
    codec_raw  = pol_vids[0].get('codec', '')
    codec      = 'x265' if 'HEVC' in codec_raw else 'x264'
    resolution = f'{height}p'

    print(f'\nMuxing -> {output.name}')
    print(f'  polsat video track {vid_id}  ({codec} {resolution})')
    print(f'  web_master: WF={wf_id}  NS={ns_id}   polsat audio: Polsat={pol_id}')
    for path, name, lang in combined_tracks:
        print(f'  {path.name}: {name} ({lang})')

    mkv = ['mkvmerge', '--output', str(output)]

    # Input 0 – polsat_master: video + chapters only
    mkv += ['--no-audio', '--no-subtitles', str(polsat_master)]

    # Input 1 – web_master: World Feed (eng, default)
    mkv += ['--no-video', '--no-subtitles', '--no-chapters',
            '--audio-tracks',  str(wf_id),
            '--track-name',    f'{wf_id}:World Feed',
            '--language',      f'{wf_id}:eng',
            '--default-track', f'{wf_id}:yes',
            str(web_master)]

    # Inputs 2..N – combined synced audio tracks
    track_order_parts = [f'0:{vid_id}', f'1:{wf_id}']
    for i, (path, name, lang) in enumerate(combined_tracks):
        inp = 2 + i
        mkv += ['--no-video', '--no-subtitles', '--no-chapters',
                '--audio-tracks',  '0',
                '--track-name',    f'0:{name}',
                '--language',      f'0:{lang}',
                '--default-track', '0:no',
                str(path)]
        track_order_parts.append(f'{inp}:0')

    # Input after combined – polsat_master: Polsat Sport audio (pol)
    polsat_audio_inp = 2 + len(combined_tracks)
    mkv += ['--no-video', '--no-subtitles', '--no-chapters',
            '--audio-tracks',  str(pol_id),
            '--track-name',    f'{pol_id}:Polsat Sport',
            '--language',      f'{pol_id}:pol',
            '--default-track', f'{pol_id}:no',
            str(polsat_master)]
    track_order_parts.append(f'{polsat_audio_inp}:{pol_id}')

    # Final input – web_master: Natural Sounds (und)
    ns_inp = polsat_audio_inp + 1
    mkv += ['--no-video', '--no-subtitles', '--no-chapters',
            '--audio-tracks',  str(ns_id),
            '--track-name',    f'{ns_id}:Natural Sounds',
            '--language',      f'{ns_id}:und',
            '--default-track', f'{ns_id}:no',
            str(web_master)]
    track_order_parts.append(f'{ns_inp}:{ns_id}')

    mkv += ['--track-order', ','.join(track_order_parts)]

    subprocess.run(mkv, check=True)
    return codec, resolution


# ── Phase 3: Frame-accurate splitting ────────────────────────────────────────

def smart_cut(src, output, t_start, webdl_dur, codec_str, exclude_audio=None):
    """
    Extract [t_start, t_start + webdl_dur] from src using the smartcut library.
    smartcut re-encodes only the partial GOPs at cut boundaries; everything
    else is stream-copied.  codec_str is 'x264' or 'x265' (from mux_combined).

    exclude_audio: optional list of 0-based audio track indices to drop from the
    output file (e.g. [1, 3] drops TNT and Polsat from a 5-track combined master).
    smartcut always cuts with all tracks; excluded tracks are stripped afterwards
    by a fast mkvmerge remux.
    """
    from fractions import Fraction
    from smartcut.media_container import MediaContainer
    from smartcut.cut_video import smart_cut as _sc, VideoSettings
    from smartcut.media_utils import VideoExportMode, VideoExportQuality
    from smartcut.misc_data import AudioExportSettings, AudioExportInfo

    t_end = t_start + webdl_dur
    print(f'  smartcut: {t_start:.3f}s - {t_end:.3f}s  ({webdl_dur:.3f}s)')

    exclude_set = set(exclude_audio or [])
    sc_output   = output.parent / ('_sc_tmp_' + output.name) if exclude_set else output

    container = MediaContainer(str(src))
    segments  = [(Fraction(t_start), Fraction(t_end))]

    sc_codec  = 'hevc' if codec_str == 'x265' else 'h264'
    vid_set   = VideoSettings(
        mode=VideoExportMode.SMARTCUT,
        quality=VideoExportQuality.NORMAL,
        codec_override=sc_codec,
    )

    n_audio  = len(container.audio_tracks)
    aud_info = AudioExportInfo(
        output_tracks=[AudioExportSettings(codec='passthru')] * n_audio,
    )

    exc = _sc(
        media_container=container,
        positive_segments=segments,
        out_path=str(sc_output),
        audio_export_info=aud_info,
        video_settings=vid_set,
    )
    if exc is not None:
        raise exc

    if exclude_set:
        # mkvmerge --audio-tracks uses absolute track IDs (as shown by mkvmerge -i),
        # not 0-based audio-type indices. Query the actual IDs from the file.
        _, audio_tracks = identify_tracks(sc_output)
        keep_ids = [t['id'] for i, t in enumerate(audio_tracks) if i not in exclude_set]
        excl_ids = [t['id'] for i, t in enumerate(audio_tracks) if i in exclude_set]
        keep_str = ','.join(str(i) for i in keep_ids)
        print(f'  Dropping audio idx {sorted(exclude_set)} (absolute IDs {excl_ids}); '
              f'keeping absolute IDs {keep_ids} -> {output.name}')
        subprocess.run(
            ['mkvmerge', '-o', str(output),
             '--audio-tracks', keep_str, str(sc_output)],
            check=True)
        sc_output.unlink()

    # Fix DefaultDuration: smartcut's partial-GOP re-encoder can write incorrect
    # fps into the H.264 SPS, causing tools like TMPGenc to misread the frame rate.
    # mkvpropedit patches the MKV track header without touching the video bitstream.
    dur_ns = round(1_000_000_000 / FPS)
    subprocess.run(
        ['mkvpropedit', str(output),
         '--edit', 'track:v1',
         '--set', f'default-duration={dur_ns}'],
        check=True)


# ── Phase 4: Remux split intermediate with individual tracks ──────────────────

def remux_with_individual_tracks(split_file, output, individual_tracks):
    """
    Remux a split intermediate file, inserting individual audio tracks
    and reordering all audio by TRACK_PRIORITY.

    split_file: intermediate MKV from smart_cut (video + combined audio tracks)
    output: final output path
    individual_tracks: list of (path, track_name, language) tuples
    """
    vid_tracks, aud_tracks = identify_tracks(split_file)
    if not vid_tracks:
        sys.exit(f'ERROR: No video track in {split_file}')

    vid_id = vid_tracks[0]['id']

    # Sort key: position in TRACK_PRIORITY, fallback 99
    def _sort_key(name):
        for i, key in enumerate(TRACK_PRIORITY):
            if key in name:
                return i
        return 99

    # Build unified list: (is_new, path, name, lang_or_none, split_track_info)
    all_audio = []
    for t in aud_tracks:
        name = t.get('properties', {}).get('track_name', '')
        all_audio.append(('split', None, name, None, t))
    for path, name, lang in individual_tracks:
        all_audio.append(('new', path, name, lang, None))

    # Sort everything together by TRACK_PRIORITY
    all_audio.sort(key=lambda x: _sort_key(x[2]))

    mkv = ['mkvmerge', '--output', str(output)]

    # Input 0 – split file: video + all combined audio tracks
    mkv += ['--no-subtitles', '--no-chapters', str(split_file)]

    track_order_parts = [f'0:{vid_id}']
    new_input_idx = 1
    for kind, path, name, lang, t_info in all_audio:
        if kind == 'split':
            track_order_parts.append(f'0:{t_info["id"]}')
        else:
            mkv += ['--no-video', '--no-subtitles', '--no-chapters',
                    '--audio-tracks',  '0',
                    '--track-name',    f'0:{name}',
                    '--language',      f'0:{lang}',
                    '--default-track', '0:no',
                    str(path)]
            track_order_parts.append(f'{new_input_idx}:0')
            new_input_idx += 1

    mkv += ['--track-order', ','.join(track_order_parts)]

    n_new = sum(1 for k, *_ in all_audio if k == 'new')
    print(f'  Remuxing with {n_new} individual tracks -> {output.name}')
    subprocess.run(mkv, check=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Mux + split MotoGP 2026 Sunday broadcast into race MKVs.')
    ap.add_argument('--round',         type=int, required=False, metavar='N',
                    help='Round number 1–22')
    ap.add_argument('--session',       required=False, choices=['Race', 'Sprint'])
    ap.add_argument('--wip-dir',       required=True, metavar='DIR',
                    help='Directory with polsat_master, web_master, synced MKAs')
    ap.add_argument('--downloads-dir', metavar='DIR',
                    help='Directory tree containing WEB-DL subfolders')
    ap.add_argument('--output-dir',    metavar='DIR',
                    help='Output directory (default: wip-dir)')
    ap.add_argument('--keep-combined',   action='store_true',
                    help='Keep combined_master.mkv after splitting')
    ap.add_argument('--combined-master', metavar='FILE',
                    help='Use existing combined MKV; skip the mux phase')
    ap.add_argument('--combined-audio-dir', metavar='DIR',
                    help='Directory with combined synced MKA files (default: wip-dir/combined_audio)')
    ap.add_argument('--individual-audio-dir', metavar='DIR',
                    help='Directory with per-race synced MKA files (default: wip-dir/individual_audio)')
    ap.add_argument('--year',            type=int, default=2026,
                    help='Broadcast year for output filenames (default: 2026)')
    ap.add_argument('--dry-run',         action='store_true',
                    help='Detect split points and print plan; no files written')
    ap.add_argument('--mux-only',        action='store_true',
                    help='Mux combined_master.mkv only; skip detection and splitting')
    ap.add_argument('--exclude-audio',   type=int, nargs='+', metavar='N',
                    help='0-based audio track indices to drop from split output files '
                         '(e.g. --exclude-audio 1 3 drops tracks 1 and 3; '
                         'does not affect combined_master.mkv)')
    args = ap.parse_args()

    wip_dir = Path(args.wip_dir)
    out_dir = Path(args.output_dir) if args.output_dir else wip_dir

    # ── Auto-detect source files ──
    polsat_master = find_file(wip_dir,
                              '*polsat*master*.mkv', '*polsat*master*.mp4',
                              '*master*.mkv')
    web_master    = find_file(wip_dir,
                              '*web*master*.mkv', '*web*master*.mp4',
                              '*web*.mkv')

    # ── Auto-detect combined audio tracks ──
    combined_audio_dir = Path(args.combined_audio_dir) if args.combined_audio_dir else wip_dir / 'combined_audio'
    combined_tracks = _detect_combined_tracks(combined_audio_dir) if combined_audio_dir.exists() else []

    # ── Auto-detect individual audio tracks ──
    individual_audio_dir = Path(args.individual_audio_dir) if args.individual_audio_dir else wip_dir / 'individual_audio'
    individual_tracks_by_race = _detect_individual_tracks(individual_audio_dir) if individual_audio_dir.exists() else {}

    # ── --mux-only: skip detection and splitting, just produce combined_master.mkv ──
    if args.mux_only:
        if not polsat_master:
            sys.exit(f'ERROR: Cannot find polsat_master in {wip_dir}')
        if not web_master:
            sys.exit(f'ERROR: Cannot find web_master in {wip_dir}')
        if not combined_tracks:
            sys.exit(f'ERROR: No combined audio tracks found in {combined_audio_dir}')
        print(f'  polsat_master : {polsat_master.name}')
        print(f'  web_master    : {web_master.name}')
        for path, name, lang in combined_tracks:
            print(f'  combined: {path.name} -> {name} ({lang})')
        out_dir.mkdir(parents=True, exist_ok=True)
        combined = out_dir / 'combined_master.mkv'
        mux_combined(polsat_master, web_master, combined_tracks, combined)
        print(f'\nDone -> {combined}')
        return

    round_num = args.round
    session   = args.session
    if not round_num or not session:
        sys.exit('ERROR: --round and --session are required unless --mux-only is set')
    calendar  = CALENDARS.get(args.year)
    if calendar is None:
        sys.exit(f'ERROR: No calendar for year {args.year} (supported: {sorted(CALENDARS)})')
    country   = calendar.get(round_num)
    if not country:
        sys.exit(f'ERROR: Round {round_num} not in {args.year} calendar (valid: 1-22)')
    round_str = f'{round_num:02d}'

    # ── Fingerprint check ──
    fp_sting    = FP_DIR / 'prerace_sting.wav'
    fp_sting_gp = FP_DIR / 'prerace_sting_motogp.wav'
    for fp in (fp_sting, fp_sting_gp):
        if not fp.exists():
            sys.exit(f'ERROR: Missing fingerprint: {fp}')

    skip_mux = args.combined_master is not None
    if not polsat_master:
        sys.exit(f'ERROR: Cannot find polsat_master in {wip_dir}')
    if not web_master:
        sys.exit(f'ERROR: Cannot find web_master in {wip_dir}')
    if not skip_mux and not combined_tracks:
        sys.exit(f'ERROR: No combined audio tracks found in {combined_audio_dir}')

    print(f'Round {round_str} {country} — {session}')
    print(f'  polsat_master : {polsat_master.name}')
    print(f'  web_master    : {web_master.name}')
    print(f'  combined tracks: {len(combined_tracks)}')
    for path, name, lang in combined_tracks:
        print(f'    {path.name} -> {name} ({lang})')

    # ── Auto-detect WEB-DL files ──
    moto3_webdl = moto2_webdl = motogp_webdl = None
    if args.downloads_dir:
        dl_dir       = Path(args.downloads_dir)
        moto3_webdl  = find_webdl(dl_dir, 'Moto3',  round_num, session)
        moto2_webdl  = find_webdl(dl_dir, 'Moto2',  round_num, session)
        motogp_webdl = find_webdl(dl_dir, 'MotoGP', round_num, session)
        print(f'  moto3_webdl   : {moto3_webdl.name  if moto3_webdl  else "not found"}')
        print(f'  moto2_webdl   : {moto2_webdl.name  if moto2_webdl  else "not found"}')
        print(f'  motogp_webdl  : {motogp_webdl.name if motogp_webdl else "not found"}')

    moto3_frames = get_frame_count(moto3_webdl) if moto3_webdl else round(MOTO3_WEBDL_DUR * FPS)
    moto3_dur    = moto3_frames / FPS

    # ── Phase 1: Detect split points ──
    moto2_start, motogp_start, moto2_dur, motogp_dur, moto2_frames, motogp_frames = \
        detect_split_points(str(web_master), polsat_master, moto2_webdl, motogp_webdl)

    print(f'\n-- Split plan -------------------------------------------')
    print(f'  Moto3 : t=0             dur={moto3_dur:.3f}s  {moto3_frames} frames')
    print(f'  Moto2 : t={moto2_start:.3f}s  dur={moto2_dur:.3f}s  {moto2_frames} frames')
    print(f'  MotoGP: t={motogp_start:.3f}s  dur={motogp_dur:.3f}s  {motogp_frames} frames')

    if args.dry_run:
        print('\n[DRY RUN] Detection complete — no files written.')
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 2: Mux (or use existing combined master) ──
    if skip_mux:
        combined = Path(args.combined_master)
        if not combined.exists():
            sys.exit(f'ERROR: --combined-master not found: {combined}')
        # Detect codec + resolution from the combined file
        pol_vids, _ = identify_tracks(combined)
        if not pol_vids:
            sys.exit(f'ERROR: No video track in {combined}')
        props      = pol_vids[0].get('properties', {})
        dims       = props.get('pixel_dimensions', '1920x1080')
        try:
            height = int(dims.split('x')[1])
        except (IndexError, ValueError):
            height = 1080
        codec_raw  = pol_vids[0].get('codec', '')
        codec      = 'x265' if 'HEVC' in codec_raw else 'x264'
        resolution = f'{height}p'
        print(f'\nUsing existing combined master: {combined.name}  ({codec} {resolution})')
    else:
        combined = out_dir / 'combined_master.mkv'
        codec, resolution = mux_combined(
            polsat_master, web_master, combined_tracks, combined)

    # Total audio track count from split intermediate: WF + combined_tracks + Polsat + NS
    n_combined_audio = len(combined_tracks) + 3  # +3 for World Feed + Polsat + Natural Sounds

    # ── Phase 3: Split ──
    exclude_audio = args.exclude_audio or []
    if exclude_audio:
        print(f'\nExcluding audio tracks {exclude_audio} from split output files.')

    intermediates = []
    for cls, t_start, dur, frames in (
        ('Moto3',  0.0,          moto3_dur,  moto3_frames),
        ('Moto2',  moto2_start,  moto2_dur,  moto2_frames),
        ('MotoGP', motogp_start, motogp_dur, motogp_frames),
    ):
        # Intermediate file for smart_cut
        intermediate = out_dir / f'_intermediate_{cls}.mkv'
        print(f'\n-- {cls}: t={t_start:.3f}s  {frames} frames')
        print(f'   -> splitting to intermediate: {intermediate.name}')
        smart_cut(combined, intermediate, t_start, dur, codec, exclude_audio)
        intermediates.append((cls, intermediate, frames))

    # ── Phase 4: Remux with individual tracks ──
    for cls, intermediate, frames in intermediates:
        race_individual = individual_tracks_by_race.get(cls, [])
        n_total_audio = n_combined_audio + len(race_individual)

        out_file = out_dir / (
            f'{cls}.{args.year}.Round{round_str}.{country}.{session}.'
            f'Polsat.HDTV.{resolution}.{codec}.Multi{n_total_audio}.mkv'
        )

        print(f'\n-- {cls}: remux with {len(race_individual)} individual tracks')
        print(f'   -> {out_file.name}')
        remux_with_individual_tracks(intermediate, out_file, race_individual)
        intermediate.unlink()  # clean up intermediate

    # Optionally delete combined master (never delete a user-provided file)
    if not skip_mux and not args.keep_combined and combined.exists():
        gc.collect()  # release PyAV file handles held by MediaContainer
        print(f'\nDeleting {combined.name}')
        combined.unlink()

    print('\nDone.')


# ── Audio track auto-detection helpers ────────────────────────────────────────

def _detect_combined_tracks(directory):
    """
    Scan directory for synced MKA files that span all 3 races.
    Returns list of (path, track_name, language) tuples in priority order.
    """
    # Priority-ordered patterns: (glob_patterns, track_name, language)
    patterns = [
        (['*dazn*synced*.mka', '*Races*synced*.mka'], 'DAZN', 'spa'),
        (['*sky_it*synced*.mka', '*sky_sport_it*synced*.mka'], 'Sky Sport MotoGP', 'ita'),
        (['*rsi*synced*.mka'], 'RSI', 'ita'),
        (['*canal*synced*.mka', '*canalplus*synced*.mka'], 'Canal+', 'fra'),
        (['*rtbf*synced*.mka'], 'RTBF', 'fra'),
        (['*sky_de*synced*.mka', '*sky_sport_de*synced*.mka'], 'Sky Sport', 'deu'),
        (['*ziggo*synced*.mka'], 'Ziggo', 'nld'),
        (['*sporttv*synced*.mka'], 'Sport TV', 'por'),
        (['*servustv*synced*.mka', '*stv*synced*.mka'], 'ServusTV', 'deu'),
    ]

    tracks = []
    for globs, name, lang in patterns:
        for pattern in globs:
            matches = sorted(directory.glob(pattern))
            if matches:
                tracks.append((matches[0], name, lang))
                break
    return tracks


def _detect_individual_tracks(directory):
    """
    Scan directory for per-race synced MKA files.
    Returns dict: {'Moto3': [...], 'Moto2': [...], 'MotoGP': [...]}
    Each entry is a list of (path, track_name, language) tuples.
    """
    result = {'Moto3': [], 'Moto2': [], 'MotoGP': []}

    all_files = list(directory.glob('*.mka'))

    # TNT (all races, one file each with race number prefix)
    for f in all_files:
        if '01.Race.Moto3' in f.name or (f.name.startswith('01') and 'tnt' in f.name.lower()):
            result['Moto3'].append((f, 'TNT Sports', 'eng'))
        elif '02.Race.Moto2' in f.name or (f.name.startswith('02') and 'tnt' in f.name.lower()):
            result['Moto2'].append((f, 'TNT Sports', 'eng'))
        elif '03.Race.MotoGP' in f.name or (f.name.startswith('03') and 'tnt' in f.name.lower()):
            result['MotoGP'].append((f, 'TNT Sports', 'eng'))

    # RTBF (Moto2 only for this round)
    for f in all_files:
        if 'rtbf' in f.name.lower() and 'moto2' in f.name.lower():
            result['Moto2'].append((f, 'RTBF', 'fra'))

    # RTS (MotoGP)
    for f in all_files:
        if 'rts' in f.name.lower() and 'moto' in f.name.lower() and 'rtbf' not in f.name.lower():
            if 'Motorrad' not in f.name:  # SRF files have "Motorrad" in name
                result['MotoGP'].append((f, 'RTS', 'fra'))

    # SRF (MotoGP)
    for f in all_files:
        if 'Motorrad' in f.name and 'rts' in f.name.lower():
            result['MotoGP'].append((f, 'SRF', 'deu'))

    # ServusTV (all races)
    for f in all_files:
        if 'servustv' in f.name.lower() or 'stv' in f.name.lower():
            if 'Moto3' in f.name:
                result['Moto3'].append((f, 'ServusTV', 'deu'))
            elif 'Moto2' in f.name:
                result['Moto2'].append((f, 'ServusTV', 'deu'))
            elif 'MotoGP' in f.name:
                result['MotoGP'].append((f, 'ServusTV', 'deu'))

    # evgenymotogp / Russian (all races)
    for f in all_files:
        if 'rus' in f.name.lower() or 'evgeny' in f.name.lower():
            if 'Moto3' in f.name:
                result['Moto3'].append((f, 'evgenymotogp', 'rus'))
            elif 'Moto2' in f.name:
                result['Moto2'].append((f, 'evgenymotogp', 'rus'))
            elif 'MotoGP' in f.name:
                result['MotoGP'].append((f, 'evgenymotogp', 'rus'))

    return result


if __name__ == '__main__':
    main()
