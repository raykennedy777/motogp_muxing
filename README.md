# motogp_muxing

Scripts for processing MotoGP broadcast audio — syncing commentary from multiple broadcasters to a Polsat master timeline, then muxing and splitting into per-race MKVs.

## Architecture

Shared utility modules are imported by all channel scripts:

| Module | Contents |
|---|---|
| `audio_utils.py` | `get_duration`, `get_audio_stream_count`, `extract_wav`, `extract_seg`, `load_fp_wav` (WAV + MKA), `_peak`, `concat_segments_to_mka` |
| `sting_detection.py` | `find_sting`, `find_all_transitions` |
| `watermark_detection.py` | `get_video_dimensions`, `build_watermark_template`, `find_break_end_via_watermark` |

---

## Workflow

### Sunday (race day)

```
Polsat broadcast          --+
Sky Sport MotoGP Italia   --|
Web videos (RFM)          --|  manual acquisition
DAZN (via Yaster)         --|
TNT Sports (smcgill1969)  --|
Canal+ / RTBF / RTS       --|
sky_sport_mix_de          --|
ServusTV / Sport TV /     --+
  Ziggo / etc.
        |
        +-- web_combine.py      (build web master from per-race web rips)
        |
        +-- sync_tnt.py       --+
        +-- sync_dazn.py        |
        +-- sync_canal.py       |  produce *_synced.mka per broadcaster
        +-- sync_sky.py         |
        +-- sync_rts.py         |
        +-- sync_sporttv.py   --+
                |
                v
        mux_and_split.py
                |
                +-- Phase 1: detect Moto3 / Moto2 / MotoGP split points via pre-race stings
                +-- Phase 2: mkvmerge -- polsat video + all synced audio -> combined_master.mkv
                +-- Phase 3: smartcut -- frame-accurate split -> 3 per-race MKVs
```

### Saturday (sprint day)
Same as Sunday except web masters don't need combining (single race) and no splitting at the end — just the final mux.

### Audio track order (per CLAUDE.md)
World Feed · TNT Sports · DAZN · ESPN · Sky Sport MotoGP · RSI · Canal+ · RTBF · RTS · Sky Sport (DE) · ServusTV · Sport TV · ESPN4 · Ziggo · Polsat Sport · Natural Sounds

---

## Scripts

### `web_combine.py`
Builds a single continuous web master MKV from three separate web-rip files (Moto3, Moto2, MotoGP).
Syncs each file to the Polsat broadcast timeline via the pre-race world-feed sting, inserting black+silent gap segments between races to match the broadcast gaps exactly.

```bash
python web_combine.py polsat.mkv moto3.mkv moto2.mkv motogp.mkv web_master.mkv
```

### `sync_tnt.py`
Syncs a TNT Sports broadcast to the web master timeline.
Detects lead-in / lead-out ad-break stings, replaces ad segments with Natural Sounds from the web master, and exports a synced MKA.

```bash
python sync_tnt.py [--dry-run] <tnt_file> <web_master.mkv> <output_dir>
```

### `sync_dazn.py`
Syncs a DAZN broadcast. Uses lead-in sting only; break-end is detected via the show-intro return sting, then watermark reappearance in video, with a fixed-duration fallback. A 7s end-of-programme sting marks the handoff back to Natural Sounds.

```bash
python sync_dazn.py [--dry-run] [--mgp-sting-dazn=S] <dazn_file> <web_master.mkv> <output_dir>
```

`--mgp-sting-dazn=S` manually sets the position of the 65s MotoGP intro sting in DAZN time (seconds) when auto-detection fails.

### `sync_canal.py`
Syncs Canal+ French commentary. Supports four race modes with different break structures and sync anchors.

```bash
python sync_canal.py --race {sprint|moto3|moto2|motogp} [--dry-run] \
  <canal_file> <master_file> <output_dir>
```

### `sync_sky.py`
Syncs Sky Sport MotoGP Italian commentary. No ad breaks — pure sting-based alignment.

```bash
python sync_sky.py [--dry-run] <sky_file> <master.mkv> <output_dir>
```

### `sync_rts.py`
Syncs RTS Swiss commentary. Optionally trims a fixed duration from the opening and/or closing of the broadcast before syncing.

```bash
python sync_rts.py [--dry-run] [--trim-start S] [--trim-end S] \
  <rts_file> <master.mkv> <output_dir>
```

### `sync_sporttv.py`
Syncs Sport TV Portuguese commentary. Detects the preshow intro sting first, then searches for the Moto3 sting. Break-end hierarchy: preshow/MotoGP return stings → watermark reappearance → fixed fallback.

```bash
python sync_sporttv.py [--dry-run] [--moto3-time=S] \
  <sporttv_file> <web_master.mkv> <output_dir>
```

`--moto3-time=S` manually sets the Moto3 sting position in Sport TV time (seconds) when auto-detection fails.

### `mux_and_split.py`
Main end-to-end script.
1. **Detects split points** in the combined master using pre-race sting fingerprints and frame-count anchoring (MotoGP end is pinned to the final frame of the broadcast).
2. **Muxes** polsat video + all synced audio tracks into `combined_master.mkv`.
3. **Splits** into three per-race MKVs using [smartcut](https://github.com/skeskinen/smartcut) — only the partial GOP at each cut boundary is re-encoded; the rest is stream-copied. A `mkvpropedit` pass corrects fps metadata in the output files.

```bash
python mux_and_split.py \
  --round N --session Race|Sprint \
  --wip-dir DIR \
  [--downloads-dir DIR] \
  [--output-dir DIR] \
  [--year 2025|2026] \
  [--combined-master FILE] \
  [--mux-only] \
  [--keep-combined] \
  [--exclude-audio N [N ...]] \
  [--dry-run]
```

| Flag | Description |
|---|---|
| `--round N` | Round number (1–22) |
| `--session Race\|Sprint` | Session type |
| `--year 2025\|2026` | Selects the broadcast calendar for output filenames (default: 2025) |
| `--downloads-dir DIR` | Directory tree containing WEB-DL MKVs for split-point detection |
| `--combined-master FILE` | Use an existing combined MKV; skip Phase 2 mux |
| `--mux-only` | Run Phase 2 only — produce `combined_master.mkv` without detection or splitting (useful for review before committing to split) |
| `--keep-combined` | Do not delete `combined_master.mkv` after splitting |
| `--exclude-audio N ...` | Drop the specified 0-based audio track indices from the split output files (does not affect `combined_master.mkv`) |
| `--dry-run` | Run Phase 1 only and print the split plan without writing files |

**Audio track indices in combined_master.mkv** (standard 5-audio build):

| Index | Track |
|---|---|
| 0 | World Feed (English) |
| 1 | TNT Sports (English) |
| 2 | DAZN (Spanish) |
| 3 | Polsat Sport (Polish) |
| 4 | Natural Sounds |

---

## Fingerprints

Place clips in `fingerprints/`. WAV unless noted otherwise.

| File | Used by | Notes |
|---|---|---|
| `prerace_sting.wav` | all sync scripts, `mux_and_split.py` | Shared Moto3/Moto2 pre-race sting |
| `prerace_sting_motogp.wav` | all sync scripts, `mux_and_split.py` | MotoGP 65s pre-race sting |
| `tnt_leadin.wav` | `sync_tnt.py` | TNT ad break lead-in |
| `tnt_leadin_alt.wav` | `sync_tnt.py` | Alternate TNT lead-in (optional) |
| `tnt_program_intro_2025.wav` | `sync_tnt.py` | 2025 TNT return-from-break sting (optional) |
| `dazn_leadin.wav` | `sync_dazn.py` | DAZN ad break lead-in |
| `dazn_leadin_alt.wav` | `sync_dazn.py` | Alternate DAZN lead-in (optional) |
| `dazn_showintro.wav` | `sync_dazn.py` | DAZN return-from-break sting |
| `dazn_showintro_alt.wav` | `sync_dazn.py` | Alternate DAZN return sting (optional) |
| `dazn_end_sting.wav` | `sync_dazn.py` | DAZN end-of-programme sting |
| `canal_opening.wav` | `sync_canal.py` | Canal+ post-break opening (10s) |
| `canal_grid_ending.wav` | `sync_canal.py` | Canal+ grid/pre-race outro (27s) |
| `canal_moto2_grid_ending.wav` | `sync_canal.py` | Shorter Canal+ grid outro for Moto2 (20s) |
| `canal_moto3_sting.wav` | `sync_canal.py` | Canal+ Moto3 sting (17s) |
| `canal_sprint_podium.wav` | `sync_canal.py` | Canal+ sprint podium anchor (20s) |
| `canal_motogp_anthem.wav` | `sync_canal.py` | Canal+ MotoGP national anthem anchor (34s) |
| `rts_sprint_opening.wav` | `sync_rts.py` | RTS sprint show opening (22s) |
| `rts_sprint_closing.wav` | `sync_rts.py` | RTS sprint show closing (12s) |
| `sporttv_leadin.mka` | `sync_sporttv.py` | Sport TV ad break lead-in |
| `preshow_intro_m2m3.mka` | `sync_sporttv.py` | Sport TV preshow intro sting |

---

## Requirements

```
ffmpeg / ffprobe  (on PATH)
mkvmerge          (MKVToolNix, on PATH)
mkvpropedit       (MKVToolNix, on PATH)
pip install numpy scipy smartcut
```

Python 3.11+.
