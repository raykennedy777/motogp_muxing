# motogp_muxing

Scripts for processing MotoGP broadcast audio — syncing commentary from multiple broadcasters to a Polsat master timeline, then muxing and splitting into per-race MKVs.

## Architecture

```
src/
├── core/              # Main workflow scripts
│   ├── web_combine.py
│   ├── mux_and_split.py
│   └── detect_race_start.py
├── sync/              # Channel-specific sync scripts
│   ├── tnt.py
│   ├── dazn.py
│   ├── canal.py
│   ├── sky_it.py
│   ├── sky_de.py
│   ├── rtbf.py
│   ├── rts.py
│   ├── rsi.py
│   ├── sporttv.py
│   ├── servustv.py
│   ├── ziggo.py
│   └── sky_templates.py
└── utils/             # Shared utility modules
    ├── audio_utils.py
    ├── sting_detection.py
    ├── watermark_detection.py
    ├── watermark_detection_canal.py
    └── canal_watermark.py
```

### Utility Modules (src/utils/)

| Module | Contents |
|---|---|
| `audio_utils.py` | `get_duration`, `get_audio_stream_count`, `extract_wav`, `extract_seg`, `load_fp_wav` (WAV + MKA), `_peak`, `concat_segments_to_mka` |
| `sting_detection.py` | `find_sting`, `find_all_transitions` |
| `watermark_detection.py` | `get_video_dimensions`, `build_watermark_template`, `find_break_end_via_watermark` |
| `watermark_detection_canal.py` | Canal+ specific watermark detection |
| `canal_watermark.py` | Canal+ watermark utilities |

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
  Ziggo / RSI / etc.
        |
        +-- python -m src.core.web_combine    (build web master from per-race web rips)
        |
        +-- python -m src.sync.tnt      --+
        +-- python -m src.sync.dazn       |
        +-- python -m src.sync.canal      |  produce *_synced.mka per broadcaster
        +-- python -m src.sync.sky_it     |
        +-- python -m src.sync.sky_de     |
        +-- python -m src.sync.rtbf       |
        +-- python -m src.sync.rts        |
        +-- python -m src.sync.rsi        |
        +-- python -m src.sync.sporttv    |
        +-- python -m src.sync.servustv   |
        +-- python -m src.sync.ziggo    --+
                |
                v
        python -m src.core.mux_and_split
                |
                +-- Phase 1: detect Moto3 / Moto2 / MotoGP split points via pre-race stings
                +-- Phase 2: mkvmerge -- polsat video + all synced audio -> combined_master.mkv
                +-- Phase 3: smartcut -- frame-accurate split -> 3 per-race MKVs
```

### Saturday (sprint day)
Same as Sunday except web masters don't need combining (single race) and no splitting at the end — just the final mux.

### Audio track order (per CLAUDE.md/AGENTS.md)
World Feed · TNT Sports · DAZN · ESPN · Sky Sport MotoGP · RSI · Canal+ · RTBF · RTS · Sky Sport (DE) · ServusTV · SRF · Sport TV · ESPN4 · Ziggo · Polsat Sport · evgenymotogp · Natural Sounds

---

## Scripts

### `python -m src.core.web_combine`
Builds a single continuous web master MKV from three separate web-rip files (Moto3, Moto2, MotoGP).
Syncs each file to the Polsat broadcast timeline via the pre-race world-feed sting, inserting black+silent gap segments between races to match the broadcast gaps exactly.

```bash
python -m src.core.web_combine polsat.mkv moto3.mkv moto2.mkv motogp.mkv web_master.mkv
```

### `python -m src.sync.tnt`
Syncs a TNT Sports broadcast to the web master timeline.
Detects lead-in / lead-out ad-break stings, replaces ad segments with Natural Sounds from the web master, and exports a synced MKA.

```bash
python -m src.sync.tnt [--dry-run] <tnt_file> <web_master.mkv> <output_dir>
```

### `python -m src.sync.dazn`
Syncs a DAZN broadcast. Uses lead-in sting only; break-end is detected via the show-intro return sting, then watermark reappearance in video, with a fixed-duration fallback. A 7s end-of-programme sting marks the handoff back to Natural Sounds.

```bash
python -m src.sync.dazn [--dry-run] [--mgp-sting-dazn=S] <dazn_file> <web_master.mkv> <output_dir>
```

`--mgp-sting-dazn=S` manually sets the position of the 65s MotoGP intro sting in DAZN time (seconds) when auto-detection fails.

### `python -m src.sync.canal`
Syncs Canal+ French commentary. Supports four race modes with different break structures and sync anchors.

```bash
python -m src.sync.canal --race {sprint|moto3|moto2|motogp} [--dry-run] \
  <canal_file> <master_file> <output_dir>
```

### `python -m src.sync.sky_it`
Syncs Sky Sport MotoGP Italian commentary. Uses a three-tier break detection strategy:
1. **Sting-pair detection** — same fingerprint marks lead-in and lead-out of each ad break
2. **PUBBLICÀ text detection** — white overlay at break start (template-matched)
3. **MGP logo watermark** — reappearance marks break end

```bash
python -m src.sync.sky_it [--dry-run] --anchor-source=S --anchor-master=S \
  <sky_it_file> <master.mkv> <output_dir>
```

### `python -m src.sync.sky_de`
Syncs Sky Sport German commentary. Uses lead-in/lead-out sting pairs for break detection.

```bash
python -m src.sync.sky_de [--dry-run] --anchor-source=S --anchor-master=S \
  <sky_de_file> <master.mkv> <output_dir>
```

### `python -m src.sync.rtbf`
Syncs RTBF TIPIK Belgian commentary. Individual race files (not combined). Uses sting pairs + watermark detection (TIPIK logo top-right, MGP logo bottom-right).

```bash
python -m src.sync.rtbf [--dry-run] [--anchor-source=S] [--anchor-master=S] [--race={moto3|moto2|motogp}] \
  <rtbf_file> <master_file> <output_dir>
```

### `python -m src.sync.rts`
Syncs RTS Swiss commentary. Optionally trims a fixed duration from the opening and/or closing of the broadcast before syncing.

```bash
python -m src.sync.rts [--dry-run] [--trim-start S] [--trim-end S] \
  <rts_file> <master.mkv> <output_dir>
```

### `python -m src.sync.rsi`
Syncs RSI Italian commentary.

```bash
python -m src.sync.rsi [--dry-run] [--trim-start S] [--trim-end S] \
  <rsi_file> <master.mkv> <output_dir>
```

### `python -m src.sync.sporttv`
Syncs Sport TV Portuguese commentary. Detects the preshow intro sting first, then searches for the Moto3 sting. Break-end hierarchy: preshow/MotoGP return stings → watermark reappearance → fixed fallback.

```bash
python -m src.sync.sporttv [--dry-run] [--moto3-time=S] \
  <sporttv_file> <web_master.mkv> <output_dir>
```

`--moto3-time=S` manually sets the Moto3 sting position in Sport TV time (seconds) when auto-detection fails.

### `python -m src.sync.servustv`
Syncs ServusTV German commentary.

```bash
python -m src.sync.servustv [--dry-run] [--anchor-source=S] [--anchor-master=S] \
  <servustv_file> <master.mkv> <output_dir>
```

### `python -m src.sync.ziggo`
Syncs Ziggo Dutch commentary. Uses lead-in sting detection for ad breaks.

```bash
python -m src.sync.ziggo [--dry-run] --anchor-source=S --anchor-master=S \
  <ziggo_file> <master.mkv> <output_dir>
```

### `python -m src.core.mux_and_split`
Main end-to-end script.
1. **Detects split points** in the combined master using pre-race sting fingerprints and frame-count anchoring (MotoGP end is pinned to the final frame of the broadcast).
2. **Muxes** polsat video + all synced audio tracks into `combined_master.mkv`.
3. **Splits** into three per-race MKVs using [smartcut](https://github.com/skeskinen/smartcut) — only the partial GOP at each cut boundary is re-encoded; the rest is stream-copied. A `mkvpropedit` pass corrects fps metadata in the output files.

```bash
python -m src.core.mux_and_split \
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
| `tnt_leadin.wav` | `tnt.py` | TNT ad break lead-in |
| `tnt_leadin_alt.wav` | `tnt.py` | Alternate TNT lead-in (optional) |
| `tnt_leadout.wav` | `tnt.py` | TNT ad break lead-out |
| `tnt_program_intro_2025.wav` | `tnt.py` | 2025 TNT return-from-break sting (optional) |
| `dazn_leadin.wav` | `dazn.py` | DAZN ad break lead-in |
| `dazn_leadin_alt.wav` | `dazn.py` | Alternate DAZN lead-in (optional) |
| `dazn_showintro.wav` | `dazn.py` | DAZN return-from-break sting |
| `dazn_showintro_alt.wav` | `dazn.py` | Alternate DAZN return sting (optional) |
| `dazn_end_sting.wav` | `dazn.py` | DAZN end-of-programme sting |
| `canal_opening.wav` | `canal.py` | Canal+ post-break opening (10s) |
| `canal_grid_ending.wav` | `canal.py` | Canal+ grid/pre-race outro (27s) |
| `canal_zarco_ad.wav` | `canal.py` | Canal+ Zarco ad watermark |
| `sky_it_leadin.wav` | `sky_it.py`, `sky_templates.py` | Sky IT ad break lead-in/lead-out sting |
| `sky_it_mgp.png` | `sky_it.py`, `sky_templates.py` | Sky IT MGP logo watermark template |
| `sky_it_pubb.png` | `sky_it.py`, `sky_templates.py` | Sky IT PUBBLICÀ text template |
| `sky_de_leadout.wav` | `sky_de.py` | Sky DE ad break lead-out sting |
| `rtbf_tipik_leadin.wav` | `rtbf.py` | RTBF TIPIK ad break lead-in |
| `rtbf_tipik_leadout.wav` | `rtbf.py` | RTBF TIPIK ad break lead-out |
| `rtbf_tipik_wm_ref.png` | `rtbf.py` | RTBF TIPIK watermark template |
| `rts_sprint_opening.wav` | `rts.py` | RTS sprint show opening (22s) |
| `rts_sprint_closing.wav` | `rts.py` | RTS sprint show closing (12s) |
| `sporttv_leadin.mka` | `sporttv.py` | Sport TV ad break lead-in (MKA format) |
| `preshow_intro_m2m3.wav` | `sporttv.py` | Sport TV preshow intro sting |
| `ziggo_leadin.wav` | `ziggo.py` | Ziggo ad break lead-in |
| `mgp_logo_wm_ref.png` | `canal.py`, `watermark_detection_canal.py` | MGP logo watermark template |

---

## Template Extraction

### `python -m src.sync.sky_templates`
Extract watermark templates for Sky Sport broadcasts. Run once per broadcast season to generate `fingerprints/sky_it_mgp.png` and `fingerprints/sky_it_pubb.png`.

```bash
python -m src.sync.sky_templates <sky_video.mkv> [--output-dir fingerprints/]
```

---

## Race Start Detection (Prototype)

### `python -m src.core.detect_race_start`
BETA prototype that detects race start and the first camera change following it. Not ready for production.

**Pipeline:**
1. **Sting detection** — finds `prerace_sting_motogp.wav` in the video
2. **Starting lights detection** — tracks 5 red lights in PiP overlay going out
3. **Camera change detection** — frame-to-frame correlation

---

## Requirements

```
ffmpeg / ffprobe  (on PATH)
mkvmerge          (MKVToolNix, on PATH)
mkvpropedit       (MKVToolNix, on PATH)
pip install numpy scipy smartcut
```

Python 3.11+.

---

## Project Guide

See `AGENTS.md` for detailed workflow documentation, audio track priority order, and reporting conventions.
