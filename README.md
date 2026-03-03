# motogp_muxing

Scripts for processing MotoGP broadcast audio — replacing ad breaks with Natural Sounds from the web master feed, then muxing and splitting a multi-audio combined broadcast into per-race MKVs.

## Workflow

```
Polsat broadcast (MKV)  ──┐
Web master (MKV)         ──┤  sync_tnt.py / sync_dazn.py
TNT Sports broadcast     ──┘       │
DAZN broadcast (MKV)   ────────────┘
        │
        ▼
  tnt_*_synced.mka
  dazn_*_synced.mka
        │
        ├── web_combine.py  (optional: build web_master from three separate rips)
        │
        ▼
  mux_and_split.py
        │
        ├── Phase 1: detect Moto3 / Moto2 / MotoGP split points via pre-race stings
        ├── Phase 2: mkvmerge — polsat video + 5 audio tracks → combined_master.mkv
        └── Phase 3: smartcut — frame-accurate split → 3 race MKVs
```

## Scripts

### `sync_tnt.py`
Syncs a TNT Sports broadcast to the Polsat master timeline.
Detects lead-in / lead-out ad-break stings, replaces ad segments with Natural Sounds audio from the web master, and exports a synced MKA.

```bash
python sync_tnt.py polsat_master.mkv tnt_broadcast.mkv web_master.mkv tnt_synced.mka
```

### `sync_dazn.py`
Same as `sync_tnt.py` but for DAZN broadcasts.
Uses lead-in sting only; break-end is detected via sliding cross-correlation against the web master (no lead-out sting available on DAZN).

```bash
python sync_dazn.py polsat_master.mkv dazn_broadcast.mkv web_master.mkv dazn_synced.mka
```

### `web_combine.py`
Builds a single continuous web master MKV from three separate web-rip files (Moto3, Moto2, MotoGP).
Syncs each file to the Polsat broadcast timeline via the pre-race world-feed sting, inserting black+silent gap segments between races to match the broadcast gaps exactly.

```bash
python web_combine.py polsat.mkv moto3.mkv moto2.mkv motogp.mkv web_master.mkv
```

### `mux_and_split.py`
Main end-to-end script.
1. **Detects split points** in the combined master using pre-race sting fingerprints and frame-count anchoring (MotoGP end is pinned to the final frame of the broadcast).
2. **Muxes** polsat video + World Feed / TNT Sports / DAZN / Polsat Sport / Natural Sounds audio into `combined_master.mkv`.
3. **Splits** into three per-race MKVs using [smartcut](https://github.com/skeskinen/smartcut) — only the partial GOP at each cut boundary is re-encoded; the rest is stream-copied.

```bash
python mux_and_split.py \
  --round 1 --session Race \
  --wip-dir /path/to/wip \
  --downloads-dir /path/to/webdl \
  [--output-dir /path/to/out] \
  [--keep-combined] \
  [--dry-run]
```

`--dry-run` runs only Phase 1 and prints the split plan without writing any files.

Output filenames follow the convention:
```
{Class}.2026.Round{NN}.{Country}.{Session}.Polsat.HDTV.{resolution}.{codec}.Multi5.mkv
```

### `2026_mux.ps1`
PowerShell helper for muxing a **single** already-split file.
Auto-detects the master and web files in the current directory, resolves codec/resolution, and calls `mkvmerge` with the full 5-track audio layout.

```powershell
.\2026_mux.ps1 -Class MotoGP -Round 1 -Session Sprint
```

## Fingerprints

Place WAV clips in `fingerprints/`:

| File | Used by | Notes |
|---|---|---|
| `prerace_sting.wav` | `mux_and_split.py`, `sync_*.py` | Shared Moto3/Moto2 pre-race sting |
| `prerace_sting_motogp.wav` | `mux_and_split.py` | MotoGP-specific pre-race sting |
| `tnt_leadin.wav` | `sync_tnt.py` | TNT ad break lead-in |
| `tnt_leadin_alt.wav` | `sync_tnt.py` | Alternate TNT lead-in |
| `tnt_leadout.wav` | `sync_tnt.py` | TNT ad break lead-out |
| `dazn_leadin.wav` | `sync_dazn.py` | DAZN ad break lead-in |
| `dazn_leadin_alt.wav` | `sync_dazn.py` | Alternate DAZN lead-in (optional) |
| `dazn_end_sting.wav` | `sync_dazn.py` | DAZN end-of-programme sting |

## Requirements

```
ffmpeg / ffprobe  (on PATH)
mkvmerge          (MKVToolNix, on PATH)
pip install numpy scipy smartcut
```

Python 3.11+.
