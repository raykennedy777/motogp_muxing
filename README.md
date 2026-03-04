# motogp_muxing

Scripts for processing MotoGP broadcast audio — syncing commentary from multiple broadcasters to a Polsat master timeline, then muxing and splitting into per-race MKVs.

## Workflow

### Sunday (race day)

```
Polsat broadcast          ──┐
Sky Sport MotoGP Italia   ──┤
Web videos (RFM)          ──┤  manual acquisition
DAZN (via Yaster)         ──┤
TNT Sports (smcgill1969)  ──┤
Canal+ / RTBF / RTS       ──┤
sky_sport_mix_de          ──┤
ServusTV / Sport TV /     ──┘
  Ziggo / etc.
        │
        ├── web_combine.py     (build web master from per-race web rips)
        │
        ├── sync_tnt.py        ──┐
        ├── sync_dazn.py         │
        ├── sync_canal.py        │  produce *_synced.mka per broadcaster
        ├── sync_sky.py          │
        └── sync_rts.py        ──┘
                │
                ▼
        mux_and_split.py
                │
                ├── Phase 1: detect Moto3 / Moto2 / MotoGP split points via pre-race stings
                ├── Phase 2: mkvmerge — polsat video + all synced audio → combined_master.mkv
                └── Phase 3: smartcut — frame-accurate split → 3 per-race MKVs
```

### Saturday (sprint day)
Same as Sunday except web masters don't need combining (single race) and no splitting at the end — just the final mux.

### Audio track order (per CLAUDE.md)
World Feed · TNT Sports · DAZN · ESPN · Sky Sport MotoGP · RSI · Canal+ · RTBF · RTS · Sky Sport (DE) · ServusTV · Sport TV · ESPN4 · Ziggo · Polsat Sport · Natural Sounds

---

## Scripts

### `sync_tnt.py`
Syncs a TNT Sports broadcast to the Polsat master timeline.
Detects lead-in / lead-out ad-break stings, replaces ad segments with Natural Sounds from the web master, and exports a synced MKA.

```bash
python sync_tnt.py polsat_master.mkv tnt_broadcast.mkv web_master.mkv tnt_synced.mka
```

### `sync_dazn.py`
Same as `sync_tnt.py` but for DAZN broadcasts.
Uses lead-in sting only; break-end is detected via sliding cross-correlation against the web master (no lead-out sting available on DAZN).

```bash
python sync_dazn.py polsat_master.mkv dazn_broadcast.mkv web_master.mkv dazn_synced.mka
```

### `sync_canal.py`
Syncs Canal+ French commentary to the Polsat master. Supports four race modes with different break structures and sync anchors.

```bash
python sync_canal.py --race {sprint|moto3|moto2|motogp} [--dry-run] \
  canal_file master_file output_dir
```

### `sync_sky.py`
Syncs Sky Sport MotoGP Italian commentary to the Polsat master. No ad breaks — pure sting-based alignment.

```bash
python sync_sky.py [--dry-run] sky_file master_file output_dir
```

### `sync_rts.py`
Syncs RTS Swiss commentary to the Polsat master. Optionally trims a fixed amount from the opening and/or closing of the RTS broadcast before syncing.

```bash
python sync_rts.py [--dry-run] [--trim-start S] [--trim-end S] \
  rts_file master_file output_dir
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
2. **Muxes** polsat video + all synced audio tracks into `combined_master.mkv`.
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

---

## Fingerprints

Place WAV clips in `fingerprints/`:

| File | Used by | Notes |
|---|---|---|
| `prerace_sting.wav` | all `sync_*.py`, `mux_and_split.py` | Shared Moto3/Moto2 pre-race sting |
| `prerace_sting_motogp.wav` | all `sync_*.py`, `mux_and_split.py` | MotoGP 65s pre-race sting |
| `tnt_leadin.wav` | `sync_tnt.py` | TNT ad break lead-in |
| `tnt_leadin_alt.wav` | `sync_tnt.py` | Alternate TNT lead-in |
| `tnt_leadout.wav` | `sync_tnt.py` | TNT ad break lead-out |
| `dazn_leadin.wav` | `sync_dazn.py` | DAZN ad break lead-in |
| `dazn_leadin_alt.wav` | `sync_dazn.py` | Alternate DAZN lead-in (optional) |
| `dazn_end_sting.wav` | `sync_dazn.py` | DAZN end-of-programme sting |
| `canal_opening.wav` | `sync_canal.py` | Canal+ post-break opening (10s) |
| `canal_grid_ending.wav` | `sync_canal.py` | Canal+ grid/pre-race outro (27s) |
| `canal_moto2_grid_ending.wav` | `sync_canal.py` | Shorter Canal+ grid outro for Moto2 (20s) |
| `canal_moto3_sting.wav` | `sync_canal.py` | Canal+ Moto3 sting (17s) |
| `canal_sprint_podium.wav` | `sync_canal.py` | Canal+ sprint podium anchor (20s) |
| `canal_motogp_anthem.wav` | `sync_canal.py` | Canal+ MotoGP national anthem anchor (34s) |
| `rts_sprint_opening.wav` | `sync_rts.py` | RTS sprint show opening (22s) |
| `rts_sprint_closing.wav` | `sync_rts.py` | RTS sprint show closing (12s) |

---

## Requirements

```
ffmpeg / ffprobe  (on PATH)
mkvmerge          (MKVToolNix, on PATH)
pip install numpy scipy smartcut
```

Python 3.11+.
