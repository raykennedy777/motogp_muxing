# MotoGP Muxing — Project Guide

## Audio Track Priority Order
When muxing final files, include tracks in this order (skip any that don't exist):

1. World Feed (English)
2. TNT Sports (English)
3. DAZN (Spanish)
4. ESPN (Spanish)
5. Sky Sport MotoGP (Italian)
6. RSI (Italian)
7. Canal+ (French)
8. RTBF (French)
9. RTS (French)
10. Sky Sport (German)
11. ServusTV (German)
12. SRF (German)
13. Sport TV (Portuguese)
14. ESPN4 (Portuguese)
15. Ziggo (Dutch)
16. Polsat Sport (Polish)
17. evgenymotogp (Russian)
18. Natural Sounds (undetermined)

---

## Sunday Workflow

### Manual steps
1. Record from Polsat
2. Record from Sky Sport MotoGP Italia
3. Download web videos from RFM
4. Download DAZN videos from Yaster
5. Download smcgill1969 TNT releases
6. Download catchup for: Canal+, RTBF (MotoGP only), RTS (MotoGP only), sky_sport_mix_de, ServusTV, Sport TV, Ziggo

### Scripted steps
7. Create master from Polsat combined with web videos *(manual)*
8. Make the web master — `python -m src.core.web_combine`
9. Create synced audio from each channel:
   - `python -m src.sync.tnt`
   - `python -m src.sync.dazn`
   - `python -m src.sync.canal`
   - `python -m src.sync.sky_it`
   - `python -m src.sync.sky_de`
   - `python -m src.sync.rtbf`
   - `python -m src.sync.rts`
   - `python -m src.sync.rsi`
   - `python -m src.sync.sporttv`
   - `python -m src.sync.servustv`
   - `python -m src.sync.ziggo`
10. Create combined master (mux all synced audio tracks together)
11. Split into individual races — `python -m src.core.mux_and_split`

### Which master to sync against
The **Class** column in the round notes determines which master each source syncs against:
- **Combined** — sync against `web_master.mkv` (the combined web master from `web_combine.py`)
- **Moto3 / Moto2 / MotoGP** (individual class) — sync against the corresponding individual web file for that class

---

## Saturday Workflow
Same as Sunday except:
- Only one race, so web masters do **not** need to be combined
- No splitting at the end — just the final muxing step

---

## Ad detection
Notes about ad detection are in "C:\Users\raisi\Desktop\WIP\notebooks\motogp_encoding\motogp_encoding\2026\ad_breaks"

---

## Key Scripts

| Script | Purpose |
|---|---|
| `src/core/web_combine.py` | Build web master from combined sources |
| `src/sync/tnt.py` | Sync TNT Sports audio (lead-in + lead-out sting pairs) |
| `src/sync/dazn.py` | Sync DAZN audio (lead-in only; break-end via sliding correlation) |
| `src/sync/canal.py` | Sync Canal+ French audio (4 race modes: sprint, moto3, moto2, motogp) |
| `src/sync/sky_it.py` | Sync Sky Sport MotoGP Italian audio (sting-pair + PUBBLICÀ text + MGP logo watermark) |
| `src/sync/sky_de.py` | Sync Sky Sport German audio (sting-based sync with lead-in/lead-out) |
| `src/sync/rtbf.py` | Sync RTBF TIPIK Belgian audio (sting pairs + watermark detection) |
| `src/sync/rts.py` | Sync RTS Swiss audio (optional opening/closing trim; sting-based sync) |
| `src/sync/rsi.py` | Sync RSI Italian audio |
| `src/sync/sporttv.py` | Sync Sport TV Portuguese audio (preshow intro sting + watermark break detection) |
| `src/sync/servustv.py` | Sync ServusTV German audio |
| `src/sync/ziggo.py` | Sync Ziggo Dutch audio |
| `src/core/mux_and_split.py` | Final mux and smartcut-based split into individual races |
| `src/sync/sky_templates.py` | Extract watermark templates for Sky Sport (MGP logo and PUBBLICÀ text) |
| `src/core/detect_race_start.py` | Detect race start and first camera change (prototype) |

## Utility Modules (src/utils/)
| Module | Contents |
|---|---|
| `audio_utils.py` | `get_duration`, `get_audio_stream_count`, `extract_wav`, `extract_seg`, `load_fp_wav`, `_peak`, `concat_segments_to_mka` |
| `sting_detection.py` | `find_sting`, `find_all_transitions` |
| `watermark_detection.py` | `get_video_dimensions`, `build_watermark_template`, `find_break_end_via_watermark` |
| `watermark_detection_canal.py` | Canal+ specific watermark detection |
| `canal_watermark.py` | Canal+ watermark utilities |

---

## Fingerprints
All fingerprint WAV/PNG/MKA clips live in `fingerprints/`. 

### Core Stings
| File | Used by | Notes |
|---|---|---|
| `prerace_sting.wav` | all sync scripts, `mux_and_split.py` | Shared Moto3/Moto2 pre-race sting |
| `prerace_sting_motogp.wav` | all sync scripts, `mux_and_split.py` | MotoGP 65s pre-race sting |

### TNT Sports
| File | Used by | Notes |
|---|---|---|
| `tnt_leadin.wav` | `tnt.py` | TNT ad break lead-in |
| `tnt_leadin_alt.wav` | `tnt.py` | Alternate TNT lead-in |
| `tnt_leadout.wav` | `tnt.py` | TNT ad break lead-out |
| `tnt_program_intro_2025.wav` | `tnt.py` | 2025 TNT return-from-break sting |

### DAZN
| File | Used by | Notes |
|---|---|---|
| `dazn_leadin.wav` | `dazn.py` | DAZN ad break lead-in |
| `dazn_leadin_alt.wav` | `dazn.py` | Alternate DAZN lead-in |
| `dazn_showintro.wav` | `dazn.py` | DAZN return-from-break sting |
| `dazn_showintro_alt.wav` | `dazn.py` | Alternate DAZN return sting |
| `dazn_end_sting.wav` | `dazn.py` | DAZN end-of-programme sting |

### Canal+
| File | Used by | Notes |
|---|---|---|
| `canal_opening.wav` | `canal.py` | Canal+ post-break opening (10s) |
| `canal_grid_ending.wav` | `canal.py` | Canal+ grid/pre-race outro (27s) |
| `canal_zarco_ad.wav` | `canal.py` | Canal+ Zarco ad watermark |
| `mgp_logo_wm_ref.png` | `canal.py`, `watermark_detection_canal.py` | MGP logo watermark template |

### Sky Sport MotoGP (IT)
| File | Used by | Notes |
|---|---|---|
| `sky_it_leadin.wav` | `sky_it.py`, `sky_templates.py` | Sky IT ad break lead-in/lead-out sting |
| `sky_it_mgp.png` | `sky_it.py`, `sky_templates.py` | Sky IT MGP logo watermark template |
| `sky_it_pubb.png` | `sky_it.py`, `sky_templates.py` | Sky IT PUBBLICÀ text template |

### Sky Sport (DE)
| File | Used by | Notes |
|---|---|---|
| `sky_de_leadout.wav` | `sky_de.py` | Sky DE ad break lead-out sting |

### RTBF
| File | Used by | Notes |
|---|---|---|
| `rtbf_tipik_leadin.wav` | `rtbf.py` | RTBF TIPIK ad break lead-in |
| `rtbf_tipik_leadout.wav` | `rtbf.py` | RTBF TIPIK ad break lead-out |
| `rtbf_tipik_wm_ref.png` | `rtbf.py` | RTBF TIPIK watermark template |

### RTS
| File | Used by | Notes |
|---|---|---|
| `rts_sprint_opening.wav` | `rts.py` | RTS sprint show opening (22s) |
| `rts_sprint_closing.wav` | `rts.py` | RTS sprint show closing (12s) |

### Sport TV
| File | Used by | Notes |
|---|---|---|
| `sporttv_leadin.mka` | `sporttv.py` | Sport TV ad break lead-in (MKA format) |
| `preshow_intro_m2m3.wav` | `sporttv.py` | Sport TV preshow intro sting |

### Ziggo
| File | Used by | Notes |
|---|---|---|
| `ziggo_leadin.wav` | `ziggo.py` | Ziggo ad break lead-in |

---

## Reporting Conventions
- Always report timestamps and durations in `hh:mm:ss` format (not raw seconds).

---

## Race Start Detection (BETA — not ready for production)

`src/core/detect_race_start.py` is a prototype that detects race start and the first camera change following it.

**Pipeline:**
1. **Sting detection** — finds `prerace_sting_motogp.wav` in the video (28-34 min window)
2. **Starting lights detection** — tracks 5 red lights in PiP overlay going out
3. **Camera change detection** — frame-to-frame correlation at 50 fps

**How it works:**
- After finding the sting, searches for the starting lights PiP overlay in a 2.5-5.5 minute window
- The 5 lights are tracked as red pixels in the overlay region
- Race start = moment when all 5 lights go out after a sustained period (3+ consecutive frames)
- Camera cut = frame where correlation drops below 0.3 (unambiguous at 50fps)

**Known limitations:**
- The PiP light positions are hardcoded for the current broadcast layout
- **Will likely need adaptation** if lights are in a different position on screen
- Different broadcasters or seasons may use different overlay graphics
- Test on new footage to verify light coordinates are correct

**Light position coordinates (640x360 scaled frame):**
- Y range: 74-90
- X ranges for 5 lights: (26,31), (39,43), (50,54), (62,65), (72,75)

**Tested accuracy (2026 Round 03 USA, MWR release):**
| Detection | Result | Ground Truth | Error |
|-----------|--------|--------------|-------|
| Sting | 00:29:00.61 | 00:28:59.46 | +1.15s |
| Race start | 00:33:35.01 | 00:33:35.00 | +0.01s |
| Camera cut | Frame 101103 | Frame 101104 | -1 frame |
