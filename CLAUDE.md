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
17. Natural Sounds (undetermined)

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
8. Make the web master — `web_combine.py`
9. Create synced audio from each channel — `sync_tnt.py`, `sync_dazn.py`, `sync_canal.py`, `sync_sky.py`, `sync_rts.py`, etc.
10. Create combined master (mux all synced audio tracks together)
11. Split into individual races — `mux_and_split.py`

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
| `web_combine.py` | Build web master from combined sources |
| `sync_tnt.py` | Sync TNT Sports audio (lead-in + lead-out sting pairs) |
| `sync_dazn.py` | Sync DAZN audio (lead-in only; break-end via sliding correlation) |
| `sync_canal.py` | Sync Canal+ French audio (4 race modes: sprint, moto3, moto2, motogp) |
| `sync_sky.py` | Sync Sky Sport Italian audio (no ad breaks; sting-based sync) |
| `sync_rts.py` | Sync RTS audio (optional opening/closing trim; sting-based sync) |
| `sync_sporttv.py` | Sync Sport TV Portuguese audio (preshow intro sting + watermark break detection) |
| `mux_and_split.py` | Final mux and smartcut-based split into individual races |

## Fingerprints
All fingerprint WAV clips live in `fingerprints/`. See `memory/MEMORY.md` for a full list.

## Reporting Conventions
- Always report timestamps and durations in `hh:mm:ss` format (not raw seconds).

---

## Race Start Detection (BETA — not ready for production)

`detect_race_start.py` is a prototype that detects race start and the first camera change following it.

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
