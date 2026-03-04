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
12. Sport TV (Portuguese)
13. ESPN4 (Portuguese)
14. Ziggo (Dutch)
15. Polsat Sport (Polish)
16. Natural Sounds (undetermined)

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

---

## Saturday Workflow
Same as Sunday except:
- Only one race, so web masters do **not** need to be combined
- No splitting at the end — just the final muxing step

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
| `mux_and_split.py` | Final mux and smartcut-based split into individual races |

## Fingerprints
All fingerprint WAV clips live in `fingerprints/`. See `memory/MEMORY.md` for a full list.
