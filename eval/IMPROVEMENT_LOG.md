# OSINT geolocation improvement log

Branch: `osint-autoimprove`. Author for auto-loop commits: Linda + Cursor.

## Section 4 diagnostics (before Phase 0)

### 1. Model setting in production
- `render.yaml` pins `VISION_MODEL=gpt-4o` and `VISION_PROVIDER=openai`.
- That overrides any default in `visionclient.py`. A local/env swap to another
  model does **not** reach Render until `render.yaml` / dashboard env change.
- `/analyze` responses now include `vision_provider` and `vision_model` so each
  result shows which model actually ran.

### 2. Confidence math
- `_fuse_confidence` starts at prior **0.2** (logit), then adds signed log-odds.
- Many named/water/solar/climate/ambiguity terms are zero or negative unless
  features match. A “missed” named place (−0.9) or water (−0.8) alone can crush
  an otherwise correct AI vote (`AI` weight 1.3 × (2c−1)).
- Temporary confidence `base * _STATUS_FACTOR * amb_factor` is computed, then
  **replaced** by fusion — so early multipliers only matter if fusion is skipped.

### 3. Stacked penalties
Places that write `result["confidence"]` on the vision path:
| Location | Effect |
|----------|--------|
| `_determine_core` temp | `base * status * amb` (then overwritten) |
| narrowing `+0.03` | before fuse → **lost** |
| `_apply_solar` ×0.6/0.9/+0.03 | before fuse → **lost**; status still fused |
| `_apply_climate` ×0.92/+0.02 | before fuse → **lost**; status still fused |
| `_fuse_confidence` | **final** for vision path |
| `apply_vehicle_signal` | re-fuses or ×0.85 |
| `_warn_if_synthetic` | caps at 0.15 |
| `_verify_and_adjust` | × `_STATUS_FACTOR` on some fallbacks |

**Verdict:** solar/climate **multipliers** are dead on the main path (double-
counting risk if someone moves fuse earlier). Fusion already applies their
statuses. Vehicle/synthetic can still stack after fusion.

### 4. Map verification rejecting correct answers
- `_select_location` hard-accepts only `verified`/`skipped`; mismatches are
  crossed out; partials/unavailable/mismatch used only as fallbacks.
- Scene booleans drive Overpass `expected` features; false positives (e.g.
  hallucinated `lake`) can mismatch a correct pin.
- Named place/water miss forces `mismatch` via `_augment_verification`.
- **To measure:** live eval will log `verified` + whether truth was among
  rejected hypotheses (follow-up).

### 5. Aug 14 commit `35d6476`
- On `main`: Landmark/water verification + eval + large osint changes.
- Ancestors of interest: `cd4c763` (verify-and-retry), `8330dc3` (Tier-1),
  `35d6476` (landmarks/water), then raven-parity commits on this line.
- Phase 0 will compare those SHAs on the same live set when feasible.

### 6. vision.py issues
| Issue | Status |
|-------|--------|
| MIME always jpeg | **Mostly fixed** in `visionclient._media_type` (magic bytes). |
| No resize | Still true — large images sent full-size. |
| temperature on new models | Handled for gpt-5/o-series; Gemini/Anthropic still pass temperature. |
| Empty analysis hides errors | Failed calls return note; empty JSON → empty candidates / low confidence. |
| “coordinates only when confident” | **Conflict:** early prompt text still says that; later rule says ALWAYS give lat/lon. |
| 3 candidates / 1 call | Still true. |
| Homonym geocode | Partially mitigated by `GEOCODE_MAX_DRIFT_KM` (300 km). |

---

## Phase 0

- **Dataset:** 150 Commons photos via `eval/terrain_bench` → `eval/images` +
  `eval/live_labels.json` (seed 42, dev=105, holdout=45). Categories:
  rural 35, urban 28, coast 29, mountain 31, other 27. EXIF GPS stripped.
- **Commit:** `12a3677` — `eval: add live image benchmark and baseline`
- **Baseline run 1** (`gpt-4o`, sha 12a3677, split=dev, n=105):

| metric | value |
|--------|------:|
| acc@1km | 8.6% |
| acc@25km | **61.0%** |
| acc@200km | 76.2% |
| acc@750km | 84.8% |
| median_km | 10.05 |
| country | 70.0% |
| mean_conf | **0.349** |
| ECE@25 | **0.343** |
| Brier@25 | 0.367 |

- **Calibration smoking gun:** among 64 correct@25km, mean conf=0.37;
  29/64 have conf&lt;0.2. Many are `verified=mismatch` (map/named checks
  contradict a pin that is still geographically right). Wrong@25 mean
  conf=0.31 — almost no separation.
- Older-commit A/B (cd4c763 / 8330dc3 / 35d6476) deferred until after
  calibration loops (same vision cache; OSINT-only re-score possible later).

---

## Loop log

| date | commit | hypothesis | dev metrics | holdout | decision |
|------|--------|------------|-------------|---------|----------|
| 2026-11-18 | (reverted) | Soften FUSION_MISMATCH −1.6→−0.4 | ECE 0.37→0.29, mean_conf 0.34→0.43, acc@25 61→62%, median p≈0.35 | — | **reject**: calibration better but median/acc improvement not p&lt;0.1 |
| 2026-11-18 | (reverted) | Named-place miss → partial not mismatch | acc@25 flat, acc@1 worse, still 58× mismatch status | — | **reject**: Overpass expected-feature misses dominate, not named-place |
| 2026-11-18 | (reverted) | Soft Overpass miss→partial; prefer vision_coords on ties | acc@25 flat, median 10.05→9.31 p≈0.30, ECE 0.37→0.21, mismatch→partial relabel | — | **reject**: calibration-only; median p&gt;0.1; pins barely changed |
| 2026-11-18 | (reverted) | Skip geocode when model already gave coords (H) | aborted: 66× `ThreadPoolExecutor` NameError from corrupted verify.py recovery | — | **reject**: eval invalid; restored HEAD verify/osint |
| 2026-11-18 | — | **recovery** | Accidental `git checkout` wiped uncommitted bands-based verify.py; reconstruction incomplete. Continuing from committed HEAD verify+osint; re-baselining. | — | note |
| 2026-11-18 | 12a3677 | **HEAD clean baseline** (presence Overpass, no bands WIP) | acc@25 **67.6%**, median 8.47, ECE 0.254, mean_conf 0.428, @1 8.6%, country 66.7% | — | new reference (WIP bands verify was worse: ~61% @25) |
| 2026-11-18 | (reverted) | Skip geocode when model already gave coords | identical to baseline (0 pin changes); Overpass mostly `unavailable` this run | — | **reject**: no effect while map verify is down |
| 2026-11-18 | (reverted) | Resize vision upload long side to 2048 (G) | all 150 bench images already ≤1933px; no-op; aborted before API rebill | — | **reject**: no effect on this dataset |
| 2026-11-18 | (reverted) | Prefer working Overpass mirrors first (osm.ch); richer UA | still 66× unavailable (parallel prefetch trips breaker); acc@25 flat | — | **reject**: connectivity order alone insufficient under parallel load |
| 2026-11-18 | (reverted) | Serialize Overpass + osm.ch first + softer breaker | with working Overpass: acc@25 67.6→61.9%, ECE worse, 16 pin flips (7↑/9↓); hard verify rejects good pins | — | **reject**: restoring map verify without softening selection hurts accuracy |
| 2026-11-18 | (reverted) | Soft OSM miss→partial + score-all selection + reliable Overpass + softer MISMATCH | acc@25 67.6→61.0%, ECE 0.25→0.21 | — | **reject**: score-all still lets map displace good AI pins |
| 2026-11-18 | (pending commit) | AI-order select (named-miss veto only) + soft MISMATCH (−1.6→−0.6) + reliable Overpass | acc@25/median flat; ECE 0.254→0.220; mean_conf 0.43→0.46; 1 pin improved; no threshold regression | — | **accept**: accuracy preserved, calibration better, Overpass usable without displacing AI pins |

---

## Session note (2026-11-18) — resumed after 8 rejects

### Start metrics (Phase 0 WIP baseline, then HEAD clean)
| set | acc@25 | median_km | ECE | mean_conf | country |
|-----|-------:|----------:|----:|----------:|--------:|
| Phase0 WIP verify (dev) | 61.0% | 10.05 | 0.34 | 0.35 | ~70% |
| **HEAD clean reference** | **67.6%** | **8.47** | **0.25** | **0.43** | 67% |

### Accepted changes
- AI-order selection + soft MISMATCH + reliable Overpass (see loop log).

### What we learned
1. **Production still pins `gpt-4o`** via `render.yaml`; confidence collapse is mostly fusion + verify status, not the model swap.
2. **Uncommitted distance-band verify WIP was worse than HEAD** (~61% vs ~67% @25). Do not restore bands without soft scoring.
3. **Overpass was effectively off during “good” HEAD baseline** (`.env` listed dead mirrors first → 66× `unavailable`). When Overpass actually answers, hard mismatch/early-accept **drops** acc@25 by ~6pp.
4. Vision oracle: any-cand@25 ≈73%, top1≈65%; only ~6 pipeline errors have a better non-top cand — selection headroom is small; ~26 errors need better vision.
5. Calibration-only fixes (softer FUSION_MISMATCH) improve ECE but fail the acc/median p&lt;0.1 gate.

### Next (for Linda / next session)
1. **Soft selection with working Overpass:** never early-accept on weak verified; treat empty OSM hits as soft partial (hypothesis C) — must pair reliability + soft reject.
2. **Fit fusion weights on dev** (hypothesis A) — may need gate exception for calibration-only if acc flat.
3. **Prompt: 5 cands + clues-before-answer** (D) for the 26 vision misses — costs API; run 3×.
4. Keep `.env` / `OVERPASS_URL` with `overpass.osm.ch` first locally (gitignored); align `render.yaml` when merging.
5. Optional: allow accept when ECE improves significantly and no accuracy threshold regresses &gt;2pp.
