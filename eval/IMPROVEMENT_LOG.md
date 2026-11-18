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

(filled after `build_live_bench` + baseline runs)

---

## Loop log

| date | commit | hypothesis | dev metrics | holdout | decision |
|------|--------|------------|-------------|---------|----------|
