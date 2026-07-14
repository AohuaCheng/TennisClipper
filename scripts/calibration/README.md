# Calibration scripts

Interactive tools for session setup. Core logic lives in `tenniscut/calibration/`.

| Script | Purpose |
|--------|---------|
| `calibrate_court.py` | Click **two endpoints per line** (singles/doubles sidelines, baselines, service lines, net) |

Equivalent CLI:

```bash
tenniscut calibrate-court sessions/test_session_7252 --time 330
```

## Court line calibration

Each line needs **two clicks** on its visible portion (left→right for horizontal lines, top→bottom for sidelines).

**Required:** singles_left, singles_right, far_baseline, net_tape  
**Optional:** press `N` to skip (near_baseline, service lines, doubles lines)

Keys: `S` save · `U` undo · `N` skip optional line · `R` reset · `Q` quit
