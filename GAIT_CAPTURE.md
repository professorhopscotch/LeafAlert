# Gait-timed capture — how it works and how to check it in the field

LeafAlert takes a photo once per stride, at the **apex** of the walking bounce:
the instant the phone is momentarily still in the vertical, so the frame is
sharpest. Everything below lives in `LeafAlert/Engines/CaptureEngine.swift`.

## The signal

CoreMotion gives `userAcceleration` (ua) and `gravity` (g, unit length, pointing
toward the earth). The engine uses

```
down = −(ua · g)          // CaptureEngine.verticalDown
```

**Why the minus sign.** `userAcceleration` is the accelerometer *residual*
(total − gravity), and an accelerometer measures *proper* acceleration, which is
the negative of kinematic acceleration along gravity. Free fall is the clean
proof: the accelerometer reads zero, so CoreMotion reports `ua = −g` while the
phone accelerates *downward* at 1 g. Without the minus sign, `down` says "up"
in free fall — and the detector fires at heel-strike (the bottom of the bounce,
the impact-laden instant) instead of the apex. This was the pre-26920bf bug; a
unit test (`testVerticalDownIsPositiveInFreeFall`) pins the convention.

## The detector

- **`ApexDetector`** — EMA-smoothed (α 0.35) local *maximum* of `down`, above a
  0.03 g floor, with a 0.25 s refractory. The apex is a maximum of `down`, not a
  zero-crossing: the body is closest to free fall at the top of the bounce.
- **`StillnessFilter`** — |ua| smoothed with a ~0.3 s time constant, seeded high.
  "Still" means sustained quiet, not one quiet sample; single-sample stillness
  used to fire on the zero-crossings of a slow walk and open the window at an
  arbitrary phase. An apex latch is never relabelled as stillness.
- **`DutyCycle.decide`** — pure decision at 20 Hz: honour cooldown, consume stale
  latches, open on a latched apex or sustained stillness (a phone held still
  is the stillness path), and as a last resort force a capture every
  `forcedCaptureInterval` so the world is still seen if neither fires.
- **Capture-time gate** — a frame is dropped if the rotation rate is above
  `rotationRateThreshold` (forced captures are exempt).
- **Battery Saver** — the sensor drops to 10 fps while no window is open and is
  restored to its exact saved rate on the next window, even if the setting was
  switched off in between.

The engine's intervals and thresholds default from `TuningDefaults` and can be
changed live from Debug → Live Controls. `ApexDetector`'s peak floor,
refractory and smoothing are compile-time constants.

## Field check (30–60 seconds)

The sign fix was made by reasoning; this measures it.

1. Debug dashboard → start a **recording**, then start a patrol.
2. Walk briskly for 30–60 s holding the phone as you normally would.
3. Stop the patrol (recording finalizes). With a sync folder set, the session
   appears under `<sync folder>/recordings/<id>/` on the Mac.
4. Run:

   ```bash
   .venv/bin/python scripts/gait_check.py "<sync folder>/recordings/<id>" --json gait.json
   ```

What good looks like:

- `SIGN  skew` clearly **negative**, with the negative extreme larger than the
  positive one → `CONSISTENT`. Heel-strike is a sharp upward jolt (negative
  `down`); the apex is a smooth positive maximum bounded by ~1 g. This is
  computed from the raw IMU columns, so it validates the *convention* on real
  data — it cannot see the app's code.
- `APP` — the check that does see the app: each apex the app used to open a
  window is stamped (`apex_ts`) and compared with the offline extrema. Stamps
  near the offline **maxima** → `APP AGREES`; near the **minima** (heel-strike)
  → `APP INVERTED`, i.e. the projection sign regressed in the app.
- `GAIT` cadence in the 90–130 steps/min range for a normal walk.
- `PHASE apex→shutter` — the median latency from the detected apex to the frame's
  presentation timestamp, and what fraction of a stride that is. Under ~15 % of a
  stride (≈ 80 ms at 110 steps/min) means the shutter is still near the apex; if
  it is much larger, the window is opening late and `maxCaptureWindowDuration`
  or the frame rate is worth a look.

`events.csv` now records, per capture, `trigger`, `pts` (frame presentation
timestamp), `apex_ts` (the apex that opened the window) and `window_ms`; per
detection, `class`, `conf`, `sev` and `trigger`. `imu.csv` carries `motion_ts`
on the same device-uptime clock as `pts`/`apex_ts`.

## Known limits

- Very slow walks have little vertical amplitude; captures will mostly come from
  the forced fallback. Expected, and visible in the trigger counts.
- The simulator has no motion sensors; none of this can be exercised there.
