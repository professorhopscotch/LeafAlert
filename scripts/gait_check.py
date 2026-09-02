#!/usr/bin/env python3
"""Offline gait check for a DataRecorder session (imu.csv + events.csv).

Answers the two questions the gait-timed capture depends on but the app cannot
verify by itself:

1. SIGN — is `down = -(userAcceleration · gravity)` really "positive = accelerating
   toward the earth"? (Computed here from the raw columns: this validates the
   CONVENTION on real data. It cannot see the app's code — the APP check below
   does that, by comparing the app's own apex stamps with the offline extrema.) Walking has a built-in asymmetry: heel-strike arrests the
   body's fall with a sharp UPWARD jolt (a sharp NEGATIVE `down` spike), while the
   top of the bounce is a smooth positive maximum bounded by ~1 g. So over walking
   samples the series should be negatively skewed, with the largest excursions on
   the negative side. Positive skew means the projection is inverted and the apex
   detector is firing at heel-strike. (A diagnostic, not a proof: hand damping and
   phone orientation soften the spikes — look at the numbers, not just the verdict.)

2. PHASE — how long after the apex does the shutter actually fire? Captures logged
   with `pts=` and `apex_ts=` (both on the device uptime clock) give this directly.
   Older logs fall back to pairing each capture with the nearest preceding
   offline-detected apex on the session's elapsed clock, which includes delivery
   latency and is only indicative.

The offline apex detector mirrors CaptureEngine.ApexDetector (EMA 0.35, peak
floor 0.03 g, refractory 0.25 s) so "apex" means the same thing in both places.

Usage:
    python scripts/gait_check.py <session_dir | imu.csv> [--events events.csv] [--json out.json]

Sessions land in Documents/recordings/<id>/ on the phone and, with a sync folder
configured, under <sync folder>/recordings/<id>/ on the Mac.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

EMA_ALPHA = 0.35
PEAK_FLOOR = 0.03
REFRACTORY_S = 0.25
ACTIVE_RMS_G = 0.05          # 1-s rolling RMS of `down` above this = walking
ACTIVE_WINDOW_S = 1.0


# ─── Loading ─────────────────────────────────────────────────────────
def load_imu(path: Path) -> dict:
    """Returns dict(t, down, mag, clock) from an imu.csv. `t` prefers the
    device uptime column (`motion_ts`) when the recorder wrote one."""
    data = np.genfromtxt(str(path), delimiter=",", names=True, dtype=float)
    if data.ndim == 0:
        data = np.atleast_1d(data)
    names = data.dtype.names or ()
    need = ["user_accel_x", "user_accel_y", "user_accel_z",
            "gravity_x", "gravity_y", "gravity_z"]
    missing = [n for n in need if n not in names]
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")
    ua = np.stack([data["user_accel_x"], data["user_accel_y"], data["user_accel_z"]], axis=1)
    g = np.stack([data["gravity_x"], data["gravity_y"], data["gravity_z"]], axis=1)
    # CaptureEngine.verticalDown: userAcceleration is the accelerometer
    # residual (negative of kinematic acceleration), hence the negation.
    down = -np.einsum("ij,ij->i", ua, g)
    mag = np.linalg.norm(ua, axis=1)
    if "motion_ts" in names and np.isfinite(data["motion_ts"]).all():
        t, clock = data["motion_ts"].astype(float), "motion_ts"
    else:
        t, clock = data["timestamp_s"].astype(float), "timestamp_s"
    return {"t": t, "down": down, "mag": mag, "clock": clock,
            "elapsed": data["timestamp_s"].astype(float)}


def load_events(path: Path) -> list[dict]:
    """events.csv rows as dicts; `details` k=v pairs are parsed into `kv`."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            kv = {}
            for tok in (r.get("details") or "").split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    kv[k] = v
            rows.append({"t": float(r["timestamp_s"]), "type": r["event_type"],
                         "details": r.get("details", ""), "kv": kv})
    return rows


# ─── Analysis ────────────────────────────────────────────────────────
def detect_apexes(t: np.ndarray, down: np.ndarray) -> np.ndarray:
    """Offline twin of ApexDetector: EMA-smoothed local maxima of `down`
    above the peak floor, with a refractory period. Returns apex times."""
    apexes = []
    smoothed = prev = prev_delta = 0.0
    last_fire = -math.inf
    for i, (ti, d) in enumerate(zip(t, down)):
        smoothed = d if i == 0 else EMA_ALPHA * d + (1 - EMA_ALPHA) * smoothed
        delta = smoothed - prev
        is_peak = prev_delta > 0 and delta <= 0 and prev >= PEAK_FLOOR
        prev, prev_delta = smoothed, delta
        if is_peak and ti - last_fire >= REFRACTORY_S:
            last_fire = ti
            # Same stamp as CaptureEngine: the sample on which the detector
            # fires (one sample AFTER the true maximum), so app apex_ts and
            # offline apexes are comparable one-to-one.
            apexes.append(ti)
    return np.asarray(apexes)


def app_alignment(apex_ts: np.ndarray, t: np.ndarray, down: np.ndarray) -> dict:
    """Does the APP's projection agree with the offline one?

    The skew test only says the offline `-(ua·g)` has the right shape; it is
    computed from raw columns and cannot see the app's code. This can: the app
    logs the CoreMotion timestamp of each apex that opened a window (`apex_ts`).
    If the app's sign were inverted again, those stamps would sit on the
    offline MINIMA (heel-strike) instead of the maxima."""
    maxima = detect_apexes(t, down)
    minima = detect_apexes(t, -down)
    if len(apex_ts) == 0 or len(maxima) == 0 or len(minima) == 0:
        return {"n": int(len(apex_ts)), "verdict": "no apogee captures / no strides to compare"}
    d_max = np.array([np.abs(maxima - a).min() for a in apex_ts])
    d_min = np.array([np.abs(minima - a).min() for a in apex_ts])
    near_max = int((d_max < d_min).sum()); near_min = int((d_min < d_max).sum())
    if near_max >= 2 * max(1, near_min):
        verdict = "APP AGREES with the offline maxima (projection sign correct)"
    elif near_min >= 2 * max(1, near_max):
        verdict = "APP INVERTED: its apex stamps sit on the offline minima (heel-strike)"
    else:
        verdict = "inconclusive (stamps split between maxima and minima)"
    return {"n": int(len(apex_ts)), "near_maxima": near_max, "near_minima": near_min,
            "median_offset_to_maxima_ms": float(np.median(d_max) * 1000), "verdict": verdict}


def active_mask(t: np.ndarray, down: np.ndarray) -> np.ndarray:
    """Samples inside a 1-s window whose RMS of `down` says the user is moving."""
    if len(t) < 2:
        return np.zeros(len(t), dtype=bool)
    dt = float(np.median(np.diff(t))) or 0.01
    w = max(1, int(round(ACTIVE_WINDOW_S / dt)))
    sq = np.convolve(down * down, np.ones(w) / w, mode="same")
    return np.sqrt(sq) > ACTIVE_RMS_G


def skewness(x: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    c = x - x.mean()
    m2 = float(np.mean(c * c))
    if m2 <= 0:
        return float("nan")
    return float(np.mean(c ** 3) / m2 ** 1.5)


def sign_verdict(skew: float, neg_ext: float, pos_ext: float) -> str:
    if not np.isfinite(skew):
        return "inconclusive (not enough walking)"
    if skew < -0.3 and neg_ext > pos_ext:
        return "CONSISTENT with kinematic-down (sharp impacts are negative)"
    if skew > 0.3 and pos_ext > neg_ext:
        return "INVERTED? (sharp impacts are positive — the apex detector would fire at heel-strike)"
    return "inconclusive (weak asymmetry — try a brisker walk, phone held normally)"


def analyze(imu: dict, events: list[dict] | None) -> dict:
    t, down = imu["t"], imu["down"]
    act = active_mask(t, down)
    apexes = detect_apexes(t, down)
    out: dict = {
        "samples": int(len(t)),
        "duration_s": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
        "clock": imu["clock"],
        "active_fraction": float(act.mean()) if len(t) else 0.0,
    }
    walking = down[act]
    skew = skewness(walking)
    neg_ext = float(np.percentile(-walking, 99.5)) if len(walking) else float("nan")
    pos_ext = float(np.percentile(walking, 99.5)) if len(walking) else float("nan")
    out["sign"] = {
        "skew": skew, "neg_extreme_g": neg_ext, "pos_extreme_g": pos_ext,
        "verdict": sign_verdict(skew, neg_ext, pos_ext),
    }
    if len(apexes) >= 2:
        periods = np.diff(apexes)
        out["cadence"] = {
            "apexes": int(len(apexes)),
            "median_period_s": float(np.median(periods)),
            "steps_per_min": float(60.0 / np.median(periods)),
        }
    else:
        out["cadence"] = {"apexes": int(len(apexes))}

    if events is not None:
        caps = [e for e in events if e["type"] == "capture"]
        by_trigger: dict[str, int] = {}
        for e in caps:
            by_trigger[e["kv"].get("trigger", "?")] = by_trigger.get(e["kv"].get("trigger", "?"), 0) + 1
        direct, fallback = [], []
        for e in caps:
            if e["kv"].get("trigger") != "apogee":
                continue
            kv = e["kv"]
            if "pts" in kv and "apex_ts" in kv:
                try:
                    direct.append((float(kv["pts"]) - float(kv["apex_ts"])) * 1000.0)
                    continue
                except ValueError:
                    pass
            # Fallback: nearest preceding offline apex on the elapsed clock.
            if len(apexes) and imu["clock"] == "timestamp_s":
                prior = apexes[apexes <= e["t"]]
                if len(prior):
                    fallback.append((e["t"] - prior[-1]) * 1000.0)
        stamps = []
        for e in caps:
            if e["kv"].get("trigger") == "apogee" and "apex_ts" in e["kv"]:
                try:
                    stamps.append(float(e["kv"]["apex_ts"]))
                except ValueError:
                    pass
        alignment = (app_alignment(np.asarray(stamps), t, down)
                     if imu["clock"] == "motion_ts" else
                     {"n": int(len(stamps)), "verdict": "needs imu.csv with motion_ts (same clock as apex_ts)"})

        def summarize(v):
            a = np.asarray(v)
            return {"n": int(len(a)), "median_ms": float(np.median(a)),
                    "p25_ms": float(np.percentile(a, 25)), "p75_ms": float(np.percentile(a, 75))} if len(a) else {"n": 0}
        ph = {"captures": int(len(caps)), "by_trigger": by_trigger,
              "direct": summarize(direct), "fallback": summarize(fallback),
              "app_alignment": alignment}
        period = out["cadence"].get("median_period_s")
        for key in ("direct", "fallback"):
            if ph[key].get("n") and period:
                ph[key]["median_fraction_of_stride"] = ph[key]["median_ms"] / 1000.0 / period
        out["phase"] = ph
    return out


def render(r: dict) -> str:
    L = [f"Session: {r['samples']} IMU samples, {r['duration_s']:.1f} s, clock={r['clock']}, "
         f"walking {100 * r['active_fraction']:.0f}% of the time"]
    s = r["sign"]
    L.append(f"SIGN   skew={s['skew']:+.2f}  extremes: -{s['neg_extreme_g']:.2f} g / +{s['pos_extreme_g']:.2f} g")
    L.append(f"       → {s['verdict']}")
    c = r["cadence"]
    if "steps_per_min" in c:
        L.append(f"GAIT   {c['apexes']} apexes, median period {c['median_period_s'] * 1000:.0f} ms "
                 f"({c['steps_per_min']:.0f} steps/min)")
    else:
        L.append(f"GAIT   {c['apexes']} apexes — too few to measure cadence")
    if "phase" in r:
        p = r["phase"]
        L.append(f"EVENTS {p['captures']} captures by trigger {p['by_trigger']}")
        for key, label in (("direct", "apex→shutter (pts − apex_ts)"), ("fallback", "apex→capture callback (elapsed clock)")):
            d = p[key]
            if d.get("n"):
                frac = f", {100 * d['median_fraction_of_stride']:.0f}% of a stride" if "median_fraction_of_stride" in d else ""
                L.append(f"PHASE  {label}: n={d['n']} median {d['median_ms']:.0f} ms "
                         f"(IQR {d['p25_ms']:.0f}–{d['p75_ms']:.0f}){frac}")
        if not p["direct"].get("n") and not p["fallback"].get("n"):
            L.append("PHASE  no apogee-triggered captures to measure")
        al = p.get("app_alignment", {})
        if al.get("n"):
            L.append(f"APP    {al['n']} apex stamps: {al.get('near_maxima', '?')} near offline maxima, "
                     f"{al.get('near_minima', '?')} near minima → {al['verdict']}")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, help="session directory or imu.csv")
    ap.add_argument("--events", type=Path, default=None, help="events.csv (default: next to imu.csv)")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)
    imu_path = a.path / "imu.csv" if a.path.is_dir() else a.path
    if not imu_path.exists():
        raise SystemExit(f"no imu.csv at {imu_path}")
    ev_path = a.events or imu_path.with_name("events.csv")
    imu = load_imu(imu_path)
    events = load_events(ev_path) if ev_path.exists() else None
    r = analyze(imu, events)
    print(render(r))
    if a.json:
        a.json.write_text(json.dumps(r, indent=2, default=float) + "\n")
        print(f"\nJSON → {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
