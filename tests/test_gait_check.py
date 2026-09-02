"""gait_check.py against synthetic walking IMU logs.

The synthetic walk encodes the physics the sign check relies on: a smooth
1.8 Hz bounce (apex = smooth positive maximum) plus a sharp NEGATIVE spike at
each heel-strike. The inverted log is the same signal with the projection sign
flipped — what the app produced before 26920bf.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gait_check  # noqa: E402

HZ, CADENCE, DUR = 100.0, 1.8, 40.0


def synthetic_walk(inverted: bool = False):
    t = np.arange(0, DUR, 1 / HZ)
    down = 0.25 * np.sin(2 * np.pi * CADENCE * t)
    strike_times = (np.arange(int(DUR * CADENCE)) + 0.75) / CADENCE   # troughs
    for ts in strike_times:
        down += -0.6 * np.exp(-((t - ts) / 0.02) ** 2)
    rng = np.random.default_rng(0)
    down += rng.normal(0, 0.01, len(t))
    if inverted:
        down = -down
    return t, down


def write_session(tmp: Path, inverted: bool = False, with_events: bool = True, motion_ts: bool = True):
    t, down = synthetic_walk(inverted)
    # Face-up phone: gravity = (0,0,-1), so -(ua·g) = ua_z  ⇒  ua_z = down.
    cols = ["timestamp_s", "accel_x", "accel_y", "accel_z", "user_accel_x", "user_accel_y", "user_accel_z",
            "gravity_x", "gravity_y", "gravity_z", "rot_x", "rot_y", "rot_z", "roll", "pitch", "yaw"]
    if motion_ts:
        cols.append("motion_ts")
    uptime0 = 5000.0
    with open(tmp / "imu.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for ti, d in zip(t, down):
            row = [ti, 0, 0, d - 1, 0, 0, d, 0, 0, -1, 0, 0, 0, 0, 0, 0]
            if motion_ts:
                row.append(uptime0 + ti)
            f.write(",".join(f"{v:.5f}" for v in row) + "\n")
    if with_events:
        apex_times = (np.arange(4, 60) + 0.25) / CADENCE          # sine peaks
        apex_times = apex_times[apex_times < DUR - 1]
        with open(tmp / "events.csv", "w") as f:
            f.write("timestamp_s,event_type,details\n")
            f.write("0.0000,session_start,\n")
            for i, at in enumerate(apex_times[::6]):
                pts = uptime0 + at + 0.080
                f.write(f"{at + 0.095:.4f},capture,trigger=apogee accel=0.100 rot=0.200 pts={pts:.4f} apex_ts={uptime0 + at:.4f}\n")
                if i % 2 == 0:
                    f.write(f"{at + 0.300:.4f},detection,class=poison_ivy conf=0.612\n")
            f.write(f"{DUR:.4f},session_end,\n")


def test_sign_check_recognises_the_correct_convention(tmp_path):
    write_session(tmp_path)
    r = gait_check.analyze(gait_check.load_imu(tmp_path / "imu.csv"), gait_check.load_events(tmp_path / "events.csv"))
    assert r["sign"]["skew"] < -0.3
    assert r["sign"]["verdict"].startswith("CONSISTENT")


def test_sign_check_flags_the_inverted_projection(tmp_path):
    write_session(tmp_path, inverted=True)
    r = gait_check.analyze(gait_check.load_imu(tmp_path / "imu.csv"), None)
    assert r["sign"]["skew"] > 0.3
    assert r["sign"]["verdict"].startswith("INVERTED")


def test_cadence_matches_the_synthetic_walk(tmp_path):
    write_session(tmp_path)
    r = gait_check.analyze(gait_check.load_imu(tmp_path / "imu.csv"), None)
    assert abs(r["cadence"]["steps_per_min"] - CADENCE * 60) < 6


def test_phase_is_measured_directly_from_pts_and_apex_ts(tmp_path):
    write_session(tmp_path)
    r = gait_check.analyze(gait_check.load_imu(tmp_path / "imu.csv"), gait_check.load_events(tmp_path / "events.csv"))
    d = r["phase"]["direct"]
    assert d["n"] >= 5
    assert abs(d["median_ms"] - 80.0) < 1.0
    assert 0.1 < d["median_fraction_of_stride"] < 0.2


def test_legacy_log_without_motion_ts_falls_back_to_elapsed_clock(tmp_path):
    write_session(tmp_path, motion_ts=False)
    # Strip pts/apex_ts to emulate a pre-instrumentation log.
    ev = tmp_path / "events.csv"
    ev.write_text("\n".join(
        line.split(" pts=")[0] if "capture" in line else line
        for line in ev.read_text().splitlines()) + "\n")
    imu = gait_check.load_imu(tmp_path / "imu.csv")
    assert imu["clock"] == "timestamp_s"
    r = gait_check.analyze(imu, gait_check.load_events(ev))
    assert r["phase"]["direct"]["n"] == 0
    fb = r["phase"]["fallback"]
    assert fb["n"] >= 5 and 40 < fb["median_ms"] < 200


def test_app_alignment_detects_an_inverted_app_projection(tmp_path):
    import gait_check as gc
    t, down = synthetic_walk()
    maxima = gc.detect_apexes(t, down)
    minima = gc.detect_apexes(t, -down)
    # App stamps on the maxima (correct sign) → agrees; on the minima → INVERTED.
    ok = gc.app_alignment(maxima[5:25] + 0.004, t, down)
    bad = gc.app_alignment(minima[5:25] + 0.004, t, down)
    assert ok["verdict"].startswith("APP AGREES") and ok["near_maxima"] > ok["near_minima"]
    assert bad["verdict"].startswith("APP INVERTED") and bad["near_minima"] > bad["near_maxima"]


def test_session_report_includes_app_alignment_when_stamps_exist(tmp_path):
    write_session(tmp_path)
    r = gait_check.analyze(gait_check.load_imu(tmp_path / "imu.csv"), gait_check.load_events(tmp_path / "events.csv"))
    al = r["phase"]["app_alignment"]
    assert al["n"] >= 5 and al["verdict"].startswith("APP AGREES")


def test_cli_renders_and_writes_json(tmp_path, capsys):
    write_session(tmp_path)
    out = tmp_path / "r.json"
    assert gait_check.main([str(tmp_path), "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "SIGN" in text and "PHASE" in text and out.exists()
