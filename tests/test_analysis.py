"""Analysis tests on synthetic laps where the correct answer is known.

Run with:  python -m pytest -q
"""

import numpy as np
import pandas as pd
import pytest

import f1_analysis as f1
from tests.fake_openf1 import LAP_START, TRACK_M, simulate


def lap(speed_fn, **kw):
    car, loc, dur = simulate(speed_fn, **kw)
    return f1.build_lap_telemetry(car, loc, LAP_START, dur), dur


def duel(fn_a, fn_b, **kw):
    a, ta = lap(fn_a, seed=1, **kw)
    b, tb = lap(fn_b, seed=2, **kw)
    meta = dict(driver_a="AAA", driver_b="BBB", lap_time_a=ta, lap_time_b=tb)
    return f1.compare_laps(a, b, meta), ta, tb


def test_integrate_distance_constant_speed():
    t = np.linspace(0, 10, 101)
    d = f1.integrate_distance(t, np.full_like(t, 180.0))  # 180 km/h = 50 m/s
    assert d[-1] == pytest.approx(500.0)


def test_build_lap_recovers_track_length_and_full_lap():
    tel, dur = lap(lambda d: 200)
    assert tel["Time"].iloc[0] == 0
    assert tel["Time"].iloc[-1] == pytest.approx(dur)
    assert tel["Distance"].iloc[-1] == pytest.approx(TRACK_M, rel=0.01)
    assert {"X", "Y"} <= set(tel.columns)


def test_build_lap_aligns_gps_onto_car_timestamps():
    tel, _ = lap(lambda d: 200)
    radius = np.hypot(tel["X"], tel["Y"])
    assert radius.between(780, 801).all()  # interpolated points stay on the circle


def test_clean_removes_duplicates_and_spikes():
    tel, _ = lap(lambda d: 200)
    dirty = pd.concat([tel, tel.iloc[::10]]).sort_index(kind="stable").reset_index(drop=True)
    dirty.loc[40, "Speed"] = 900
    dirty.loc[5, "Distance"] = np.nan
    clean = f1.clean_telemetry(dirty)
    assert clean["Distance"].diff().dropna().gt(0).all()
    assert clean["Speed"].max() < 210


def test_constant_speed_gap_matches_physics():
    res, ta, tb = duel(lambda d: 200, lambda d: 180)
    expected = TRACK_M / (180 / 3.6) - TRACK_M / (200 / 3.6)
    assert tb - ta == pytest.approx(expected, abs=0.02)
    assert res.delta[-1] == pytest.approx(tb - ta, abs=0.05)
    assert np.all(np.diff(res.delta) >= -0.005)  # A gains everywhere (to within 5 ms)


def corner(at_m, slow_kmh, fast_kmh=280, half_width=120):
    """Speed profile with one corner: brake from fast_kmh down to slow_kmh at `at_m`."""
    def fn(d):
        x = abs(d - at_m)
        return slow_kmh + (fast_kmh - slow_kmh) * min(1.0, x / half_width)
    return fn


def test_detects_corners_and_attributes_the_gain():
    # Same track, two corners; B is 20 km/h slower through the second one only.
    a_fn = lambda d: min(corner(1500, 100)(d), corner(3500, 120)(d))
    b_fn = lambda d: min(corner(1500, 100)(d), corner(3500, 100)(d))
    res, ta, tb = duel(a_fn, b_fn)

    gains = res.corner_gain
    assert len(gains) == 2
    assert gains["distance"].iloc[0] == pytest.approx(1500, abs=60)
    assert gains["distance"].iloc[1] == pytest.approx(3500, abs=60)
    # Nearly all of the gap is won at corner 2, by A.
    assert gains["gain_a_seconds"].iloc[1] > 0.1
    assert abs(gains["gain_a_seconds"].iloc[0]) < 0.05
    assert gains["gain_a_seconds"].sum() == pytest.approx(tb - ta, abs=0.05)


def test_dominance_map_marks_the_slow_corner():
    a_fn = corner(2500, 100)  # A is slower through the corner
    b_fn = corner(2500, 130)
    res, _, _ = duel(a_fn, b_fn)
    who = f1.dominance_from_delta(res.distance, res.delta)
    zone = (res.distance > 2440) & (res.distance < 2560)
    assert (who[zone] == "B").mean() > 0.9


def test_noisy_telemetry_still_gives_correct_gap():
    res, ta, tb = duel(lambda d: 200, lambda d: 180, noise=4)
    assert res.delta[-1] == pytest.approx(tb - ta, abs=0.05)


def test_no_corners_gives_empty_breakdown():
    res, _, _ = duel(lambda d: 250, lambda d: 249)
    assert res.corner_gain.empty


def test_format_laptime():
    assert f1.format_laptime(82.207) == "1:22.207"
    assert f1.format_laptime(pd.Timedelta(seconds=75.123)) == "1:15.123"
    assert f1.format_laptime(None) == "—"


def test_gps_alignment_removes_speed_sensor_drift():
    # Identical driving; B's speed sensor reads 4% high in the first half of the lap.
    fn = lambda d: min(corner(1200, 110)(d), corner(2600, 90)(d), corner(4000, 130)(d))
    a, ta = lap(fn, seed=1)
    b, tb = lap(fn, seed=2, sensor_scale=lambda d: 1.04 if d < 2500 else 1.0)
    meta = dict(driver_a="AAA", driver_b="BBB", lap_time_a=ta, lap_time_b=tb)

    # Without GPS alignment the drift shows up as a fake mid-lap gap...
    track = (a["Distance"].iloc[-1] + b["Distance"].iloc[-1]) / 2
    _, ca, cb = f1.align_laps(
        f1.clean_telemetry(f1.scale_to_length(a, track)),
        f1.clean_telemetry(f1.scale_to_length(b, track)),
    )
    assert np.abs(cb["Time"] - ca["Time"]).max() > 0.5

    # ...with it, the laps are recognised as the same lap all the way round.
    res = f1.compare_laps(a, b, meta)
    assert np.abs(res.delta).max() < 0.1


def test_gps_jitter_does_not_make_the_gap_curve_noisy():
    # Real GPS jitters by a few meters per sample. The gap curve must stay smooth.
    fn = lambda d: min(corner(1200, 110)(d), corner(2600, 90)(d), corner(4000, 130)(d))
    a, ta = lap(fn, seed=1)
    b, tb = lap(fn, seed=2, gps_noise_m=4.0, sensor_scale=lambda d: 1.02 if d < 2500 else 1.0)
    res = f1.compare_laps(a, b, dict(driver_a="A", driver_b="B", lap_time_a=ta, lap_time_b=tb))
    assert np.abs(res.delta).max() < 0.06        # identical driving stays level
    assert np.abs(np.diff(res.delta)).max() < 0.01  # no jagged jumps between 2 m steps
