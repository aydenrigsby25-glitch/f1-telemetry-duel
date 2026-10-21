"""Tests for the analysis pipeline using synthetic laps with known answers.

Run with:  python -m pytest -q
"""

import numpy as np
import pandas as pd
import pytest

import f1_analysis as f1

TRACK_M = 5000.0


def make_lap(speed_fn, hz=8, noise=0.0, seed=0, duplicate_every=0):
    """Simulate one lap of telemetry: drive around a 5 km circle at speed_fn(d) km/h."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / hz
    d, t, rows = 0.0, 0.0, []
    while d < TRACK_M:
        v = speed_fn(d)
        rows.append((d, t, v))
        d += v / 3.6 * dt
        t += dt
    df = pd.DataFrame(rows, columns=["Distance", "Time", "Speed"])
    if noise:
        df["Speed"] += rng.normal(0, noise, len(df))
    df["Throttle"] = 100.0
    df["Brake"] = False
    df["nGear"] = 7
    ang = df["Distance"] / TRACK_M * 2 * np.pi
    df["X"] = np.cos(ang) * 800
    df["Y"] = np.sin(ang) * 800
    df["Time"] = pd.to_timedelta(df["Time"], unit="s")
    if duplicate_every:
        df = pd.concat([df, df.iloc[::duplicate_every]]).sort_index(kind="stable")
    return df.reset_index(drop=True)


def test_clean_removes_duplicates_and_makes_distance_increasing():
    lap = make_lap(lambda d: 200, duplicate_every=10)
    clean = f1.clean_telemetry(lap)
    assert clean["Distance"].is_monotonic_increasing
    assert clean["Distance"].diff().dropna().gt(0).all()
    assert clean["Distance"].iloc[0] == 0 and clean["Time"].iloc[0] == 0


def test_clean_drops_missing_rows_and_filters_spikes():
    lap = make_lap(lambda d: 200)
    lap.loc[5, "Distance"] = np.nan
    lap.loc[50, "Speed"] = 900  # single-sample sensor spike
    clean = f1.clean_telemetry(lap)
    assert clean["Distance"].notna().all()
    assert clean["Speed"].max() < 210  # spike removed by the rolling median


def test_constant_speed_gap_matches_physics():
    # A at 200 km/h, B at 180 km/h over 5 km: exact lap-time gap is known.
    a = f1.clean_telemetry(make_lap(lambda d: 200))
    b = f1.clean_telemetry(make_lap(lambda d: 180))
    grid, ca, cb = f1.align_laps(a, b)
    delta = cb["Time"] - ca["Time"]
    expected = grid[-1] / (180 / 3.6) - grid[-1] / (200 / 3.6)
    assert delta[-1] == pytest.approx(expected, abs=0.05)
    assert np.all(np.diff(delta) >= -1e-9)  # A gains everywhere


def test_corner_attribution_finds_where_time_is_gained():
    # Identical laps except B is much slower between 2000 m and 2500 m.
    slow_zone = lambda d: 120 if 2000 <= d < 2500 else 250
    a = f1.clean_telemetry(make_lap(lambda d: 250))
    b = f1.clean_telemetry(make_lap(slow_zone))
    grid, ca, cb = f1.align_laps(a, b)
    delta = cb["Time"] - ca["Time"]

    corners = pd.DataFrame(
        {"Number": [1, 2, 3, 4], "Letter": ["", "", "", ""],
         "Distance": [800.0, 2250.0, 3500.0, 4500.0]}
    )
    gains = f1.corner_time_gain(grid, delta, corners)
    best = gains.loc[gains["gain_a_seconds"].idxmax()]
    assert best["corner_number"] == 2
    # Time lost in the slow zone: 500 m at 120 vs 250 km/h.
    expected = 500 / (120 / 3.6) - 500 / (250 / 3.6)
    assert gains["gain_a_seconds"].sum() == pytest.approx(expected, abs=0.1)


def test_dominance_map_marks_slow_zone_for_faster_driver():
    slow_zone = lambda d: 120 if 2000 <= d < 2500 else 250
    a = f1.clean_telemetry(make_lap(slow_zone))   # A is slow in the zone
    b = f1.clean_telemetry(make_lap(lambda d: 250))
    grid, ca, cb = f1.align_laps(a, b)
    who = f1.dominance_from_delta(grid, cb["Time"] - ca["Time"])
    in_zone = (grid > 2100) & (grid < 2400)
    assert (who[in_zone] == "B").all()


def test_noisy_laps_still_give_correct_gap():
    a = f1.clean_telemetry(make_lap(lambda d: 200, noise=3, seed=1))
    b = f1.clean_telemetry(make_lap(lambda d: 180, noise=3, seed=2))
    grid, ca, cb = f1.align_laps(a, b)
    expected = grid[-1] / (180 / 3.6) - grid[-1] / (200 / 3.6)
    assert (cb["Time"] - ca["Time"])[-1] == pytest.approx(expected, abs=0.1)


def test_missing_corner_info_returns_empty_table():
    grid = np.arange(0, 100, 2.0)
    out = f1.corner_time_gain(grid, np.zeros_like(grid), None)
    assert out.empty


def test_format_laptime():
    assert f1.format_laptime(pd.Timedelta(seconds=75.123)) == "1:15.123"
    assert f1.format_laptime(None) == "—"
