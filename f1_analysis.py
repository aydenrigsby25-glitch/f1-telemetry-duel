"""
f1_analysis.py

Data + analysis layer for the F1 Telemetry Duel app.

Everything here works on REAL F1 session data pulled via FastF1, which sources
official F1 timing/telemetry feeds (car ECU data: speed, throttle, brake, gear,
RPM, DRS + car GPS position, sampled multiple times per second).

The interesting/"unique" piece is `compare_drivers()`: instead of just plotting
two speed traces on top of each other, it:
  1. Puts both drivers' laps on a shared distance axis (cars don't cross the
     line at the same instant, so raw telemetry timestamps aren't comparable).
  2. Derives a time-delta-vs-distance curve from first principles (not
     FastF1's built-in `delta_time`, which FastF1's own docs flag as
     imprecise) by integrating each driver's speed trace into elapsed time
     over a fixed distance grid.
  3. Uses the *slope* of that delta curve, cross-referenced against the
     circuit's real corner locations (also from FastF1), to attribute how
     much time each driver gained or lost through each individual corner.
     That's the same kind of corner-by-corner trace-comparison race
     engineers actually look at.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import streamlit as st

import fastf1
import fastf1.plotting as ff1_plotting

CACHE_DIR = "cache"

# Fixed, always-distinguishable duel colors (independent of team livery, so a
# teammate-vs-teammate comparison is still readable).
DRIVER_A_COLOR = "#00D9FF"  # electric blue
DRIVER_B_COLOR = "#FF3366"  # signal red

DISTANCE_STEP_M = 2.0  # resolution of the common distance grid, in meters


def enable_cache():
    import os

    os.makedirs(CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def get_event_schedule(year: int) -> pd.DataFrame:
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    return schedule[["RoundNumber", "EventName", "Location", "Country", "EventDate"]]


@st.cache_resource(show_spinner=False)
def load_session(year: int, event_name: str, session_code: str):
    session = fastf1.get_session(year, event_name, session_code)
    session.load(laps=True, telemetry=True, weather=False, messages=False)
    return session


def list_drivers(session) -> list[dict]:
    """Drivers who set a valid lap in this session, fastest lap first."""
    out = []
    for drv in session.drivers:
        laps = session.laps.pick_drivers([drv])
        if laps.empty or laps["LapTime"].isna().all():
            continue
        info = session.get_driver(drv)
        fastest = laps.pick_fastest()
        out.append(
            {
                "code": info["Abbreviation"],
                "name": f"{info['FirstName']} {info['LastName']}",
                "team": info["TeamName"],
                "fastest_time": fastest["LapTime"],
            }
        )
    out.sort(key=lambda d: d["fastest_time"])
    return out


@dataclass
class DuelResult:
    driver_a: str
    driver_b: str
    team_a: str
    team_b: str
    lap_time_a: pd.Timedelta
    lap_time_b: pd.Timedelta
    distance: np.ndarray            # common distance grid (m)
    speed_a: np.ndarray
    speed_b: np.ndarray
    throttle_a: np.ndarray
    throttle_b: np.ndarray
    brake_a: np.ndarray
    brake_b: np.ndarray
    gear_a: np.ndarray
    gear_b: np.ndarray
    x: np.ndarray                   # track shape, from driver A's GPS trace
    y: np.ndarray
    delta: np.ndarray               # +ve => A ahead of B at this point on track (seconds)
    corners: pd.DataFrame           # circuit corner metadata
    corner_gain: pd.DataFrame       # time gained/lost by A through each corner


def _telemetry_for_fastest_lap(session, driver_code: str):
    laps = session.laps.pick_drivers([driver_code])
    lap = laps.pick_fastest()
    tel = lap.get_telemetry().add_distance()
    return lap, tel


def _time_vs_distance(tel: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    """Elapsed lap time (s) as a function of distance, resampled onto `grid`."""
    t = (tel["Time"] - tel["Time"].iloc[0]).dt.total_seconds().to_numpy()
    d = tel["Distance"].to_numpy()
    # Distance must be strictly increasing for np.interp; guard against noise.
    order = np.argsort(d)
    d, t = d[order], t[order]
    d_unique, idx = np.unique(d, return_index=True)
    t_unique = t[idx]
    return np.interp(grid, d_unique, t_unique)


def _channel_vs_distance(tel: pd.DataFrame, col: str, grid: np.ndarray) -> np.ndarray:
    d = tel["Distance"].to_numpy()
    v = tel[col].to_numpy(dtype=float)
    order = np.argsort(d)
    d, v = d[order], v[order]
    d_unique, idx = np.unique(d, return_index=True)
    v_unique = v[idx]
    return np.interp(grid, d_unique, v_unique)


@st.cache_data(show_spinner=False)
def compare_drivers(session_key: tuple, driver_a: str, driver_b: str, _session) -> DuelResult:
    """`session_key` (hashable tuple) is what Streamlit hashes to key the
    cache; `_session` (leading underscore) is excluded from hashing since the
    real FastF1 Session object isn't cheaply hashable."""
    session = _session

    lap_a, tel_a = _telemetry_for_fastest_lap(session, driver_a)
    lap_b, tel_b = _telemetry_for_fastest_lap(session, driver_b)

    max_dist = min(tel_a["Distance"].max(), tel_b["Distance"].max())
    grid = np.arange(0, max_dist, DISTANCE_STEP_M)

    time_a = _time_vs_distance(tel_a, grid)
    time_b = _time_vs_distance(tel_b, grid)
    delta = time_b - time_a  # +ve => A is ahead (B took longer to reach this point)

    speed_a = _channel_vs_distance(tel_a, "Speed", grid)
    speed_b = _channel_vs_distance(tel_b, "Speed", grid)
    throttle_a = _channel_vs_distance(tel_a, "Throttle", grid)
    throttle_b = _channel_vs_distance(tel_b, "Throttle", grid)
    brake_a = _channel_vs_distance(tel_a, "Brake", grid).round().astype(bool)
    brake_b = _channel_vs_distance(tel_b, "Brake", grid).round().astype(bool)
    gear_a = _channel_vs_distance(tel_a, "nGear", grid)
    gear_b = _channel_vs_distance(tel_b, "nGear", grid)
    x = _channel_vs_distance(tel_a, "X", grid)
    y = _channel_vs_distance(tel_a, "Y", grid)

    corners = session.get_circuit_info().corners.copy()
    corner_gain = _corner_time_gain(grid, delta, corners)

    info_a = session.get_driver(driver_a)
    info_b = session.get_driver(driver_b)

    return DuelResult(
        driver_a=driver_a,
        driver_b=driver_b,
        team_a=info_a["TeamName"],
        team_b=info_b["TeamName"],
        lap_time_a=lap_a["LapTime"],
        lap_time_b=lap_b["LapTime"],
        distance=grid,
        speed_a=speed_a,
        speed_b=speed_b,
        throttle_a=throttle_a,
        throttle_b=throttle_b,
        brake_a=brake_a,
        brake_b=brake_b,
        gear_a=gear_a,
        gear_b=gear_b,
        x=x,
        y=y,
        delta=delta,
        corners=corners,
        corner_gain=corner_gain,
    )


def _corner_time_gain(grid: np.ndarray, delta: np.ndarray, corners: pd.DataFrame) -> pd.DataFrame:
    """
    Attribute time gained/lost to each corner by looking at how the delta
    trace changes across a window that brackets the corner's apex distance.

    Window = midpoint-to-midpoint between this corner's apex and the
    previous/next corner's apex, which approximates "entry through exit"
    for that corner without needing braking-point detection.
    """
    corner_distances = corners["Distance"].to_numpy()
    n = len(corner_distances)
    rows = []
    for i in range(n):
        apex = corner_distances[i]
        prev_apex = corner_distances[i - 1] if i > 0 else 0.0
        next_apex = corner_distances[i + 1] if i < n - 1 else grid[-1]
        start = (prev_apex + apex) / 2
        end = (apex + next_apex) / 2
        start_delta = np.interp(start, grid, delta)
        end_delta = np.interp(end, grid, delta)
        # positive gain => A gained time (relative to B) through this corner
        gain_a = end_delta - start_delta
        rows.append(
            {
                "corner_number": int(corners.iloc[i]["Number"]),
                "letter": corners.iloc[i]["Letter"] or "",
                "distance": apex,
                "gain_a_seconds": gain_a,
            }
        )
    return pd.DataFrame(rows)


def dominance_segments(res: DuelResult):
    """Split the track shape into contiguous colored segments based on who's
    gaining time (rising vs falling delta) at each point, for the track map."""
    d_delta = np.gradient(res.delta, res.distance)
    # smooth a bit so we don't get flickering pixel-scale segments
    window = 9
    kernel = np.ones(window) / window
    d_delta_smooth = np.convolve(d_delta, kernel, mode="same")
    faster = np.where(d_delta_smooth >= 0, "A", "B")
    return faster
