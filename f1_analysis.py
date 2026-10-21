"""
f1_analysis.py

Data + analysis layer for the F1 Telemetry Duel app.

Everything here works on REAL F1 session data pulled via FastF1, which sources
official F1 timing/telemetry feeds (car ECU data: speed, throttle, brake, gear,
RPM + car GPS position, sampled multiple times per second).

Pipeline for a duel:
  1. Load the session and confirm lap + telemetry data actually arrived.
  2. Clean each driver's fastest-lap telemetry (drop bad rows, remove
     duplicate distance samples, filter single-sample speed spikes).
  3. Align both laps on a shared distance grid. Cars don't cross the line at
     the same instant and log samples at different moments, so raw rows
     aren't comparable until they're resampled to the same points on track.
  4. Compute a time-delta-vs-distance curve from each driver's elapsed lap
     time on that grid.
  5. Use the delta curve plus the circuit's real corner locations to
     attribute how much time each driver gained or lost through each corner.

The pure functions (clean_telemetry, align_laps, corner_time_gain,
dominance_from_delta) take plain DataFrames/arrays so they can be unit
tested without network access.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

import fastf1

CACHE_DIR = "cache"

# Fixed, always-distinguishable duel colors (independent of team livery, so a
# teammate-vs-teammate comparison is still readable).
DRIVER_A_COLOR = "#00D9FF"  # electric blue
DRIVER_B_COLOR = "#FF3366"  # signal red

DISTANCE_STEP_M = 2.0  # resolution of the common distance grid, in meters
SPEED_FILTER_WINDOW = 5  # samples in the rolling median used to remove spikes

# Channels the app plots. Distance and Time are required for alignment.
CHANNELS = ["Speed", "Throttle", "Brake", "nGear", "X", "Y"]


class SessionLoadError(RuntimeError):
    """Raised when FastF1 returns a session without usable lap/telemetry data."""


# --------------------------------------------------------------- loading ----

def enable_cache() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def get_event_schedule(year: int) -> pd.DataFrame:
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    return schedule[["RoundNumber", "EventName", "Location", "Country", "EventDate"]]


@st.cache_resource(show_spinner=False, ttl=60 * 60 * 6, max_entries=8)
def load_session(year: int, event_name: str, session_code: str):
    """Load a session and verify the data actually arrived.

    FastF1's `session.load()` does not raise when a download fails; it logs a
    warning and leaves `session.laps` unloaded. The old version of this app
    cached that half-loaded session, so one failed download broke the app for
    every visitor. Raising here means Streamlit never caches a bad session and
    the next request retries.
    """
    session = fastf1.get_session(year, event_name, session_code)
    session.load(laps=True, telemetry=True, weather=False, messages=False)

    try:
        laps = session.laps
    except Exception as exc:  # fastf1.exceptions.DataNotLoadedError
        raise SessionLoadError(
            "Lap data for this session isn't available from the F1 timing feed yet."
        ) from exc

    if laps is None or laps.empty or laps["LapTime"].isna().all():
        raise SessionLoadError("This session has no timed laps yet.")
    return session


def _driver_laps(session, driver: str):
    """`pick_drivers` is the current FastF1 API; `pick_driver` is the old one."""
    laps = session.laps
    if hasattr(laps, "pick_drivers"):
        return laps.pick_drivers([driver])
    return laps.pick_driver(driver)


def list_drivers(session) -> list[dict]:
    """Drivers who set a valid lap in this session, fastest lap first."""
    out = []
    for drv in session.drivers:
        laps = _driver_laps(session, drv)
        if laps.empty or laps["LapTime"].isna().all():
            continue
        fastest = laps.pick_fastest()
        if fastest is None or pd.isna(fastest["LapTime"]):
            continue
        info = session.get_driver(drv)
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


# --------------------------------------------------- pure analysis steps ----

def clean_telemetry(tel: pd.DataFrame) -> pd.DataFrame:
    """Clean one lap of raw telemetry so it can be interpolated safely.

    1. Keep only the channels we use and convert Time to elapsed seconds.
    2. Drop rows with missing Distance or Time.
    3. Sort by distance and drop samples where distance doesn't increase
       (duplicates and GPS jitter), so the distance axis is strictly
       increasing, which interpolation requires.
    4. Fill small gaps in the other channels by interpolating over distance.
    5. Apply a short rolling median to Speed to remove single-sample spikes
       without shifting where braking points happen.
    """
    missing = [c for c in ("Distance", "Time") if c not in tel.columns]
    if missing:
        raise ValueError(f"Telemetry is missing required columns: {missing}")

    cols = ["Distance", "Time"] + [c for c in CHANNELS if c in tel.columns]
    df = tel[cols].copy()

    t = df["Time"]
    df["Time"] = t.dt.total_seconds() if pd.api.types.is_timedelta64_dtype(t) else t.astype(float)
    df["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    if "Brake" in df.columns:
        df["Brake"] = df["Brake"].astype(float).clip(0, 1)

    df = df.dropna(subset=["Distance", "Time"]).sort_values("Distance", kind="stable")
    df = df[df["Distance"].diff().fillna(1.0) > 0].reset_index(drop=True)

    # Start both laps at distance 0, time 0.
    df["Distance"] -= df["Distance"].iloc[0]
    df["Time"] -= df["Time"].iloc[0]

    others = [c for c in CHANNELS if c in df.columns]
    if others:
        df[others] = df[others].astype(float).interpolate(limit_direction="both")

    if "Speed" in df.columns:
        df["Speed"] = (
            df["Speed"].rolling(SPEED_FILTER_WINDOW, center=True, min_periods=1).median()
        )

    if len(df) < 20:
        raise ValueError("Not enough clean telemetry points to analyse this lap.")
    return df


def align_laps(a: pd.DataFrame, b: pd.DataFrame, step_m: float = DISTANCE_STEP_M):
    """Resample two cleaned laps onto one shared distance grid.

    Returns (grid, channels_a, channels_b) where channels_* map a channel name
    (including "Time") to an array of values at each grid point.
    """
    end = min(a["Distance"].iloc[-1], b["Distance"].iloc[-1])
    grid = np.arange(0.0, end, step_m)
    names = ["Time"] + [c for c in CHANNELS if c in a.columns and c in b.columns]
    ch_a = {n: np.interp(grid, a["Distance"], a[n]) for n in names}
    ch_b = {n: np.interp(grid, b["Distance"], b[n]) for n in names}
    return grid, ch_a, ch_b


def corner_time_gain(grid: np.ndarray, delta: np.ndarray, corners: pd.DataFrame) -> pd.DataFrame:
    """Attribute time gained/lost to each corner.

    Each corner's window runs from the midpoint with the previous corner's
    apex to the midpoint with the next one, which approximates entry through
    exit. The change in the delta curve across that window is the time
    driver A gained (positive) or lost (negative) there.
    """
    if corners is None or corners.empty:
        return pd.DataFrame(columns=["corner_number", "letter", "distance", "gain_a_seconds"])

    corners = corners.sort_values("Distance")
    apexes = corners["Distance"].to_numpy(dtype=float)
    n = len(apexes)
    rows = []
    for i in range(n):
        apex = apexes[i]
        prev_apex = apexes[i - 1] if i > 0 else 0.0
        next_apex = apexes[i + 1] if i < n - 1 else grid[-1]
        start = (prev_apex + apex) / 2
        end = (apex + next_apex) / 2
        gain_a = np.interp(end, grid, delta) - np.interp(start, grid, delta)
        letter = corners.iloc[i].get("Letter", "")
        rows.append(
            {
                "corner_number": int(corners.iloc[i]["Number"]),
                "letter": letter if isinstance(letter, str) else "",
                "distance": apex,
                "gain_a_seconds": gain_a,
            }
        )
    return pd.DataFrame(rows)


def dominance_from_delta(grid: np.ndarray, delta: np.ndarray, window: int = 9) -> np.ndarray:
    """For each grid point, which driver is gaining time there ("A" or "B").

    Uses the slope of the delta curve, lightly smoothed so the track map
    doesn't flicker between colors on tiny noise.
    """
    slope = np.gradient(delta, grid)
    kernel = np.ones(window) / window
    smooth = np.convolve(slope, kernel, mode="same")
    return np.where(smooth >= 0, "A", "B")


# -------------------------------------------------------- full comparison ---

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
    delta: np.ndarray               # +ve => A ahead of B at this point (seconds)
    corner_gain: pd.DataFrame       # time gained/lost by A through each corner


def _fastest_lap_telemetry(session, driver: str):
    lap = _driver_laps(session, driver).pick_fastest()
    if lap is None or pd.isna(lap["LapTime"]):
        raise ValueError(f"{driver} has no timed lap in this session.")
    tel = lap.get_telemetry()
    if "Distance" not in tel.columns:
        tel = tel.add_distance()
    return lap, tel


@st.cache_data(show_spinner=False, max_entries=64)
def compare_drivers(session_key: tuple, driver_a: str, driver_b: str, _session) -> DuelResult:
    """`session_key` is what Streamlit hashes for the cache; `_session`
    (leading underscore) is skipped because a FastF1 Session isn't hashable."""
    session = _session

    lap_a, raw_a = _fastest_lap_telemetry(session, driver_a)
    lap_b, raw_b = _fastest_lap_telemetry(session, driver_b)

    grid, a, b = align_laps(clean_telemetry(raw_a), clean_telemetry(raw_b))
    delta = b["Time"] - a["Time"]  # +ve => A ahead (B took longer to get here)

    try:
        corners = session.get_circuit_info().corners
    except Exception:
        corners = None  # circuit info missing: skip the per-corner breakdown

    info_a = session.get_driver(driver_a)
    info_b = session.get_driver(driver_b)
    zeros = np.zeros_like(grid)

    return DuelResult(
        driver_a=driver_a,
        driver_b=driver_b,
        team_a=info_a["TeamName"],
        team_b=info_b["TeamName"],
        lap_time_a=lap_a["LapTime"],
        lap_time_b=lap_b["LapTime"],
        distance=grid,
        speed_a=a.get("Speed", zeros),
        speed_b=b.get("Speed", zeros),
        throttle_a=a.get("Throttle", zeros),
        throttle_b=b.get("Throttle", zeros),
        brake_a=a.get("Brake", zeros).round().astype(bool),
        brake_b=b.get("Brake", zeros).round().astype(bool),
        gear_a=a.get("nGear", zeros),
        gear_b=b.get("nGear", zeros),
        x=a.get("X", zeros),
        y=a.get("Y", zeros),
        delta=delta,
        corner_gain=corner_time_gain(grid, delta, corners),
    )


def dominance_segments(res: DuelResult) -> np.ndarray:
    return dominance_from_delta(res.distance, res.delta)


def format_laptime(td) -> str:
    """Timedelta -> '1:15.123'."""
    if td is None or pd.isna(td):
        return "—"
    total = pd.Timedelta(td).total_seconds()
    m, s = divmod(total, 60)
    return f"{int(m)}:{s:06.3f}"
