"""
f1_analysis.py

Analysis layer for the F1 Telemetry Duel app. Everything here is plain
pandas/NumPy, so it can be unit tested without network access.

Pipeline for a duel:
  1. Build each lap's telemetry table from the raw OpenF1 feeds: car data
     (speed, throttle, brake, gear) and GPS position arrive as separate
     streams at different timestamps, so position is interpolated onto the
     car-data timestamps.
  2. Compute distance traveled by integrating speed over time (trapezoid
     rule), then re-measure the second lap's distance along the first lap's
     GPS racing line, so the same distance means the same spot on track
     for both drivers.
  3. Clean the telemetry (drop bad rows, keep distance strictly increasing,
     fill small gaps, filter single-sample speed spikes).
  4. Align both laps on a shared distance grid and compute the time gap
     between the drivers at every point on track.
  5. Detect braking zones from the speed trace and attribute the time each
     driver gained or lost through each one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Fixed, always-distinguishable duel colors (independent of team livery, so a
# teammate-vs-teammate comparison is still readable).
DRIVER_A_COLOR = "#00D9FF"  # electric blue
DRIVER_B_COLOR = "#FF3366"  # signal red

# Bump when the analysis changes, so cached results from older code are discarded.
ANALYSIS_VERSION = 3

DISTANCE_STEP_M = 2.0  # resolution of the common distance grid, in meters
SPEED_FILTER_WINDOW = 3  # samples in the rolling median used to remove spikes
GPS_MEDIAN_WINDOW = 5    # samples; drops single GPS outliers
GPS_SMOOTH_WINDOW = 15   # samples (~4 s at 3.7 Hz); keeps only slow drift

# Channels the app plots. Distance and Time are required for alignment.
CHANNELS = ["Speed", "Throttle", "Brake", "nGear", "X", "Y"]


# -------------------------------------------- 1-2. raw feeds -> lap table ---

def build_lap_telemetry(car: pd.DataFrame, loc: pd.DataFrame, lap_start, lap_duration_s: float) -> pd.DataFrame:
    """Turn raw OpenF1 car_data + location rows for one lap into one table.

    Output columns: Time (s from lap start), Distance (m), Speed (km/h),
    Throttle (%), Brake (0/1), nGear, X, Y.
    """
    start = pd.Timestamp(lap_start)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")

    car = car.copy()
    car["t"] = (pd.to_datetime(car["date"], utc=True, format="ISO8601") - start).dt.total_seconds()
    car = car.sort_values("t").drop_duplicates("t")
    car = car[(car["t"] >= 0) & (car["t"] <= lap_duration_s)]
    if len(car) < 20:
        raise ValueError("Not enough car telemetry on this lap.")

    df = pd.DataFrame(
        {
            "Time": car["t"].to_numpy(dtype=float),
            "Speed": car["speed"].to_numpy(dtype=float),
            "Throttle": car["throttle"].to_numpy(dtype=float),
            "Brake": (car["brake"].to_numpy(dtype=float) > 0).astype(float),
            "nGear": car["n_gear"].to_numpy(dtype=float),
        }
    )

    # Telemetry starts slightly after the line and ends slightly before it.
    # Anchor the lap at t=0 and t=lap_duration using the nearest samples, so
    # the integrated distance and elapsed time cover the whole lap.
    first, last = df.iloc[[0]].copy(), df.iloc[[-1]].copy()
    first["Time"], last["Time"] = 0.0, float(lap_duration_s)
    df = pd.concat([first, df, last]).drop_duplicates("Time").sort_values("Time").reset_index(drop=True)

    # GPS position comes on its own timestamps: interpolate onto ours.
    if loc is not None and not loc.empty and {"x", "y", "date"} <= set(loc.columns):
        lt = (pd.to_datetime(loc["date"], utc=True, format="ISO8601") - start).dt.total_seconds()
        order = np.argsort(lt.to_numpy())
        lt = lt.to_numpy()[order]
        df["X"] = np.interp(df["Time"], lt, loc["x"].to_numpy(dtype=float)[order])
        df["Y"] = np.interp(df["Time"], lt, loc["y"].to_numpy(dtype=float)[order])

    df["Distance"] = integrate_distance(df["Time"].to_numpy(), df["Speed"].to_numpy())
    return df


def integrate_distance(t_s: np.ndarray, speed_kmh: np.ndarray) -> np.ndarray:
    """Distance traveled (m) by trapezoid-rule integration of speed over time."""
    v = np.asarray(speed_kmh, dtype=float) / 3.6
    dt = np.diff(np.asarray(t_s, dtype=float))
    return np.concatenate([[0.0], np.cumsum((v[1:] + v[:-1]) / 2 * dt)])


def scale_to_length(lap: pd.DataFrame, length_m: float) -> pd.DataFrame:
    """Stretch a lap's distance axis so it ends exactly at `length_m`."""
    out = lap.copy()
    out["Distance"] *= length_m / out["Distance"].iloc[-1]
    return out


def project_onto_reference(ref: pd.DataFrame, other: pd.DataFrame, window_m: float = 250.0) -> pd.DataFrame:
    """Re-measure `other`'s distance along `ref`'s racing line using GPS.

    Integrated speed drifts a little differently on every lap, so "3000 m"
    on one lap isn't quite the same place as "3000 m" on another. That shows
    up as fake time swings in braking zones. Instead, each GPS point of the
    other lap is matched to the nearest point on the reference lap's path,
    and takes the reference lap's distance at that spot. The search is
    limited to a window around where the point should be, so a nearby
    stretch of track (for example a hairpin or a parallel straight) can't
    be matched by mistake.
    """
    if not {"X", "Y"} <= set(ref.columns) or not {"X", "Y"} <= set(other.columns):
        return other

    ref_end = ref["Distance"].iloc[-1]
    d_ref = np.arange(0.0, ref_end, 1.0)
    x_ref = np.interp(d_ref, ref["Distance"], ref["X"])
    y_ref = np.interp(d_ref, ref["Distance"], ref["Y"])

    guess = other["Distance"].to_numpy() * (ref_end / other["Distance"].iloc[-1])
    ox, oy = other["X"].to_numpy(), other["Y"].to_numpy()
    w = int(window_m)
    matched = np.empty(len(other))
    for i, g in enumerate(guess):
        c = int(round(g))
        lo, hi = max(0, c - w), min(len(d_ref), c + w + 1)
        if hi <= lo:
            matched[i] = g
            continue
        j = lo + int(np.argmin((x_ref[lo:hi] - ox[i]) ** 2 + (y_ref[lo:hi] - oy[i]) ** 2))
        matched[i] = d_ref[j]
        # Refine below 1 m: fit a parabola through the squared distances at
        # j-1, j, j+1 and take its minimum.
        if 0 < j < len(d_ref) - 1:
            f = (x_ref[j - 1:j + 2] - ox[i]) ** 2 + (y_ref[j - 1:j + 2] - oy[i]) ** 2
            curv = f[0] - 2 * f[1] + f[2]
            if curv > 0:
                matched[i] += float(np.clip(0.5 * (f[0] - f[2]) / curv, -0.5, 0.5))

    # Complementary filter: GPS matching has no drift but a few meters of
    # jitter per sample; integrated speed is smooth but drifts. Keep the
    # integrated distance and correct it with only the slow-moving part of
    # the GPS difference (median to drop outliers, then a ~4 s moving mean).
    integrated = other["Distance"].to_numpy() * (ref_end / other["Distance"].iloc[-1])
    correction = pd.Series(matched - integrated)
    correction.iloc[0] = 0.0
    correction.iloc[-1] = 0.0
    smooth = (
        correction.rolling(GPS_MEDIAN_WINDOW, center=True, min_periods=1).median()
        .rolling(GPS_SMOOTH_WINDOW, center=True, min_periods=1).mean()
        .to_numpy()
    )
    # The lap starts and ends on the line by definition: remove whatever
    # correction is left at the ends with a straight ramp, so there's no step.
    smooth -= np.linspace(smooth[0], smooth[-1], len(smooth))

    out = other.copy()
    out["Distance"] = integrated + smooth
    return out


# ------------------------------------------------------------- 3. clean ----

def clean_telemetry(tel: pd.DataFrame) -> pd.DataFrame:
    """Clean one lap of telemetry so it can be interpolated safely.

    1. Keep only the channels we use; Time becomes elapsed seconds.
    2. Drop rows with missing Distance or Time.
    3. Sort by distance and drop samples where distance doesn't increase, so
       the distance axis is strictly increasing (interpolation requires it).
    4. Fill small gaps in the other channels by interpolating.
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
        df["Brake"] = (df["Brake"].astype(float) > 0).astype(float)

    df = df.dropna(subset=["Distance", "Time"]).sort_values("Distance", kind="stable")
    df = df[df["Distance"].diff().fillna(1.0) > 0].reset_index(drop=True)

    df["Distance"] -= df["Distance"].iloc[0]
    df["Time"] -= df["Time"].iloc[0]

    others = [c for c in CHANNELS if c in df.columns]
    if others:
        df[others] = df[others].astype(float).interpolate(limit_direction="both")

    if "Speed" in df.columns:
        df["Speed"] = df["Speed"].rolling(SPEED_FILTER_WINDOW, center=True, min_periods=1).median()

    if len(df) < 20:
        raise ValueError("Not enough clean telemetry points to analyse this lap.")
    return df


# ------------------------------------------------------------- 4. align ----

def align_laps(a: pd.DataFrame, b: pd.DataFrame, step_m: float = DISTANCE_STEP_M):
    """Resample two cleaned laps onto one shared distance grid.

    Returns (grid, channels_a, channels_b); channels_* map a channel name
    (including "Time") to its values at each grid point.
    """
    end = min(a["Distance"].iloc[-1], b["Distance"].iloc[-1])
    grid = np.append(np.arange(0.0, end, step_m), end)
    names = ["Time"] + [c for c in CHANNELS if c in a.columns and c in b.columns]
    ch_a = {n: np.interp(grid, a["Distance"], a[n]) for n in names}
    ch_b = {n: np.interp(grid, b["Distance"], b[n]) for n in names}
    return grid, ch_a, ch_b


# ------------------------------------------------- 5. braking zones/gains ---

def detect_braking_zones(grid: np.ndarray, speed: np.ndarray, min_drop_kmh: float = 25.0,
                         min_gap_m: float = 150.0) -> pd.DataFrame:
    """Find corners as the slowest points after a significant braking event.

    A point counts as a corner apex when it's the minimum speed within
    `min_gap_m` either side and the car slowed by at least `min_drop_kmh`
    from the preceding peak. Returns columns Number, Letter, Distance.
    """
    v = pd.Series(speed).rolling(5, center=True, min_periods=1).mean().to_numpy()
    half = max(1, int(min_gap_m / max(grid[1] - grid[0], 1e-6)))
    apexes = []
    for i in range(1, len(v) - 1):
        lo, hi = max(0, i - half), min(len(v), i + half + 1)
        if v[i] != v[lo:hi].min():
            continue
        before = v[max(0, i - 4 * half):i + 1]
        if before.max() - v[i] < min_drop_kmh:
            continue
        if apexes and grid[i] - apexes[-1] < min_gap_m:
            continue
        apexes.append(grid[i])
    return pd.DataFrame(
        {"Number": range(1, len(apexes) + 1), "Letter": [""] * len(apexes), "Distance": apexes}
    )


def corner_time_gain(grid: np.ndarray, delta: np.ndarray, corners: pd.DataFrame | None) -> pd.DataFrame:
    """Attribute time gained/lost to each corner.

    Each corner's window runs from the midpoint with the previous apex to the
    midpoint with the next one (approximately entry through exit). The change
    in the gap across that window is the time driver A gained (positive) or
    lost (negative) there.
    """
    cols = ["corner_number", "letter", "distance", "gain_a_seconds"]
    if corners is None or corners.empty:
        return pd.DataFrame(columns=cols)

    corners = corners.sort_values("Distance")
    apexes = corners["Distance"].to_numpy(dtype=float)
    n = len(apexes)
    rows = []
    for i in range(n):
        apex = apexes[i]
        prev_apex = apexes[i - 1] if i > 0 else 0.0
        next_apex = apexes[i + 1] if i < n - 1 else grid[-1]
        start = (prev_apex + apex) / 2 if i > 0 else 0.0
        end = (apex + next_apex) / 2 if i < n - 1 else grid[-1]
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
    return pd.DataFrame(rows, columns=cols)


def dominance_from_delta(grid: np.ndarray, delta: np.ndarray, window: int = 15) -> np.ndarray:
    """For each grid point, which driver is gaining time there ("A" or "B").

    Uses the slope of the gap curve, smoothed so the track map doesn't
    flicker between colors on small noise.
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
    lap_time_a: float               # seconds
    lap_time_b: float
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
    corner_gain: pd.DataFrame       # time gained/lost by A through each braking zone


def compare_laps(lap_a: pd.DataFrame, lap_b: pd.DataFrame, meta: dict) -> DuelResult:
    """Compare two laps built by build_lap_telemetry().

    `meta` holds driver_a, driver_b, team_a, team_b, lap_time_a, lap_time_b.
    """
    track_len = (lap_a["Distance"].iloc[-1] + lap_b["Distance"].iloc[-1]) / 2
    ref = scale_to_length(lap_a, track_len)
    other = project_onto_reference(ref, scale_to_length(lap_b, track_len))
    a_clean = clean_telemetry(ref)
    b_clean = clean_telemetry(other)

    grid, a, b = align_laps(a_clean, b_clean)
    delta = b["Time"] - a["Time"]  # +ve => A ahead (B took longer to get here)

    corners = detect_braking_zones(grid, (a["Speed"] + b["Speed"]) / 2)
    zeros = np.zeros_like(grid)

    return DuelResult(
        driver_a=meta["driver_a"],
        driver_b=meta["driver_b"],
        team_a=meta.get("team_a", ""),
        team_b=meta.get("team_b", ""),
        lap_time_a=float(meta["lap_time_a"]),
        lap_time_b=float(meta["lap_time_b"]),
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


def format_laptime(seconds) -> str:
    """82.207 -> '1:22.207'."""
    if seconds is None:
        return "—"
    if isinstance(seconds, pd.Timedelta):
        seconds = seconds.total_seconds()
    if not np.isfinite(seconds):
        return "—"
    m, s = divmod(float(seconds), 60)
    return f"{int(m)}:{s:06.3f}"
