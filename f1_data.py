"""
f1_data.py

Data access for F1 Telemetry Duel, using the OpenF1 API (https://openf1.org).

Why OpenF1: in October 2026 Formula 1 put its live-timing archive behind an
F1 TV login, which broke the FastF1 library this app originally used
(FastF1 issue #978). OpenF1 serves the same official timing and car
telemetry for 2023 onward, free and without authentication.

The free tier allows about 3 requests/second and 30/minute, so every request
is spaced out, retried politely on HTTP 429, and cached so repeat views of a
session don't hit the API again.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import streamlit as st

BASE_URL = "https://api.openf1.org/v1"
FIRST_SEASON = 2023  # OpenF1 historical coverage starts here
MIN_INTERVAL_S = 0.4  # stay under the free tier's 3 requests/second
TIMEOUT_S = 30
MAX_RETRIES = 4

_lock = threading.Lock()
_last_call = [0.0]


class DataUnavailableError(RuntimeError):
    """The API had no usable data for what was asked."""


def _get(endpoint: str, params: dict | None = None) -> list[dict]:
    """GET an OpenF1 endpoint with spacing between calls and retries on 429/5xx."""
    url = f"{BASE_URL}/{endpoint}"
    for attempt in range(MAX_RETRIES):
        with _lock:
            wait = MIN_INTERVAL_S - (time.monotonic() - _last_call[0])
            if wait > 0:
                time.sleep(wait)
            _last_call[0] = time.monotonic()
        try:
            resp = requests.get(url, params=params, timeout=TIMEOUT_S)
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES - 1:
                raise DataUnavailableError(f"Couldn't reach OpenF1: {exc}") from exc
            time.sleep(2 ** attempt)
            continue

        if resp.status_code == 429 or resp.status_code >= 500:
            retry_after = resp.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 2 ** (attempt + 1)
            time.sleep(min(delay, 20))
            continue
        if resp.status_code == 404:
            return []
        if not resp.ok:
            raise DataUnavailableError(f"OpenF1 returned HTTP {resp.status_code} for {endpoint}.")
        data = resp.json()
        # OpenF1 returns {"detail": "No results found."} instead of [] sometimes.
        return data if isinstance(data, list) else []
    raise DataUnavailableError("OpenF1 is rate-limiting requests right now. Try again in a minute.")


def _ts(s) -> pd.Timestamp:
    return pd.to_datetime(s, utc=True, format="ISO8601")


# ------------------------------------------------------------- calendar ----

@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def get_meetings(year: int) -> pd.DataFrame:
    """Grand Prix weekends for a season that have already finished at least one session."""
    rows = _get("meetings", {"year": year})
    if not rows:
        return pd.DataFrame(columns=["meeting_key", "meeting_name", "location", "date_start"])
    df = pd.DataFrame(rows)
    df = df[~df["meeting_name"].str.contains("Testing", case=False, na=False)]
    if "is_cancelled" in df.columns:
        df = df[~df["is_cancelled"].fillna(False).astype(bool)]
    df["date_start"] = _ts(df["date_start"])
    now = datetime.now(timezone.utc)
    df = df[df["date_start"] <= now]
    return df.sort_values("date_start")[["meeting_key", "meeting_name", "location", "date_start"]]


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def get_sessions(meeting_key: int) -> pd.DataFrame:
    """Sessions of a weekend that ended over an hour ago (free data is historical only)."""
    rows = _get("sessions", {"meeting_key": meeting_key})
    if not rows:
        return pd.DataFrame(columns=["session_key", "session_name", "date_start", "date_end"])
    df = pd.DataFrame(rows)
    if "is_cancelled" in df.columns:
        df = df[~df["is_cancelled"].fillna(False).astype(bool)]
    df["date_start"] = _ts(df["date_start"])
    df["date_end"] = _ts(df["date_end"])
    cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
    df = df[df["date_end"] <= cutoff]
    return df.sort_values("date_start")[["session_key", "session_name", "date_start", "date_end"]]


# ------------------------------------------------------------ session data --

@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def get_drivers(session_key: int) -> pd.DataFrame:
    rows = _get("drivers", {"session_key": session_key})
    if not rows:
        raise DataUnavailableError("No driver list for this session yet.")
    df = pd.DataFrame(rows).drop_duplicates("driver_number")
    return df[["driver_number", "name_acronym", "full_name", "team_name", "team_colour"]]


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def get_laps(session_key: int) -> pd.DataFrame:
    """Every timed lap in the session (all drivers), from one request."""
    rows = _get("laps", {"session_key": session_key})
    if not rows:
        raise DataUnavailableError("No lap data for this session yet.")
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["lap_duration", "date_start"])
    if df.empty:
        raise DataUnavailableError("No timed laps in this session yet.")
    df["date_start"] = _ts(df["date_start"])
    return df[["driver_number", "lap_number", "lap_duration", "date_start"]]


def fastest_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Each driver's fastest timed lap."""
    idx = laps.groupby("driver_number")["lap_duration"].idxmin()
    return laps.loc[idx].sort_values("lap_duration").reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24, max_entries=128)
def get_lap_telemetry(session_key: int, driver_number: int, start_iso: str, duration_s: float):
    """Raw car data (speed, throttle, brake, gear, rpm) and GPS position for one lap.

    A small margin is fetched on each side so interpolation has data right up
    to the start and finish line; it's trimmed back to the lap afterwards.
    """
    start = _ts(start_iso)
    end = start + pd.Timedelta(seconds=duration_s)
    margin = pd.Timedelta(seconds=1)
    window = {
        "session_key": session_key,
        "driver_number": driver_number,
        "date>": (start - margin).isoformat(),
        "date<": (end + margin).isoformat(),
    }
    car = pd.DataFrame(_get("car_data", window))
    loc = pd.DataFrame(_get("location", window))
    if car.empty:
        raise DataUnavailableError(f"No car telemetry for driver #{driver_number} on this lap.")
    return car, loc
