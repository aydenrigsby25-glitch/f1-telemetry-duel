# F1 Telemetry Duel

An interactive web app that pulls **real Formula 1 telemetry** (official car
data: speed, throttle, brake, gear, plus GPS position, sampled several times
per second) and compares two drivers' fastest laps to show exactly where one
gains or loses time on track.

**Live app:** https://f1-telemetry-duel.streamlit.app

## How it works

1. **Builds each lap from raw telemetry feeds.** Car data and GPS position
   arrive as separate streams on different timestamps, so position is
   interpolated onto the car-data timestamps.
2. **Computes distance by integrating speed over time** (trapezoid rule),
   then re-measures the second lap along the first lap's GPS racing line.
   Integrated speed drifts slightly differently on every lap, which shows up
   as fake time swings in braking zones; matching GPS positions makes the
   same distance mean the same spot on track for both drivers.
3. **Cleans the data.** Rows with missing values are dropped, duplicate and
   out-of-order samples are removed so the distance axis only moves forward,
   small gaps are filled, and a short rolling median removes single-sample
   speed spikes.
4. **Aligns both laps on a shared distance grid** (every 2 m). Cars log
   samples at different points on track, so raw rows aren't comparable until
   they're resampled to the same places.
5. **Builds a time-gap-vs-distance curve**: at every point on track, driver
   B's elapsed lap time minus driver A's.
6. **Detects corners from the speed trace** (each major braking zone) and
   measures how much time each driver gained or lost through each one.
7. **Renders a track dominance map**: the circuit outline colored by which
   driver is *gaining* time at each point (the slope of the gap curve), not
   just who's faster in a straight line.

## Data source

[OpenF1](https://openf1.org), a free API serving official F1 timing and car
telemetry from the 2023 season onward.

The app originally used the FastF1 library. In October 2026 Formula 1 put
its live-timing archive behind an F1 TV login, which stopped FastF1 from
downloading sessions ([FastF1 issue #978](https://github.com/theOehrly/Fast-F1/issues/978)),
so the data layer was moved to OpenF1. Requests are spaced out to respect
OpenF1's free-tier rate limit, retried on HTTP 429, and cached.

## Reliability

- A session is only used once its lap data has actually loaded; a failed
  load is retried rather than cached.
- On first visit the app loads the latest qualifying session, and steps back
  to earlier weekends if the newest one isn't published yet.
- Dependencies are pinned to bounded version ranges so a future major
  release can't silently break the app.

## Running it locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Tests

The analysis is tested on simulated laps shaped exactly like OpenF1's
responses, where the correct answer is known: two cars at constant speeds
have an exact, computable gap, a car that's slower through one corner
should lose all its time there, and two identical laps where one speed
sensor reads 4% high must still come out level after GPS alignment. The full app is also run against fake API
data, including the cases where the newest session fails to load and where
the API rate-limits requests.

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

## Stack

Python · pandas · NumPy · Streamlit · Plotly · OpenF1 API
