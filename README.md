# F1 Telemetry Duel

An interactive web app that pulls **real Formula 1 telemetry** (official car
ECU + GPS data: speed, throttle, brake, gear, position, sampled multiple
times per second) and compares two drivers' fastest laps to show exactly
where one gains or loses time on track.

**Live app:** https://f1-telemetry-duel.streamlit.app

## How it works

1. **Cleans the raw telemetry.** Each lap's data is filtered before any
   comparison: rows with missing distance or time are dropped, duplicate and
   out-of-order distance samples are removed so the distance axis only moves
   forward, small gaps in other channels are filled, and a short rolling
   median removes single-sample speed spikes.
2. **Aligns both laps on a shared distance axis.** Cars cross the line at
   different moments and log samples at different points on track, so raw
   rows aren't comparable. Both laps are resampled onto the same 2 m
   distance grid.
3. **Builds a time-delta-vs-distance curve.** On that shared grid, the gap at
   every point is driver B's elapsed lap time minus driver A's.
4. **Attributes time gained or lost to individual corners.** Using each
   circuit's real corner locations (from FastF1), the delta curve is split
   into windows around every corner, and the change in the gap across each
   window is the time won or lost there.
5. **Renders a track dominance map.** The circuit outline is colored by which
   driver is *gaining* time at each point (the slope of the delta curve),
   not just who's faster in a straight line.

## Reliability

FastF1's `session.load()` doesn't raise an error when a download fails; it
leaves the session half-loaded. The app checks that lap data actually
arrived before using or caching a session. On first visit it loads the most
recent completed session, and if that one isn't available yet it steps back
to earlier events instead of showing an error. Dependencies are pinned to
bounded version ranges so a future major release can't silently break it.

## Data source

[FastF1](https://github.com/theOehrly/Fast-F1), which sources official F1
live-timing and car telemetry feeds. Data is cached locally after the first
fetch, so repeat views of the same session are fast.

## Running it locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Tests

The analysis is tested on simulated laps where the correct answer is known
(for example, two cars at constant speeds have an exact, computable gap),
and the full app is run against a fake session, including the case where
the newest session fails to load.

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

## Stack

Python · pandas · NumPy · FastF1 · Streamlit · Plotly
