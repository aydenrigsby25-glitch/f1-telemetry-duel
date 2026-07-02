# F1 Telemetry Duel

An interactive web app that pulls **real Formula 1 telemetry** (official car
ECU + GPS data: speed, throttle, brake, gear, position — sampled multiple
times per second) and overlays two drivers' fastest laps to show exactly
where one gains or loses time on track.

Built to demonstrate applied data-science thinking on motorsport data, not
just a stats dashboard.

## What makes it more than a chart

Most "F1 data" projects plot a speed trace and stop there. This one:

1. **Puts both drivers on a shared distance axis.** Cars don't cross the
   line at the same instant, so raw timestamps aren't directly comparable —
   telemetry is resampled onto a common 2m distance grid for both drivers.
2. **Derives a time-delta-vs-distance curve from first principles**, by
   integrating each driver's speed trace into elapsed lap time over that
   shared grid, rather than relying on FastF1's built-in `delta_time` helper
   (which its own docs flag as imprecise).
3. **Attributes time gained/lost to individual corners.** Using the
   circuit's real corner coordinates (also sourced from FastF1), the delta
   curve is segmented apex-to-apex to produce a "seconds gained per corner"
   breakdown — the same kind of trace comparison race engineers use.
4. **Renders a track dominance map** — the circuit outline colored, segment
   by segment, by which driver is *gaining* time there (via the slope of the
   delta curve), not just who's faster in a straight line.

## Data source

[FastF1](https://github.com/theOehrly/Fast-F1), which sources official F1
live-timing and car telemetry feeds. Data is cached locally after first
fetch, so repeat views of the same session are instant.

## Running it

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the printed local URL, pick a season / Grand Prix / session in the
sidebar, and choose two drivers to compare.

## Stack

Python · FastF1 · Streamlit · Plotly · pandas/numpy
