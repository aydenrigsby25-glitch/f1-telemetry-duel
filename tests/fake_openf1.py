"""Synthetic laps shaped exactly like OpenF1 car_data / location responses."""

import numpy as np
import pandas as pd

TRACK_M = 5000.0
LAP_START = pd.Timestamp("2025-12-06T14:59:31.374", tz="UTC")


def simulate(speed_fn, hz=3.7, loc_offset_s=0.11, noise=0.0, seed=0):
    """Drive a 5 km circular track at speed_fn(distance) km/h.

    Returns (car_df, loc_df, lap_duration_s) in OpenF1's raw format. Car data
    and GPS are sampled on different timestamps, like the real feeds.
    """
    rng = np.random.default_rng(seed)
    dt_fine = 0.005
    t, d, ts, ds, vs = 0.0, 0.0, [], [], []
    while d < TRACK_M:
        v = speed_fn(d)
        ts.append(t); ds.append(d); vs.append(v)
        d += v / 3.6 * dt_fine
        t += dt_fine
    ts, ds, vs = np.array(ts), np.array(ds), np.array(vs)
    duration = t

    def sample(times):
        return np.interp(times, ts, ds), np.interp(times, ts, vs)

    car_t = np.arange(0.03, duration, 1 / hz)
    _, car_v = sample(car_t)
    car_v = car_v + rng.normal(0, noise, len(car_v)) if noise else car_v
    braking = np.gradient(car_v) < -2
    car = pd.DataFrame(
        {
            "date": [(LAP_START + pd.Timedelta(seconds=float(x))).isoformat() for x in car_t],
            "speed": np.round(car_v).astype(int),
            "throttle": np.where(braking, 0, 100),
            "brake": np.where(braking, 100, 0),
            "n_gear": np.clip((car_v / 45).astype(int) + 1, 1, 8),
            "rpm": 11000,
            "drs": 8,
            "driver_number": 1,
        }
    )

    loc_t = np.arange(loc_offset_s, duration, 1 / hz)
    loc_d, _ = sample(loc_t)
    ang = loc_d / TRACK_M * 2 * np.pi
    loc = pd.DataFrame(
        {
            "date": [(LAP_START + pd.Timedelta(seconds=float(x))).isoformat() for x in loc_t],
            "x": np.cos(ang) * 800,
            "y": np.sin(ang) * 800,
            "z": 0,
        }
    )
    return car, loc, duration
