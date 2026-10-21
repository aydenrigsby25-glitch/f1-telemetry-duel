"""
F1 Telemetry Duel
An interactive tool that compares two drivers' real telemetry from their
fastest laps to show exactly where one gains or loses time on track.

Data: official F1 timing and car telemetry via the OpenF1 API (2023 onward).
"""

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import f1_analysis as f1
import f1_data as data

st.set_page_config(
    page_title="F1 Telemetry Duel",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

A_COLOR = f1.DRIVER_A_COLOR
B_COLOR = f1.DRIVER_B_COLOR
BG = "#0E1117"
GRID = "#2A2E39"

CUSTOM_CSS = f"""
<style>
.stApp {{ background-color: {BG}; }}
[data-testid="stMetricValue"] {{ font-size: 1.6rem; }}
.duel-header {{
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin-bottom: 0;
}}
.duel-sub {{
    color: #8A8F98;
    font-size: 0.95rem;
    margin-top: 0.1rem;
}}
.callout {{
    background: linear-gradient(90deg, rgba(0,217,255,0.08), rgba(255,51,102,0.08));
    border-left: 4px solid {A_COLOR};
    padding: 0.9rem 1.1rem;
    border-radius: 6px;
    font-size: 1.05rem;
    margin: 0.6rem 0 1.2rem 0;
}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ------------------------------------------------------------ data helpers --

def load_session(session_key: int):
    """Laps + drivers for a session. Raises DataUnavailableError if missing."""
    laps = data.get_laps(session_key)
    drivers = data.get_drivers(session_key)
    best = data.fastest_laps(laps).merge(drivers, on="driver_number", how="inner")
    if len(best) < 2:
        raise data.DataUnavailableError("Fewer than two drivers set a timed lap in this session.")
    return best


@st.cache_data(show_spinner=False, max_entries=64)
def run_duel(session_key: int, row_a: dict, row_b: dict) -> f1.DuelResult:
    laps = []
    for row in (row_a, row_b):
        car, loc = data.get_lap_telemetry(
            session_key, int(row["driver_number"]), row["date_start"], float(row["lap_duration"])
        )
        laps.append(f1.build_lap_telemetry(car, loc, row["date_start"], float(row["lap_duration"])))
    meta = {
        "driver_a": row_a["name_acronym"], "driver_b": row_b["name_acronym"],
        "team_a": row_a["team_name"], "team_b": row_b["team_name"],
        "lap_time_a": row_a["lap_duration"], "lap_time_b": row_b["lap_duration"],
    }
    return f1.compare_laps(laps[0], laps[1], meta)


def pick_default_session(sessions: pd.DataFrame):
    """Qualifying if the weekend has it, otherwise the latest session."""
    quali = sessions[sessions["session_name"] == "Qualifying"]
    return (quali if not quali.empty else sessions).iloc[-1]


# ---------------------------------------------------------------- sidebar --
st.sidebar.title("🏎️ F1 Telemetry Duel")
st.sidebar.caption("Real F1 telemetry · powered by OpenF1")

current_year = dt.date.today().year
years = list(range(current_year, data.FIRST_SEASON - 1, -1))
year = st.sidebar.selectbox("Season", years, index=0)

try:
    meetings = data.get_meetings(year)
except Exception as e:
    st.sidebar.error(f"Couldn't load the {year} calendar: {e}")
    st.stop()
if meetings.empty:
    st.sidebar.info(f"No {year} race weekends have happened yet. Pick an earlier season.")
    st.stop()

meeting_names = meetings["meeting_name"].tolist()
meeting_name = st.sidebar.selectbox("Grand Prix", meeting_names, index=len(meeting_names) - 1)
meeting_key = int(meetings.loc[meetings["meeting_name"] == meeting_name, "meeting_key"].iloc[-1])

try:
    sessions = data.get_sessions(meeting_key)
except Exception as e:
    st.sidebar.error(f"Couldn't load sessions: {e}")
    st.stop()
if sessions.empty:
    st.sidebar.info("No completed sessions for this weekend yet.")
    st.stop()

session_names = sessions["session_name"].tolist()
default_session = pick_default_session(sessions)["session_name"]
session_name = st.sidebar.selectbox(
    "Session", session_names, index=session_names.index(default_session)
)
session_key = int(sessions.loc[sessions["session_name"] == session_name, "session_key"].iloc[-1])

load_clicked = st.sidebar.button("Load session", type="primary", width="stretch")

if "loaded" not in st.session_state:
    st.session_state.loaded = None  # (session_key, label)
    st.session_state.best = None

if load_clicked:
    label = f"{year} {meeting_name} · {session_name}"
    with st.spinner(f"Pulling {label} data..."):
        try:
            st.session_state.best = load_session(session_key)
            st.session_state.loaded = (session_key, label)
        except Exception as e:
            st.sidebar.error(f"That session isn't available right now: {e}")

elif st.session_state.loaded is None:
    # First visit: load the newest weekend's qualifying. If its data isn't
    # published yet, step back through earlier weekends instead of erroring.
    with st.spinner("Loading the latest session..."):
        for _, m in meetings.iloc[::-1].head(4).iterrows():
            try:
                sess = data.get_sessions(int(m["meeting_key"]))
                if sess.empty:
                    continue
                s = pick_default_session(sess)
                st.session_state.best = load_session(int(s["session_key"]))
                st.session_state.loaded = (
                    int(s["session_key"]), f"{year} {m['meeting_name']} · {s['session_name']}"
                )
                break
            except Exception:
                continue

if st.session_state.loaded is None:
    st.title("🏎️ F1 Telemetry Duel")
    st.write(
        "Pick a season, Grand Prix and session in the sidebar, then hit "
        "**Load session** to pull real telemetry and compare two drivers "
        "lap-for-lap."
    )
    st.stop()

loaded_key, loaded_label = st.session_state.loaded
best = st.session_state.best
if loaded_key != session_key:
    st.info(f"Showing {loaded_label}. Hit **Load session** to switch to your selection.")

codes = best["name_acronym"].tolist()
labels = {
    r.name_acronym: f"{r.name_acronym} — {r.full_name.title()} ({r.team_name})"
    for r in best.itertuples()
}

st.sidebar.markdown("---")
driver_a = st.sidebar.selectbox("Driver A", codes, index=0, format_func=lambda c: labels[c])
driver_b = st.sidebar.selectbox("Driver B", codes, index=1, format_func=lambda c: labels[c])

if driver_a == driver_b:
    st.warning("Pick two different drivers to run a duel.")
    st.stop()


def row_for(code: str) -> dict:
    r = best[best["name_acronym"] == code].iloc[0]
    return {
        "driver_number": int(r["driver_number"]),
        "name_acronym": r["name_acronym"],
        "team_name": r["team_name"],
        "lap_duration": float(r["lap_duration"]),
        "date_start": pd.Timestamp(r["date_start"]).isoformat(),
    }


with st.spinner(f"Crunching telemetry: {driver_a} vs {driver_b}..."):
    try:
        res = run_duel(loaded_key, row_for(driver_a), row_for(driver_b))
    except Exception as e:
        st.error(f"Couldn't build this comparison: {e}")
        st.stop()

# ------------------------------------------------------------------ header --
gap = res.lap_time_b - res.lap_time_a
leader = res.driver_a if gap > 0 else res.driver_b
gap_abs = abs(gap)

st.markdown(
    f'<div class="duel-header">'
    f'<span style="color:{A_COLOR}">{res.driver_a}</span> vs '
    f'<span style="color:{B_COLOR}">{res.driver_b}</span>'
    f"</div>",
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="duel-sub">{loaded_label} · fastest laps compared</div>',
    unsafe_allow_html=True,
)
st.write("")

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"{res.driver_a} lap", f1.format_laptime(res.lap_time_a))
c2.metric(f"{res.driver_b} lap", f1.format_laptime(res.lap_time_b))
c3.metric("Gap", f"{gap_abs:.3f}s", delta=f"{leader} ahead", delta_color="off")
c4.metric("Top speed A / B", f"{res.speed_a.max():.0f} / {res.speed_b.max():.0f} km/h")

# biggest single-corner swing, for a punchy narrative line
gains = res.corner_gain
if not gains.empty:
    best_a = gains.loc[gains["gain_a_seconds"].idxmax()]
    best_b = gains.loc[gains["gain_a_seconds"].idxmin()]
    parts = []
    if best_a.gain_a_seconds > 0:
        parts.append(
            f'<b style="color:{A_COLOR}">{res.driver_a}</b> gains the most at '
            f'<b>corner {int(best_a.corner_number)}</b> (+{best_a.gain_a_seconds:.2f}s)'
        )
    if best_b.gain_a_seconds < 0:
        parts.append(
            f'<b style="color:{B_COLOR}">{res.driver_b}</b> gains the most at '
            f'<b>corner {int(best_b.corner_number)}</b> (+{-best_b.gain_a_seconds:.2f}s)'
        )
    if parts:
        st.markdown(
            '<div class="callout">🔑 Biggest swing: ' + " · ".join(parts) + "</div>",
            unsafe_allow_html=True,
        )

# ------------------------------------------------------------ track map ----
st.subheader("Track dominance map")
st.caption(
    "Colored by who is *gaining* time at each point on track — not just who's "
    "faster in a straight line. Derived from the slope of the time-delta curve."
)

faster = f1.dominance_segments(res)
fig_map = go.Figure()

# draw contiguous same-color runs as separate line segments (with 1pt overlap
# so the line looks continuous rather than dashed)
start_i = 0
for i in range(1, len(faster) + 1):
    if i == len(faster) or faster[i] != faster[start_i]:
        end_i = min(i + 1, len(faster))
        color = A_COLOR if faster[start_i] == "A" else B_COLOR
        fig_map.add_trace(
            go.Scatter(
                x=res.x[start_i:end_i],
                y=res.y[start_i:end_i],
                mode="lines",
                line=dict(color=color, width=6),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        start_i = i

fig_map.update_layout(
    height=520,
    plot_bgcolor=BG,
    paper_bgcolor=BG,
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
    yaxis=dict(visible=False),
)
fig_map.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                              line=dict(color=A_COLOR, width=6), name=f"{res.driver_a} faster here"))
fig_map.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                              line=dict(color=B_COLOR, width=6), name=f"{res.driver_b} faster here"))
fig_map.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.0,
                                   font=dict(color="white"), bgcolor="rgba(0,0,0,0)"))
st.plotly_chart(fig_map, width="stretch", config={"displayModeBar": False})

# -------------------------------------------------------- speed + delta ----
st.subheader("Speed trace & time delta")
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38],
    vertical_spacing=0.05,
    subplot_titles=("Speed (km/h)", f"Time delta — gap to {res.driver_a} (s)"),
)
fig.add_trace(go.Scatter(x=res.distance, y=res.speed_a, name=res.driver_a,
                          line=dict(color=A_COLOR, width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=res.distance, y=res.speed_b, name=res.driver_b,
                          line=dict(color=B_COLOR, width=2)), row=1, col=1)

delta_color = np.where(res.delta >= 0, A_COLOR, B_COLOR)
fig.add_trace(
    go.Scatter(
        x=res.distance, y=res.delta, mode="lines", line=dict(color="#AAAAAA", width=1.5),
        fill="tozeroy", fillcolor="rgba(0,217,255,0.15)", name="delta", showlegend=False,
    ),
    row=2, col=1,
)
fig.add_hline(y=0, line=dict(color=GRID, width=1), row=2, col=1)

fig.update_layout(
    height=560, plot_bgcolor=BG, paper_bgcolor=BG,
    legend=dict(orientation="h", y=1.12, font=dict(color="white")),
    font=dict(color="#CCCCCC"),
    margin=dict(l=10, r=10, t=60, b=10),
)
fig.update_xaxes(title_text="Distance (m)", gridcolor=GRID, row=2, col=1)
fig.update_yaxes(gridcolor=GRID)
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

# ------------------------------------------------------ throttle & brake ---
st.subheader("Throttle & brake")
fig2 = make_subplots(rows=1, cols=1)
fig2.add_trace(go.Scatter(x=res.distance, y=res.throttle_a, name=f"{res.driver_a} throttle",
                           line=dict(color=A_COLOR, width=1.6)))
fig2.add_trace(go.Scatter(x=res.distance, y=res.throttle_b, name=f"{res.driver_b} throttle",
                           line=dict(color=B_COLOR, width=1.6)))
fig2.add_trace(go.Scatter(x=res.distance, y=res.brake_a.astype(int) * -15, name=f"{res.driver_a} brake",
                           line=dict(color=A_COLOR, width=1), fill="tozeroy",
                           fillcolor="rgba(0,217,255,0.35)"))
fig2.add_trace(go.Scatter(x=res.distance, y=res.brake_b.astype(int) * -15, name=f"{res.driver_b} brake",
                           line=dict(color=B_COLOR, width=1), fill="tozeroy",
                           fillcolor="rgba(255,51,102,0.35)"))
fig2.update_layout(
    height=320, plot_bgcolor=BG, paper_bgcolor=BG, font=dict(color="#CCCCCC"),
    legend=dict(orientation="h", y=1.15, font=dict(color="white")),
    margin=dict(l=10, r=10, t=10, b=10),
    yaxis=dict(title="Throttle % (brake shaded below 0)", gridcolor=GRID),
    xaxis=dict(title="Distance (m)", gridcolor=GRID),
)
st.plotly_chart(fig2, width="stretch", config={"displayModeBar": False})

# --------------------------------------------------- corner-by-corner gain -
st.subheader("Time gained / lost by corner")
if gains.empty:
    st.caption("No clear braking zones were detected on this lap, so the per-corner breakdown is skipped.")
else:
    st.caption(
        f"Corners are detected from the speed trace (each major braking zone) and numbered in lap order, "
        f"so numbers may differ from official turn numbers. Positive bars = {res.driver_a} gains time "
        f"through that corner. Negative = {res.driver_b} does."
    )
    gains_sorted = gains.sort_values("distance")
    bar_colors = [A_COLOR if v >= 0 else B_COLOR for v in gains_sorted["gain_a_seconds"]]
    fig3 = go.Figure(
        go.Bar(
            x=[f"C{int(n)}" for n in gains_sorted["corner_number"]],
            y=gains_sorted["gain_a_seconds"],
            marker_color=bar_colors,
        )
    )
    fig3.update_layout(
        height=340, plot_bgcolor=BG, paper_bgcolor=BG, font=dict(color="#CCCCCC"),
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(title="Seconds gained by " + res.driver_a, gridcolor=GRID, zerolinecolor=GRID),
        xaxis=dict(gridcolor=GRID),
    )
    st.plotly_chart(fig3, width="stretch", config={"displayModeBar": False})

st.markdown("---")
st.caption(
    "Data: OpenF1 (official F1 timing & car telemetry, 2023 onward). Distance is "
    "computed by integrating each car's speed over time; corner attribution "
    "segments the time-gap curve around braking zones detected in the speed "
    "trace. Not an official F1 statistic."
)
