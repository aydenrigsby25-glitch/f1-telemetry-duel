"""
F1 Telemetry Duel
An interactive tool that overlays two drivers' real telemetry from the same
lap to show exactly where one gains or loses time on track.

Data: official F1 timing/car telemetry via the FastF1 library.
"""

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import f1_analysis as f1

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

f1.enable_cache()

# ---------------------------------------------------------------- sidebar --
st.sidebar.title("🏎️ F1 Telemetry Duel")
st.sidebar.caption("Real session data · powered by FastF1")

current_year = dt.date.today().year
years = list(range(current_year, 2017, -1))
year = st.sidebar.selectbox("Season", years, index=0)

with st.spinner("Loading race calendar..."):
    try:
        schedule = f1.get_event_schedule(year)
    except Exception as e:
        st.sidebar.error(f"Couldn't load {year} calendar: {e}")
        st.stop()

event_names = schedule["EventName"].tolist()
# Default to the most recently *completed* race weekend, not the season
# finale — future events on the calendar have no data yet.
now = pd.Timestamp.now(tz=schedule["EventDate"].dt.tz)
completed = schedule[schedule["EventDate"] <= now]
if not completed.empty:
    default_event = completed.iloc[-1]["EventName"]
elif not schedule.empty:
    default_event = schedule.iloc[0]["EventName"]
else:
    default_event = None
default_idx = event_names.index(default_event) if default_event in event_names else 0
event_name = st.sidebar.selectbox("Grand Prix", event_names, index=default_idx)

session_labels = {
    "FP1": "Practice 1", "FP2": "Practice 2", "FP3": "Practice 3",
    "Q": "Qualifying", "S": "Sprint", "SQ": "Sprint Qualifying", "R": "Race",
}
session_code = st.sidebar.selectbox(
    "Session",
    list(session_labels.keys()),
    format_func=lambda c: session_labels[c],
    index=3,  # Qualifying
)

load_clicked = st.sidebar.button("Load session", type="primary", width="stretch")

if "session_key" not in st.session_state:
    st.session_state.session_key = None
    st.session_state.session_obj = None

if load_clicked or st.session_state.session_key is None:
    key = (year, event_name, session_code)
    if st.session_state.session_key != key:
        with st.spinner(f"Pulling {year} {event_name} {session_labels[session_code]} data..."):
            try:
                sess = f1.load_session(year, event_name, session_code)
                st.session_state.session_key = key
                st.session_state.session_obj = sess
            except Exception as e:
                st.sidebar.error(f"Session not available: {e}")
                st.stop()

session = st.session_state.session_obj
session_key = st.session_state.session_key

if session is None:
    st.title("🏎️ F1 Telemetry Duel")
    st.write(
        "Pick a season, Grand Prix and session in the sidebar, then hit "
        "**Load session** to pull real telemetry and compare two drivers "
        "lap-for-lap."
    )
    st.stop()

drivers = f1.list_drivers(session)
if len(drivers) < 2:
    st.error("Not enough driver data in this session yet.")
    st.stop()

driver_codes = [d["code"] for d in drivers]
driver_labels = {d["code"]: f"{d['code']} — {d['name']} ({d['team']})" for d in drivers}

st.sidebar.markdown("---")
driver_a = st.sidebar.selectbox(
    "Driver A", driver_codes, index=0, format_func=lambda c: driver_labels[c]
)
default_b_idx = 1 if len(driver_codes) > 1 else 0
driver_b = st.sidebar.selectbox(
    "Driver B", driver_codes, index=default_b_idx, format_func=lambda c: driver_labels[c]
)

if driver_a == driver_b:
    st.warning("Pick two different drivers to run a duel.")
    st.stop()

with st.spinner(f"Crunching telemetry: {driver_a} vs {driver_b}..."):
    try:
        res = f1.compare_drivers(session_key, driver_a, driver_b, session)
    except Exception as e:
        st.error(f"Couldn't build comparison: {e}")
        st.stop()

# ------------------------------------------------------------------ header --
gap = (res.lap_time_b - res.lap_time_a).total_seconds()
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
    f'<div class="duel-sub">{session.event["EventName"]} {year} · '
    f'{session_labels[session_code]} · fastest laps compared</div>',
    unsafe_allow_html=True,
)
st.write("")

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"{res.driver_a} lap", str(res.lap_time_a)[10:21])
c2.metric(f"{res.driver_b} lap", str(res.lap_time_b)[10:21])
c3.metric("Gap", f"{gap_abs:.3f}s", delta=f"{leader} ahead", delta_color="off")
c4.metric("Top speed A / B", f"{res.speed_a.max():.0f} / {res.speed_b.max():.0f} km/h")

# biggest single-corner swing, for a punchy narrative line
gains = res.corner_gain
best_a = gains.loc[gains["gain_a_seconds"].idxmax()]
best_b = gains.loc[gains["gain_a_seconds"].idxmin()]
st.markdown(
    f'<div class="callout">🔑 Biggest swing: <b style="color:{A_COLOR}">{res.driver_a}</b> '
    f'gains the most at <b>Turn {int(best_a.corner_number)}</b> '
    f'(+{best_a.gain_a_seconds:.2f}s) · '
    f'<b style="color:{B_COLOR}">{res.driver_b}</b> gains the most at '
    f'<b>Turn {int(best_b.corner_number)}</b> '
    f'(+{-best_b.gain_a_seconds:.2f}s)</div>',
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
st.caption(f"Positive bars = {res.driver_a} gains time through that corner. Negative = {res.driver_b} does.")

gains_sorted = gains.sort_values("distance")
bar_colors = [A_COLOR if v >= 0 else B_COLOR for v in gains_sorted["gain_a_seconds"]]
fig3 = go.Figure(
    go.Bar(
        x=[f"T{int(n)}" for n in gains_sorted["corner_number"]],
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
    "Data: FastF1 (official F1 timing & car telemetry). Corner attribution is "
    "computed by segmenting the time-delta-vs-distance curve around each "
    "circuit's real corner coordinates — not an official F1 statistic."
)
