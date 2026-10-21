"""Run the whole Streamlit app against fake OpenF1 data (no network)."""

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import f1_data as data
from tests.fake_openf1 import LAP_START, simulate

NOW = pd.Timestamp.now(tz="UTC")
MEETINGS = pd.DataFrame(
    {
        "meeting_key": [1, 2],
        "meeting_name": ["Old Grand Prix", "Newest Grand Prix"],
        "location": ["A", "B"],
        "date_start": [NOW - pd.Timedelta(days=30), NOW - pd.Timedelta(days=5)],
    }
)
SESSIONS = {
    1: pd.DataFrame({"session_key": [11, 12], "session_name": ["Qualifying", "Race"],
                     "date_start": [NOW] * 2, "date_end": [NOW] * 2}),
    2: pd.DataFrame({"session_key": [21, 22], "session_name": ["Qualifying", "Race"],
                     "date_start": [NOW] * 2, "date_end": [NOW] * 2}),
}
SPEEDS = {1: 250, 4: 245, 16: 240}
TEL = {n: simulate(lambda d, s=s: s, seed=n) for n, s in SPEEDS.items()}
DRIVERS = pd.DataFrame(
    {"driver_number": [1, 4, 16], "name_acronym": ["VER", "NOR", "LEC"],
     "full_name": ["Max VERSTAPPEN", "Lando NORRIS", "Charles LECLERC"],
     "team_name": ["Red Bull Racing", "McLaren", "Ferrari"], "team_colour": ["", "", ""]}
)


def fake_laps(session_key):
    return pd.DataFrame(
        {"driver_number": list(SPEEDS), "lap_number": [5, 5, 5],
         "lap_duration": [TEL[n][2] for n in SPEEDS], "date_start": [LAP_START] * 3}
    )


@pytest.fixture(autouse=True)
def fake_api(monkeypatch):
    monkeypatch.setattr(data, "get_meetings", lambda year: MEETINGS)
    monkeypatch.setattr(data, "get_sessions", lambda mk: SESSIONS[mk])
    monkeypatch.setattr(data, "get_laps", fake_laps)
    monkeypatch.setattr(data, "get_drivers", lambda sk: DRIVERS)
    monkeypatch.setattr(
        data, "get_lap_telemetry", lambda sk, num, start, dur: (TEL[num][0], TEL[num][1])
    )


def run_app():
    at = AppTest.from_file("../app.py", default_timeout=120)
    at.run()
    return at


def test_app_renders_a_duel():
    at = run_app()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    header = at.markdown[1].value
    assert "VER" in header and "NOR" in header
    assert at.metric[0].value.startswith("1:")
    assert "Newest Grand Prix" in at.markdown[2].value


def test_app_falls_back_when_newest_session_fails(monkeypatch):
    calls = []

    def flaky(session_key):
        calls.append(session_key)
        if session_key == 21:
            raise data.DataUnavailableError("not published yet")
        return fake_laps(session_key)

    monkeypatch.setattr(data, "get_laps", flaky)
    at = run_app()
    assert not at.exception, at.exception
    assert calls[:2] == [21, 11]
    assert any("Old Grand Prix" in i.value for i in at.info)


def test_app_shows_instructions_instead_of_crashing_when_nothing_loads(monkeypatch):
    def always_fail(session_key):
        raise data.DataUnavailableError("API down")

    monkeypatch.setattr(data, "get_laps", always_fail)
    at = run_app()
    assert not at.exception, at.exception
    assert any("Load session" in m.value for m in at.markdown)


def test_switching_drivers_works():
    at = run_app()
    at.sidebar.selectbox[3].set_value("LEC").run()
    assert not at.exception, at.exception
    assert "LEC" in at.markdown[1].value


def test_client_retries_after_rate_limit(monkeypatch):
    class Resp:
        def __init__(self, code, body):
            self.status_code, self._body, self.headers = code, body, {"Retry-After": "0"}
            self.ok = code < 400

        def json(self):
            return self._body

    replies = [Resp(429, None), Resp(200, [{"meeting_key": 1}])]
    monkeypatch.setattr(data.requests, "get", lambda *a, **k: replies.pop(0))
    monkeypatch.setattr(data, "MIN_INTERVAL_S", 0)
    assert data._get("meetings", {"year": 2025}) == [{"meeting_key": 1}]
    assert not replies


def test_client_treats_no_results_as_empty(monkeypatch):
    class Resp:
        status_code, ok, headers = 200, True, {}

        def json(self):
            return {"detail": "No results found."}

    monkeypatch.setattr(data.requests, "get", lambda *a, **k: Resp())
    monkeypatch.setattr(data, "MIN_INTERVAL_S", 0)
    assert data._get("laps", {"session_key": 1}) == []
