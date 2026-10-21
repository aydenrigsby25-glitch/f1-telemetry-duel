"""Run the whole Streamlit app against a fake FastF1 session (no network)."""

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import f1_analysis as f1
from tests.test_analysis import make_lap


class FakeLap(dict):
    def __init__(self, laptime, tel):
        super().__init__(LapTime=laptime)
        self._tel = tel

    def get_telemetry(self):
        return self._tel.copy()


class FakeLaps(pd.DataFrame):
    _metadata = ["_store"]

    @property
    def _constructor(self):
        return FakeLaps

    def pick_drivers(self, drivers):
        out = FakeLaps(self[self["Driver"].isin(drivers)])
        out._store = self._store
        return out

    def pick_fastest(self):
        row = self.sort_values("LapTime").iloc[0]
        return FakeLap(row["LapTime"], self._store[row["Driver"]])


class FakeSession:
    def __init__(self):
        speeds = {"VER": 250, "NOR": 245, "LEC": 240}
        self.tel = {d: make_lap(lambda x, s=s: s) for d, s in speeds.items()}
        laps = FakeLaps(
            {"Driver": list(speeds),
             "LapTime": [t["Time"].iloc[-1] for t in self.tel.values()]}
        )
        laps._store = self.tel
        self.laps = laps
        self.drivers = list(speeds)
        self.event = {"EventName": "Test Grand Prix"}

    def get_driver(self, d):
        return {"Abbreviation": d, "FirstName": d.title(), "LastName": "Driver", "TeamName": "Team"}

    def get_circuit_info(self):
        class CI:
            corners = pd.DataFrame(
                {"Number": [1, 2, 3], "Letter": ["", "", ""], "Distance": [1000.0, 2500.0, 4000.0]}
            )
        return CI()


SCHEDULE = pd.DataFrame(
    {
        "RoundNumber": [1, 2],
        "EventName": ["Old Grand Prix", "Newest Grand Prix"],
        "Location": ["A", "B"],
        "Country": ["X", "Y"],
        "EventDate": [pd.Timestamp.now() - pd.Timedelta(days=30),
                      pd.Timestamp.now() - pd.Timedelta(days=5)],
    }
)


@pytest.fixture(autouse=True)
def fake_fastf1(monkeypatch):
    monkeypatch.setattr(f1, "get_event_schedule", lambda year: SCHEDULE)
    monkeypatch.setattr(f1, "enable_cache", lambda: None)


def run_app():
    at = AppTest.from_file("../app.py", default_timeout=60)
    at.run()
    return at


def test_app_renders_a_duel(monkeypatch):
    monkeypatch.setattr(f1, "load_session", lambda y, e, c: FakeSession())
    at = run_app()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    assert "VER" in at.markdown[1].value and "NOR" in at.markdown[1].value
    assert len(at.metric) == 4


def test_app_falls_back_when_newest_session_fails(monkeypatch):
    calls = []

    def flaky_load(y, event, code):
        calls.append(event)
        if event == "Newest Grand Prix":
            raise f1.SessionLoadError("not published yet")
        return FakeSession()

    monkeypatch.setattr(f1, "load_session", flaky_load)
    at = run_app()
    assert not at.exception, at.exception
    assert calls[:2] == ["Newest Grand Prix", "Old Grand Prix"]
    assert any("Old Grand Prix" in i.value for i in at.info)


def test_app_shows_message_instead_of_crashing_when_nothing_loads(monkeypatch):
    def always_fail(y, e, c):
        raise f1.SessionLoadError("feed down")

    monkeypatch.setattr(f1, "load_session", always_fail)
    at = run_app()
    assert not at.exception, at.exception
    assert any("Load session" in m.value for m in at.markdown)


def test_half_loaded_session_is_rejected_not_cached(monkeypatch):
    """Reproduces the production crash: session.load() 'succeeds' but laps never load."""
    from fastf1.exceptions import DataNotLoadedError

    class HalfLoaded:
        def load(self, **kw):
            pass  # FastF1 logs the download failure and returns normally

        @property
        def laps(self):
            raise DataNotLoadedError("The data you are trying to access has not been loaded yet.")

    attempts = []

    def fake_get_session(y, e, c):
        attempts.append(1)
        return HalfLoaded()

    monkeypatch.setattr(f1.fastf1, "get_session", fake_get_session)
    f1.load_session.clear()
    for _ in range(2):
        with pytest.raises(f1.SessionLoadError):
            f1.load_session(2026, "Some Grand Prix", "Q")
    assert len(attempts) == 2  # failure was not cached; second call retried
