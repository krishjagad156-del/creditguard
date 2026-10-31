"""UI smoke test: the page renders, and an unreachable API is handled gracefully."""
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "ui" / "app.py")


@pytest.fixture(autouse=True)
def _no_waiting(monkeypatch):
    """Do not wait for a sleeping API during tests."""
    monkeypatch.setenv("WAKE_TIMEOUT_S", "0")


def test_page_renders_without_exceptions():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception


def test_api_down_shows_friendly_error():
    import requests
    import streamlit as st

    st.cache_data.clear()

    with patch("requests.post", side_effect=requests.ConnectionError()), patch(
        "requests.get", side_effect=requests.ConnectionError()
    ):
        at = AppTest.from_file(APP, default_timeout=30).run()
        at.button[0].click().run()
    assert not at.exception
    assert any("Cannot reach" in e.value for e in at.error)


class _Resp:
    def __init__(self, data, status=200):
        self._d, self.status_code, self.ok = data, status, status < 400

    def json(self):
        return self._d

    def raise_for_status(self):
        pass


def test_selected_operating_point_is_sent_to_the_api():
    import requests
    import streamlit as st

    st.cache_data.clear()
    points = [
        {"id": "capacity_10", "label": "Review top 10%", "threshold": 0.6, "test_flag_rate": 0.1,
         "test_recall": 0.3, "test_precision": 0.7, "is_default": False},
        {"id": "cost_5", "label": "Cost ratio 5:1", "threshold": 0.15, "test_flag_rate": 0.49,
         "test_recall": 0.78, "test_precision": 0.35, "is_default": True},
    ]
    sent = {}

    def fake_get(url, **kw):
        if url.endswith("/health"):
            return _Resp({"status": "ok", "model_loaded": True})
        if url.endswith("/operating-points"):
            return _Resp({"default_threshold": 0.15, "operating_points": points})
        raise requests.ConnectionError()

    def fake_post(url, **kw):
        sent.update(kw)
        return _Resp({"default_probability": 0.4, "risk_band": "Elevated", "flagged_for_review": True,
                      "threshold": kw["params"]["threshold"], "model_version": "t", "disclaimer": "d",
                      "average_default_rate": 0.22})

    with patch("requests.get", side_effect=fake_get), patch("requests.post", side_effect=fake_post):
        at = AppTest.from_file(APP, default_timeout=30).run()
        assert at.selectbox[0].index == 1                     # default operating point preselected
        at.selectbox[0].select_index(0).run()
        at.button[0].click().run()
    assert not at.exception
    assert sent["params"]["threshold"] == 0.6
    st.cache_data.clear()


def test_ui_waits_for_a_sleeping_api(monkeypatch):
    import requests
    import streamlit as st

    st.cache_data.clear()
    monkeypatch.setenv("WAKE_TIMEOUT_S", "5")
    calls = {"n": 0}

    def fake_get(url, **kw):
        if url.endswith("/health"):
            calls["n"] += 1
            if calls["n"] < 4:
                raise requests.ConnectionError()   # still waking up
            return _Resp({"status": "ok", "model_loaded": True})
        if url.endswith("/operating-points"):
            return _Resp({"default_threshold": 0.15, "operating_points": []})
        raise requests.ConnectionError()

    with patch("requests.get", side_effect=fake_get), patch("time.sleep"):
        at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert calls["n"] == 4
    assert at.session_state["api_ready"] is True
    st.cache_data.clear()


def test_failed_fetch_is_not_cached(monkeypatch):
    """A failure while the API sleeps must not be remembered as 'no operating points'."""
    import requests
    import streamlit as st

    st.cache_data.clear()
    good = [{"id": "cost_5", "label": "Cost ratio 5:1", "threshold": 0.15, "test_flag_rate": 0.49,
             "test_recall": 0.78, "test_precision": 0.35, "is_default": True}]
    state = {"up": False}

    def fake_get(url, **kw):
        if not state["up"]:
            raise requests.ConnectionError()
        if url.endswith("/health"):
            return _Resp({"status": "ok", "model_loaded": True})
        return _Resp({"default_threshold": 0.15, "operating_points": good})

    with patch("requests.get", side_effect=fake_get):
        at = AppTest.from_file(APP, default_timeout=30).run()
        assert len(at.selectbox) == 1           # only the Education box: no operating-point menu yet
        state["up"] = True
        at.run()
    assert len(at.selectbox) == 2               # menu appears once the API is up
    st.cache_data.clear()
