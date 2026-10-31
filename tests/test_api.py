import pytest


def test_root_and_health(client):
    assert client.get("/").json()["service"] == "CreditGuard API"
    health = client.get("/health").json()
    assert health == {"status": "ok", "model_loaded": True}


def test_valid_prediction(client, valid_payload):
    r = client.post("/predict", json=valid_payload)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["default_probability"] <= 1
    assert body["risk_band"] in {"Below average", "Around average", "Elevated", "Very high"}
    assert 0 < body["average_default_rate"] < 1
    assert body["flagged_for_review"] == (body["default_probability"] >= body["threshold"])
    assert "disclaimer" in body


def test_worse_repayment_history_scores_higher_on_synthetic_model(client, valid_payload):
    good = client.post("/predict", json=valid_payload).json()["default_probability"]
    bad_payload = {**valid_payload, "PAY_1": 3, "PAY_2": 3, "PAY_3": 2}
    bad = client.post("/predict", json=bad_payload).json()["default_probability"]
    assert bad > good


@pytest.mark.parametrize(
    "change",
    [
        {"LIMIT_BAL": -5},
        {"LIMIT_BAL": 0},
        {"EDUCATION": 9},
        {"PAY_1": 50},
        {"PAY_AMT1": -1},
        {"BILL_AMT1": "abc"},
        {"AGE": 40},      # audit-only attribute must be rejected
        {"SEX": 1},       # audit-only attribute must be rejected
    ],
)
def test_invalid_inputs_return_422(client, valid_payload, change):
    r = client.post("/predict", json={**valid_payload, **change})
    assert r.status_code == 422


def test_missing_field_returns_422(client, valid_payload):
    valid_payload.pop("PAY_3")
    assert client.post("/predict", json=valid_payload).status_code == 422


def test_empty_body_returns_422(client):
    assert client.post("/predict", json={}).status_code == 422


def test_boundary_values_are_accepted(client, valid_payload):
    edge = {**valid_payload, "PAY_1": -2, "PAY_2": 9, "BILL_AMT1": -1000, "PAY_AMT1": 0, "EDUCATION": 1}
    assert client.post("/predict", json=edge).status_code == 200


def test_model_info_exposes_metrics_and_fairness(client):
    info = client.get("/model-info").json()
    assert info["test_metrics"]["roc_auc"] > 0
    assert set(info["fairness_audit"]) == {"SEX", "AGE_BAND", "MARRIAGE"}


def test_degraded_when_model_missing(client_no_model, valid_payload):
    assert client_no_model.get("/health").json() == {"status": "degraded", "model_loaded": False}
    assert client_no_model.post("/predict", json=valid_payload).status_code == 503
    assert client_no_model.get("/model-info").status_code == 503


def test_operating_points_endpoint(client):
    body = client.get("/operating-points").json()
    assert body["operating_points"]
    assert sum(o["is_default"] for o in body["operating_points"]) == 1
    assert 0 < body["default_threshold"] < 1


def test_threshold_override_controls_flag(client, valid_payload):
    strict = client.post("/predict?threshold=0.99", json=valid_payload).json()
    loose = client.post("/predict?threshold=0.01", json=valid_payload).json()
    assert strict["threshold"] == 0.99 and strict["flagged_for_review"] is False
    assert loose["threshold"] == 0.01 and loose["flagged_for_review"] is True
    assert strict["default_probability"] == pytest.approx(loose["default_probability"])


@pytest.mark.parametrize("bad", ["0", "1", "5", "-0.2", "abc"])
def test_invalid_threshold_returns_422(client, valid_payload, bad):
    assert client.post(f"/predict?threshold={bad}", json=valid_payload).status_code == 422


def test_operating_points_unavailable_without_model(client_no_model):
    assert client_no_model.get("/operating-points").status_code == 503


def test_risk_band_does_not_depend_on_operating_point(client, valid_payload):
    strict = client.post("/predict?threshold=0.99", json=valid_payload).json()
    loose = client.post("/predict?threshold=0.01", json=valid_payload).json()
    assert strict["risk_band"] == loose["risk_band"]


def test_relative_risk_band_boundaries():
    from creditguard.api.model_service import relative_risk_band

    base = 0.2
    assert relative_risk_band(0.05, base) == "Below average"     # 0.25x
    assert relative_risk_band(0.20, base) == "Around average"    # 1x
    assert relative_risk_band(0.45, base) == "Elevated"          # 2.25x
    assert relative_risk_band(0.70, base) == "Very high"         # 3.5x
