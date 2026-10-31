import pytest
from fastapi.testclient import TestClient

from creditguard import train
from creditguard.api.main import create_app

VALID_PAYLOAD = {
    "LIMIT_BAL": 50000, "EDUCATION": 2,
    "PAY_1": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
    "BILL_AMT1": 20000, "BILL_AMT2": 19000, "BILL_AMT3": 18000,
    "BILL_AMT4": 17000, "BILL_AMT5": 16000, "BILL_AMT6": 15000,
    "PAY_AMT1": 2000, "PAY_AMT2": 2000, "PAY_AMT3": 2000,
    "PAY_AMT4": 1500, "PAY_AMT5": 1500, "PAY_AMT6": 1500,
}


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """Train once on SYNTHETIC data into a temp dir (fast, offline, no real results)."""
    out = tmp_path_factory.mktemp("artifacts")
    meta = train.run(
        synthetic=True, n_synthetic=2000, cv_folds=3,
        model_path=out / "model.joblib", meta_path=out / "model_meta.json",
        make_figures=False,
    )
    return {"model": out / "model.joblib", "meta_path": out / "model_meta.json", "meta": meta}


@pytest.fixture()
def client(trained):
    app = create_app(trained["model"], trained["meta_path"])
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def client_no_model(tmp_path):
    app = create_app(tmp_path / "missing.joblib", tmp_path / "missing.json")
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def valid_payload():
    return dict(VALID_PAYLOAD)
