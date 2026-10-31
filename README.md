# CreditGuard: credit card default risk scorer

An end-to-end machine learning project: data, training pipeline, FastAPI backend, Streamlit front end, Docker, tests.

> **Status: scaffold.** The code in this repository has been tested on **synthetic data only**. No metrics from the real UCI dataset exist yet. Fill in the "Results" section below only with numbers from your own runs.

## The problem

A card issuer wants to flag customers who may miss their next payment so a credit risk analyst can review them early. The model outputs a **probability of default**; a **cost-aware threshold** turns it into a review flag. It supports an analyst's judgement and is not an automated lending decision.

| | |
|---|---|
| Task | Binary classification (target `default`: 1 = default next month) |
| Data | UCI *Default of Credit Card Clients*, 30,000 customers, Taiwan 2005 (CC BY 4.0). See `data/README.md` |
| Metrics | ROC-AUC, PR-AUC, precision/recall at threshold, Brier score (calibration), expected cost, subgroup audit |
| Fairness decision | `SEX`, `AGE`, `MARRIAGE` are **not model inputs**. They are used only to audit error rates by group. The API rejects them. |

## Architecture

```
                 +-------------------+   HTTP/JSON   +--------------------+
  analyst ---->  | Streamlit (8501)  | ------------> |  FastAPI  (8000)   |
                 |  ui/app.py        |  /predict     |  creditguard/api   |
                 +-------------------+  /model-info  |  loads model ONCE  |
                                                     +---------+----------+
                                                               |
                                              artifacts/model.joblib  +  model_meta.json
                                                               ^
                              python -m creditguard.train  ----+
                                (load -> clean -> split -> CV compare ->
                                 calibrate -> threshold -> test -> save)
```

Model inference lives only in the API. The UI never loads the model.

## Repository layout

```
creditguard/        package: config, data, features, evaluation, train, synthetic
creditguard/api/    FastAPI app, Pydantic schemas, model service
ui/app.py           Streamlit front end
scripts/            download_data.py
notebooks/          01_eda.ipynb (run it and record your findings)
tests/              pytest suite (API, features, training, UI smoke test)
artifacts/          trained model + metadata (written by training)
requirements/       pinned dependencies split by purpose
Dockerfile.api  Dockerfile.ui  docker-compose.yml
```

## Quick start (Windows PowerShell)

```powershell
# 1. environment (Python 3.12 recommended; that is what was tested)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. get the real data (needs internet)
python scripts/download_data.py

# 3. explore it
jupyter lab notebooks/01_eda.ipynb

# 4. train (writes artifacts/model.joblib and artifacts/model_meta.json)
python -m creditguard.train            # add --tune for a small hyper-parameter search

# 5. run the API, then the UI in a second terminal
uvicorn creditguard.api.main:app --reload
streamlit run ui/app.py

# 6. tests
python -m pytest -q
```

If PowerShell blocks activation: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
On macOS/Linux activate with `source .venv/bin/activate`.

API docs: http://localhost:8000/docs · UI: http://localhost:8501

## API

| Endpoint | Purpose |
|---|---|
| `GET /` | service info |
| `GET /health` | `ok` when the model is loaded, `degraded` otherwise |
| `GET /model-info` | metrics, comparison, calibration, fairness audit, operating points |
| `GET /operating-points` | menu of flag thresholds with their held-out test results |
| `POST /predict` | validated input -> probability, risk band, review flag. Optional `?threshold=0.25` overrides the default cut-off |

```powershell
curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "@sample_request.json"
```
Status codes: `200` success, `422` invalid input (out-of-range values, missing fields, or audit-only fields such as `AGE`), `503` model not loaded, `500` unexpected error.

## Modelling approach

1. **Split first.** A stratified 20% test set (`random_state=42`) is set aside before any modelling.
2. **Pipeline.** Feature engineering (utilisation, payment ratios, delay statistics, trends) is row-wise and stateless; imputation, scaling and one-hot encoding are fit inside a scikit-learn `Pipeline`, so they only ever learn from training folds.
3. **Compare** a prior-probability baseline, logistic regression (numeric and one-hot repayment status), random forest and gradient boosting using 5-fold stratified CV on the training set. Selection uses PR-AUC because the classes are imbalanced (confirm the rate in the EDA).
4. **Calibrate** probabilities (sigmoid) with cross-validation.
5. **Operating points.** Turning probabilities into flags is a business choice, so the model exposes a menu: cost-ratio settings (1:1 to 10:1) and review-capacity settings (flag the top 10% to 50%). Every threshold is chosen on **out-of-fold training predictions**; the test set only reports what each setting would have achieved. The default is the 5:1 cost setting. The costs (`COST_FALSE_NEGATIVE=5`, `COST_FALSE_POSITIVE=1` in `creditguard/config.py`) are **assumptions**, not measured values. The Streamlit app lets the user pick the operating point. For a calibrated model the optimal cut-off for cost ratio r is about 1/(1+r), a useful sanity check.
6. **Evaluate once** on the untouched test set, plus a per-group fairness audit and permutation importance.

Trade-offs: logistic regression is the most interpretable and fastest; gradient boosting is usually more accurate on tabular data but harder to explain; random forest sits between. The training script reports all of them so you can justify your choice.

**Risk level vs. flag.** The API returns two separate things. `risk_band` compares the customer with the average customer in the training data (below 0.5x, up to 1.5x, up to 3x, or more than 3x the average default rate) and does not change with the operating point. `flagged_for_review` does depend on the chosen operating point. The 0.5x/1.5x/3x cut-points are an **assumption** chosen for readability, not learned from data.

## Experiments (real run, 5-fold CV on the training set, test set untouched)

Every variant uses the same folds, so differences are compared fold by fold.

| Model | Variant | CV PR-AUC | CV ROC-AUC | PR-AUC change vs current | Folds better |
|---|---|---|---|---|---|
| Logistic regression | raw columns only | 0.5078 | 0.7265 | -0.0023 (spread 0.0068) | 2 of 5 |
| Logistic regression | engineered (current) | 0.5101 | 0.7571 | 0 | n/a |
| Logistic regression | engineered + PAY one-hot | 0.5429 | 0.7712 | **+0.0328** (spread 0.0069) | **5 of 5** |
| Gradient boosting | raw columns only | 0.5582 | 0.7803 | -0.0041 (spread 0.0033) | 1 of 5 |
| Gradient boosting | engineered (current) | 0.5623 | 0.7860 | 0 | n/a |
| Gradient boosting | engineered + PAY one-hot | 0.5625 | 0.7856 | +0.0002 (spread 0.0028) | 2 of 5 |

What this shows (5 folds is a small sample, so treat these as indications):
- Repayment status is not a straight line, and a linear model needs it encoded as categories: the logistic model gains 0.033 PR-AUC in every fold and closes about 63% of its gap to gradient boosting.
- Tree models handle that shape on their own, so one-hot encoding adds nothing for gradient boosting.
- Engineered features add a small gain for gradient boosting (about 0.004 PR-AUC, better in 4 of 5 folds) and clearly help the logistic model's ROC-AUC (0.727 to 0.757). They are cheap, so they stay.

**Tuning.** A randomized search over 12 gradient-boosting settings reached a CV PR-AUC of 0.5619, against 0.5623 for the untuned model. The acceptance rule (set before the run) required a gain of more than 0.0077, the fold-to-fold spread, so the untuned model was kept. Random forest, untuned and tuned gradient boosting all land near 0.56, so the ceiling here is set by the information in the data, not by hyper-parameters.

## Results (fill in from YOUR real run)

| Metric (test set) | Value |
|---|---|
| ROC-AUC | _run training first_ |
| PR-AUC | |
| Recall / precision at threshold | |
| Expected cost per customer vs. flagging nobody | |

## Docker

Train first so `artifacts/` contains a model built from **real** data, then:

```powershell
docker compose up --build
docker compose logs -f api        # inspect logs
docker compose ps                 # health status
curl.exe http://localhost:8000/health
```
Open http://localhost:8501. Common failures: `artifacts/` empty (the API starts but reports `degraded`), port already in use (change the left side of `8000:8000`), or a scikit-learn version mismatch between training and the image (versions are pinned in `requirements/`). The Docker setup has **not been run** by the author of this scaffold, so expect to debug it.

## Limitations and responsible use

- 2005 Taiwan data; not validated for any other market or period.
- `EDUCATION` remains a model input and can act as a socioeconomic proxy. Decide whether to keep it and document why.
- Removing `SEX`/`AGE`/`MARRIAGE` does not guarantee fairness, because other features can proxy for them. The audit table describes gaps but does not certify fairness.
- Cost assumptions drive the threshold; change them for a different business case.
- Predictions are estimates, never guarantees, and should not be the sole basis for a lending decision.

## Data license and credit

Yeh, I-Cheng (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H. Licensed CC BY 4.0.

## Next steps (not done yet)

Real-data EDA and results, cloud deployment and public demo link, architecture screenshots, blog post.
