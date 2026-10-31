"""Streamlit front end. All model inference happens in the FastAPI backend."""
from __future__ import annotations

import os
import time

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT_S = 30  # per request once the API is awake
# How long to wait for a sleeping API (free cloud hosts spin idle services down).
WAKE_TIMEOUT_S = float(os.getenv("WAKE_TIMEOUT_S", "120"))

st.set_page_config(page_title="CreditGuard", page_icon="💳", layout="wide")

PAY_HELP = "-1: paid duly; 1-9: months of payment delay. -2 and 0 are not defined in the official description (commonly read as no consumption and revolving credit)."


def wait_for_api(timeout_s: float, interval_s: float = 3.0) -> bool:
    """Poll /health until the model is loaded. Always tries at least once."""
    deadline = time.time() + timeout_s
    while True:
        try:
            r = requests.get(f"{API_URL}/health", timeout=10)
            if r.ok and r.json().get("model_loaded"):
                return True
        except (requests.RequestException, ValueError):
            pass
        if time.time() >= deadline:
            return False
        time.sleep(interval_s)


# These raise on failure on purpose: st.cache_data does not cache exceptions, so a
# failure while the API is asleep is never remembered as an empty result.
@st.cache_data(ttl=60, show_spinner=False)
def fetch_model_info() -> dict:
    r = requests.get(f"{API_URL}/model-info", timeout=TIMEOUT_S)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=60, show_spinner=False)
def fetch_operating_points() -> list[dict]:
    r = requests.get(f"{API_URL}/operating-points", timeout=TIMEOUT_S)
    r.raise_for_status()
    return r.json().get("operating_points", [])


def call_predict(payload: dict, threshold: float | None = None) -> tuple[dict | None, str | None]:
    """Return (result, error_message). Never raises to the UI."""
    params = {"threshold": threshold} if threshold is not None else None
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, params=params, timeout=TIMEOUT_S)
    except requests.ConnectionError:
        return None, f"Cannot reach the prediction service at {API_URL}. Is the API running?"
    except requests.Timeout:
        return None, "The prediction service took too long to respond. Please try again."
    if r.status_code == 422:
        problems = "; ".join(
            f"{'.'.join(str(x) for x in e['loc'][1:])}: {e['msg']}" for e in r.json().get("detail", [])
        )
        return None, f"The service rejected the input: {problems}"
    if r.status_code == 503:
        return None, "The model is not loaded on the server yet."
    if not r.ok:
        return None, f"Unexpected error from the service (HTTP {r.status_code})."
    return r.json(), None


st.title("💳 CreditGuard")
st.caption("Credit card default risk estimates for analyst decision support")

WAKE_MESSAGE = "Connecting to the prediction service. If it was idle, waking it can take about a minute..."
if not st.session_state.get("api_ready"):
    with st.spinner(WAKE_MESSAGE):
        st.session_state["api_ready"] = wait_for_api(WAKE_TIMEOUT_S)
    if not st.session_state["api_ready"]:
        st.warning("The prediction service is not responding yet. Wait a minute, then reload this page.")

tab_predict, tab_perf, tab_about = st.tabs(["Assess a customer", "Model performance", "About & limits"])

# ---------------------------------------------------------------- predict
with tab_predict:
    st.markdown(
        "**Problem:** card issuers want to spot customers who may miss their next payment so an "
        "analyst can review them early. Enter a customer's recent history below. Month 1 is the most recent month."
    )
    try:
        points = fetch_operating_points()
    except requests.RequestException:
        points = []
    chosen_threshold = None
    if points:
        default_idx = next((i for i, p in enumerate(points) if p["is_default"]), 0)
        idx = st.selectbox(
            "Operating point: how many customers should be flagged?",
            range(len(points)),
            index=default_idx,
            format_func=lambda i: f"{points[i]['label']}  (flags about {points[i]['test_flag_rate']:.0%} of customers)",
        )
        op = points[idx]
        chosen_threshold = op["threshold"]
        st.caption(
            f"On held-out test data this setting flagged {op['test_flag_rate']:.0%} of customers, caught "
            f"{op['test_recall']:.0%} of the customers who defaulted, and {op['test_precision']:.0%} of flagged "
            "customers actually defaulted. Flagging fewer customers means fewer false alarms but more missed defaults."
        )

    with st.form("customer"):
        c1, c2 = st.columns(2)
        limit = c1.number_input("Credit limit (NT$)", min_value=1, max_value=10_000_000, value=50_000)
        education = c2.selectbox(
            "Education", [1, 2, 3, 4], index=1,
            format_func=lambda v: {1: "Graduate school", 2: "University", 3: "High school", 4: "Other"}[v],
        )

        st.subheader("Repayment status")
        st.caption(PAY_HELP)
        cols = st.columns(6)
        pay = [cols[i].number_input(f"Month {i + 1}", min_value=-2, max_value=9, value=0, key=f"pay{i}") for i in range(6)]

        st.subheader("Bill statement amounts (NT$)")
        cols = st.columns(6)
        bills = [cols[i].number_input(f"Month {i + 1}", min_value=-1_000_000, max_value=10_000_000, value=20_000 - i * 1_000, key=f"bill{i}") for i in range(6)]

        st.subheader("Payments made (NT$)")
        cols = st.columns(6)
        paid = [cols[i].number_input(f"Month {i + 1}", min_value=0, max_value=10_000_000, value=2_000, key=f"paid{i}") for i in range(6)]

        submitted = st.form_submit_button("Assess risk", type="primary")

    if submitted:
        payload = {"LIMIT_BAL": limit, "EDUCATION": education}
        payload.update({f"PAY_{i + 1}": int(pay[i]) for i in range(6)})
        payload.update({f"BILL_AMT{i + 1}": float(bills[i]) for i in range(6)})
        payload.update({f"PAY_AMT{i + 1}": float(paid[i]) for i in range(6)})
        with st.spinner(WAKE_MESSAGE):
            ready = wait_for_api(WAKE_TIMEOUT_S)  # fast when awake; waits if the API went back to sleep
        st.session_state["api_ready"] = ready
        if ready:
            with st.spinner("Scoring..."):
                result, error = call_predict(payload, chosen_threshold)
        else:
            result = None
            error = f"Cannot reach the prediction service at {API_URL}. It may still be waking up: wait a minute and try again."
        if error:
            st.error(error)
        else:
            prob = result["default_probability"]
            m1, m2, m3 = st.columns(3)
            m1.metric("Estimated default probability", f"{prob:.1%}")
            m2.metric("Risk compared with average customer", result["risk_band"])
            m3.metric("Flag for review (depends on operating point)", "Yes" if result["flagged_for_review"] else "No")
            avg = result.get("average_default_rate")
            if avg is not None:
                st.caption(
                    f"The average customer in the training data defaulted {avg:.0%} of the time. "
                    "'Risk compared with average' does not change with the operating point; the flag does."
                )
            st.info(
                f"With the selected operating point, customers are flagged when the estimate reaches "
                f"{result['threshold']:.1%}. The cut-off is a business choice, not a fact about the customer. "
                "The number is a statistical estimate, not a guarantee."
            )
            st.caption(result["disclaimer"] + f"  ·  model {result['model_version']}")

# ------------------------------------------------------------ performance
with tab_perf:
    try:
        info = fetch_model_info()
    except requests.RequestException:
        info = None
    if info is None:
        st.warning("Model information is unavailable (API not reachable or model not loaded).")
    else:
        if info.get("trained_on_synthetic_data"):
            st.error("This model was trained on SYNTHETIC data. The numbers below are not real results.")
        tm = info["test_metrics"]
        st.subheader("Held-out test set")
        a, b, c, d = st.columns(4)
        a.metric("ROC-AUC", f"{tm['roc_auc']:.3f}")
        b.metric("PR-AUC", f"{tm['pr_auc']:.3f}", help=f"Random-guess level equals the default rate, {tm['base_rate']:.3f}")
        c.metric("Recall at threshold", f"{tm['recall']:.1%}")
        d.metric("Precision at threshold", f"{tm['precision']:.1%}")
        st.write(
            f"Expected cost per customer: **{tm['expected_cost_per_customer']:.3f}** "
            f"versus **{tm['cost_if_nobody_flagged']:.3f}** if nobody were flagged (under the stated cost assumptions)."
        )

        if info.get("operating_points"):
            st.subheader("Operating points (held-out test set)")
            ops = pd.DataFrame(info["operating_points"])[
                ["label", "threshold", "test_flag_rate", "test_recall", "test_precision"]
            ]
            st.dataframe(ops, hide_index=True, use_container_width=True)
            st.caption("Thresholds were chosen on out-of-fold training predictions, never on the test set.")

        st.subheader("Model comparison (cross-validation on training data)")
        st.dataframe(pd.DataFrame(info["cv_comparison"]), hide_index=True, use_container_width=True)

        st.subheader("Calibration (test set)")
        cal = pd.DataFrame(info["calibration"]).set_index("mean_predicted")
        st.line_chart(cal)
        st.caption("A well-calibrated model sits close to the diagonal: predicted 20% means about 20% actually default.")

        st.subheader("Most influential inputs")
        imp = pd.Series(info["permutation_importance"]).sort_values(ascending=False).head(10)
        st.bar_chart(imp)

        st.subheader("Fairness audit (SEX, age band, marital status are NOT model inputs)")
        for attr, rows in info["fairness_audit"].items():
            st.markdown(f"**{attr}**")
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ about
with tab_about:
    st.markdown(
        """
**Data:** UCI *Default of Credit Card Clients* (Taiwan, April to September 2005), CC BY 4.0.

**Limits to keep in mind**
- The data is old and from one market; the model is not validated for any other population.
- Estimates support an analyst's judgement. They must not be the sole basis of a lending decision.
- The fairness table describes gaps between groups; it does not prove the model is fair.
        """
    )
