import numpy as np
import pandas as pd

from creditguard.data import clean_data, standardize_columns
from creditguard.error_analysis import PAY_ORDER, run, segment_table
from creditguard.synthetic import make_synthetic


def test_segment_table_counts_are_consistent():
    y = np.array([1, 1, 0, 0, 1, 0])
    p = np.array([0.9, 0.2, 0.1, 0.8, 0.7, 0.3])
    flag = (p >= 0.5).astype(int)
    seg = pd.Series(pd.Categorical(list("aabbcc"), categories=list("abc")))
    t = segment_table(y, p, flag, seg).set_index("segment")
    assert t["n"].sum() == 6
    assert t.loc["a", "recall"] == 0.5 and t.loc["a", "missed_defaulters"] == 1
    assert t.loc["b", "precision"] == 0.0


def test_run_end_to_end_on_synthetic(trained, tmp_path):
    df = clean_data(standardize_columns(make_synthetic(2000)))
    res = run(model_path=trained["model"], meta_path=trained["meta_path"], df=df, out_dir=tmp_path)
    assert (tmp_path / "error_analysis_segments.csv").exists()
    status = res["tables"]["Most recent repayment status (PAY_1)"]
    assert set(status["segment"]) <= set(PAY_ORDER)
    # n across any slicing must equal the test-set size (20% of 2000)
    for table in res["tables"].values():
        assert table["n"].sum() == 400
    assert res["n_missed"] >= 0
