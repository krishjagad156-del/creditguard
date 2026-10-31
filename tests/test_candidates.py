from creditguard.train import candidate_models


def test_comparison_includes_logistic_with_pay_onehot(trained):
    names = {r["model"] for r in trained["meta"]["cv_comparison"]}
    assert {"baseline_prior", "logistic_regression", "logistic_regression_pay_onehot",
            "random_forest", "hist_gradient_boosting"} <= names


def test_candidate_specs_carry_their_preprocessing():
    specs = candidate_models(0)
    assert specs["logistic_regression_pay_onehot"][1] == {"pay_onehot": True}
    assert specs["hist_gradient_boosting"][1] == {}


def test_metadata_records_preprocessing_options(trained):
    assert isinstance(trained["meta"]["preprocessing"], dict)
