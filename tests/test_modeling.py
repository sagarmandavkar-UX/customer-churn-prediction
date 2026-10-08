from pathlib import Path

import numpy as np
import pandas as pd

from churn.modeling import load_and_clean, make_models, top_fraction_metrics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"


def test_dataset_contract() -> None:
    features, target = load_and_clean(DATA)
    assert features.shape == (7043, 19)
    assert target.isin([0, 1]).all()
    assert 0.20 < target.mean() < 0.35
    assert features["TotalCharges"].isna().sum() == 11


def test_all_models_fit_and_predict_probabilities() -> None:
    features, target = load_and_clean(DATA)
    sample = features.iloc[:500]
    labels = target.iloc[:500]
    for model in make_models(features).values():
        model.fit(sample, labels)
        probabilities = model.predict_proba(sample.iloc[:20])[:, 1]
        assert probabilities.shape == (20,)
        assert np.all((0 <= probabilities) & (probabilities <= 1))


def test_top_fraction_metrics() -> None:
    truth = pd.Series([1, 0, 1, 0, 0])
    scores = np.array([0.9, 0.2, 0.8, 0.1, 0.3])
    result = top_fraction_metrics(truth, scores, fraction=0.4)
    assert result["customers_targeted"] == 2
    assert result["churners_captured"] == 2
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0

