from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from churn.modeling import load_and_clean, make_models, top_fraction_metrics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
COLORS = {"Logistic Regression": "#2463A9", "Random Forest": "#C58B24", "Gradient Boosting": "#8A5A9B"}


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()


def main(quick: bool = False) -> None:
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    X, y = load_and_clean(DATA)
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    folds = 3 if quick else 5
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    models = make_models(X)
    results: list[dict[str, float | str]] = []
    oof_scores: dict[str, np.ndarray] = {}

    for name, model in models.items():
        oof = cross_val_predict(model, X_dev, y_dev, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
        oof_scores[name] = oof
        results.append(
            {
                "model": name,
                "cv_pr_auc": average_precision_score(y_dev, oof),
                "cv_roc_auc": roc_auc_score(y_dev, oof),
            }
        )

    comparison = pd.DataFrame(results).sort_values("cv_pr_auc", ascending=False).reset_index(drop=True)
    best_name = str(comparison.loc[0, "model"])
    precision, recall, thresholds = precision_recall_curve(y_dev, oof_scores[best_name])
    f1_values = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    threshold = float(thresholds[int(np.nanargmax(f1_values))])

    test_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    fitted = {}
    for name, model in models.items():
        model.fit(X_dev, y_dev)
        fitted[name] = model
        scores = model.predict_proba(X_test)[:, 1]
        test_curves[name] = (y_test.to_numpy(), scores)
        mask = comparison["model"].eq(name)
        comparison.loc[mask, "test_pr_auc"] = average_precision_score(y_test, scores)
        comparison.loc[mask, "test_roc_auc"] = roc_auc_score(y_test, scores)

    best_scores = test_curves[best_name][1]
    predictions = (best_scores >= threshold).astype(int)
    targeting = top_fraction_metrics(y_test, best_scores, 0.20)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    summary = {
        "rows": int(len(X)),
        "features": int(X.shape[1]),
        "overall_churn_rate": float(y.mean()),
        "development_rows": int(len(X_dev)),
        "test_rows": int(len(X_test)),
        "selected_model": best_name,
        "selection_metric": "out-of-fold average precision",
        "decision_threshold": threshold,
        "test_pr_auc": float(average_precision_score(y_test, best_scores)),
        "test_roc_auc": float(roc_auc_score(y_test, best_scores)),
        "test_precision": float(precision_score(y_test, predictions)),
        "test_recall": float(recall_score(y_test, predictions)),
        "test_f1": float(f1_score(y_test, predictions)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "top_20_percent_targeting": targeting,
        "caveat": "This IBM sample describes a fictional telco and is not a production validation population.",
    }

    raw = pd.read_csv(DATA)
    raw["churn_flag"] = raw["Churn"].eq("Yes").astype(int)
    contract = raw.groupby("Contract", as_index=False).agg(
        churn_rate=("churn_flag", "mean"), customers=("churn_flag", "size")
    ).rename(columns={"Contract": "contract"})
    contract.to_csv(REPORTS / "churn_by_contract.csv", index=False)

    perm = permutation_importance(
        fitted[best_name], X_test, y_test, scoring="average_precision", n_repeats=12 if not quick else 3,
        random_state=42, n_jobs=1,
    )
    importance = pd.DataFrame(
        {"feature": X.columns, "importance_mean": perm.importances_mean, "importance_std": perm.importances_std}
    ).sort_values("importance_mean", ascending=False)
    importance.to_csv(REPORTS / "feature_importance.csv", index=False)
    comparison.to_csv(REPORTS / "model_comparison.csv", index=False)
    pd.DataFrame({"actual_churn": y_test.to_numpy(), "predicted_probability": best_scores, "predicted_class": predictions}).to_csv(
        REPORTS / "test_predictions.csv", index=False
    )
    (REPORTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ordered = contract.sort_values("churn_rate")
    ax.barh(ordered["contract"], ordered["churn_rate"] * 100, color="#2463A9")
    ax.set(title="Observed churn rate by contract", xlabel="Customers who churned (%)", ylabel="")
    ax.set_xlim(0, max(60, ordered["churn_rate"].max() * 115))
    for i, value in enumerate(ordered["churn_rate"] * 100):
        ax.text(value + 1, i, f"{value:.1f}%", va="center")
    save_figure(FIGURES / "churn_by_contract.png")

    fig, ax = plt.subplots(figsize=(8, 5.2))
    for name, (truth, scores) in test_curves.items():
        p, r, _ = precision_recall_curve(truth, scores)
        ax.plot(r, p, lw=2.2, color=COLORS[name], label=f"{name} (AP={average_precision_score(truth, scores):.3f})")
    ax.axhline(y_test.mean(), color="#555555", linestyle="--", label=f"No-skill ({y_test.mean():.3f})")
    ax.set(title="Precision–recall curves on the untouched test set", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    ax.legend(frameon=False)
    save_figure(FIGURES / "precision_recall_curves.png")

    top = importance.head(10).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#C58B24", alpha=0.9)
    ax.set(title=f"Top permutation importance — {best_name}", xlabel="Decrease in average precision when shuffled", ylabel="")
    save_figure(FIGURES / "feature_importance.png")

    ranked = pd.DataFrame({"actual": y_test.to_numpy(), "score": best_scores}).sort_values("score", ascending=False).reset_index(drop=True)
    ranked["decile"] = pd.qcut(ranked.index, 10, labels=False, duplicates="drop") + 1
    gains = ranked.groupby("decile", as_index=False).agg(customers=("actual", "size"), churners=("actual", "sum"))
    gains["cumulative_recall"] = gains["churners"].cumsum() / ranked["actual"].sum()
    gains.to_csv(REPORTS / "targeting_gains.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(gains["decile"] * 10, gains["cumulative_recall"] * 100, marker="o", color="#2463A9", lw=2.2, label="Model ranking")
    ax.plot([10, 100], [10, 100], linestyle="--", color="#777777", label="Random targeting")
    ax.set(title="Cumulative churners captured by risk-ranked outreach", xlabel="Customers targeted (%)", ylabel="Churners captured (%)", xlim=(10, 100), ylim=(0, 105))
    ax.legend(frameon=False)
    save_figure(FIGURES / "targeting_gains.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use fewer CV folds and repeats for CI")
    args = parser.parse_args()
    main(quick=args.quick)
