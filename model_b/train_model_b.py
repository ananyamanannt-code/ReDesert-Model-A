"""Re:Desert Model B — reproducible satellite-derived retained-greenness proxy.

Adapted from the original submitted research script. This script reproduces the
model training and geographic holdout evaluation using the included CSV.
It does NOT download satellite imagery, verify native plant survival, or predict
irrigation-free planting success.

Run: python train_model_b.py
Optional: python train_model_b.py --data training_data.csv --output outputs
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier

FEATURES = [
    "ndvi_early", "msavi_early", "ndvi_variability", "swir_early",
    "rain_mean_month_mm", "lst_c", "elevation_m", "slope_deg",
]
MODEL_SETTINGS = dict(n_estimators=150, min_samples_leaf=3,
                      class_weight="balanced_subsample", random_state=42, n_jobs=-1)


def evaluate(model, cols, train, test):
    model.fit(train[cols], train["persistence"])
    y_true = test["persistence"]
    pred = model.predict(test[cols])
    proba = model.predict_proba(test[cols])[:, list(model.classes_).index(1)]
    return model, pred, proba, {
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, pred)), 4),
        "macro_f1": round(float(f1_score(y_true, pred, average="macro", zero_division=0)), 4),
        "persistent_f1": round(float(f1_score(y_true, pred, pos_label=1, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "brier": round(float(brier_score_loss(y_true, proba)), 4),
        "test_n": int(len(test)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("training_data.csv"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("outputs"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data).replace([np.inf, -np.inf], np.nan)
    required = FEATURES + ["longitude", "latitude", "persistence"]
    df = df.dropna(subset=required).reset_index(drop=True)
    assert len(df) == 360, f"Expected 360 eligible records; got {len(df)}"
    assert set(df.persistence.unique()) == {0, 1}
    assert not df.duplicated(["longitude", "latitude"]).any()

    gx = np.floor((df.longitude - 55.535) / 0.015).astype(int)
    gy = np.floor((df.latitude - 24.990) / 0.015).astype(int)
    df["block"] = gx.astype(str) + "_" + gy.astype(str)

    split = None
    for seed in range(100):
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25,
                                        random_state=seed).split(df, df.persistence, groups=df.block))
        ytr, yte = df.persistence.iloc[tr], df.persistence.iloc[te]
        if (set(ytr.unique()) == {0, 1} and set(yte.unique()) == {0, 1}
                and ytr.value_counts().min() >= 10 and yte.value_counts().min() >= 5):
            split = (tr, te, seed)
            break
    assert split is not None, "Could not find a stratified-enough grouped split"
    tr, te, seed = split
    train, test = df.iloc[tr], df.iloc[te]
    assert not (set(train.block) & set(test.block))

    spec = {
        "Most-frequent baseline": (DummyClassifier(strategy="most_frequent"), FEATURES),
        "NDVI-only Random Forest": (RandomForestClassifier(**MODEL_SETTINGS), ["ndvi_early"]),
        "Full Random Forest": (RandomForestClassifier(**MODEL_SETTINGS), FEATURES),
    }
    scores = {}
    for name, (model, columns) in spec.items():
        fitted, prediction, probability, scores[name] = evaluate(model, columns, train, test)
        if name == "Full Random Forest":
            full_rf, full_predictions, full_probabilities = fitted, prediction, probability
        print(name, scores[name])

    # Reproduce geographic-block cross-validation. Neighbouring blocks may still be correlated.
    cv = {}
    for name in ["NDVI-only Random Forest", "Full Random Forest"]:
        cols = spec[name][1]
        rows = []
        for fold, (i_tr, i_te) in enumerate(
                GroupKFold(n_splits=5).split(df[cols], df.persistence, groups=df.block), start=1):
            fitted = RandomForestClassifier(**MODEL_SETTINGS).fit(df.iloc[i_tr][cols], df.persistence.iloc[i_tr])
            pred = fitted.predict(df.iloc[i_te][cols])
            actual = df.persistence.iloc[i_te]
            rows.append({
                "fold": fold, "n": int(len(i_te)),
                "class_0": int((actual == 0).sum()), "class_1": int((actual == 1).sum()),
                "balanced_accuracy": round(float(balanced_accuracy_score(actual, pred)), 4),
                "macro_f1": round(float(f1_score(actual, pred, average="macro", zero_division=0)), 4),
            })
        cv[name] = rows

    cv_summary = {
        name: {
            "mean_macro_f1": round(float(np.mean([row["macro_f1"] for row in rows])), 4),
            "range_macro_f1": [float(np.min([row["macro_f1"] for row in rows])),
                               float(np.max([row["macro_f1"] for row in rows]))],
            "mean_balanced_accuracy": round(float(np.mean([row["balanced_accuracy"] for row in rows])), 4),
        } for name, rows in cv.items()
    }

    # Simple learned NDVI threshold — essential comparator for the 8-feature model.
    stump = DecisionTreeClassifier(max_depth=1, random_state=42)
    stump.fit(train[["ndvi_early"]], train.persistence)
    stump_predictions = stump.predict(test[["ndvi_early"]])
    stump_folds = []
    for i_tr, i_te in GroupKFold(n_splits=5).split(df[["ndvi_early"]], df.persistence, groups=df.block):
        fit = DecisionTreeClassifier(max_depth=1, random_state=42).fit(
            df.iloc[i_tr][["ndvi_early"]], df.persistence.iloc[i_tr])
        stump_folds.append(float(f1_score(df.persistence.iloc[i_te],
                                           fit.predict(df.iloc[i_te][["ndvi_early"]]),
                                           average="macro", zero_division=0)))

    importance = pd.DataFrame({"feature": FEATURES, "importance": full_rf.feature_importances_})
    importance = importance.sort_values("importance", ascending=False)
    importance.to_csv(args.output / "feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.barh(importance.feature.iloc[::-1], importance.importance.iloc[::-1])
    ax.set_title("Model B feature importance")
    ax.set_xlabel("Random Forest impurity importance (not causal)")
    fig.tight_layout()
    fig.savefig(args.output / "feature_importance.png", dpi=160)
    plt.close(fig)

    matrix = confusion_matrix(test.persistence, full_predictions, labels=[0, 1])
    pd.DataFrame(matrix, columns=["Predicted 0", "Predicted 1"],
                 index=["Actual 0", "Actual 1"]).to_csv(args.output / "confusion_matrix.csv")
    ConfusionMatrixDisplay.from_predictions(
        test.persistence, full_predictions, display_labels=["Not persistent", "Persistent"], cmap="Blues")
    plt.title("Model B — Geographic holdout")
    plt.tight_layout()
    plt.savefig(args.output / "confusion_matrix.png", dpi=160)
    plt.close()

    test_points = test[["longitude", "latitude", "block", "persistence"]].copy()
    test_points["predicted"] = full_predictions
    test_points["probability_persistence"] = full_probabilities
    test_points.to_csv(args.output / "test_predictions.csv", index=False)

    joblib.dump({
        "model": full_rf, "features": FEATURES,
        "research_window": "2019–2021 inputs, 2023–2025 proxy outcome",
        "algorithm": "Random Forest Classifier", "holdout_seed": seed,
        "grid_block_degrees": 0.015,
        "warning": "Satellite-proxy prediction only, not native plant survival or planting suitability",
    }, args.output / "model_B_rf.joblib")

    result = {
        "n": int(len(df)),
        "classes": {str(int(k)): int(v) for k, v in df.persistence.value_counts().sort_index().items()},
        "unique_location_count": int(len(df[["longitude", "latitude"]].drop_duplicates())),
        "distinct_rainfall_values": int(df.rain_mean_month_mm.nunique()),
        "split": {
            "seed": int(seed), "n_train": int(len(tr)), "n_test": int(len(te)),
            "n_train_blocks": int(train.block.nunique()), "n_test_blocks": int(test.block.nunique()),
            "train_counts": {str(int(k)): int(v) for k, v in train.persistence.value_counts().items()},
            "test_counts": {str(int(k)): int(v) for k, v in test.persistence.value_counts().items()},
            "block_overlap": len(set(train.block) & set(test.block)),
        },
        "holdout_results": scores, "cross_validation_5fold": cv, "cv_summary": cv_summary,
        "limitations": [
            "Proxy label uses later satellite NDVI thresholds, not native plant survival",
            "Balanced class selection does not describe landscape prevalence",
            "Geographic blocks have no separation buffer; spatial autocorrelation remains possible",
            "Only Lahbab and one before/after window, with no temporally independent evaluation",
            "Rainfall and land-surface temperature inputs are coarser than the HLS features",
            "No field validation or irrigation-status verification; not a calibrated planting score",
        ],
        "simple_threshold_comparison": {
            "decision_tree_threshold_ndvi": round(float(stump.tree_.threshold[0]), 5),
            "test_accuracy": round(float(accuracy_score(test.persistence, stump_predictions)), 4),
            "test_macro_f1": round(float(f1_score(test.persistence, stump_predictions, average="macro")), 4),
            "spatial_cv_macro_f1_mean": round(float(np.mean(stump_folds)), 4),
            "spatial_cv_macro_f1_folds": [round(x, 4) for x in stump_folds],
        },
    }
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved evaluation artifacts in: {args.output}")


if __name__ == "__main__":
    main()
