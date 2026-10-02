"""Tune the Random Forest candidates for AI Health Guardian.

This script deliberately keeps the original production models unchanged.
It:
1. creates the same stratified 80/20 split used by the project;
2. runs GridSearchCV on the TRAINING portion only;
3. selects hyperparameters using 3-fold cross-validation and weighted F1;
4. evaluates the selected configuration once on the untouched test set;
5. refits the selected configuration on the full dataset and saves it as a
   candidate tuned model (not an automatic production replacement).

Run from health_ai_v4/models:
    python tune_random_forest.py
"""

import json, os, time, joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)

BASE = os.path.dirname(__file__)
DATA = os.path.join(BASE, "..", "data")
OUT = BASE

df = pd.read_csv(os.path.join(DATA, "symptom_disease_dataset.csv"))
with open(os.path.join(DATA, "symptom_columns.json"), encoding="utf-8") as f:
    SC = json.load(f)

X = df[SC].to_numpy(dtype=np.int8)

PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [10, 14, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}

def metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    mp, mr, mf, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "weighted_precision": round(float(p), 6),
        "weighted_recall": round(float(r), 6),
        "weighted_f1": round(float(f), 6),
        "macro_precision": round(float(mp), 6),
        "macro_recall": round(float(mr), 6),
        "macro_f1": round(float(mf), 6),
    }

def tune_task(target, task):
    encoder = LabelEncoder().fit(target)
    y = encoder.transform(target)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    estimator = RandomForestClassifier(
        random_state=42, class_weight="balanced", n_jobs=-1
    )
    search = GridSearchCV(
        estimator=estimator,
        param_grid=PARAM_GRID,
        cv=3,
        scoring="f1_weighted",
        refit=True,
        n_jobs=-1,
        return_train_score=False,
    )
    started = time.time()
    search.fit(Xtr, ytr)
    elapsed = time.time() - started

    test_pred = search.best_estimator_.predict(Xte)
    test_metrics = metrics(yte, test_pred)

    # Refit the selected hyperparameters on ALL available records to create
    # a candidate deployment artifact. The untouched test result above is
    # retained as the honest selection/evaluation record.
    final_model = RandomForestClassifier(
        **search.best_params_,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    final_model.fit(X, y)

    model_path = os.path.join(OUT, f"{task}_model_tuned_candidate.joblib")
    joblib.dump(final_model, model_path)

    return {
        "task": task,
        "selection": {
            "method": "GridSearchCV",
            "cv_folds": 3,
            "scoring": "weighted_f1",
            "parameter_grid": PARAM_GRID,
            "best_params": search.best_params_,
            "best_cv_weighted_f1": round(float(search.best_score_), 6),
            "search_seconds": round(elapsed, 2),
        },
        "test_split": {
            "test_size": 0.20,
            "random_state": 42,
            "stratified": True,
            "metrics": test_metrics,
            "confusion_matrix": confusion_matrix(yte, test_pred).tolist(),
            "labels": list(encoder.classes_),
        },
        "candidate_artifact": model_path,
        "production_model_changed": False,
    }

report = {
    "purpose": "Reproducible Random Forest hyperparameter tuning",
    "dataset": {
        "records": int(len(df)),
        "symptoms": int(len(SC)),
        "diseases": int(df.prognosis.nunique()),
        "risk_classes": list(LabelEncoder().fit(df.risk_level).classes_),
    },
    "important_note": (
        "This tuning script does not automatically replace the production "
        "models. Candidate tuned artifacts are saved separately so the "
        "project team can validate them before deployment."
    ),
    "disease": tune_task(df["prognosis"], "disease"),
    "risk": tune_task(df["risk_level"], "risk"),
}

with open(os.path.join(OUT, "random_forest_tuning_report.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print(json.dumps(report, indent=2))
