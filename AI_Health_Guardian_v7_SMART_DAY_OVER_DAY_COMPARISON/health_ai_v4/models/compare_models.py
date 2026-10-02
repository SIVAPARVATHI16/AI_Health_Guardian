"""Compare candidate classifiers for the AI Health Guardian symptom/risk datasets.

Uses the SAME stratified 80/20 split and feature matrix as train_all.py so that
model comparisons are reproducible and directly comparable. This script does
not replace the production Random Forest models; it creates evaluation reports.
"""
import json, os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

BASE = os.path.dirname(__file__)
DATA = os.path.join(BASE, "..", "data")
OUT = BASE

df = pd.read_csv(os.path.join(DATA, "symptom_disease_dataset.csv"))
with open(os.path.join(DATA, "symptom_columns.json"), encoding="utf-8") as f:
    SC = json.load(f)

X = df[SC].to_numpy(dtype=np.int8)

def make_models(task_name, n_classes=None):
    models = {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced"))
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced", max_depth=16),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "SVM (RBF)": Pipeline([
            ("scale", StandardScaler()),
            ("model", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42))
        ]),
        "Random Forest (production)": RandomForestClassifier(
            n_estimators=120 if task_name == "disease" else 100,
            max_depth=14 if task_name == "disease" else 10,
            random_state=42, class_weight="balanced", n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(n_estimators=120, max_depth=18, random_state=42, class_weight="balanced", n_jobs=-1),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=31, random_state=42),
    }
    if HAVE_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=180, max_depth=5, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            objective="multi:softprob", eval_metric="mlogloss",
            tree_method="hist", random_state=42, n_jobs=4
        )
    return models

def evaluate(target, task_name, encoder=None):
    y = target
    if encoder is None:
        encoder = __import__("sklearn.preprocessing", fromlist=["LabelEncoder"]).LabelEncoder().fit(y)
    y = encoder.transform(y)
    labels = list(encoder.classes_)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    rows = []
    cms = {}
    for name, model in make_models(task_name, len(labels)).items():
        # XGBoost expects compact integer labels; other sklearn models accept them too.
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        acc = accuracy_score(yte, pred)
        p, r, f1, _ = precision_recall_fscore_support(
            yte, pred, average="weighted", zero_division=0
        )
        mp, mr, mf1, _ = precision_recall_fscore_support(
            yte, pred, average="macro", zero_division=0
        )
        rows.append({
            "task": task_name,
            "model": name,
            "accuracy": round(float(acc), 6),
            "weighted_precision": round(float(p), 6),
            "weighted_recall": round(float(r), 6),
            "weighted_f1": round(float(f1), 6),
            "macro_precision": round(float(mp), 6),
            "macro_recall": round(float(mr), 6),
            "macro_f1": round(float(mf1), 6),
        })
        cm = confusion_matrix(yte, pred, labels=list(range(len(labels))))
        cms[name] = {
            "labels": labels,
            "matrix": cm.tolist(),
        }

    return rows, cms

from sklearn.preprocessing import LabelEncoder
risk_encoder = LabelEncoder().fit(df["risk_level"])

disease_rows, disease_cms = evaluate(df["prognosis"], "disease", None)
risk_rows, risk_cms = evaluate(df["risk_level"], "risk", risk_encoder)

all_rows = disease_rows + risk_rows
pd.DataFrame(all_rows).to_csv(os.path.join(OUT, "model_comparison.csv"), index=False)

with open(os.path.join(OUT, "model_comparison_report.json"), "w", encoding="utf-8") as f:
    json.dump({
        "dataset": {
            "records": int(len(df)),
            "symptoms": int(len(SC)),
            "diseases": int(df.prognosis.nunique()),
            "risk_classes": list(risk_encoder.classes_),
            "split": "stratified 80/20, random_state=42"
        },
        "notes": [
            "These are dataset/test-split metrics, not clinical diagnostic accuracy.",
            "The production Random Forest model was not replaced by this comparison script.",
            "Models use the same symptom feature matrix and split for a fair comparison."
        ],
        "disease": disease_rows,
        "risk": risk_rows,
        "confusion_matrices": {"disease": disease_cms, "risk": risk_cms}
    }, f, indent=2)

# Human-readable review notes.
dfres = pd.DataFrame(all_rows)
with open(os.path.join(OUT, "MODEL_COMPARISON_README.md"), "w", encoding="utf-8") as f:
    f.write("# Symptom/Risk Model Comparison\n\n")
    f.write("All models use the same 80/20 stratified split (`random_state=42`).\n\n")
    f.write("**Important:** these metrics measure performance on the project's dataset/test split. They are not clinical diagnostic accuracy.\n\n")
    for task in ["disease", "risk"]:
        f.write(f"## {task.title()} classification\n\n")
        sub = dfres[dfres.task == task].sort_values("accuracy", ascending=False)
        f.write(sub[["model","accuracy","weighted_precision","weighted_recall","weighted_f1","macro_f1"]].to_markdown(index=False))
        f.write("\n\n")
    f.write("## Why Random Forest remains the production model\n\n")
    f.write("Random Forest is retained for the deployed symptom/risk pipeline because it is a strong tabular baseline, works naturally with binary symptom features, is straightforward to run locally/offline, and is easy to explain in a project review. The comparison report is evidence for discussion; it does not by itself establish clinical suitability.\n")

print(pd.DataFrame(all_rows).to_string(index=False))
print("\nSaved model_comparison.csv, model_comparison_report.json and MODEL_COMPARISON_README.md")
