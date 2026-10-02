# v7 ML/Tuning Update

Added:
- `health_ai_v4/models/tune_random_forest.py`
- `health_ai_v4/models/random_forest_tuning_report.json`
- `health_ai_v4/models/disease_model_tuned_candidate.joblib`
- `health_ai_v4/models/risk_model_tuned_candidate.joblib`
- `health_ai_v4/models/MODEL_TRAINING_AND_SELECTION.md` (expanded)
- `health_ai_v4/models/MODEL_INTERVIEW_GUIDE.md`

The original production Random Forest artifacts remain unchanged. The tuned
models are candidates for validation; the application is not silently switched
to them.
