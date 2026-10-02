# Symptom/Risk Model Comparison

All models use the same 80/20 stratified split (`random_state=42`).

**Important:** these metrics measure performance on the project's dataset/test split. They are not clinical diagnostic accuracy.

## Disease classification

| model                      |   accuracy |   weighted_precision |   weighted_recall |   weighted_f1 |   macro_f1 |
|:---------------------------|-----------:|---------------------:|------------------:|--------------:|-----------:|
| XGBoost                    |   0.985484 |             0.986785 |          0.985484 |      0.985566 |   0.985566 |
| Logistic Regression        |   0.983871 |             0.985407 |          0.983871 |      0.98384  |   0.98384  |
| Hist Gradient Boosting     |   0.983871 |             0.984707 |          0.983871 |      0.983885 |   0.983885 |
| KNN                        |   0.982258 |             0.983427 |          0.982258 |      0.982304 |   0.982304 |
| SVM (RBF)                  |   0.979032 |             0.980557 |          0.979032 |      0.979069 |   0.979069 |
| Random Forest (production) |   0.974194 |             0.975494 |          0.974194 |      0.974154 |   0.974154 |
| Extra Trees                |   0.964516 |             0.966354 |          0.964516 |      0.964504 |   0.964504 |
| Decision Tree              |   0.582258 |             0.75878  |          0.582258 |      0.635812 |   0.635812 |

## Risk classification

| model                      |   accuracy |   weighted_precision |   weighted_recall |   weighted_f1 |   macro_f1 |
|:---------------------------|-----------:|---------------------:|------------------:|--------------:|-----------:|
| KNN                        |   0.995161 |             0.995177 |          0.995161 |      0.995165 |   0.995156 |
| XGBoost                    |   0.98871  |             0.988837 |          0.98871  |      0.988725 |   0.988799 |
| Extra Trees                |   0.987097 |             0.987148 |          0.987097 |      0.987095 |   0.986966 |
| SVM (RBF)                  |   0.987097 |             0.987122 |          0.987097 |      0.987098 |   0.987121 |
| Hist Gradient Boosting     |   0.982258 |             0.982255 |          0.982258 |      0.982253 |   0.98219  |
| Random Forest (production) |   0.966129 |             0.9664   |          0.966129 |      0.966121 |   0.9663   |
| Logistic Regression        |   0.964516 |             0.964812 |          0.964516 |      0.964483 |   0.964716 |
| Decision Tree              |   0.925806 |             0.92668  |          0.925806 |      0.925902 |   0.925624 |

## Why Random Forest remains the production model

Random Forest is retained for the deployed symptom/risk pipeline because it is a strong tabular baseline, works naturally with binary symptom features, is straightforward to run locally/offline, and is easy to explain in a project review. The comparison report is evidence for discussion; it does not by itself establish clinical suitability.
