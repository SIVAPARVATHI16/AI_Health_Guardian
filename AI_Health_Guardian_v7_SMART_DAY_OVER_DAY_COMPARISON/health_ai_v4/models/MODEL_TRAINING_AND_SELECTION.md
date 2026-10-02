# AI Health Guardian — ML Algorithms, Tuning & Model Selection (v7 update)

## 1. Is Random Forest the only algorithm for binary tabular data?

No. A binary tabular feature matrix can be used with many supervised classifiers.
In this project the symptom vector has 79 binary features (`0 = symptom not
detected`, `1 = symptom detected`). The ZIP already contains a fair comparison
experiment using:

- Logistic Regression
- Decision Tree
- KNN
- SVM (RBF)
- Random Forest
- Extra Trees
- Hist Gradient Boosting
- XGBoost (when the optional package is installed)

So the correct viva answer is **not** "binary tabular data only works with
Random Forest."

## 2. Why can these algorithms work?

Binary features are still numerical features. A classifier only needs a
consistent feature vector and target label.

- Logistic Regression learns a linear relationship between symptom indicators
  and class probabilities.
- Decision Tree learns if/then symptom splits.
- KNN compares a new symptom vector with nearby training vectors.
- SVM finds a separating decision boundary; the RBF kernel can model
  nonlinear boundaries.
- Random Forest combines many decision trees.
- Extra Trees is another randomized tree ensemble.
- Hist Gradient Boosting builds trees sequentially to correct earlier errors.
- XGBoost is a gradient-boosted tree implementation.
- Neural networks could also process the same tabular vector, although that
  is not automatically necessary for this dataset.

## 3. What did the original v7 project actually do?

`train_all.py` trains the production Random Forest models with explicitly
specified hyperparameters:

Disease:
`n_estimators=120, max_depth=14, class_weight="balanced", random_state=42`

Risk:
`n_estimators=100, max_depth=10, class_weight="balanced", random_state=42`

The original training script did **not** use GridSearchCV or
RandomizedSearchCV. These values were manually specified.

The separate `compare_models.py` evaluates multiple algorithms on the same
stratified 80/20 split, but it is a comparison experiment; it does not tune
the production Random Forest with cross-validation.

## 4. What is hyperparameter tuning?

A hyperparameter is a setting chosen before/during model training rather than
learned directly from the training rows.

For Random Forest examples:
- `n_estimators`: number of trees.
- `max_depth`: maximum depth of each tree.
- `min_samples_split`: minimum samples needed to split a node.
- `min_samples_leaf`: minimum samples allowed in a leaf.

Tuning means trying sensible combinations of these settings and selecting the
configuration that performs best under a predefined validation procedure.

### Why tune?

Poor settings can cause:
- underfitting (model too simple),
- overfitting (model memorizes training patterns),
- unnecessary computation,
- weaker generalization.

Tuning tries to find a better balance between model complexity and
generalization.

## 5. What tuning was added in this update?

A new `models/tune_random_forest.py` script uses **GridSearchCV**.

Process:

1. Make the same stratified 80/20 train/test split.
2. Keep the test set untouched during tuning.
3. Run 3-fold cross-validation only on the training set.
4. Try the parameter grid:
   - `n_estimators`: 100, 200
   - `max_depth`: 10, 14, None
   - `min_samples_split`: 2, 5
   - `min_samples_leaf`: 1, 2
5. Select the combination with the highest cross-validated weighted F1.
6. Evaluate that selected configuration once on the untouched test set.
7. Refit the selected configuration on all records and save it as a
   **candidate tuned model**.

The test set is not used to choose the hyperparameters. This avoids leaking
test information into model selection.

## 6. Actual tuning results in this update

| Task | Best parameters | CV weighted F1 | Test accuracy | Test weighted F1 |
|---|---|---:|---:|---:|
| Disease | `n_estimators=200, max_depth=None, min_samples_split=5, min_samples_leaf=1` | 97.7802% | 98.3871% | 98.3877% |
| Risk | `n_estimators=100, max_depth=None, min_samples_split=5, min_samples_leaf=1` | 98.5896% | 99.0323% | 99.0329% |

These are performance measurements on the project's dataset/test split, not
clinical diagnostic accuracy.

The tuned candidates are saved separately:
- `disease_model_tuned_candidate.joblib`
- `risk_model_tuned_candidate.joblib`

The existing production `disease_model.joblib` and `risk_model.joblib` are
not silently replaced.

## 7. How to explain the production Random Forest choice

The ZIP's comparison experiment shows that other algorithms can outperform
the current production Random Forest on this particular held-out split. For
example, the recorded disease test accuracy is 98.5484% for XGBoost versus
97.4194% for the production Random Forest; for risk it is 99.5161% for KNN
versus 96.6129% for the production Random Forest.

Therefore, do **not** say:
"Random Forest was chosen because it had the highest accuracy."

A truthful explanation is:

> "We compared multiple algorithms on the same binary symptom feature
> matrix and the same stratified train/test split. Random Forest was selected
> as the initial production baseline because it handles nonlinear interactions
> in tabular symptom data, requires relatively little preprocessing, provides
> probability estimates through `predict_proba`, and is practical for our
> local/offline application. We also retained the comparison results instead
> of claiming Random Forest had the highest score. In this dataset, some
> alternatives scored higher. We have now added GridSearchCV-based tuning for
> Random Forest so its hyperparameters are selected systematically before a
> future production replacement."

## 8. Interview question: "Can you use another algorithm?"

Answer:

> "Yes. Binary tabular features are not restricted to Random Forest. We tested
> Logistic Regression, Decision Tree, KNN, RBF-SVM, Random Forest, Extra Trees,
> Hist Gradient Boosting and XGBoost. The choice depends on the data,
> preprocessing requirements, probability output needs, computational
> constraints and validation results."

## 9. Interview question: "What tuning technique did you use?"

For the **original v7**:

> "The original production model used manually specified Random Forest
> hyperparameters; it did not use GridSearchCV or RandomizedSearchCV."

For this **updated v7 package**:

> "I added GridSearchCV with 3-fold cross-validation. I tune the number of
> trees, tree depth, minimum samples for splitting and minimum samples per
> leaf, using weighted F1 as the selection metric. The test set remains
> untouched until final evaluation."

## 10. Interview question: "Why weighted F1 instead of only accuracy?"

> "Accuracy can hide class-specific errors, especially in multiclass
> classification. F1 combines precision and recall. Weighted F1 accounts for
> each class while weighting by its number of samples. We also report macro
> precision, macro recall and macro F1 so class balance can be inspected."

For a safety-oriented health application, class-specific recall and confusion
matrices should also be inspected rather than relying on one aggregate number.

## 11. Important project limitation

The dataset is a project dataset, not a clinically validated dataset. High
test metrics should not be described as medical diagnostic accuracy. A real
clinical system would require independent external validation, calibration,
clinical review and safety evaluation.
