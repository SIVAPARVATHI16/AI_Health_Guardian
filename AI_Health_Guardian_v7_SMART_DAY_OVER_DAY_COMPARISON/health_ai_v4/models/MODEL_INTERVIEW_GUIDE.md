# AI Health Guardian — ML Viva / Interview Questions

## Q1. Is Random Forest the only algorithm for binary tabular data?
No. Binary tabular data can be handled by Logistic Regression, Decision Tree,
KNN, SVM, Random Forest, Extra Trees, Gradient Boosting, XGBoost and others.

## Q2. Why does binary data work with these algorithms?
A binary feature is simply a numeric 0/1 feature. For example:
`fever=1, cough=1, nausea=0`. The complete patient becomes a fixed-length
feature vector.

## Q3. Why Random Forest?
In this project it was used as a practical production baseline because it can
learn nonlinear symptom interactions, needs little preprocessing, is robust
to individual-tree errors through ensembling, runs locally, and supports
`predict_proba()` for probability outputs.

Do not claim it had the highest score: the project's own comparison shows
higher test scores for some alternatives on this dataset.

## Q4. Which algorithms did you compare?
Logistic Regression, Decision Tree, KNN, SVM (RBF), Random Forest, Extra Trees,
Hist Gradient Boosting and XGBoost when available.

## Q5. What is hyperparameter tuning?
Hyperparameters are model settings chosen before fitting. Tuning means
systematically trying combinations and selecting one using a validation
procedure.

## Q6. What tuning did the original project use?
The original production training script manually specified Random Forest
settings. It did not use GridSearchCV or RandomizedSearchCV.

## Q7. What tuning did you add?
GridSearchCV with 3-fold cross-validation and weighted F1 scoring. The search
covers `n_estimators`, `max_depth`, `min_samples_split` and `min_samples_leaf`.

## Q8. Why keep the test set untouched?
If the test set is repeatedly used to choose hyperparameters, information from
the test set leaks into model selection. The final test score then becomes
less trustworthy.

## Q9. Why use F1?
F1 combines precision and recall. Weighted F1 gives more influence to classes
with more samples; macro F1 treats classes equally. Both are useful alongside
accuracy and confusion matrices.

## Q10. Why `class_weight="balanced"`?
It tells the classifier to give greater training importance to classes that
have fewer samples. It is a way to reduce the influence of class imbalance.

## Q11. What is `predict_proba()`?
It returns a probability-like distribution over the model's classes. The
application uses these values to show disease/risk probabilities. They should
not automatically be interpreted as calibrated medical probabilities.

## Q12. What is your strongest truthful answer to "Why only Random Forest?"
> "It is not the only possible algorithm. We evaluated several classifiers.
> Random Forest was our initial production baseline because it fits the
> binary tabular symptom representation, captures nonlinear interactions,
> requires limited preprocessing, works offline and provides probability
> estimates. We also measured alternatives and now have a reproducible
> GridSearchCV tuning workflow. We do not claim Random Forest was the highest
> scoring model on this dataset."

## Q13. If the interviewer asks, "Why didn't you just choose the highest accuracy?"
> "Model selection should not be based on accuracy alone. We also inspect
> precision, recall, F1 and confusion matrices, and we have application
> constraints such as local inference, maintainability and probability output.
> For a health-oriented system, class-specific errors matter."

## Q14. What should you never say?
Do not say:
- "Random Forest is the only algorithm for binary data."
- "Random Forest had the highest accuracy" — the comparison does not support
  that.
- "The model is 97% medically accurate" — these are project-dataset metrics,
  not clinical validation.
- "We used GridSearchCV in the original version" — that was added in this
  update.

## Q15. What is the current architecture?
Text → cleaning/symptom extraction → 79-dimensional binary symptom vector →
disease classifier + risk classifier → class probabilities → application
guidance.

The system is assistive and should not be presented as a diagnostic device.
