# Diabetes Risk Prediction: an End-to-End ML Mini Project

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ayethuzar22/diabetes-risk-prediction/blob/main/notebooks/diabetes_ml_colab.ipynb)
[![Live Demo](https://img.shields.io/badge/demo-Hugging%20Face%20Spaces-yellow)](https://huggingface.co/spaces/YOUR_HF_USERNAME/diabetes-risk-prediction)

Predicts whether a patient is likely to have diabetes from eight clinical measurements, following the full ML lifecycle: problem framing, data selection, EDA, preparation, model comparison, tuning, evaluation, explainability and a Gradio app.

> **Disclaimer:** educational project only. Not a medical device and not medical advice.

## 1. Problem
Binary classification on the Pima Indians Diabetes dataset. In screening, a missed case (false negative) costs more than a false alarm, so I optimize for **recall** and **PR-AUC** instead of accuracy. A model that always predicts "no diabetes" would already reach about 65% accuracy while finding nobody.

## 2. Data
- Source: [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) (Kaggle), 768 rows, 8 numeric features, ~35% positive.
- **Data-quality finding:** zeros in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin` and `BMI` are impossible values, so they are missing data (Insulin ~49%, SkinThickness ~30%). I converted them to `NaN` and imputed inside the pipeline.

## 3. Approach
1. Stratified 80/20 train/test split **before** any fitting, to avoid leakage.
2. `Pipeline`: median imputation, standard scaling, model.
3. Compared Dummy, Logistic Regression, SVM, Random Forest and Gradient Boosting with 5-fold stratified CV.
4. Tuned Random Forest with randomized search (scoring: PR-AUC) and picked the winner by CV PR-AUC.
5. Chose the decision threshold from cross-validated predictions on the training set (target recall ≥ 0.80), so the test set stayed untouched until the final check.
6. Permutation importance for explainability.

## 4. Results
<!-- Replace with the numbers from YOUR final run. -->
| Metric | CV (train) | Test |
|---|---|---|
| PR-AUC | 0.748 | 0.673 |
| ROC-AUC | n/a | 0.813 |
| Recall @ chosen threshold | ~0.80 | 0.81 |
| Precision @ chosen threshold | n/a | 0.57 |

Selected model: Logistic Regression. It matched a tuned Random Forest (CV PR-AUC 0.748 vs 0.743), which is a good reminder that a simple, interpretable model can be enough on 768 rows. The random-guess PR-AUC baseline is about 0.35.

Add your images here:
`images/confusion_matrix.png`, `images/pr_curve.png`, `images/app_screenshot.png`

## 5. Limitations
- Small test set (154 rows), so metrics are noisy. Differences of a few points are not reliable.
- The data covers one population (Pima women aged 21+), so results may not generalize.
- Precision is modest because the threshold favors recall.
- Not clinically validated.

## 6. Run it
- **Colab:** click the badge above, then Runtime → Run all.
- **Locally:**
  ```bash
  pip install -r requirements.txt
  python app/app.py
  ```
  `models/diabetes_model.joblib` is created by the notebook.

## 7. Next steps
- Missing-value indicators and feature engineering, XGBoost/LightGBM, repeated CV with confidence intervals.
- Second experiment on the larger BRFSS diabetes dataset.
- Apply the same workflow to credit-risk / loan-default prediction for microfinance.
