# Model Card: Diabetes Risk Predictor

## Model details
- Type: scikit-learn `Pipeline` (median imputer, standard scaler, Logistic Regression). Selected by CV PR-AUC against a tuned Random Forest.
- Output: probability of diabetes plus a binary flag at a decision threshold chosen for recall ≥ 0.80 on cross-validated training predictions.
- Author: ayethuzar22

## Intended use
- Educational demonstration of an end-to-end ML workflow.
- **Not** for diagnosis, treatment or any clinical or insurance decision.

## Training data
- Pima Indians Diabetes Database (Kaggle), 768 rows, 8 features.
- Zeros in Glucose, BloodPressure, SkinThickness, Insulin, BMI treated as missing.
- Stratified 80/20 split, random_state=42.

## Evaluation
- Metrics: recall, precision, F1, ROC-AUC, PR-AUC (primary).
- See README results table. Test set has 154 rows, so uncertainty is large.

## Ethical considerations and limitations
- Single population (Pima women aged 21+); may not generalize to other groups.
- Small dataset and noisy measurements.
- False negatives mean a missed case; false positives mean unnecessary worry or testing. The threshold favors fewer misses.
- Probabilities are not calibrated or clinically validated.

## Caveats
Always consult a medical professional for health decisions.
