import os
import numpy as np, pandas as pd, joblib
import gradio as gr

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "diabetes_model.joblib")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "diabetes_model.joblib"  # for Hugging Face Spaces, file at repo root

bundle = joblib.load(MODEL_PATH)
pipe, thr, feats = bundle["pipeline"], bundle["threshold"], bundle["features"]
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

def predict(pregnancies, glucose, bp, skin, insulin, bmi, dpf, age):
    row = pd.DataFrame([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]], columns=feats)
    row[ZERO_AS_MISSING] = row[ZERO_AS_MISSING].replace(0, np.nan)
    p = float(pipe.predict_proba(row)[0, 1])
    verdict = "Higher risk: consider a clinical check" if p >= thr else "Lower risk"
    return {"Diabetes": p, "No diabetes": 1 - p}, f"{verdict}\n(risk score {p:.0%}, decision threshold {thr:.0%})"

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(0, 17, value=2, step=1, label="Pregnancies"),
        gr.Slider(0, 200, value=120, label="Glucose (mg/dL, 0 = unknown)"),
        gr.Slider(0, 122, value=70, label="Blood pressure (mm Hg, 0 = unknown)"),
        gr.Slider(0, 99, value=20, label="Skin thickness (mm, 0 = unknown)"),
        gr.Slider(0, 846, value=80, label="Insulin (µU/mL, 0 = unknown)"),
        gr.Slider(0, 67, value=28, label="BMI (0 = unknown)"),
        gr.Slider(0.05, 2.5, value=0.4, label="Diabetes pedigree function"),
        gr.Slider(21, 81, value=33, step=1, label="Age"),
    ],
    outputs=[gr.Label(label="Prediction"), gr.Textbox(label="Interpretation")],
    title="Diabetes Risk Predictor (educational demo)",
    description="Trained on the Pima Indians Diabetes dataset. Not medical advice.",
)

if __name__ == "__main__":
    demo.launch()
