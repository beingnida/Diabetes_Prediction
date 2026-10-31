import os

import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.datasets import load_diabetes

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_diabetes_model.joblib")

FEATURE_INFO = {
    "age": ("Age", "Age in years (normalized)"),
    "sex": ("Sex", "Sex (normalized)"),
    "bmi": ("BMI", "Body mass index (normalized)"),
    "bp": ("Blood Pressure", "Average blood pressure (normalized)"),
    "s1": ("S1 - Total Cholesterol", "tc, total serum cholesterol (normalized)"),
    "s2": ("S2 - LDL", "ldl, low-density lipoproteins (normalized)"),
    "s3": ("S3 - HDL", "hdl, high-density lipoproteins (normalized)"),
    "s4": ("S4 - TCH/HDL", "total cholesterol / HDL (normalized)"),
    "s5": ("S5 - Log Triglycerides", "possibly log of serum triglycerides level (normalized)"),
    "s6": ("S6 - Blood Sugar", "glu, blood sugar level (normalized)"),
}

FEATURE_ORDER = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at: {MODEL_PATH}")
        st.info("Please run the notebook `diabetes_regression_project.ipynb` first to train and save the model.")
        st.stop()
    return joblib.load(MODEL_PATH)


@st.cache_data
def get_feature_bounds():
    data = load_diabetes(as_frame=True)
    df = data.frame
    bounds = {}
    for col in FEATURE_ORDER:
        bounds[col] = (float(df[col].min()), float(df[col].max()), float(df[col].mean()))
    return bounds


def main():
    st.set_page_config(page_title="Diabetes Progression Predictor", page_icon="🩺", layout="wide")

    st.title("🩺 Diabetes Progression Predictor")
    st.markdown(
        "Predict the **quantitative measure of disease progression one year after baseline** "
        "using the Ridge regression model trained in `diabetes_regression_project.ipynb`."
    )

    model = load_model()
    bounds = get_feature_bounds()

    with st.sidebar:
        st.header("Input Features")
        st.caption("Values are pre-normalized (mean-centered & scaled) as in the sklearn Diabetes dataset.")
        use_sample = st.checkbox("Use a sample patient from the dataset", value=False)
        sample_idx = st.slider("Sample patient index", 0, 441, 0, disabled=not use_sample)

        if use_sample:
            data = load_diabetes(as_frame=True)
            sample = data.frame.iloc[sample_idx]
            values = {col: float(sample[col]) for col in FEATURE_ORDER}
        else:
            values = {}
            for col in FEATURE_ORDER:
                lo, hi, mean = bounds[col]
                label, desc = FEATURE_INFO[col]
                values[col] = st.slider(
                    label,
                    min_value=round(lo, 4),
                    max_value=round(hi, 4),
                    value=round(mean, 4),
                    step=0.001,
                    help=desc,
                )

        predict_btn = st.button("Predict", type="primary", use_container_width=True)

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.subheader("Feature Values")
        input_df = pd.DataFrame([values])[FEATURE_ORDER]
        st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

        st.subheader("Feature Descriptions")
        desc_rows = [{"Feature": FEATURE_INFO[c][0], "Description": FEATURE_INFO[c][1]} for c in FEATURE_ORDER]
        st.table(pd.DataFrame(desc_rows))

    with col_right:
        st.subheader("Prediction")
        if predict_btn or use_sample:
            X_input = pd.DataFrame([values])[FEATURE_ORDER]
            prediction = float(model.predict(X_input)[0])
            st.metric(label="Disease Progression Score", value=f"{prediction:.2f}")

            if prediction > 200:
                st.warning("⚠️ High predicted progression. Consider clinical follow-up.")
            elif prediction < 100:
                st.success("✅ Low predicted progression.")
            else:
                st.info("ℹ️ Moderate predicted progression.")

            st.markdown("---")
            st.markdown("**Model Performance (test set)**")
            perf = pd.DataFrame(
                {
                    "Metric": ["R2", "RMSE", "MAE"],
                    "Value": [0.4541, 53.78, 42.81],
                }
            )
            st.table(perf)
        else:
            st.info("Click **Predict** in the sidebar to get a prediction.")

    st.markdown("---")
    st.caption(
        "Model: Ridge regression (alpha=1) inside a StandardScaler pipeline. "
        "Trained on the sklearn Diabetes dataset (442 samples, 10 features)."
    )


if __name__ == "__main__":
    main()