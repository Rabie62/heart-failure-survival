import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="CardioCare: Heart Failure Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BUNDLE_PATH = r"C:\Users\rabi3\.gemini\antigravity\scratch\heart-failure-survival\heart_failure_bundle.joblib"
DATA_PATH = r"C:\Users\rabi3\.gemini\antigravity\scratch\heart-failure-survival\heart_failure_clinical_records_dataset.csv"

@st.cache_resource
def load_bundle():
    if not os.path.exists(BUNDLE_PATH):
        st.error("Model bundle not found!")
        st.stop()
    return joblib.load(BUNDLE_PATH)

bundle = load_bundle()
models = bundle['models']
eval_results = bundle['eval_results']
best_params = bundle.get('best_params', {})
feature_importance = pd.DataFrame(bundle['feature_importance'])
feature_summary = bundle['feature_summary']

# App Title
st.title("❤️ CardioCare: Heart Failure Mortality Risk Predictor")

# Sidebar
with st.sidebar:
    st.header("Tuned Model Selection")
    
    selected_model_name = st.selectbox(
        "Select Model:",
        list(models.keys()),
        index=0
    )
    current_model = models[selected_model_name]
    m_info = eval_results[selected_model_name]

    threshold = st.slider(
        "Decision Threshold:",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Probability threshold for predicting a positive mortality event."
    )

    st.markdown("---")
    st.subheader("Model Performance")
    c1, c2 = st.columns(2)
    c1.metric("5-Fold CV AUC", f"{m_info['5-Fold CV ROC-AUC']:.3f}")
    c2.metric("Test AUROC", f"{m_info['Test AUROC']:.3f}")
    c1.metric("Accuracy", f"{m_info['Accuracy']:.1%}")
    c2.metric("F1 Score", f"{m_info['F1 Score']:.3f}")

tabs = st.tabs([
    "🩺 Bedside Patient Evaluation",
    "⚙️ Model Benchmarks and Optimal Parameters",
    "📈 Feature Importance Rankings",
    "📂 Explore Clinical Cohort Data"
])

# ---------------- TAB 1: Patient Evaluation ----------------
with tabs[0]:
    st.subheader(f"Patient Risk Assessment (`{selected_model_name}`)")
    
    preset = st.selectbox(
        "Quick Load Clinical Case Preset:",
        [
            "Custom Patient Input",
            "Case 1: Low Risk (Preserved EF, Normal Creatinine, Long Follow-up)",
            "Case 2: Moderate Risk (Elderly, Moderate Systolic Dysfunction)",
            "Case 3: High Risk (Severe Systolic Dysfunction, Acute Renal Impairment)"
        ]
    )

    defaults = {
        'age': 60, 'ejection_fraction': 38, 'serum_creatinine': 1.1,
        'serum_sodium': 137, 'creatinine_phosphokinase': 250,
        'platelets': 260000, 'high_blood_pressure': 0, 'time': 120
    }

    if preset == "Case 1: Low Risk (Preserved EF, Normal Creatinine, Long Follow-up)":
        defaults.update({
            'age': 52, 'ejection_fraction': 50, 'serum_creatinine': 0.8,
            'serum_sodium': 140, 'creatinine_phosphokinase': 150,
            'platelets': 280000, 'high_blood_pressure': 0, 'time': 210
        })
    elif preset == "Case 2: Moderate Risk (Elderly, Moderate Systolic Dysfunction)":
        defaults.update({
            'age': 70, 'ejection_fraction': 35, 'serum_creatinine': 1.6,
            'serum_sodium': 134, 'creatinine_phosphokinase': 380,
            'platelets': 210000, 'high_blood_pressure': 1, 'time': 80
        })
    elif preset == "Case 3: High Risk (Severe Systolic Dysfunction, Acute Renal Impairment)":
        defaults.update({
            'age': 75, 'ejection_fraction': 20, 'serum_creatinine': 3.2,
            'serum_sodium': 128, 'creatinine_phosphokinase': 850,
            'platelets': 140000, 'high_blood_pressure': 1, 'time': 15
        })

    st.markdown("#### Patient Attributes")
    row1_c1, row1_c2, row1_c3, row1_c4 = st.columns(4)
    with row1_c1:
        age = st.number_input("Age (years)", 18, 100, int(defaults['age']))
    with row1_c2:
        ef = st.number_input("Ejection Fraction (%)", 10, 85, int(defaults['ejection_fraction']))
    with row1_c3:
        hbp_str = st.selectbox("High Blood Pressure", ["No (0)", "Yes (1)"], index=int(defaults['high_blood_pressure']))
        hbp = 1 if "Yes" in hbp_str else 0
    with row1_c4:
        time_days = st.number_input("Follow-Up Duration (Days)", 1, 300, int(defaults['time']))

    row2_c1, row2_c2, row2_c3, row2_c4 = st.columns(4)
    with row2_c1:
        creatinine = st.number_input("Serum Creatinine (mg/dL)", 0.4, 10.0, float(defaults['serum_creatinine']), step=0.1)
    with row2_c2:
        sodium = st.number_input("Serum Sodium (mEq/L)", 110, 155, int(defaults['serum_sodium']))
    with row2_c3:
        cpk = st.number_input("CPK Enzyme (mcg/L)", 20, 8000, int(defaults['creatinine_phosphokinase']))
    with row2_c4:
        platelets = st.number_input("Platelets (kiloplatelets/mL)", 20000, 900000, int(defaults['platelets']), step=5000)

    cardiorenal_idx = creatinine / (ef / 100.0)
    age_to_ef_val = age / (ef + 1e-4)
    na_creat_ratio = sodium / (creatinine + 1e-4)

    patient_df = pd.DataFrame([{
        'time': time_days,
        'serum_creatinine': creatinine,
        'ejection_fraction': ef,
        'age': age,
        'serum_sodium': sodium,
        'creatinine_phosphokinase': cpk,
        'platelets': platelets,
        'high_blood_pressure': hbp,
        'cardiorenal_index': cardiorenal_idx,
        'age_to_ef': age_to_ef_val,
        'sodium_creatinine_ratio': na_creat_ratio
    }])

    mortality_prob = float(current_model.predict_proba(patient_df)[0, 1])
    is_high_risk = mortality_prob >= threshold

    st.markdown("---")
    res1, spacer, res2 = st.columns([1.2, 0.4, 2.0])

    with res1:
        st.subheader("Prediction Result")
        if mortality_prob >= 0.60:
            st.error(f"🚨 **HIGH RISK PATIENT** ({mortality_prob:.1%})")
            st.markdown("**Action:** Intensive monitoring and medication adjustment.")
        elif mortality_prob >= threshold:
            st.warning(f"⚠️ **ELEVATED RISK** ({mortality_prob:.1%})")
            st.markdown("**Action:** Inpatient monitoring, follow renal/electrolyte trends.")
        else:
            st.success(f"✅ **LOW RISK / STABLE** ({mortality_prob:.1%})")
            st.markdown("**Action:** Standard outpatient management.")

    with res2:
        st.subheader("Key Clinical Indicators")
        b1, b2, b3 = st.columns(3)
        b1.metric("Serum Creatinine", f"{creatinine:.1f} mg/dL", delta="Elevated (>1.3)" if creatinine > 1.3 else "Normal", delta_color="inverse")
        b2.metric("Ejection Fraction", f"{ef}%", delta="Reduced (<40%)" if ef < 40 else "Preserved", delta_color="normal" if ef >= 50 else "inverse")
        b3.metric("Cardiorenal Index", f"{cardiorenal_idx:.2f}", delta="Severe (>3.0)" if cardiorenal_idx > 3.0 else "Normal", delta_color="inverse")

# ---------------- TAB 2: Model Benchmarks & Optimal Parameters ----------------
with tabs[1]:
    st.subheader("5-Fold Stratified GridSearchCV Performance")
    
    col_bench1, col_bench2 = st.columns([1.6, 1.4])
    with col_bench1:
        st.markdown("#### Cross-Validation and Test Holdout Scores")
        bench_df = pd.DataFrame(eval_results).T.reset_index().rename(columns={'index': 'Model'})
        st.dataframe(bench_df, use_container_width=True)

    with col_bench2:
        st.markdown("#### Optimal Parameters (via GridSearch)")
        for m_name, params in best_params.items():
            with st.expander(f"🔹 {m_name}", expanded=True):
                st.write(params)

# ---------------- TAB 3: Feature Importance ----------------
with tabs[2]:
    st.subheader("Clinical Feature Importance Rankings")
    fig_full_fi = px.bar(
        feature_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale="Purples"
    )
    fig_full_fi.update_layout(yaxis={'categoryorder': 'total ascending'}, height=420)
    st.plotly_chart(fig_full_fi, use_container_width=True)

# ---------------- TAB 4: Explore Data ----------------
with tabs[3]:
    st.subheader("Explore the UCI Heart Failure Cohort")
    if os.path.exists(DATA_PATH):
        raw_df = pd.read_csv(DATA_PATH)
        st.dataframe(raw_df, use_container_width=True)
