# ❤️ CardioCare: Heart Failure Clinical Mortality Risk Predictor

A streamlined Machine Learning classification project built on the official **UCI Heart Failure Clinical Records Dataset**.

---

## 🎯 Clinical Motivation & Objectives
Heart failure occurs when the cardiac muscle cannot pump sufficient blood to meet systemic needs. In acute and emergency admission settings, clinicians need reliable risk stratification tools to:
- Identify high-risk patients who require intensive telemetry and closer monitoring.
- Target early intervention and optimize guideline-directed medical therapy (GDMT).

---

## 🏗️ Technical Architecture

### 1. Data (Real Dataset)
* **Source:** [UCI Heart Failure Clinical Records](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records)
* **File:** Single clean CSV (`heart_failure_clinical_records_dataset.csv`) with 299 patients, 11 clinical admission features, and 0 missing values.
* **Core Biomarkers:**
  * **Ejection Fraction (EF):** Percentage of blood leaving the left ventricle with each contraction ($<40\%$ indicates systolic failure).
  * **Serum Creatinine & Sodium:** Markers of kidney function and cardiorenal syndrome.
  * **Creatinine Phosphokinase (CPK):** Cardiac/muscle tissue injury enzyme.
  * **Age & Comorbidities:** High blood pressure, anaemia, diabetes, and smoking.

### 2. Modeling & Benchmarks
Standard supervised binary classification predicting in-hospital mortality (`DEATH_EVENT`):
* **Algorithms Evaluated:**
  * **Random Forest** (Non-linear ensemble)
  * **XGBoost** (Gradient boosted trees)
  * **LightGBM** (Histogram gradient boosting)
  * **Logistic Regression** (Standardized linear baseline)
* **Performance:**
  * **AUROC:** ~0.81
  * **AUPRC:** ~0.65
  * **Sensitivity / Recall:** ~92% (at optimal triage threshold)

---

## 📁 Repository Structure

```text
heart-failure-survival/
├── heart_failure_clinical_records_dataset.csv   # Real UCI dataset (12 KB, 299 patients)
├── heart_failure_clinical_pipeline.ipynb        # Single end-to-end Jupyter Notebook
├── app.py                                       # Interactive Streamlit Web Application
├── train_pipeline.py                            # Standalone model training & serialization script
├── heart_failure_bundle.joblib                  # Serialized ML models bundle
└── README.md                                    # Documentation & clinical guide
```

---

## 🚀 How to Run

### 1. Run the Jupyter Notebook
Open and execute all cells in Jupyter Lab or VS Code:
```bash
jupyter lab heart_failure_clinical_pipeline.ipynb
```

### 2. Run the Streamlit Application
Launch the web interface locally:
```bash
streamlit run app.py
```

### App Features:
1. **🩺 Bedside Patient Risk Scoring:**
   * Interactive input form for demographics, cardiac metrics, and renal lab values.
   * Pre-configured clinical cases (*Low Risk*, *Elderly Renal Impairment*, *Severe Systolic Failure*).
   * Visual risk gauge, threshold adjustment slider, and clinical guidance.
2. **📊 Model Benchmarking & Feature Rankings:**
   * Side-by-side performance table for all 4 models and feature importance bar plots.
3. **📂 Explore Clinical Cohort Data:**
   * Interactive table and cardiorenal scatter plot (Ejection Fraction vs. Serum Creatinine).
