# ❤️ CardioCare: Heart Failure Mortality Risk Predictor

A machine learning project predicting patient mortality risk in heart failure cases using the official **UCI Heart Failure Clinical Records Dataset**. 

The pipeline incorporates **Cardiorenal Interaction Feature Engineering**, **SMOTE Imbalance Resampling**, and **5-Fold Stratified GridSearchCV** hyperparameter optimization, deployed through an interactive **Streamlit** web application.

---

## 🎯 Clinical Motivation and Problem Formulation
Heart failure occurs when the myocardium cannot pump sufficient blood to sustain systemic oxygen demands. In emergency admission and inpatient cardiology units, reliable risk stratification helps clinicians:
- Identify high-risk patients who require intensive cardiac telemetry and closer hemodynamic observation.
- Prevent multi-organ damage through early intervention and medication titration (GDMT).

---

## 🏗️ Technical Architecture

### 1. Dataset
* **Source:** [UCI Heart Failure Clinical Records](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records)
* **File:** Clean tabular dataset (`heart_failure_clinical_records_dataset.csv`) with 299 patients and 0 missing values.
* **Core Biomarkers:**
  * **Ejection Fraction (EF):** Percentage of blood leaving the left ventricle with each contraction ($<40\%$ indicates systolic failure).
  * **Serum Creatinine and Sodium:** Key indicators of renal perfusion and cardiorenal syndrome.
  * **Creatinine Phosphokinase (CPK):** Cardiac and muscle injury enzyme.
  * **Age, Platelets, High Blood Pressure, and Follow-up Duration (`time`).**

### 2. Clinical Feature Engineering
To capture physiological interactions governing patient survival:
* **Cardiorenal Index:** $\frac{\text{Serum Creatinine}}{\text{Ejection Fraction} / 100}$ (captures acute renal hypoperfusion from low cardiac output).
* **Age-to-EF Ratio:** $\frac{\text{Age}}{\text{Ejection Fraction}}$ (cardiovascular aging stress).
* **Sodium-to-Creatinine Ratio:** $\frac{\text{Serum Sodium}}{\text{Serum Creatinine}}$ (electrolyte-renal balance).

### 3. Class Imbalance Handling and Hyperparameter Optimization
* **SMOTE (Synthetic Minority Over-sampling Technique):** Applied to the training split to synthesize minority mortality samples, completely eliminating test holdout leakage.
* **5-Fold Stratified GridSearchCV:** All models are tuned across parameter grids on **ROC-AUC**:
  * **Random Forest** (`n_estimators`, `max_depth`, `min_samples_leaf`)
  * **XGBoost** (`n_estimators`, `max_depth`, `learning_rate`, `subsample`)
  * **LightGBM** (`n_estimators`, `max_depth`, `learning_rate`, `num_leaves`)
  * **Logistic Regression** (`C`)

---

## 📊 Model Benchmark Results

| Model | 5-Fold CV ROC-AUC | Test Holdout AUROC | Test Accuracy | Test F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | **0.9589** | **0.8627** | **80.0%** | **0.6809** |
| **LightGBM** | **0.9590** | **0.8619** | **84.0%** | **0.7391** |
| **Random Forest** | **0.9452** | **0.8693** | **82.7%** | **0.7347** |
| **Logistic Regression** | **0.9234** | **0.8709** | **78.7%** | **0.6667** |

---

## 📁 Repository Structure

```text
heart-failure-survival/
├── app.py                                       # Interactive Streamlit Web Application
├── heart_failure_survival.ipynb                 # End-to-end Jupyter Notebook (EDA, SMOTE, GridSearch)
├── heart_failure_clinical_records_dataset.csv   # Real UCI dataset (299 patients)
├── heart_failure_bundle.joblib                  # Serialized tuned models and metadata
├── README.md                                    # Project documentation
└── .gitignore                                   # Standard exclusions
```

---

## 🚀 How to Run

### 1. Prerequisites
Install required dependencies:
```bash
pip install streamlit scikit-learn xgboost lightgbm imbalanced-learn plotly pandas numpy joblib
```

### 2. Run the Jupyter Notebook
Open and execute all cells in Jupyter Lab, Notebook, or VS Code:
```bash
jupyter lab heart_failure_survival.ipynb
```

### 3. Launch the Streamlit Web Application
Run the app locally:
```bash
streamlit run app.py
```

### Dashboard Features:
1. **🩺 Bedside Patient Evaluation:**
   * Input patient parameters in a clean 4-column grid.
   * Pre-configured clinical presets (*Low Risk*, *Elderly Moderate Risk*, *High Risk Acute Failure*).
   * Real-time risk probability output, clinical recommendations, and cardiorenal traffic-light metrics.
2. **⚙️ Model Benchmarks and Optimal Parameters:**
   * Side-by-side performance comparison of all 4 tuned models along with their exact GridSearch hyperparameters.
3. **📈 Feature Importance Rankings:**
   * Visual Gini feature importance rankings from the tuned ensemble.
4. **📂 Explore Clinical Cohort Data:**
   * Interactive, scrollable patient records from the UCI cohort.
