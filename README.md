# Customer Churn Prediction & Risk Analytics Dashboard

An end-to-end Machine Learning web application built with **Streamlit**, **Scikit-Learn**, and **Plotly** to analyze, predict, and visualize customer churn risk in real time.

---

## 📌 Project Overview

Customer churn is a critical metric for subscription-based businesses. This interactive dashboard enables business analysts and retention teams to:
- **Analyze Risk Drivers:** Inspect global feature importance to understand what causes customers to churn.
- **Predict Real-Time Churn Risk:** Input live customer attributes (age, tenure, support calls, spend) to calculate individual churn probability scores.
- **Formulate Retention Strategies:** Instantly classify risk levels (High vs. Low) and deliver actionable recommendations for retention offers.

---

## 🛠️ Tech Stack & Dependencies

- **Frontend / Dashboard Framework:** Streamlit
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning & Preprocessing:** Scikit-Learn, Joblib
- **Data Visualization:** Plotly Express / Graph Objects

---

## 📁 Repository Structure

```text
.
├── customer.py          # Main Streamlit web application
├── model.pkl            # Trained Scikit-Learn Classification Model (Random Forest / XGBoost)
├── scaler.pkl           # Fitted ColumnTransformer / StandardScaler / Preprocessor
├── requirements.txt     # Python package requirements
└── README.md            # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/customer-churn-dashboard.git
cd customer-churn-dashboard
```

### 2. Create and Activate a Virtual Environment
```bash
# On Linux / macOS
python3 -m venv myenv
source myenv/bin/activate

# On Windows
python -m venv myenv
myenv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage Instructions

Launch the Streamlit dashboard using:
```bash
streamlit run customer.py
```

Navigate to `http://localhost:8501` in your browser.

---

## 📊 Modules & Features

### 1. 🧠 Feature Importance Drivers
- Visually plots top features driving churn (e.g., *Support Calls*, *Payment Delay*, *Contract Length*).
- Dynamically extracts feature names directly from the fitted Scikit-Learn `ColumnTransformer` / model pipeline.

### 2. 🔮 Live Churn Risk Predictor
- Interactive input form featuring numerical controls (with custom increment/decrement steps) and categorical selectors.
- Computes risk percentage scores in real time:
  - **🚨 HIGH RISK (> 50%):** Prompts retention discount recommendations.
  - **✅ LOW RISK (≤ 50%):** Flags account as stable.

---

## 📝 License
This project is open-source and available under the [MIT License](LICENSE).