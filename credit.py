import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Industry-Oriented Machine Learning System for Credit Risk Assessment", layout="wide")

# ---------------- HERO SECTION ----------------
st.markdown("""
    <div style="
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        padding: 40px;
        border-radius: 10px;
        text-align:center;
        color:white;
        margin-bottom:20px;
    ">
        <h1 style="font-size:50px;">💳 Industry-Oriented Machine Learning System for Credit Risk Assessment</h1>
        <p style="font-size:20px;">Smart AI-Based Loan Risk Prediction</p>
    </div>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">💳 Credit Risk Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">AI-powered Loan Approval System using Machine Learning</p>', unsafe_allow_html=True)
st.divider()

# ---------------- LOAD DATA ----------------
data = fetch_openml(name='credit-g', version=1, as_frame=True)
df = data.frame

df.rename(columns={"class": "Loan_Default"}, inplace=True)
df["Loan_Default"] = df["Loan_Default"].map({"good": 0, "bad": 1})

# Convert categorical
df["checking_status"] = df["checking_status"].map({
    "no checking": 0,
    "<0": 1,
    "0<=X<200": 2,
    ">=200": 3
})

df["credit_history"] = df["credit_history"].map({
    "critical/other existing credit": 0,
    "existing paid": 1,
    "delayed previously": 2
})

# ---------------- FEATURES ----------------
features = ["duration", "credit_amount", "age", "checking_status", "credit_history"]

X = df[features]
y = df["Loan_Default"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🧍 Prediction", "📈 Insights"])

# ---------------- DASHBOARD ----------------
with tab1:
    st.subheader("📊 Model Performance")
    st.info(f"Accuracy: {accuracy*100:.2f}%")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Loan Distribution")
        fig, ax = plt.subplots(figsize=(4,3))
        sns.countplot(x=y, ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt="d", ax=ax)
        st.pyplot(fig)

    if st.checkbox("Show Classification Report"):
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

# ---------------- PREDICTION ----------------
with tab2:
    st.subheader("🏦 Enter Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        duration = st.slider("Loan Duration (months)", 4, 72, 24)
        age = st.slider("Age", 18, 75, 30)

    with col2:
        credit_amount = st.number_input("Credit Amount", 250, 20000, 5000)

    checking_status = st.selectbox(
        "Checking Account Status",
        ["no checking", "<0", "0<=X<200", ">=200"]
    )

    credit_history = st.selectbox(
        "Credit History",
        ["critical/other existing credit", "existing paid", "delayed previously"]
    )

    # Mapping
    checking_map = {
        "no checking": 0,
        "<0": 1,
        "0<=X<200": 2,
        ">=200": 3
    }

    credit_map = {
        "critical/other existing credit": 0,
        "existing paid": 1,
        "delayed previously": 2
    }

    input_df = pd.DataFrame({
        "duration": [duration],
        "credit_amount": [credit_amount],
        "age": [age],
        "checking_status": [checking_map[checking_status]],
        "credit_history": [credit_map[credit_history]]
    })

    input_scaled = scaler.transform(input_df)

    if st.button("🔍 Predict Loan Risk"):
        probability = model.predict_proba(input_scaled)[0][1]

        st.write(f"📈 Risk Score: {probability:.2f}")

        if probability > 0.6:
            st.error("⚠ High Risk: Loan should NOT be approved")
        elif probability > 0.4:
            st.warning("⚠ Medium Risk: Needs manual review")
        else:
            st.success("✅ Low Risk: Safe to approve")

# ---------------- INSIGHTS ----------------
with tab3:
    st.subheader("🔥 Feature Importance")

    importance = model.feature_importances_

    imp_df = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    st.dataframe(imp_df)
