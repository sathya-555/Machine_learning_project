# 💳 Credit Risk Assessment System

## 📌 Project Overview

This project is an industry-oriented Machine Learning system designed to evaluate credit risk and predict the probability of loan default. It helps banks and financial institutions make accurate, data-driven decisions while approving loans.

---

## ❗ Problem Statement

Banks and financial institutions face significant risk when approving loans without proper risk evaluation. Traditional manual credit assessment methods are:

* Time-consuming
* Subjective
* Prone to human error

This often leads to approving loans for high-risk customers, resulting in financial losses.

---

## 🎯 Objective

* Build a machine learning model to predict loan default risk
* Generate a risk score for each customer
* Automate the loan approval decision process
* Improve accuracy and reduce human bias

---

## 🛠️ Tools & Technologies

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Streamlit

---

## 📊 Dataset

* German Credit Dataset (real-world financial dataset)
* Contains customer financial information such as:

  * Loan duration
  * Credit amount
  * Age
  * Checking account status
  * Credit history

---

## ⚙️ Project Workflow

### 1. Data Ingestion

* Loaded dataset from OpenML

### 2. Data Preprocessing

* Converted categorical values into numerical format
* Handled string values
* Normalized data using StandardScaler

### 3. Feature Engineering

* Selected key financial indicators:

  * Duration
  * Credit amount
  * Age
  * Checking account status
  * Credit history

### 4. Exploratory Data Analysis (EDA)

* Visualized loan default distribution
* Identified patterns in the dataset

### 5. Model Training

* Used Random Forest Classifier
* Split data into training and testing sets

### 6. Model Evaluation

* Accuracy Score
* Precision, Recall, F1-score
* Confusion Matrix

### 7. Risk Score Generation

* Used probability prediction (`predict_proba`)
* Generated risk score between 0 and 1

### 8. Visualization

* Loan distribution graph
* Confusion matrix
* Feature importance analysis

### 9. User Interface

* Built using Streamlit
* Manual input system for real-time prediction
* Organized UI using tabs and clean layout

---

## 🔍 Features

* Real-world dataset integration
* Machine Learning-based prediction system
* Risk score generation
* Low / Medium / High risk classification
* Interactive web interface
* Dashboard with visual insights

---

## 📈 Sample Output

**Example 1:**

* Risk Score: 0.72
* ⚠ High Risk → Loan should NOT be approved

**Example 2:**

* Risk Score: 0.15
* ✅ Low Risk → Safe to approve

---

## 🚀 How to Run the Project

### Step 1: Install Dependencies

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn
```

### Step 2: Run the Application

```bash
streamlit run credit.py
```

### Step 3: Open in Browser

```
http://localhost:8501
```

---

## 💡 Real-World Impact

* Helps banks reduce financial risk
* Automates loan approval decisions
* Improves prediction accuracy using data
* Saves time and reduces manual effort

---

## 🎯 Future Improvements

* Add more financial features for better prediction
* Use advanced algorithms like XGBoost or Deep Learning
* Deploy as a cloud-based web application
* Integrate real-time financial APIs

  Deployment link: https://credit-risk-assessment-project.streamlit.app/

---

This project demonstrates how Machine Learning can be effectively applied in the financial domain to solve real-world problems like credit risk assessment. It showcases the complete pipeline from data preprocessing to model deployment with a user-friendly interface.
