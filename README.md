# HR-Analytics
# HR Analytics – Employee Attrition Prediction 🚀

This project explores the factors influencing employee attrition using machine learning and visualization techniques. By identifying trends in the dataset, we aim to assist HR departments in understanding why employees leave and what can be done to retain them.

---

## 🔍 Objective

Analyze the HR dataset to:
- Identify key factors that lead to employee attrition
- Build a predictive model using Logistic Regression
- Create data visualizations and a Power BI dashboard

---

## 📁 Dataset

- File: `"C:\Users\varra\OneDrive\Desktop\my project\WA_Fn-UseC_-HR-Employee-Attrition.csv"`
- Source: IBM HR Analytics Employee Attrition & Performance dataset (available on Kaggle)

---

## 🧹 Data Cleaning

Removed the following non-informative or constant columns:
- `EmployeeCount`
- `Over18`
- `StandardHours`
- `EmployeeNumber`

---

## 📊 Exploratory Data Analysis (EDA)

Analyzed attrition across different features:
- **Gender**
- **Department**
- **Job Role**
- **Monthly Income**
- **Years at Company**

Visualizations include:
- Count plots for categorical variables
- Box plots for numerical variables vs attrition

---

## 🧠 Model Building

**Model Used:** Logistic Regression

Steps:
1. Encoded categorical features using Label Encoding
2. Train-Test Split (80-20)
3. Built a baseline logistic regression model
4. Evaluated using Accuracy, Confusion Matrix, and Classification Report

---

## 🔁 Cross Validation

Used **StratifiedKFold** to ensure consistent distribution of attrition cases across folds.

---

## 🛠️ Hyperparameter Tuning

Used **GridSearchCV** to tune:
- `C`: [0.01, 0.1, 1, 10]
- `penalty`: ['l1', 'l2']
- `solver`: 'liblinear'

---

## 📦 Project Structure

