
# HR Analytics – Employee Attrition

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load Dataset
df = pd.read_csv(r"C:\Users\varra\OneDrive\Desktop\my project\WA_Fn-UseC_-HR-Employee-Attrition.csv")

# 2. Drop Irrelevant Columns
df.drop(['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber'], axis=1, inplace=True)

# 3. Remove Outliers using IQR
def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)]

for col in ['MonthlyIncome', 'YearsAtCompany', 'TotalWorkingYears']:
    df = remove_outliers(df, col)

# 4. EDA
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Attrition')
plt.title('Attrition Distribution')
plt.show()

eda_cols = ['Gender', 'Department', 'JobRole', 'MonthlyIncome', 'YearsAtCompany']
for col in eda_cols:
    plt.figure(figsize=(8, 4))
    if df[col].dtype == 'object':
        sns.countplot(data=df, x=col, hue='Attrition')
    else:
        sns.boxplot(data=df, x='Attrition', y=col)
    plt.title(f'Attrition by {col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 5. Label Encode Categorical Variables
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# 6. Define X and y
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# 7. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 8. Model Training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 9. Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 10. Cross Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
print("CV Scores:", scores)
print("Mean CV Accuracy:", scores.mean())

# 11. Hyperparameter Tuning
params = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid=params, cv=cv, scoring='accuracy')
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best CV Score:", grid.best_score_)
