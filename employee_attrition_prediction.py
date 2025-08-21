import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib

# ========================
# 1. Load Dataset
# ========================
CSV_FILE = "employee_attrition_dataset.csv"
df = pd.read_csv("employee_attrition_data.csv")


print(f"Loaded: {CSV_FILE} — rows: {df.shape[0]} cols: {df.shape[1]}\n")
print("=== Head ===")
print(df.head(), "\n")
print("=== Info ===")
print(df.info(), "\n")
print("=== Missing values ===")
print(df.isnull().sum(), "\n")
print("=== Numeric Summary ===")
print(df.describe(), "\n")

# ========================
# 2. Create EDA Charts Folder
# ========================
eda_dir = "eda_charts"
os.makedirs(eda_dir, exist_ok=True)

def save_and_show(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, filename), bbox_inches="tight")
    plt.show()

# ========================
# 3. EDA + Insights
# ========================

# Attrition distribution
plt.figure(figsize=(5,4))
sns.countplot(x="Attrition", data=df)
plt.title("Attrition distribution")
save_and_show("attrition_distribution.png")
attrition_rate = df["Attrition"].value_counts(normalize=True)["Yes"] * 100
print(f"Insight: {attrition_rate:.1f}% of employees have left, while {100-attrition_rate:.1f}% stayed.\n")

# Attrition by Department
plt.figure(figsize=(6,4))
sns.countplot(x="Department", hue="Attrition", data=df)
plt.title("Attrition by Department")
plt.xticks(rotation=45)
save_and_show("attrition_by_department.png")
dept_attr = df.groupby("Department")["Attrition"].value_counts(normalize=True).unstack()["Yes"]*100
print(f"Insight: {dept_attr.idxmax()} department shows the highest attrition rate at {dept_attr.max():.1f}%.\n")

# Attrition by Education Level
plt.figure(figsize=(6,4))
sns.countplot(x="Education", hue="Attrition", data=df)
plt.title("Attrition by Education Level")
plt.xticks(rotation=45)
save_and_show("attrition_by_education.png")
edu_attr = df.groupby("Education")["Attrition"].value_counts(normalize=True).unstack()["Yes"]*100
print(f"Insight: Education level '{edu_attr.idxmax()}' has the highest attrition rate at {edu_attr.max():.1f}%.\n")

# Monthly Income vs Attrition
plt.figure(figsize=(6,4))
sns.boxplot(x="Attrition", y="MonthlyIncome", data=df)
plt.title("Monthly Income vs Attrition")
save_and_show("income_vs_attrition.png")
avg_income_left = df[df["Attrition"]=="Yes"]["MonthlyIncome"].mean()
avg_income_stayed = df[df["Attrition"]=="No"]["MonthlyIncome"].mean()
print(f"Insight: Employees who left had an average income of ₹{avg_income_left:,.0f}, compared to ₹{avg_income_stayed:,.0f} for those who stayed.\n")

# Job Satisfaction vs Attrition
plt.figure(figsize=(6,4))
sns.countplot(x="JobSatisfaction", hue="Attrition", data=df)
plt.title("Job Satisfaction vs Attrition")
save_and_show("jobsatisfaction_vs_attrition.png")
low_sat_attr = df[df["JobSatisfaction"] <= 2]["Attrition"].value_counts(normalize=True)["Yes"]*100
print(f"Insight: Employees with job satisfaction ≤ 2 have an attrition rate of {low_sat_attr:.1f}%.\n")

# Age distribution by Attrition
plt.figure(figsize=(6,4))
sns.histplot(data=df, x="Age", hue="Attrition", bins=15, kde=True)
plt.title("Age Distribution by Attrition")
save_and_show("age_distribution_by_attrition.png")
young_attr = df[df["Age"] < 35]["Attrition"].value_counts(normalize=True)["Yes"]*100
print(f"Insight: Employees under 35 have an attrition rate of {young_attr:.1f}%.\n")

# Correlation Heatmap
plt.figure(figsize=(10,6))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
save_and_show("correlation_heatmap.png")
print("Insight: Monthly Income and Total Working Years are highly correlated, suggesting experienced employees earn more.\n")

# ========================
# 4. Data Preprocessing
# ========================
target_map = {"No": 0, "Yes": 1}
print(f"Target mapping: {target_map}\n")
df["Attrition_enc"] = df["Attrition"].map(target_map)

categorical_cols = df.select_dtypes(include=["object"]).drop(columns=["Attrition"]).columns.tolist()
print(f"Categorical columns to encode: {categorical_cols}\n")

# Encode categorical features
df_encoded = df.copy()
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])

X = df_encoded.drop(columns=["Attrition", "Attrition_enc"])
y = df_encoded["Attrition_enc"]
print(f"Feature matrix shape: {X.shape}")
print("Target distribution:\n", y.value_counts(), "\n")

# Balance with SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
print("After SMOTE:\n", y_res.value_counts())

# ========================
# 5. Train-Test Split
# ========================
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
print(f"Train shape: {X_train.shape} Test shape: {X_test.shape}")

# ========================
# 6. Model Training
# ========================
param_grid = {
    "n_estimators": [100, 150],
    "max_depth": [8, 12, None],
    "min_samples_split": [2, 5]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
print(f"Best params: {grid_search.best_params_}\n")

# ========================
# 7. Evaluation
# ========================
y_pred = best_rf.predict(X_test)
acc = (y_pred == y_test).mean()
print(f"Accuracy: {acc:.4f}\n")
print("Classification report:\n", classification_report(y_test, y_pred))
roc_auc = roc_auc_score(y_test, best_rf.predict_proba(X_test)[:,1])
print(f"ROC-AUC: {roc_auc:.4f}\n")

# ========================
# 8. Save Model & Encoder
# ========================
joblib.dump(best_rf, "employee_attrition_rf_model.joblib")
joblib.dump(LabelEncoder().fit(df["Attrition"]), "label_encoder_target.joblib")
print("Model & encoder saved.\n")

# ========================
# 9. Model Charts (Confusion Matrix & Feature Importance)
# ========================
model_dir = "model_charts"
os.makedirs(model_dir, exist_ok=True)

# Confusion Matrix
disp = ConfusionMatrixDisplay.from_estimator(best_rf, X_test, y_test, display_labels=["No", "Yes"], cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(model_dir, "confusion_matrix.png"), bbox_inches="tight")
plt.show()

# Feature Importance
importances = best_rf.feature_importances_
feat_names = X.columns
feat_imp_df = pd.DataFrame({"Feature": feat_names, "Importance": importances}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(x="Importance", y="Feature", data=feat_imp_df, hue="Feature", dodge=False, palette="viridis", legend=False)
plt.title("Feature Importance")
plt.savefig(os.path.join(model_dir, "feature_importance.png"), bbox_inches="tight")
plt.show()



