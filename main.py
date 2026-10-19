import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    classification_report
) 


# ============================================================
# 1. LOAD DATASET
# ============================================================

# Your CSV is TAB-SEPARATED, so we use sep="\t"
df = pd.read_csv("https://github.com/npradaschnor/Pima-Indians-Diabetes-Dataset/blob/master/diabetes.csv")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

print("\n========================================")
print("DIABETES ONSET PREDICTION")
print("========================================")

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nOutcome distribution:")
print(df["Outcome"].value_counts())


# ============================================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

feature_names = list(X.columns)

print("\nNumber of features:", len(feature_names))
print("Features:", feature_names)


# ============================================================
# 3. HANDLE BIOLOGICALLY IMPOSSIBLE ZERO VALUES
# ============================================================

# In the Pima dataset, zero is not biologically meaningful
# for these measurements, so we treat zero as missing data.

zero_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

X[zero_columns] = X[zero_columns].replace(0, np.nan)

print("\nImpossible zero values converted to missing values.")
print("Columns affected:", zero_columns)


# ============================================================
# 4. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 5. HANDLE MISSING VALUES
# ============================================================

# Median imputation is used because it is less affected
# by extreme values than mean imputation.

imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

print("\nMissing values handled using median imputation.")


# ============================================================
# 6. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Feature scaling completed.")


# ============================================================
# 7. TRAIN RANDOM FOREST CLASSIFIER
# ============================================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

print("\nRandom Forest model trained successfully!")


# ============================================================
# 8. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 9. MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print("\n========================================")
print("MODEL EVALUATION")
print("========================================")

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"Precision : {precision:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================

importance = model.feature_importances_

print("\n========================================")
print("FEATURE IMPORTANCE")
print("========================================")

for name, value in zip(feature_names, importance):
    print(f"{name:30s}: {value:.4f}")


# ============================================================
# 11. CREATE FEATURE IMPORTANCE CHART
# ============================================================

# Sort features from least important to most important
sorted_indices = np.argsort(importance)

sorted_features = [
    feature_names[i]
    for i in sorted_indices
]

sorted_importance = importance[sorted_indices]

plt.figure(figsize=(10, 6))

plt.barh(
    sorted_features,
    sorted_importance
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Diabetes Prediction - Feature Importance")

plt.tight_layout()

plt.savefig(
    "feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nFeature importance chart created:")
print("feature_importance.png")


# ============================================================
# 12. SAVE MODEL AND PREPROCESSING OBJECTS
# ============================================================

with open("diabetes_model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("imputer.pkl", "wb") as file:
    pickle.dump(imputer, file)

with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

# ============================================================
# 13. FINAL MESSAGE
# ============================================================

print("\n========================================")
print("TRAINING COMPLETED SUCCESSFULLY!")
print("========================================")

print("\nFiles created:")

print("1. diabetes_model.pkl")
print("2. imputer.pkl")
print("3. scaler.pkl")
print("4. feature_importance.png")

print("\nYou can now run the Flask application using:")
print("python app.py")