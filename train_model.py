import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "diabetes_prediction_dataset.csv")

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

print("Dataset Shape:", df.shape)
print("\nTarget Distribution:\n", df["diabetes"].value_counts())

X = df.drop("diabetes", axis=1)
y = df["diabetes"]

numeric_features = [
    "age",
    "hypertension",
    "heart_disease",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
]
categorical_features = ["gender", "smoking_history"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_weight = neg_count / pos_count

lr_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42,
)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    scale_pos_weight=scale_weight,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1,
)

voting_model = VotingClassifier(
    estimators=[
        ("lr", lr_model),
        ("rf", rf_model),
        ("xgb", xgb_model),
    ],
    voting="soft",
    n_jobs=-1,
)

models = {
    "Logistic Regression": lr_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model,
    "Soft Voting Ensemble": voting_model,
}

results = []

print("\n" + "=" * 60)
print("TRAINING & TESTING MODELS (FOCUS: RECALL)")
print("=" * 60)

for name, clf in models.items():
    print(f"\nTraining {name}...")
    clf.fit(X_train_processed, y_train)
    y_pred = clf.predict(X_test_processed)

    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append(
        {
            "Model": name,
            "Recall": round(rec, 4),
            "Precision": round(prec, 4),
            "Accuracy": round(acc, 4),
            "F1-Score": round(f1, 4),
        }
    )

    if name == "Soft Voting Ensemble":
        print("\n--- Soft Voting Ensemble Detailed Test Report ---")
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("\nClassification Report:\n", classification_report(y_test, y_pred))

results_df = pd.DataFrame(results)
print("\n" + "=" * 60)
print("MODEL COMPARISON TABLE (TEST SET)")
print("=" * 60)
print(results_df.to_string(index=False))

# 5-Fold Stratified Cross-Validation on Training Set
print("\n" + "=" * 60)
print("5-FOLD STRATIFIED CROSS-VALIDATION (SOFT VOTING ENSEMBLE)")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_recall = cross_val_score(
    voting_model, X_train_processed, y_train, cv=cv, scoring="recall", n_jobs=-1
)

print("Fold Recall Scores:", np.round(cv_recall, 4))
print(f"Mean CV Recall:     {cv_recall.mean():.4f} (+/- {cv_recall.std():.4f})")

# Feature Importance Chart (Averaged from Random Forest & XGBoost)
feature_names = preprocessor.get_feature_names_out()
clean_feature_names = [
    f.replace("num__", "").replace("cat__", "") for f in feature_names
]

avg_importance = (rf_model.feature_importances_ + xgb_model.feature_importances_) / 2
sorted_idx = np.argsort(avg_importance)

plt.figure(figsize=(10, 7))
plt.barh(
    np.array(clean_feature_names)[sorted_idx],
    avg_importance[sorted_idx],
    color="steelblue",
)
plt.xlabel("Average Relative Importance (Random Forest + XGBoost)")
plt.ylabel("Clinical & Demographic Features")
plt.title("Diabetes Onset Prediction - Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "feature_importance.png"), dpi=300, bbox_inches="tight")
plt.close()

with open(os.path.join(BASE_DIR, "preprocessor.pkl"), "wb") as f:
    pickle.dump(preprocessor, f)

with open(os.path.join(BASE_DIR, "voting_model.pkl"), "wb") as f:
    pickle.dump(voting_model, f)

print("\nFiles created successfully:")
print("1. preprocessor.pkl")
print("2. voting_model.pkl")
print("3. feature_importance.png")