import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, classification_report, confusion_matrix

# Step 1: Load the dataset
# Assuming columns: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigree, Age, Outcome
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome']
df = pd.read_csv(url, names=columns)

# Step 2: Handle biologically impossible zeros
# Pregnancies can be 0, but Glucose, BP, SkinThickness, Insulin, and BMI cannot.
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[zero_cols] = df[zero_cols].replace(0, np.nan)

# Impute missing values with the median of each column
imputer = SimpleImputer(strategy='median')
df[zero_cols] = imputer.fit_transform(df[zero_cols])

# Separate features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split into training and testing sets (stratified to maintain class balance)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the classifier
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

# Focus strictly on Recall
recall = recall_score(y_test, y_pred)
print(f"Recall Score: {recall:.2f}\n")
print("Full Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix to visualize False Negatives
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Focus on Bottom-Left: False Negatives)")
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# Extract feature importances
importances = model.feature_importances_
feature_names = X.columns

# Sort and plot
indices = np.argsort(importances)
plt.figure(figsize=(8, 6))
plt.title('Feature Importances for Diabetes Risk')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()