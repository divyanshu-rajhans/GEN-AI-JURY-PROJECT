# Diabetes Onset Prediction

A supervised-learning example using the Pima Indians Diabetes dataset to predict the binary `Outcome` label. The project demonstrates data inspection, missing-value handling, model training, recall-focused evaluation, and feature-importance visualization.

> **Health and ethics notice:** This is an educational demonstration, not a diagnostic or clinical decision-making tool. A false negative means a person who has diabetes is predicted as not having it. Although recall is emphasized to reduce missed cases, a model prediction must not replace clinical assessment. The dataset is limited and may not represent other populations; model performance and errors can differ between groups.

## Project files

| File | Purpose |
| --- | --- |
| [`prima-indians-diabetes-database.ipynb`](./prima-indians-diabetes-database.ipynb) | Jupyter notebook for exploratory analysis and model experimentation. Its first data-loading cells expect the dataset to be available under Kaggle's `/kaggle/input` directory. |
| [`ta_2.py`](./ta_2.py) | Standalone recall-focused workflow. Downloads the dataset from a raw GitHub URL, trains a random forest, prints metrics, and displays a confusion-matrix heatmap and feature-importance chart. |
| [`main.py`](./main.py) | Alternate workflow that reports accuracy, recall, precision, and a classification report; saves a feature-importance chart and serialized model/preprocessing objects. Its current dataset URL points to a GitHub web page rather than the raw CSV, so it may fail to load the data until that URL is corrected. |

## Dataset and preparation

The target column is `Outcome` (`1` = diabetes, `0` = no diabetes). The predictors are pregnancies, glucose, blood pressure, skin thickness, insulin, BMI, diabetes pedigree, and age.

In `ta_2.py`, zero values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` are treated as missing and replaced with each column's median. Zero pregnancies are retained because they are valid. The data is split into stratified 80% training and 20% test sets with `random_state=42`.

## Run the standalone workflow

Use Python 3 with the required packages installed:

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn
python ta_2.py
```

An internet connection is needed to download the dataset. The script prints recall and a classification report, then opens plots for the confusion matrix and random-forest feature importances. In the confusion matrix, rows are actual labels and columns are predicted labels; the false negatives are actual-positive cases predicted negative.

`main.py` is an alternate script and currently uses a non-raw GitHub URL for `pd.read_csv`. It also saves `feature_importance.png`, `diabetes_model.pkl`, `imputer.pkl`, and `scaler.pkl` if its data-loading step succeeds. The script mentions a Flask application, but this repository does not currently include `app.py`.

## Evaluation and interpretation

Recall is the proportion of actual positive cases identified by the model (`true positives / (true positives + false negatives)`). It is highlighted because missed positive cases can be consequential in a screening context. Recall alone is not sufficient: review precision, the confusion matrix, and other metrics to understand the trade-off, especially the number of false positives.

The random forest's feature-importance values summarize how the trained model used features to make splits. They are not evidence that a feature causes diabetes, nor are they a medical risk ranking. Treat them as exploratory model interpretation.

## Supervised-learning lifecycle

1. Load and inspect labeled examples and their features.
2. Separate predictors from the `Outcome` label.
3. Mark biologically implausible zero measurements as missing and impute them.
4. Split the data into stratified training and test sets.
5. Fit a classifier on the training data.
6. Evaluate on held-out data, emphasizing recall and reviewing false negatives.
7. Visualize feature importance and consider limitations, ethics, and appropriate use before drawing conclusions.
