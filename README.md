# Diabetes Prediction System

A Flask web application that uses a trained Random Forest model to predict the diabetes outcome class from eight patient measurements.

> This project is for educational and demonstration purposes only. It is not a medical diagnostic tool.

## Features

- Responsive web interface for entering health measurements
- Server-side input validation and form value retention
- Zero-value handling for measurements that may represent missing data
- Fitted imputer and feature scaler preprocessing
- Positive and negative prediction result cards
- Clear medical disclaimer

## Input Fields

| Field | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Glucose level |
| BloodPressure | Blood pressure |
| SkinThickness | Skin thickness |
| Insulin | Insulin level |
| BMI | Body mass index |
| DiabetesPedigree | Diabetes pedigree function |
| Age | Age in years |

The application also accepts the Pima Indian Diabetes dataset file, `diabetes.csv`, as the project dataset.

## Project Structure

```text
.
|-- app.py
|-- diabetes.csv
|-- static/
|   `-- style.css
|-- templates/
|   `-- index.html
`-- README.md
```

## Requirements

- Python 3.9 or newer
- Flask
- NumPy
- scikit-learn
- A compatible trained model and preprocessing artifacts:
  - `diabetes_model.pkl`
  - `imputer.pkl`
  - `scaler.pkl`

The `.pkl` files must be placed in the project root next to `app.py`.

## Setup

From the project root, create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the Python packages:

```powershell
python -m pip install -r requirements.txt
```

Place `diabetes_model.pkl`, `imputer.pkl`, and `scaler.pkl` in the project root.

## Run the Application

```powershell
python app.py
```

Open `http://127.0.0.1:5000/` in a browser.

## How Prediction Works

1. The form sends the eight measurements to the `/` route.
2. Values are checked against the application's validation limits.
3. Zero is rejected for Glucose, BloodPressure, SkinThickness, Insulin, BMI, Diabetes Pedigree, and Age.
4. The fitted imputer fills missing values.
5. The fitted scaler transforms the values.
6. The Random Forest model returns the predicted class.

## Medical Disclaimer

The result is a model-based educational prediction. It does not confirm or rule out diabetes and is not a substitute for professional medical advice, diagnosis, or testing. Consult a qualified healthcare professional for medical concerns.
