import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_pickle(filename):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Required artifact '{filename}' not found. Run train_model.py first."
        )
    with open(path, "rb") as file:
        return pickle.load(file)


preprocessor = load_pickle("preprocessor.pkl")
model = load_pickle("voting_model.pkl")

NUMERIC_LIMITS = {
    "age": (0.1, 120),
    "bmi": (10.0, 100.0),
    "HbA1c_level": (3.0, 15.0),
    "blood_glucose_level": (50.0, 400.0),
}

BINARY_FIELDS = ["hypertension", "heart_disease"]

CATEGORICAL_OPTIONS = {
    "gender": ["Female", "Male", "Other"],
    "smoking_history": ["never", "No Info", "current", "former", "ever", "not current"],
}


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    form_data = {}

    if request.method == "POST":
        try:
            record = {}

            for field, (minimum, maximum) in NUMERIC_LIMITS.items():
                raw_value = request.form.get(field, "").strip()
                form_data[field] = raw_value

                if not raw_value:
                    raise ValueError(f"Please enter a value for {field}.")

                value = float(raw_value)
                if not np.isfinite(value):
                    raise ValueError(f"{field} must be a valid number.")
                if not minimum <= value <= maximum:
                    raise ValueError(f"{field} must be between {minimum} and {maximum}.")

                record[field] = value

            for field in BINARY_FIELDS:
                raw_value = request.form.get(field, "").strip()
                form_data[field] = raw_value

                if raw_value not in ("0", "1"):
                    raise ValueError(f"Please select a valid option for {field}.")

                record[field] = int(raw_value)

            for field, allowed in CATEGORICAL_OPTIONS.items():
                raw_value = request.form.get(field, "").strip()
                form_data[field] = raw_value

                if raw_value not in allowed:
                    raise ValueError(f"Please select a valid option for {field}.")

                record[field] = raw_value

            input_df = pd.DataFrame([record])
            input_processed = preprocessor.transform(input_df)

            result = int(model.predict(input_processed)[0])
            probability = float(model.predict_proba(input_processed)[0][1]) * 100

            if result == 1:
                prediction = {
                    "label": "High Risk (Positive Prediction)",
                    "probability": f"{probability:.1f}%",
                    "message": (
                        "The Soft Voting Ensemble predicts a high likelihood of diabetes. "
                        "A clinical follow-up test is recommended."
                    ),
                    "status": "positive",
                }
            else:
                prediction = {
                    "label": "Low Risk (Negative Prediction)",
                    "probability": f"{probability:.1f}%",
                    "message": (
                        "The Soft Voting Ensemble predicts a low likelihood of diabetes "
                        "based on the provided indicators."
                    ),
                    "status": "negative",
                }

        except ValueError as exc:
            error = str(exc)
        except Exception:
            app.logger.exception("Prediction failed")
            error = "The prediction could not be completed. Please check the server logs."

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        form_data=form_data,
        categorical_options=CATEGORICAL_OPTIONS,
    )


if __name__ == "__main__":
    app.run(debug=True)