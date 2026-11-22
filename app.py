
from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# --------------------------------------------------
# 1. LOAD TRAINED MODEL AND PREPROCESSING OBJECTS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_pickle(filename):
    path = os.path.join(BASE_DIR, filename)

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Required model artifact '{filename}' was not found in "
            f"the project directory."
        )

    with open(path, "rb") as file:
        return pickle.load(file)

model = load_pickle("diabetes_model.pkl")
imputer = load_pickle("imputer.pkl")
scaler = load_pickle("scaler.pkl")


# --------------------------------------------------
# 2. FEATURE NAMES AND VALIDATION LIMITS
# --------------------------------------------------

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigree",
    "Age"
]

# These are broad input-validation limits, not
# clinical diagnostic thresholds.
LIMITS = {
    "Pregnancies": (0, 30),
    "Glucose": (1, 500),
    "BloodPressure": (1, 250),
    "SkinThickness": (1, 150),
    "Insulin": (1, 3000),
    "BMI": (0.1, 100),
    "DiabetesPedigree": (0.001, 5),
    "Age": (1, 120)
}


# --------------------------------------------------
# 3. HOME PAGE
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    error = None
    form_data = {}

    if request.method == "POST":

        try:
            values = []

            # Read and validate user inputs
            for feature in FEATURES:

                raw_value = request.form.get(feature, "").strip()

                if not raw_value:
                    raise ValueError(
                        f"Please enter a value for {feature}."
                    )

                value = float(raw_value)

                if not np.isfinite(value):
                    raise ValueError(
                        f"{feature} must be a valid number."
                    )

                minimum, maximum = LIMITS[feature]

                if not minimum <= value <= maximum:
                    raise ValueError(
                        f"{feature} must be between "
                        f"{minimum} and {maximum}."
                    )

                # Store values to repopulate the form
                form_data[feature] = raw_value
                values.append(value)

            # Convert input to a 2D NumPy array
            input_data = np.array(values).reshape(1, -1)

            # Match training preprocessing:
            # 1. Replace selected zeros with missing values
            zero_columns = [
                "Glucose",
                "BloodPressure",
                "SkinThickness",
                "Insulin",
                "BMI"
            ]

            for column in zero_columns:
                index = FEATURES.index(column)

                if input_data[0, index] == 0:
                    input_data[0, index] = np.nan

            # 2. Apply the fitted imputer
            input_data = imputer.transform(input_data)

            # 3. Apply the fitted scaler
            input_data = scaler.transform(input_data)

            # 4. Predict the outcome
            result = int(model.predict(input_data)[0])

            if result == 1:
                prediction = {
                    "label": "Positive Prediction",
                    "message": (
                        "The model predicts the positive class "
                        "for diabetes in this record."
                    ),
                    "status": "positive"
                }
            else:
                prediction = {
                    "label": "Negative Prediction",
                    "message": (
                        "The model predicts the negative class "
                        "for diabetes in this record."
                    ),
                    "status": "negative"
                }

        except ValueError as exc:
            error = str(exc)

        except Exception:
            app.logger.exception("Prediction failed")

            error = (
                "The prediction could not be completed. "
                "Please check the server configuration."
            )

    return render_template(
        "index.html",
        features=FEATURES,
        prediction=prediction,
        error=error,
        form_data=form_data
    )


# --------------------------------------------------
# 4. RUN FLASK APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)