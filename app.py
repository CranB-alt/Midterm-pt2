from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model + features
model = pickle.load(open("model.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Collect form inputs
    education_num = int(request.form['education-num'])
    hours_per_week = int(request.form['hours-per-week'])
    occupation = request.form['occupation']
    age = int(request.form['age'])

    # Build dataframe for model
    input_df = pd.DataFrame([[
        education_num,
        hours_per_week,
        occupation,
        age
    ]], columns=features)

    # Predict
    prediction = model.predict(input_df)[0]

    result = "Income > 50K" if prediction == 1 else "Income <= 50K"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)

