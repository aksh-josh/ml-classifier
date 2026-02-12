
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    f1 = float(request.form["f1"])
    f2 = float(request.form["f2"])
    f3 = float(request.form["f3"])
    f4 = float(request.form["f4"])

    data = np.array([[f1,f2,f3,f4]])
    result = model.predict(data)[0]

    return f"Prediction Class: {result}"

if __name__ == "__main__":
    app.run(debug=True)
