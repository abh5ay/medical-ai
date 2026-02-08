from flask import Flask, render_template, request, jsonify
import pickle
import warnings
import numpy as np
import os
import pickle

if not os.path.exists("model/disease_model.pkl"):
    print("⚙ Training model on server...")
    import train_model   # this will run training file automatically

disease_model = pickle.load(open("model/disease_model.pkl","rb"))
category_model = pickle.load(open("model/category_model.pkl","rb"))
encoders = pickle.load(open("model/encoders.pkl","rb"))

warnings.filterwarnings("ignore")

app = Flask(__name__)

disease_model = pickle.load(open("model/disease_model.pkl","rb"))
category_model = pickle.load(open("model/category_model.pkl","rb"))
encoders = pickle.load(open("model/encoders.pkl","rb"))

print("Medical AI system loaded")

@app.route("/")
def home():
    return render_template("index.html")

def safe_encode(column, value):
    try:
        return encoders[column].transform([value])[0]
    except:
        raise ValueError(f"Invalid {column}")

def get_risk(age, bp, chol, breath):
    score = 0

    if age > 55:
        score += 2
    if bp == "High":
        score += 2
    if chol == "High":
        score += 2
    if breath == "Yes":
        score += 3

    if score >= 5:
        return "High Risk"
    elif score >= 3:
        return "Medium Risk"
    else:
        return "Low Risk"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        fever = request.form["Fever"]
        cough = request.form["Cough"]
        fatigue = request.form["Fatigue"]
        breath = request.form["Difficulty Breathing"]
        age = int(request.form["Age"])
        gender = request.form["Gender"]
        bp = request.form["Blood Pressure"]
        chol = request.form["Cholesterol"]

        if age < 1 or age > 120:
            return jsonify({"error": "Invalid age"})

        input_data = [
            safe_encode("Fever", fever),
            safe_encode("Cough", cough),
            safe_encode("Fatigue", fatigue),
            safe_encode("Difficulty Breathing", breath),
            age,
            safe_encode("Gender", gender),
            safe_encode("Blood Pressure", bp),
            safe_encode("Cholesterol Level", chol),
            safe_encode("Outcome Variable", "Positive")
        ]

        cat_pred = category_model.predict([input_data])[0]
        category = encoders["Category"].inverse_transform([cat_pred])[0]

        probs = disease_model.predict_proba([input_data])[0]
        top3_idx = np.argsort(probs)[-3:][::-1]

        top3 = []
        for i in top3_idx:
            disease_name = encoders["Disease"].inverse_transform([i])[0]
            conf = round(probs[i] * 100, 2)
            top3.append({
                "disease": disease_name,
                "confidence": conf
            })

        risk = get_risk(age, bp, chol, breath)

        return jsonify({
            "category": category,
            "risk": risk,
            "top3": top3
        })

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": "Prediction failed"})

if __name__ == "__main__":
    app.run()

