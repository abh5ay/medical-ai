import pickle
import random
import pandas as pd
from sklearn.metrics import accuracy_score

model = pickle.load(open("model.pkl","rb"))
encoders = pickle.load(open("encoders.pkl","rb"))

print("\nModel testing started\n")

fever_vals = ["Yes","No"]
cough_vals = ["Yes","No"]
fatigue_vals = ["Yes","No"]
breath_vals = ["Yes","No"]
gender_vals = ["Male","Female"]
bp_vals = ["Low","Normal","High"]
chol_vals = ["Low","Normal","High"]

total = 0
errors = 0

print("Running stress test on random inputs...\n")

for i in range(1000):
    try:
        f = random.choice(fever_vals)
        c = random.choice(cough_vals)
        fa = random.choice(fatigue_vals)
        b = random.choice(breath_vals)
        g = random.choice(gender_vals)
        bp = random.choice(bp_vals)
        ch = random.choice(chol_vals)
        age = random.randint(1,90)

        input_data = [
            encoders["Fever"].transform([f])[0],
            encoders["Cough"].transform([c])[0],
            encoders["Fatigue"].transform([fa])[0],
            encoders["Difficulty Breathing"].transform([b])[0],
            age,
            encoders["Gender"].transform([g])[0],
            encoders["Blood Pressure"].transform([bp])[0],
            encoders["Cholesterol Level"].transform([ch])[0],
            encoders["Outcome Variable"].transform(["Positive"])[0]
        ]

        pred = model.predict([input_data])[0]
        disease = encoders["Disease"].inverse_transform([pred])[0]
        total += 1

    except Exception as e:
        print("Crash:", e)
        errors += 1

print("Stress test complete")
print("Total tested:", total)
print("Errors:", errors)

print("\nChecking accuracy on dataset...\n")

df = pd.read_csv("Disease_symptom_and_patient_profile_dataset.csv")

cat_cols = [
"Fever","Cough","Fatigue","Difficulty Breathing",
"Gender","Blood Pressure","Cholesterol Level",
"Outcome Variable","Disease"
]

for col in cat_cols:
    df[col] = encoders[col].transform(df[col])

X = df.drop("Disease", axis=1)
y = df["Disease"]

preds = model.predict(X)
acc = accuracy_score(y,preds)

print("Model accuracy:", round(acc*100,2),"%")

print("\nTesting invalid inputs...\n")

invalid_tests = [
["Maybe","Yes","No","Yes",25,"Male","High","Normal"],
["Yes","Alien","No","Yes",25,"Male","High","Normal"],
["Yes","Yes","Yes","Yes",-5,"Male","High","Normal"],
]

for test in invalid_tests:
    try:
        input_data = [
            encoders["Fever"].transform([test[0]])[0],
            encoders["Cough"].transform([test[1]])[0],
            encoders["Fatigue"].transform([test[2]])[0],
            encoders["Difficulty Breathing"].transform([test[3]])[0],
            test[4],
            encoders["Gender"].transform([test[5]])[0],
            encoders["Blood Pressure"].transform([test[6]])[0],
            encoders["Cholesterol Level"].transform([test[7]])[0],
            encoders["Outcome Variable"].transform(["Positive"])[0]
        ]

        pred = model.predict([input_data])[0]
        disease = encoders["Disease"].inverse_transform([pred])[0]
        print("Invalid passed but predicted:", disease)

    except:
        print("Rejected invalid input:", test)

print("\nTesting completed\n")
