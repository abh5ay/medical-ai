
import os


import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

os.makedirs("model", exist_ok=True)
print("Training medical AI system...")

df = pd.read_csv("Disease_symptom_and_patient_profile_dataset.csv")

def get_category(disease):
    respiratory = ["Asthma","Pneumonia","Bronchitis","Tuberculosis","Influenza","Common Cold"]
    heart = ["Hypertension","Stroke","Heart","Myocardial"]
    mental = ["Depression","Anxiety","Bipolar"]
    infection = ["Dengue","Malaria","Hepatitis","HIV","COVID","Flu"]
    skin = ["Eczema","Psoriasis","Acne"]
    cancer = ["Cancer","Tumor","Lymphoma"]

    for r in respiratory:
        if r.lower() in disease.lower():
            return "Respiratory"

    for h in heart:
        if h.lower() in disease.lower():
            return "Cardio"

    for m in mental:
        if m.lower() in disease.lower():
            return "Mental"

    for i in infection:
        if i.lower() in disease.lower():
            return "Infection"

    for s in skin:
        if s.lower() in disease.lower():
            return "Skin"

    for c in cancer:
        if c.lower() in disease.lower():
            return "Cancer"

    return "General"

df["Category"] = df["Disease"].apply(get_category)

df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

encoders = {}

cat_cols = [
    "Fever","Cough","Fatigue","Difficulty Breathing",
    "Gender","Blood Pressure","Cholesterol Level",
    "Outcome Variable","Disease","Category"
]

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df.drop(["Disease","Category"], axis=1)
y_disease = df["Disease"]
y_category = df["Category"]

disease_model = RandomForestClassifier(n_estimators=400, random_state=42)
category_model = RandomForestClassifier(n_estimators=300, random_state=42)

disease_model.fit(X, y_disease)
category_model.fit(X, y_category)

print("Models trained successfully")

pickle.dump(disease_model, open("model/disease_model.pkl","wb"))
pickle.dump(category_model, open("model/category_model.pkl","wb"))
pickle.dump(encoders, open("model/encoders.pkl","wb"))

print("Medical AI system ready")
