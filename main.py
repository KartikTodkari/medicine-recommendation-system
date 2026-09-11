from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets"
MODEL_DIR = BASE_DIR / "models"

app = Flask(__name__)

# Load datasets using absolute paths so the app works regardless of the
# directory from which `python main.py` is executed.
symptoms_df = pd.read_csv(DATA_DIR / "symtoms_df.csv")
precautions_df = pd.read_csv(DATA_DIR / "precautions_df.csv")
workout_df = pd.read_csv(DATA_DIR / "workout_df.csv")
description_df = pd.read_csv(DATA_DIR / "description.csv")
medications_df = pd.read_csv(DATA_DIR / "medications.csv")
diets_df = pd.read_csv(DATA_DIR / "diets.csv")

with open(MODEL_DIR / "svc.pkl", "rb") as model_file:
    svc = pickle.load(model_file)

# The model was trained with the 132 symptom columns of Training.csv.
# Building the dictionary from the dataset prevents path/order mistakes caused
# by manually maintaining 132 symptom indexes.
training_df = pd.read_csv(DATA_DIR / "Training.csv")
symptom_columns = [c for c in training_df.columns if c != "prognosis"]
symptoms_dict = {name.strip().lower(): index for index, name in enumerate(symptom_columns)}

# Disease class IDs used by the supplied model. These IDs are preserved from
# the original project so predictions keep the same disease mapping.
diseases_list = {
    15: "Fungal infection", 4: "Allergy", 16: "GERD",
    9: "Chronic cholestasis", 14: "Drug Reaction", 33: "Peptic ulcer diseae",
    1: "AIDS", 12: "Diabetes ", 17: "Gastroenteritis",
    6: "Bronchial Asthma", 23: "Hypertension ", 30: "Migraine",
    7: "Cervical spondylosis", 32: "Paralysis (brain hemorrhage)",
    28: "Jaundice", 29: "Malaria", 8: "Chicken pox", 11: "Dengue",
    37: "Typhoid", 40: "hepatitis A", 19: "Hepatitis B", 20: "Hepatitis C",
    21: "Hepatitis D", 22: "Hepatitis E", 3: "Alcoholic hepatitis",
    36: "Tuberculosis", 10: "Common Cold", 34: "Pneumonia",
    13: "Dimorphic hemmorhoids(piles)", 18: "Heart attack", 39: "Varicose veins",
    26: "Hypothyroidism", 24: "Hyperthyroidism", 25: "Hypoglycemia",
    31: "Osteoarthristis", 5: "Arthritis",
    0: "(vertigo) Paroymsal  Positional Vertigo", 2: "Acne",
    38: "Urinary tract infection", 35: "Psoriasis", 27: "Impetigo"
}


def normalize_symptom(symptom: str) -> str:
    """Normalize user-entered symptom text to the dataset's symptom key."""
    return "_".join(symptom.strip().lower().split())


def get_prediction(patient_symptoms):
    """Convert symptoms to a 132-feature vector and predict a disease."""
    input_vector = np.zeros(len(symptoms_dict), dtype=int)
    unknown = []

    for raw_item in patient_symptoms:
        symptom = normalize_symptom(raw_item)
        if not symptom:
            continue
        if symptom not in symptoms_dict:
            unknown.append(raw_item.strip())
        else:
            input_vector[symptoms_dict[symptom]] = 1

    if not input_vector.any():
        raise ValueError("Please enter at least one valid symptom.")

    prediction = int(svc.predict([input_vector])[0])
    predicted_disease = diseases_list[prediction]
    return predicted_disease, unknown


def get_recommendations(disease):
    """Fetch project recommendations from the supplied CSV datasets."""
    desc_series = description_df.loc[
        description_df["Disease"] == disease, "Description"
    ]
    desc = " ".join(desc_series.astype(str).tolist())

    pre_row = precautions_df.loc[
        precautions_df["Disease"] == disease,
        ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"],
    ]
    precautions = []
    if not pre_row.empty:
        precautions = [
            str(value).strip()
            for value in pre_row.iloc[0].tolist()
            if pd.notna(value) and str(value).strip()
        ]

    medications = medications_df.loc[
        medications_df["Disease"] == disease, "Medication"
    ].dropna().astype(str).tolist()

    diets = diets_df.loc[diets_df["Disease"] == disease, "Diet"].dropna().astype(str).tolist()

    workouts = workout_df.loc[
        workout_df["disease"] == disease, "workout"
    ].dropna().astype(str).tolist()

    return {
        "description": desc,
        "precautions": precautions,
        "medications": medications,
        "diets": diets,
        "workouts": workouts,
    }


def parse_symptoms(raw_symptoms):
    if not raw_symptoms:
        return []
    # Accept comma-separated symptoms and also tolerate accidental list-style
    # characters from copied input.
    cleaned = raw_symptoms.replace("[", "").replace("]", "").replace("'", "")
    return [item.strip() for item in cleaned.split(",") if item.strip()]


@app.route("/")
def index():
    return render_template("index.html", symptom_options=sorted(symptom_columns))


@app.route("/predict", methods=["POST"])
def predict():
    raw_symptoms = request.form.get("symptoms", "")
    patient_symptoms = parse_symptoms(raw_symptoms)

    try:
        disease, unknown = get_prediction(patient_symptoms)
        recommendations = get_recommendations(disease)
        return render_template(
            "index.html",
            symptom_options=sorted(symptom_columns),
            input_symptoms=raw_symptoms,
            predicted_disease=disease,
            dis_des=recommendations["description"],
            my_precautions=recommendations["precautions"],
            medications=recommendations["medications"],
            my_diet=recommendations["diets"],
            workout=recommendations["workouts"],
            unknown_symptoms=unknown,
        )
    except ValueError as exc:
        return render_template(
            "index.html",
            symptom_options=sorted(symptom_columns),
            input_symptoms=raw_symptoms,
            message=str(exc),
        ), 400


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint for Postman/mobile/frontend integration."""
    data = request.get_json(silent=True) or {}
    raw_symptoms = data.get("symptoms", "")

    if isinstance(raw_symptoms, list):
        patient_symptoms = [str(item) for item in raw_symptoms]
    else:
        patient_symptoms = parse_symptoms(str(raw_symptoms))

    try:
        disease, unknown = get_prediction(patient_symptoms)
        result = get_recommendations(disease)
        result.update({"predicted_disease": disease, "unknown_symptoms": unknown})
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/developer")
def developer():
    return render_template("developer.html")


@app.route("/blog")
def blog():
    return render_template("blog.html")


if __name__ == "__main__":
    # debug=True is convenient for local development. For deployment, use a
    # production WSGI server instead.
    app.run(host="127.0.0.1", port=5000, debug=True)
