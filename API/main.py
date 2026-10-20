from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import logging
from typing import List

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(
    title="Delivery Type Prediction API",
    description="Predict whether delivery will be Vaginal or C-Section",
    version="1.0"
)

# -----------------------------
# Load Model
# -----------------------------
try:
    model = joblib.load("model/delivery_model.pkl")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Model loading failed: {e}")
    raise RuntimeError("Model could not be loaded.")


# -----------------------------
# Input Schema
# -----------------------------
class PatientData(BaseModel):

    PrimaryIndicationforCaesarean_fetal_compromise: int
    PrimaryIndicationforCaesarean_previous_uterine_surgery: int
    No_Of_previous_Csections: int
    Risk_Factors_twins_or_more: int
    Height: float
    Parity: int
    Gestation: float
    PrimaryIndicationforCaesarean_multiple_pregnancy: int
    PrimaryIndicationforCaesarean_antepartum_haemorrhage: int
    PrimaryIndicationforCaesarean_pre_eclampsia: int
    SystolicBloodPressureCuff: float
    DiastolicBloodPressure: float   # used only for MAP calculation


# -----------------------------
# Root Endpoint
# -----------------------------
@app.get("/")
def home():
    return {"message": "Delivery Prediction API is running"}


# -----------------------------
# Health Endpoint
# -----------------------------
@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


# -----------------------------
# Prediction Function
# -----------------------------
@app.post("/predict")
def predict(data: PatientData):

    try:

        # Convert input to dict
        patient = data.dict()

        # Extract BP values
        systolic = patient["SystolicBloodPressureCuff"]
        diastolic = patient["DiastolicBloodPressure"]

        # Calculate MAP
        map_value = (systolic + 2 * diastolic) / 3

        # Build model input dictionary
        model_input = {
            "PrimaryIndicationforCaesarean_fetal compromise": patient["PrimaryIndicationforCaesarean_fetal_compromise"],
            "PrimaryIndicationforCaesarean_previous uterine surgery": patient["PrimaryIndicationforCaesarean_previous_uterine_surgery"],
            "No_Of_previous_Csections": patient["No_Of_previous_Csections"],
            "Risk Factors_twins or more": patient["Risk_Factors_twins_or_more"],
            "Height": patient["Height"],
            "Parity": patient["Parity"],
            "Gestation": patient["Gestation"],
            "PrimaryIndicationforCaesarean_multiple pregnancy": patient["PrimaryIndicationforCaesarean_multiple_pregnancy"],
            "PrimaryIndicationforCaesarean_antepartum haemorrhage": patient["PrimaryIndicationforCaesarean_antepartum_haemorrhage"],
            "PrimaryIndicationforCaesarean_pre eclampsia": patient["PrimaryIndicationforCaesarean_pre_eclampsia"],
            "SystolicBloodPressureCuff": systolic,
            "MAP": map_value
        }

        # Convert to dataframe
        input_df = pd.DataFrame([model_input])

        # Prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        result = "C-Section" if prediction == 1 else "Vaginal"

        return {
            "prediction": result,
            "c_section_probability": float(probability),
        }

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# Batch Prediction
# -----------------------------
@app.post("/predict_batch")
def predict_batch(data: List[PatientData]):

    try:

        records = []

        for patient in data:

            p = patient.dict()

            systolic = p["SystolicBloodPressureCuff"]
            diastolic = p["DiastolicBloodPressure"]

            map_value = (systolic + 2 * diastolic) / 3

            records.append({
                "PrimaryIndicationforCaesarean_fetal compromise": p["PrimaryIndicationforCaesarean_fetal_compromise"],
                "PrimaryIndicationforCaesarean_previous uterine surgery": p["PrimaryIndicationforCaesarean_previous_uterine_surgery"],
                "No_Of_previous_Csections": p["No_Of_previous_Csections"],
                "Risk Factors_twins or more": p["Risk_Factors_twins_or_more"],
                "Height": p["Height"],
                "Parity": p["Parity"],
                "Gestation": p["Gestation"],
                "PrimaryIndicationforCaesarean_multiple pregnancy": p["PrimaryIndicationforCaesarean_multiple_pregnancy"],
                "PrimaryIndicationforCaesarean_antepartum haemorrhage": p["PrimaryIndicationforCaesarean_antepartum_haemorrhage"],
                "PrimaryIndicationforCaesarean_pre eclampsia": p["PrimaryIndicationforCaesarean_pre_eclampsia"],
                "SystolicBloodPressureCuff": systolic,
                "MAP": map_value
            })

        df = pd.DataFrame(records)

        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:,1]

        results = []

        for pred, prob in zip(predictions, probabilities):
            results.append({
                "prediction": "C-Section" if pred == 1 else "Vaginal",
                "c_section_probability": float(prob)
            })

        return {"results": results}

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))