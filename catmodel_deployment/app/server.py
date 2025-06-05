from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

@app.on_event("startup")
def load_model(): 
       global model
       model = joblib.load(....) t



model = joblib.load("app/catmodel.pkl")

app = FastAPI()

smoking_map = {'never': 0, 'formerly': 1, 'smokes': 2}
work_type_map = {'Private': 0, 'Self-employed': 1,
                 'Govt_job': 2, 'children': 3}
married_map = {'No': 0, 'Yes': 1}


def simulate_scaling(value, mean, std):
    return (value - mean) / std


def simulate_encoding(value, mapping):
    return mapping.get(value, -1)


label_map = {
    0: "no_stroke",
    1: "stroke"
}


class InputData(BaseModel):
    age: int
    bmi: float
    smoking_status: str
    work_type: str
    ever_married: str


@app.get('/')
def read_root():
    return {'message': 'stroke model API'}


@app.post('/predict')
def predict(input_data: InputData):
    """
    Takes raw patient data and returns stroke prediction.
    """

    scaled_age = simulate_scaling(input_data.age, mean=50, std=17)
    scaled_bmi = simulate_scaling(input_data.bmi, mean=30, std=7)

    encoded_smoking = simulate_encoding(input_data.smoking_status, smoking_map)
    encoded_work = simulate_encoding(input_data.work_type, work_type_map)
    encoded_married = simulate_encoding(input_data.ever_married, married_map)

    df = pd.DataFrame([{
        'scaler__age': scaled_age,
        'scaler__bmi': scaled_bmi,
        'encoder__smoking_status': encoded_smoking,
        'encoder__work_type': encoded_work,
        'encoder__ever_married': encoded_married
    }])

    prediction = model.predict(df)


    try:
        predicted_class = int(np.ravel(prediction)[0])
    except Exception as e:
        return {'error': f"Prediction error: {str(e)}"}
    
    predicted_label = label_map.get(predicted_class, "unknown")

    message = (
        "⚠️ The patient is likely to have a stroke"
        if predicted_class == 1
        else "✅ The patient is unlikely to have a stroke"
    )

    return {
        'prediction': predicted_label,
        'raw_output': predicted_class, 
        'message': message
    }
