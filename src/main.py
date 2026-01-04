from fast_api import FastAPI
from pydantic import BaseModel
from typing import Union
import joblib
import pandas as pd
import numpy as np
import model

app = FastAPI()


class ModelInfo(BaseModel):
    """Model information structure"""
    mode_name: str
    model_type: str
    model_version: str
    model_features: list[str]
    model_author: str
    model_description: str
    model_accuracy: float
    model_precision: float
    model_recall: float
    model_f1_score: float


class EmployeeData(BaseModel):
    department: str
    region: str
    education: str
    gender: str
    recruitment_channel: str
    no_of_trainings: int
    age: int
    previous_year_rating: int
    length_of_service: int
    KPIs_met_gt_80: int
    awards_won: int
    avg_training_score: int


def load_model():
    global model_pipeline
    model_pipeline = joblib.load('../model/xgb_hr_model.pkl')
    return model_pipeline


@app.on_event("startup")
def startup_event():
    load_model()
    print("Model loaded successfully")


@app.get("/model_info", response_model=ModelInfo)
def get_model_info():
    return ModelInfo(
        mode_name="XGBoost Classifier for HR Promotion Prediction",
        model_type="XGBoost Classifier",
        model_version="1.0.0",
        model_features=["department", "region", "education", "gender", "recruitment_channel",
                        "no_of_trainings", "age", "previous_year_rating", "length_of_service",
                        "KPIs_met_gt_80", "awards_won", "avg_training_score"],
        model_author="Muniappan Mohanraj",
        model_description="An XGBoost Classifier model to predict employee promotions based on various features.",
        model_accuracy=model.accuracy_score,
        model_precision=model.precision,
        model_recall=model.recall,
        model_f1_score=model.f1
    )


@app.get("/")
def read_root():
    return \
        {
            "Description": "Welcome to the HR Analytics Promotion Prediction API",
            "Version": "1.0.0",
            "Author": "Muniappan Mohanraj",
            "Endpoints":
                {
                    "/predict": "POST endpoint to predict employee promotion",
                    "/docs": "API documentation",
                    "/model_info": "GET endpoint to retrieve model information"
                }
        }


class PredictionResponse(BaseModel):
    is_promoted: int


@app.post("/predict", response_model=PredictionResponse)
def predict_promotion(data: EmployeeData):
    if not model_pipeline:
        print("Model not loaded, loading now...")
        return {"error": "Model not loaded"}
    else:
        input_data = pd.DataFrame([data.dict()])
        model_columns = model.X.columns
        print("model_columns: ", model_columns)
        prediction = model_pipeline.predict(input_data[model_columns])
        return PredictionResponse(is_promoted=int(prediction[0]))
