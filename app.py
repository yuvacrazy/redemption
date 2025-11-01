# app.py - SmartPay Salary Prediction API
import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Header, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import logging
import uvicorn

# ---------------------------
# CONFIGURATION
# ---------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "salary_model_pipeline.pkl")
API_KEY = os.getenv("API_KEY", "*")  # change this to your own
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

# ---------------------------
# LOGGING
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("smartpay-api")

# ---------------------------
# LOAD MODEL PIPELINE
# ---------------------------
try:
    model = joblib.load(MODEL_PATH)
    logger.info("✅ Salary prediction pipeline loaded successfully.")
except Exception as e:
    logger.exception("❌ Failed to load model pipeline: %s", e)
    raise RuntimeError("Model loading failed. Check model path or file integrity.")

# ---------------------------
# FASTAPI INITIALIZATION
# ---------------------------
app = FastAPI(title="SmartPay AI Salary Prediction API", version="3.0")

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOWED_ORIGINS == "*" else [ALLOWED_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# AUTHENTICATION
# ---------------------------
def api_key_auth(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized access")
    return True

# ---------------------------
# REQUEST SCHEMA
# ---------------------------
class PredictRequest(BaseModel):
    age: float = Field(..., ge=15, le=100)
    gender: str
    education: str
    marital_status: str
    experience_level: str
    employment_type: str
    job_title: str
    hours_per_week: float = Field(..., ge=0, le=168)
    employee_residence: str
    company_location: str
    remote_ratio: float = Field(..., ge=0, le=100)
    company_size: str

    @validator('*', pre=True)
    def validate_text(cls, v):
        if isinstance(v, str):
            return v.strip()
        return v

class PredictResponse(BaseModel):
    predicted_salary_usd: float

# ---------------------------
# PREDICT ENDPOINT
# ---------------------------
@app.post("/predict", response_model=PredictResponse)
async def predict_salary(req: PredictRequest, auth: bool = Depends(api_key_auth)):
    try:
        # Convert to DataFrame (model expects DataFrame input)
        data = pd.DataFrame([req.dict()])

        # Predict salary
        prediction = model.predict(data)[0]

        return PredictResponse(predicted_salary_usd=float(prediction))
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# ---------------------------
# HEALTH CHECK
# ---------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}

# ---------------------------
# ROOT ENDPOINT
# ---------------------------
@app.get("/")
def root():
    return {"service": "SmartPay Salary Prediction API", "status": "running"}

# ---------------------------
# RUN LOCALLY
# ---------------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
