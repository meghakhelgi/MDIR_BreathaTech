"""
BreathaTech FastAPI Backend v5
===============================
Endpoints:
  POST /predict      — run agent + severity prediction
  GET  /health       — liveness + model-loaded status
  GET  /model/info   — metadata from breathatech_model_metadata.json

Start:
    uvicorn api:app --reload --port 8000
"""

import json
import os
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from inference import BreathaTechInference

# ── app setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title='BreathaTech Clinical Decision Support',
    version='5.0',
    description='AI triage for breath-based chemical exposure detection.',
)

# Allow the frontend dev server (any localhost port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['GET', 'POST'],
    allow_headers=['*'],
)

# Load models once at startup
_inference = BreathaTechInference()


# ── request / response schemas ─────────────────────────────────────────────────
class SensorReading(BaseModel):
    # Electrochemical sensor channels
    eco_ppm:   float = Field(1.26,  description='Exhaled CO (ppm)')
    eno_ppb:   float = Field(15.0,  description='Exhaled NO (ppb)')
    eco2_pct:  float = Field(4.1,   description='Exhaled CO₂ (%)')
    op_score:  float = Field(2.0,   description='OP aptamer score (AU)')

    # Clinician-measured vitals
    spo2_pct:  float = Field(98.0,  description='SpO₂ (%)')
    hr_bpm:    float = Field(72.0,  description='Heart rate (bpm)')
    rr_rpm:    float = Field(14.0,  description='Respiratory rate (rpm)')
    sbp_mmhg:  float = Field(120.0, description='Systolic BP (mmHg)')
    dbp_mmhg:  float = Field(80.0,  description='Diastolic BP (mmHg)')
    temp:      Optional[float] = Field(37.0, description='Temperature (°C) — optional')

    # Symptom flags (0/1) — OP-specific
    symp_miosis:          int = 0
    symp_salivation:      int = 0
    symp_lacrimation:     int = 0
    symp_bronchospasm:    int = 0
    symp_bronchorrhea:    int = 0
    symp_fasciculations:  int = 0
    symp_muscle_weakness: int = 0
    symp_diaphoresis:     int = 0

    # Symptom flags — phosgene-specific
    symp_cough:           int = 0
    symp_eye_irritation:  int = 0
    symp_pulmonary_edema: int = 0
    symp_dyspnea:         int = 0

    # Symptom flags — general
    symp_headache:              int = 0
    symp_nausea:                int = 0
    symp_vomiting:              int = 0
    symp_dizziness:             int = 0
    symp_chest_pain:            int = 0
    symp_confusion:             int = 0
    symp_seizure:               int = 0
    symp_loss_of_consciousness: int = 0

    # Demographics
    is_smoker:               int   = Field(0,    description='1 = smoker (adjusts eCO baseline)')
    age:                     float = Field(35.0, description='Age (years)')
    weight_kg:               float = Field(75.0, description='Weight (kg) — informational only')
    time_since_exposure_min: float = Field(30.0, description='Minutes since exposure (clinician estimate)')


class PredictRequest(BaseModel):
    reading:    SensorReading
    notes:      str = ''
    patient_id: str = ''


class PredictionResponse(BaseModel):
    agent:             str
    agent_name:        str
    agent_confidence:  float
    agent_breakdown:   Dict[str, float]
    triage:            str
    severity:          int
    nerve_alert:       bool
    phosgene_occult:   bool
    treatment_hint:    str
    model_version:     str


# ── endpoints ──────────────────────────────────────────────────────────────────
@app.post('/predict', response_model=PredictionResponse)
def predict(req: PredictRequest):
    if not _inference.model_loaded:
        raise HTTPException(
            status_code=503,
            detail='Models not loaded. Run: python train_models.py'
        )

    reading_dict = req.reading.model_dump()
    try:
        result = _inference.predict(reading_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result


@app.get('/health')
def health():
    return {
        'status':       'ok',
        'model_loaded': _inference.model_loaded,
        'version':      '1.0',
    }


@app.get('/model/info')
def model_info():
    meta_path = os.path.join(os.path.dirname(__file__), 'models', 'metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {'version': 'v1.0', 'note': 'metadata file not found'}
