# BreathaTech — Codebase Context

**What it is:** AI clinical decision support for a handheld breath-based chemical triage device. Patient exhales → electrochemical sensors measure biomarkers → ML model classifies exposure agent → treatment recommendation displayed.

**Team:** Drebin, Rubinchik, Nagarkar, Khelgi, Cooper — CMU BME MS, Spring 2026

---

## Agents Detected

| Agent | Sensor | Why it works |
|-------|--------|-------------|
| CO | Direct electrochemical oxidation, Pt electrode, no aptamer | eCO tracks HbCO in real-time; corrects SpO₂ paradox (pulse ox reads falsely normal in CO poisoning) |
| Nerve agents (OP class: sarin, soman, VX) | E-AB aptamer on Au electrode, methylene blue reporter | Detectable in seconds; pralidoxime window closes as enzyme ages (soman: ~2 min, sarin: ~5h, VX: ~40h) |
| Phosgene | eNO via direct oxidation, Pt/Nafion electrode — NOT aptamer | eNO rises 30–60 min during silent latent phase before fatal pulmonary edema develops |

**Excluded:** HCN (washout t½ 10–22 sec, patient incapacitated before sampling), Cl₂ (parent compound consumed in airway, not exhaled), biologics (TB/anthrax/tularemia/plague — days-to-weeks detection, wrong timescale).

---

## File Map — Use These

```
model_v5/
  generate_data.py      ← Monte Carlo synthetic data generator
  training_data.csv     ← 2,400 patients (600 × CO/OP/PHOSGENE/NONE)
  agent_model.pkl       ← XGBoost agent classifier (AUC 0.9999)
  severity_model.pkl    ← XGBoost severity classifier (AUC 0.9986)
  agent_le.pkl / severity_le.pkl
  model_metadata.json

inference.py            ← BreathaTechInference class; loads models; runs predict()
api.py                  ← FastAPI: POST /predict, GET /health, GET /model/info
sensor.py               ← SensorManager; hardware + simulation mode; to_dict() merge
main.py                 ← CLI orchestrator

frontend/
  index.html            ← structure only
  css/styles.css
  data/demo_cases.json  ← 4 real rows from training_data.csv
  js/config.js          ← all constants (load first)
  js/api.js             ← fetch calls to FastAPI + buildReading()
  js/sensors.js         ← sensor card rendering + alarm score
  js/symptoms.js        ← symptom grid state + burden score
  js/results.js         ← ML output panel rendering
  js/app.js             ← main controller (load last)
```

Ignore any `_v1`, `_v2`, `_v3`, `_v4` suffixed files — all superseded.

---

## Key Architectural Decisions

**`exposure_ppm` is intentionally absent from features.** Early versions included it; it caused data leakage (clinicians cannot measure ambient ppm in the field). Removed from both training data and inference.

**`time_since_exposure_min` is clinician-entered, not measured.** Defaults to 30 if missing. Must be 0 for true baseline (no exposure) cases.

**`sensor.py:to_dict(reading, clinician)`** merges two data sources: sensor readings (measured by device) and clinician inputs (vitals, symptoms, demographics entered via UI). These must stay separate — the sensor pipeline should never contain clinician-entered data and vice versa.

**Frontend is vanilla JS, no build step.** Script load order matters: `config.js` → `api.js` → `sensors.js` → `symptoms.js` → `results.js` → `app.js`.

**Demo cases are real training data rows**, not hardcoded. `data/demo_cases.json` is extracted from `training_data.csv`.

---

## API — Quick Reference

```bash
# Start backend
uvicorn api:app --reload --port 8000

# Start frontend
cd frontend && python3 -m http.server 3000
```

**POST /predict** — key fields:
```json
{
  "reading": {
    "eco_ppm": 31.3, "eno_ppb": 21.7, "eco2_pct": 4.35, "op_score": 2.4,
    "spo2_pct": 94, "hr_bpm": 105, "rr_rpm": 22, "sbp_mmhg": 112, "dbp_mmhg": 72,
    "symp_miosis": 0, "symp_salivation": 0, "symp_lacrimation": 0,
    "symp_bronchospasm": 0, "symp_bronchorrhea": 0, "symp_fasciculations": 0,
    "symp_muscle_weakness": 0, "symp_diaphoresis": 0, "symp_cough": 0,
    "symp_eye_irritation": 0, "symp_pulmonary_edema": 0, "symp_dyspnea": 0,
    "symp_headache": 1, "symp_nausea": 1, "symp_vomiting": 0, "symp_dizziness": 1,
    "symp_chest_pain": 0, "symp_confusion": 0, "symp_seizure": 0,
    "symp_loss_of_consciousness": 0,
    "is_smoker": 1, "age": 45, "weight_kg": 75, "time_since_exposure_min": 20
  }
}
```

**Response includes:** `agent`, `agent_confidence`, `agent_breakdown` (all 4 probs), `triage`, `severity`, `nerve_alert` (bool), `phosgene_occult` (bool), `treatment_hint`.

---

## Treatment Rules (agent × severity → treatment)

| | Severity 1 | Severity 2 | Severity 3 |
|---|---|---|---|
| CO | 100% O₂ NRB | ECG + HBO if HbCO >25% | HBO + ICU |
| OP | Atropine 2mg + pralidoxime 1g IV | Atropine titrated + 2-PAM STAT + diazepam | Full airway + ICU |
| PHOSGENE | Strict rest + O₂ + admit 24h | ICU monitoring | Ventilate |

---

## Clinical Rules — Do Not Break These

1. **SpO₂ is unreliable in CO** — never show it as reassuring alongside elevated eCO.
2. **`nerve_alert` = true → highest-priority UI alert** — pralidoxime window closes in minutes for soman.
3. **`phosgene_occult` = true → patient looks fine but is not** — strict rest is the treatment; discharge = death.
4. **Smoker baseline eCO is 15–20 ppm** — this is normal; `is_smoker` flag adjusts model interpretation.
5. **OP can cause tachycardia OR bradycardia** — both presentations are in training data; do not assume one direction.
