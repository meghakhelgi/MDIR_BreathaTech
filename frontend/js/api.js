// api.js — handles all communication with the FastAPI backend
// Exposes: API.predict(payload) → prediction result
//          API.health()         → server health check
//          API.modelInfo()      → model metadata

const API = (() => {

  const BASE = Config.API_BASE;

  // ── health check ─────────────────────────────────────────────────────
  async function health() {
    const r = await fetch(`${BASE}/health`);
    if (!r.ok) throw new Error(`Health check failed: ${r.status}`);
    return r.json();
  }

  // ── model metadata ────────────────────────────────────────────────────
  async function modelInfo() {
    const r = await fetch(`${BASE}/model/info`);
    if (!r.ok) throw new Error(`Model info failed: ${r.status}`);
    return r.json();
  }

  // ── predict ───────────────────────────────────────────────────────────
  // payload must match the SensorReading schema in api.py:
  //   eco_ppm, eno_ppb, eco2_pct, op_score,
  //   spo2_pct, hr_bpm, rr_rpm, sbp_mmhg, dbp_mmhg,
  //   all 20 symptom flags,
  //   is_smoker, age, weight_kg, time_since_exposure_min
  async function predict(reading, notes = '', patientId = '') {
    const body = {
      reading,
      notes,
      patient_id: patientId,
    };
    const r = await fetch(`${BASE}/predict`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(body),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: r.statusText }));
      throw new Error(err.detail || `Prediction failed: ${r.status}`);
    }
    return r.json();
  }

  // ── build reading from form + sensor state ────────────────────────────
  // Merges clinician-entered vitals/symptoms with current sensor readings.
  // This is the field-level mapping between the UI form and the API schema.
  function buildReading(sensors, vitals, symptoms) {
    return {
      // sensor channels
      eco_ppm:   sensors.eco_ppm  ?? 1.26,
      eno_ppb:   sensors.eno_ppb  ?? 15.0,
      eco2_pct:  sensors.eco2_pct ?? 4.1,
      op_score:  sensors.op_score ?? 2.0,

      // clinician-entered vitals
      spo2_pct:  vitals.spo2  ?? 98,
      hr_bpm:    vitals.hr    ?? 72,
      rr_rpm:    vitals.rr    ?? 14,
      sbp_mmhg:  vitals.sbp   ?? 120,
      dbp_mmhg:  vitals.dbp   ?? 80,

      // all 20 symptom flags
      symp_miosis:                 symptoms.miosis              ?? 0,
      symp_salivation:             symptoms.salivation          ?? 0,
      symp_lacrimation:            symptoms.lacrimation         ?? 0,
      symp_bronchospasm:           symptoms.bronchospasm        ?? 0,
      symp_bronchorrhea:           symptoms.bronchorrhea        ?? 0,
      symp_fasciculations:         symptoms.fasciculations      ?? 0,
      symp_muscle_weakness:        symptoms.muscle_weakness     ?? 0,
      symp_diaphoresis:            symptoms.diaphoresis         ?? 0,
      symp_cough:                  symptoms.cough               ?? 0,
      symp_eye_irritation:         symptoms.eye_irritation      ?? 0,
      symp_pulmonary_edema:        symptoms.pulmonary_edema     ?? 0,
      symp_dyspnea:                symptoms.dyspnea             ?? 0,
      symp_headache:               symptoms.headache            ?? 0,
      symp_nausea:                 symptoms.nausea              ?? 0,
      symp_vomiting:               symptoms.vomiting            ?? 0,
      symp_dizziness:              symptoms.dizziness           ?? 0,
      symp_chest_pain:             symptoms.chest_pain          ?? 0,
      symp_confusion:              symptoms.confusion           ?? 0,
      symp_seizure:                symptoms.seizure             ?? 0,
      symp_loss_of_consciousness:  symptoms.loss_of_consciousness ?? 0,

      // demographics
      is_smoker:               vitals.is_smoker ?? 0,
      age:                     vitals.age       ?? 35,
      weight_kg:               vitals.weight_kg ?? 75,
      time_since_exposure_min: vitals.texp      ?? 0,
    };
  }

  return { health, modelInfo, predict, buildReading };
})();
