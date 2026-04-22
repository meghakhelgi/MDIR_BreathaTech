"""
BreathaTech Inference v6 — sensor-first cascade architecture.

Agent classifier:   sensor readings + vitals + symptoms → CO / OP / PHOSGENE / NONE
Severity classifier: all agent features + agent probability outputs → severity 0–3

time_since_exposure_min is NOT a model feature. It is used only in _treatment()
as a clinician-entered annotation for OP urgency annotation.
"""

import os
import pickle
import json
import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# ── feature tiers — must match train_models.py exactly ────────────────────────
SENSOR_FEATURES = [
    'eco_ppm', 'eno_ppb', 'eco2_pct', 'op_score',
    'eco_ppm_z', 'eno_ppb_z', 'eco2_pct_z', 'op_score_z',
    'max_sensor_z', 'eno_eco2_product', 'spo2_paradox',
]

VITAL_FEATURES = [
    'hr', 'sbp', 'dbp', 'rr', 'spo2', 'temp', 'age', 'is_smoker',
    'map', 'pulse_pressure', 'shock_index', 'brady_flag', 'tachy_flag',
    'hypoxia_flag',
]

SYMPTOM_FEATURES = [
    'headache', 'nausea', 'vomiting', 'dizziness', 'confusion',
    'loss_of_consciousness', 'chest_pain', 'dyspnea', 'diaphoresis',
    'miosis', 'salivation', 'lacrimation', 'bronchospasm', 'bronchorrhea',
    'fasciculations', 'muscle_weakness', 'seizure', 'cough', 'eye_irritation',
    'pulmonary_edema',
    'cholinergic_score', 'sludge_score', 'symptom_burden',
    'eno_eco_ratio', 'eno_x_rr', 'op_chol_product',
]

FEATURES = SENSOR_FEATURES + VITAL_FEATURES + SYMPTOM_FEATURES

AGENT_CLASSES = ['CO', 'NONE', 'OP', 'PHOSGENE']
SEV_FEATURES  = FEATURES + [f'agent_prob_{c}' for c in AGENT_CLASSES]

SENSOR_BASELINES = {
    'eco_ppm':  {'mu': 1.26, 'sd': 5.0},
    'eno_ppb':  {'mu': 15.0, 'sd': 8.0},
    'eco2_pct': {'mu':  4.1, 'sd': 0.3},
    'op_score': {'mu':  2.0, 'sd': 2.0},
}

SYMPTOM_WEIGHTS = {
    'miosis': 2, 'salivation': 1, 'lacrimation': 1, 'bronchospasm': 2,
    'bronchorrhea': 2, 'fasciculations': 2, 'muscle_weakness': 1, 'diaphoresis': 1,
    'cough': 1, 'eye_irritation': 1, 'pulmonary_edema': 3, 'dyspnea': 2,
    'headache': 1, 'nausea': 1, 'vomiting': 1, 'dizziness': 1,
    'chest_pain': 2, 'confusion': 3, 'seizure': 4, 'loss_of_consciousness': 4,
}

# API field name → training column name
FIELD_MAP = {
    'spo2_pct':  'spo2',
    'hr_bpm':    'hr',
    'rr_rpm':    'rr',
    'sbp_mmhg':  'sbp',
    'dbp_mmhg':  'dbp',
}

TRIAGE_MAP = {0: 'Clear', 1: 'Monitor', 2: 'Escalate', 3: 'Immediate'}

AGENT_NAMES = {
    'NONE':     'No exposure detected',
    'CO':       'Carbon Monoxide (CO)',
    'OP':       'Nerve Agent — Organophosphate',
    'PHOSGENE': 'Phosgene — eNO signal',
}

TREATMENT_RULES = {
    'CO': {
        1: 'Remove from source. 100% O₂ via non-rebreather mask. Monitor SpO₂ and HbCO. Reassess every 30 min.',
        2: 'Remove from source. 100% O₂ NRB. Obtain ECG. Consider hyperbaric oxygen (HBO) if HbCO >25% or neurological symptoms present.',
        3: 'Immediate HBO therapy. ICU admission. Continuous cardiac monitoring. Neuroprotective measures.',
    },
    'OP': {
        1: 'Decontaminate. Atropine 2 mg IV/IM — repeat q5–10 min until secretions dry. Pralidoxime 1 g IV over 15–30 min.',
        2: 'Decontaminate. Atropine titrated IV (target: dry secretions). Pralidoxime 2 g IV STAT — time-critical, before enzyme aging. Diazepam PRN for seizure prophylaxis.',
        3: 'NERVE AGENT — IMMEDIATE ACTION. Decontaminate. Atropine 4–6 mg IV bolus, repeat q5 min. Pralidoxime 2 g IV STAT (soman ages in ~2 min, sarin ~5 h, VX ~40 h). Diazepam 10 mg IV. Secure airway — intubate. ICU.',
    },
    'PHOSGENE': {
        1: 'STRICT REST — exertion accelerates pulmonary edema. High-flow O₂. Admit for 24 h observation. Serial eNO and eCO₂ every hour. Do NOT discharge even if patient feels well.',
        2: 'Strict rest. High-flow O₂. ICU monitoring. Serial ABG and CXR. Early CPAP if SpO₂ declining. Do NOT discharge.',
        3: 'Mechanical ventilation (lung-protective strategy: TV 6 mL/kg, PEEP 8–12 cmH₂O). ICU. Early corticosteroids consider. Prone positioning if refractory hypoxaemia.',
    },
    'NONE': {
        0: 'No chemical exposure detected. Routine monitoring. Reassess if symptoms develop.',
    },
}


class BreathaTechInference:
    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        self._agent_model    = None
        self._severity_model = None
        self._agent_le       = None
        self._severity_le    = None
        self._meta           = {}
        self._loaded         = False
        self._load()

    def _load(self):
        try:
            def _pkl(name):
                with open(os.path.join(self.model_dir, name), 'rb') as f:
                    return pickle.load(f)

            self._agent_model    = _pkl('agent_model.pkl')
            self._severity_model = _pkl('severity_model.pkl')
            self._agent_le       = _pkl('agent_le.pkl')
            self._severity_le    = _pkl('severity_le.pkl')

            meta_path = os.path.join(BASE_DIR, 'models', 'metadata.json')
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    self._meta = json.load(f)

            self._loaded = True
            print(f"[BreathaTech] Models loaded from {self.model_dir}")
        except FileNotFoundError as e:
            print(f"[BreathaTech] WARNING: Model file not found — {e}")
            print("[BreathaTech] Run: python train_models.py")

    @property
    def model_loaded(self) -> bool:
        return self._loaded

    # ── field name remapping (API names → training names) ─────────────────────
    def _remap_fields(self, reading: dict) -> dict:
        d = {}
        for k, v in reading.items():
            mapped = FIELD_MAP.get(k, k)
            if mapped.startswith('symp_'):
                mapped = mapped[5:]
            d[mapped] = v

        d.setdefault('temp', 37.0)
        return d

    # ── feature engineering (must match train_models.py exactly) ──────────────
    def _engineer(self, d: dict) -> dict:
        # Sensor z-scores
        d['eco_ppm_z']  = (d['eco_ppm']  - SENSOR_BASELINES['eco_ppm']['mu'])  / SENSOR_BASELINES['eco_ppm']['sd']
        d['eno_ppb_z']  = (d['eno_ppb']  - SENSOR_BASELINES['eno_ppb']['mu'])  / SENSOR_BASELINES['eno_ppb']['sd']
        d['eco2_pct_z'] = (d['eco2_pct'] - SENSOR_BASELINES['eco2_pct']['mu']) / SENSOR_BASELINES['eco2_pct']['sd']
        d['op_score_z'] = (d['op_score'] - SENSOR_BASELINES['op_score']['mu']) / SENSOR_BASELINES['op_score']['sd']

        # Sensor interaction features
        d['max_sensor_z']     = max(d['eco_ppm_z'], d['eno_ppb_z'], d['eco2_pct_z'], d['op_score_z'])
        d['eno_eco2_product'] = d['eno_ppb_z'] * d['eco2_pct_z']

        # CO SpO2 paradox: pulse-ox reads normal despite elevated eCO
        d['spo2_paradox'] = int(d['eco_ppm'] > 6 and d['spo2'] >= 95)

        # Haemodynamic derived vitals
        sbp = max(d['sbp'], 1)
        d['map']            = d['dbp'] + (d['sbp'] - d['dbp']) / 3.0
        d['pulse_pressure'] = d['sbp'] - d['dbp']
        d['shock_index']    = d['hr'] / sbp
        d['brady_flag']     = int(d['hr'] < 60)
        d['tachy_flag']     = int(d['hr'] > 100)
        d['hypoxia_flag']   = int(d['spo2'] < 94)

        # Cholinergic / SLUDGE scores
        cholinergic_cols = ['miosis', 'salivation', 'lacrimation',
                            'bronchospasm', 'bronchorrhea', 'diaphoresis']
        d['cholinergic_score'] = sum(d.get(c, 0) for c in cholinergic_cols)

        sludge_cols = ['salivation', 'lacrimation', 'nausea', 'vomiting', 'diaphoresis']
        d['sludge_score'] = sum(d.get(c, 0) for c in sludge_cols)

        d['symptom_burden'] = sum(
            d.get(col, 0) * w for col, w in SYMPTOM_WEIGHTS.items()
        )

        # Sensor ratio and interaction features
        d['eno_eco_ratio']  = d['eno_ppb'] / (d['eco_ppm'] + 0.1)
        d['eno_x_rr']       = d['eno_ppb'] * d['rr']
        d['op_chol_product'] = d['op_score_z'] * d['cholinergic_score']

        return d

    # ── clinical treatment hint ────────────────────────────────────────────────
    def _treatment(self, agent: str, severity: int,
                   time_since_exposure_min: float = 30) -> str:
        rules = TREATMENT_RULES.get(agent, {})
        text  = rules.get(severity, rules.get(max(rules.keys(), default=0), ''))

        if agent == 'OP' and severity >= 2:
            if time_since_exposure_min <= 2:
                text = '⚠ SOMAN AGING WINDOW — pralidoxime may already be ineffective. ' + text
            elif time_since_exposure_min <= 5:
                text = '⚠ Within sarin aging window (~5 h). Pralidoxime STAT. ' + text

        return text

    # ── main prediction entry point ────────────────────────────────────────────
    def predict(self, reading: dict) -> dict:
        if not self._loaded:
            raise RuntimeError("Models not loaded. Run: python train_models.py")

        # 1. Remap API field names to training names
        d = self._remap_fields(reading)

        # 2. Engineer features
        d = self._engineer(d)

        # 3. Build agent feature vector
        row = {f: d.get(f, 0) for f in FEATURES}
        X   = pd.DataFrame([row])[FEATURES]

        # 4. Agent prediction
        agent_probs = self._agent_model.predict_proba(X)[0]
        agent_idx   = int(np.argmax(agent_probs))
        agent       = self._agent_le.inverse_transform([agent_idx])[0]
        agent_conf  = float(agent_probs[agent_idx]) * 100

        agent_breakdown = {
            cls: round(float(agent_probs[i]) * 100, 1)
            for i, cls in enumerate(self._agent_le.classes_)
        }

        # 5. Cascade: inject agent probabilities as features for severity model
        for i, cls in enumerate(AGENT_CLASSES):
            d[f'agent_prob_{cls}'] = float(agent_probs[i])

        sev_row = {f: d.get(f, 0) for f in SEV_FEATURES}
        X_sev   = pd.DataFrame([sev_row])[SEV_FEATURES]

        # 6. Severity prediction
        sev_probs = self._severity_model.predict_proba(X_sev)[0]
        sev_idx   = int(np.argmax(sev_probs))
        severity  = int(self._severity_le.inverse_transform([sev_idx])[0])

        # 7. Triage from severity
        triage = TRIAGE_MAP.get(severity, 'Monitor')

        # 8. Clinical flags
        nerve_alert     = (agent == 'OP')
        phosgene_occult = (agent == 'PHOSGENE' and d.get('spo2', 100) > 93)

        # 9. Treatment hint (time_since_exposure_min used only here, not in model)
        tse = reading.get('time_since_exposure_min', 30) or 30
        treatment_hint = self._treatment(agent, severity, tse)

        return {
            'agent':             agent,
            'agent_name':        AGENT_NAMES.get(agent, agent),
            'agent_confidence':  round(agent_conf, 1),
            'agent_breakdown':   agent_breakdown,
            'triage':            triage,
            'severity':          severity,
            'nerve_alert':       nerve_alert,
            'phosgene_occult':   phosgene_occult,
            'treatment_hint':    treatment_hint,
            'model_version':     self._meta.get('version', 'v1.0'),
        }
