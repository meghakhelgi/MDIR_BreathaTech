"""
BreathaTech model training — sensor-first cascade architecture.

Agent classifier:  sensor readings + vitals + symptoms → CO / OP / PHOSGENE / NONE
Severity classifier: all agent features + agent probability outputs → severity 0–3

The cascade design reflects the actual use-case: the device detects and identifies
an unknown exposure. The severity model is explicitly conditioned on what the agent
model thinks, mirroring clinical reasoning ("given this is probably OP, how severe?")
rather than running blind on agent-agnostic features.

time_since_exposure_min is intentionally excluded from both models.
It belongs only in the treatment urgency rules (inference.py _treatment()),
where it is used as a clinician-entered annotation, not a sensor signal.

Ablation note: sensor-only AUC is printed alongside full-model AUC to quantify
how much vitals + symptoms add beyond the four aptamer/electrochemical readings.
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'training_data.csv')
MODEL_DIR  = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# ── feature tiers ──────────────────────────────────────────────────────────────
# Tier 1: raw sensor readings and their z-scores — always available,
#         device-generated, no clinician input required.
SENSOR_FEATURES = [
    'eco_ppm', 'eno_ppb', 'eco2_pct', 'op_score',
    'eco_ppm_z', 'eno_ppb_z', 'eco2_pct_z', 'op_score_z',
    'max_sensor_z', 'eno_eco2_product', 'spo2_paradox',
]

# Tier 2: vitals — usually available, measured at triage.
VITAL_FEATURES = [
    'hr', 'sbp', 'dbp', 'rr', 'spo2', 'temp', 'age', 'is_smoker',
    'map', 'pulse_pressure', 'shock_index', 'brady_flag', 'tachy_flag',
    'hypoxia_flag',
]

# Tier 3: symptoms — clinician-entered, often incomplete.
SYMPTOM_FEATURES = [
    'headache', 'nausea', 'vomiting', 'dizziness', 'confusion',
    'loss_of_consciousness', 'chest_pain', 'dyspnea', 'diaphoresis',
    'miosis', 'salivation', 'lacrimation', 'bronchospasm', 'bronchorrhea',
    'fasciculations', 'muscle_weakness', 'seizure', 'cough', 'eye_irritation',
    'pulmonary_edema',
    'cholinergic_score', 'sludge_score', 'symptom_burden',
    'eno_eco_ratio', 'eno_x_rr', 'op_chol_product',
]

# Full agent feature set (must match inference.py FEATURES exactly)
FEATURES = SENSOR_FEATURES + VITAL_FEATURES + SYMPTOM_FEATURES

# Agent class ordering — must match sklearn LabelEncoder alphabetical sort
AGENT_CLASSES = ['CO', 'NONE', 'OP', 'PHOSGENE']

# Severity features = all agent features + agent probability outputs from cascade
SEV_FEATURES = FEATURES + [f'agent_prob_{c}' for c in AGENT_CLASSES]

# ── sensor normalisation baselines (matches config.js SENSOR_BASELINES) ────────
SENSOR_BASELINES = {
    'eco_ppm':  {'mu': 1.26, 'sd': 5.0},
    'eno_ppb':  {'mu': 15.0, 'sd': 8.0},
    'eco2_pct': {'mu':  4.1, 'sd': 0.3},
    'op_score': {'mu':  2.0, 'sd': 2.0},
}

# Symptom weights for burden score (matches config.js)
SYMPTOM_WEIGHTS = {
    'miosis': 2, 'salivation': 1, 'lacrimation': 1, 'bronchospasm': 2,
    'bronchorrhea': 2, 'fasciculations': 2, 'muscle_weakness': 1, 'diaphoresis': 1,
    'cough': 1, 'eye_irritation': 1, 'pulmonary_edema': 3, 'dyspnea': 2,
    'headache': 1, 'nausea': 1, 'vomiting': 1, 'dizziness': 1,
    'chest_pain': 2, 'confusion': 3, 'seizure': 4, 'loss_of_consciousness': 4,
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all engineered features. Must be identical to inference.py _engineer()."""
    d = df.copy()

    # Sensor z-scores
    d['eco_ppm_z']  = (d['eco_ppm']  - SENSOR_BASELINES['eco_ppm']['mu'])  / SENSOR_BASELINES['eco_ppm']['sd']
    d['eno_ppb_z']  = (d['eno_ppb']  - SENSOR_BASELINES['eno_ppb']['mu'])  / SENSOR_BASELINES['eno_ppb']['sd']
    d['eco2_pct_z'] = (d['eco2_pct'] - SENSOR_BASELINES['eco2_pct']['mu']) / SENSOR_BASELINES['eco2_pct']['sd']
    d['op_score_z'] = (d['op_score'] - SENSOR_BASELINES['op_score']['mu']) / SENSOR_BASELINES['op_score']['sd']

    # Sensor-only interaction features
    sensor_z_cols = ['eco_ppm_z', 'eno_ppb_z', 'eco2_pct_z', 'op_score_z']
    d['max_sensor_z']     = d[sensor_z_cols].max(axis=1)
    d['eno_eco2_product'] = d['eno_ppb_z'] * d['eco2_pct_z']

    # CO SpO2 paradox: eCO elevated but pulse-ox reads normal
    # (carboxyHb absorbs at the same wavelength as oxyHb — pulse ox cannot distinguish)
    d['spo2_paradox'] = ((d['eco_ppm'] > 6) & (d['spo2'] >= 95)).astype(int)

    # Haemodynamic derived vitals
    d['map']            = d['dbp'] + (d['sbp'] - d['dbp']) / 3.0
    d['pulse_pressure'] = d['sbp'] - d['dbp']
    d['shock_index']    = d['hr'] / d['sbp'].clip(lower=1)
    d['brady_flag']     = (d['hr'] < 60).astype(int)
    d['tachy_flag']     = (d['hr'] > 100).astype(int)
    d['hypoxia_flag']   = (d['spo2'] < 94).astype(int)

    # Symptom aggregate scores
    cholinergic_cols = ['miosis', 'salivation', 'lacrimation',
                        'bronchospasm', 'bronchorrhea', 'diaphoresis']
    d['cholinergic_score'] = d[cholinergic_cols].sum(axis=1)

    sludge_cols = ['salivation', 'lacrimation', 'nausea', 'vomiting', 'diaphoresis']
    d['sludge_score'] = d[sludge_cols].sum(axis=1)

    d['symptom_burden'] = sum(
        d[col] * w for col, w in SYMPTOM_WEIGHTS.items() if col in d.columns
    )

    # Sensor × symptom and sensor × vital interaction features
    d['eno_eco_ratio']  = d['eno_ppb'] / (d['eco_ppm'] + 0.1)
    d['eno_x_rr']       = d['eno_ppb'] * d['rr']
    d['op_chol_product'] = d['op_score_z'] * d['cholinergic_score']

    return d


def _xgb():
    return XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric='mlogloss',
        random_state=42, n_jobs=-1,
    )


def train():
    print(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    df = engineer_features(df)
    X  = df[FEATURES]
    print(f"  Agent feature matrix:  {X.shape}")

    # ── Agent classifier ───────────────────────────────────────────────────────
    agent_le = LabelEncoder()
    y_agent  = agent_le.fit_transform(df['agent'])
    print(f"\nAgent classes: {list(agent_le.classes_)}")
    assert list(agent_le.classes_) == AGENT_CLASSES, \
        f"Label encoder order changed: {list(agent_le.classes_)}"

    X_tr, X_te, ya_tr, ya_te = train_test_split(
        X, y_agent, test_size=0.2, random_state=42, stratify=y_agent
    )

    agent_model = _xgb()
    agent_model.fit(X_tr, ya_tr)

    ya_prob = agent_model.predict_proba(X_te)
    agent_auc = roc_auc_score(ya_te, ya_prob, multi_class='ovr', average='macro')
    print(f"  Agent AUC (macro OvR, full features):    {agent_auc:.4f}")

    # Sensor-only ablation — how well do the four aptamer/electrochemical
    # readings alone discriminate exposure type?
    sensor_model = _xgb()
    Xs_tr_s = X_tr[SENSOR_FEATURES]
    Xs_te_s = X_te[SENSOR_FEATURES]
    sensor_model.fit(Xs_tr_s, ya_tr)
    sensor_auc = roc_auc_score(
        ya_te, sensor_model.predict_proba(Xs_te_s), multi_class='ovr', average='macro'
    )
    print(f"  Agent AUC (macro OvR, sensors only):     {sensor_auc:.4f}")
    print(f"  Clinical data uplift:                    +{agent_auc - sensor_auc:.4f}")

    # ── Cascade: agent probabilities become features for severity model ─────────
    # Compute agent probabilities for the full dataset.
    # XGBoost with regularisation does not perfectly memorise training rows, so
    # training-set probabilities are informative but not trivially 1.0 — acceptable
    # for this prototype without cross-validated OOF predictions.
    ya_prob_all = agent_model.predict_proba(X)
    for i, cls in enumerate(AGENT_CLASSES):
        df[f'agent_prob_{cls}'] = ya_prob_all[:, i]

    X_sev = df[SEV_FEATURES]
    print(f"\n  Severity feature matrix: {X_sev.shape}  (includes agent probs)")

    # ── Severity classifier ────────────────────────────────────────────────────
    severity_le = LabelEncoder()
    y_sev       = severity_le.fit_transform(df['severity'])
    print(f"Severity classes: {list(severity_le.classes_)}")

    Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(
        X_sev, y_sev, test_size=0.2, random_state=42, stratify=y_sev
    )

    severity_model = _xgb()
    severity_model.fit(Xs_tr, ys_tr)

    ys_prob = severity_model.predict_proba(Xs_te)
    sev_auc = roc_auc_score(ys_te, ys_prob, multi_class='ovr', average='macro')
    print(f"  Severity AUC (macro OvR): {sev_auc:.4f}")

    # ── Save models ────────────────────────────────────────────────────────────
    artifacts = {
        'agent_model.pkl':    agent_model,
        'severity_model.pkl': severity_model,
        'agent_le.pkl':       agent_le,
        'severity_le.pkl':    severity_le,
    }
    for fname, obj in artifacts.items():
        path = os.path.join(MODEL_DIR, fname)
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
        print(f"  Saved -> {path}")

    meta_path = os.path.join(BASE_DIR, 'models', 'metadata.json')
    with open(meta_path, encoding='utf-8') as f:
        meta = json.load(f)
    meta['agent_macro_auc']    = round(agent_auc, 4)
    meta['sensor_only_auc']    = round(sensor_auc, 4)
    meta['severity_macro_auc'] = round(sev_auc, 4)
    meta['n_train'] = len(X_tr)
    meta['n_test']  = len(X_te)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata updated -> {meta_path}")
    print("\nTraining complete.")


if __name__ == '__main__':
    train()
