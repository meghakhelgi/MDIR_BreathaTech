"""
BreathaTech Model Trainer v5
=============================
Reads breathatech_training_data.csv, engineers features identical to those
listed in breathatech_model_metadata.json, trains XGBoost classifiers for
agent and severity, saves to model_v5/.

Run:
    python train_models.py
"""

import os
import pickle
import json
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

# ── feature list (must match inference.py exactly) ─────────────────────────────
FEATURES = [
    'eco_ppm', 'eno_ppb', 'eco2_pct', 'op_score',
    'hr', 'sbp', 'dbp', 'rr', 'spo2', 'temp', 'age', 'is_smoker',
    'headache', 'nausea', 'vomiting', 'dizziness', 'confusion',
    'loss_of_consciousness', 'chest_pain', 'dyspnea', 'diaphoresis',
    'miosis', 'salivation', 'lacrimation', 'bronchospasm', 'bronchorrhea',
    'fasciculations', 'muscle_weakness', 'seizure', 'cough', 'eye_irritation',
    'pulmonary_edema',
    'spo2_paradox', 'op_z', 'cholinergic_score', 'eno_eco_ratio', 'eno_x_rr',
    'map', 'pulse_pressure', 'shock_index', 'brady_flag', 'tachy_flag',
    'hypoxia_flag', 'symptom_burden', 'sludge_score',
    'eco_ppm_z', 'eno_ppb_z', 'eco2_pct_z', 'op_score_z',
]

# Sensor baselines for z-score normalisation (matches config.js SENSOR_BASELINES)
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
    """Add all engineered features. Must be identical to inference.py."""
    d = df.copy()

    # Sensor z-scores
    d['eco_ppm_z']  = (d['eco_ppm']  - SENSOR_BASELINES['eco_ppm']['mu'])  / SENSOR_BASELINES['eco_ppm']['sd']
    d['eno_ppb_z']  = (d['eno_ppb']  - SENSOR_BASELINES['eno_ppb']['mu'])  / SENSOR_BASELINES['eno_ppb']['sd']
    d['eco2_pct_z'] = (d['eco2_pct'] - SENSOR_BASELINES['eco2_pct']['mu']) / SENSOR_BASELINES['eco2_pct']['sd']
    d['op_score_z'] = (d['op_score'] - SENSOR_BASELINES['op_score']['mu']) / SENSOR_BASELINES['op_score']['sd']
    d['op_z']       = d['op_score_z']

    # CO SpO2 paradox: eCO elevated but pulse-ox reads normal
    # (pulse ox cannot distinguish oxyHb from carboxyHb)
    d['spo2_paradox'] = ((d['eco_ppm'] > 6) & (d['spo2'] >= 95)).astype(int)

    # Haemodynamic derived
    d['map']            = d['dbp'] + (d['sbp'] - d['dbp']) / 3.0
    d['pulse_pressure'] = d['sbp'] - d['dbp']
    d['shock_index']    = d['hr'] / d['sbp'].clip(lower=1)
    d['brady_flag']     = (d['hr'] < 60).astype(int)
    d['tachy_flag']     = (d['hr'] > 100).astype(int)
    d['hypoxia_flag']   = (d['spo2'] < 94).astype(int)

    # Cholinergic syndrome score (muscarinic signs — core OP findings)
    cholinergic_cols = ['miosis', 'salivation', 'lacrimation',
                        'bronchospasm', 'bronchorrhea', 'diaphoresis']
    d['cholinergic_score'] = d[cholinergic_cols].sum(axis=1)

    # SLUDGE score (mnemonic for cholinergic toxidrome)
    sludge_cols = ['salivation', 'lacrimation', 'nausea', 'vomiting', 'diaphoresis']
    d['sludge_score'] = d[sludge_cols].sum(axis=1)

    # Weighted symptom burden across all 20 symptoms
    d['symptom_burden'] = sum(
        d[col] * w for col, w in SYMPTOM_WEIGHTS.items() if col in d.columns
    )

    # Cross-sensor ratios
    d['eno_eco_ratio'] = d['eno_ppb'] / (d['eco_ppm'] + 0.1)
    d['eno_x_rr']      = d['eno_ppb'] * d['rr']

    return d


def train():
    print(f"Loading {DATA_PATH} …")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    df = engineer_features(df)

    X = df[FEATURES]
    print(f"  Feature matrix: {X.shape}")

    # ── Agent classifier ───────────────────────────────────────────────────────
    agent_le = LabelEncoder()
    y_agent  = agent_le.fit_transform(df['agent'])
    print(f"\nAgent classes: {list(agent_le.classes_)}")

    X_tr, X_te, ya_tr, ya_te = train_test_split(
        X, y_agent, test_size=0.2, random_state=42, stratify=y_agent
    )

    agent_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1,
    )
    agent_model.fit(X_tr, ya_tr)

    ya_prob = agent_model.predict_proba(X_te)
    agent_auc = roc_auc_score(ya_te, ya_prob, multi_class='ovr', average='macro')
    print(f"  Agent AUC (macro OvR): {agent_auc:.4f}")

    # ── Severity classifier ────────────────────────────────────────────────────
    severity_le = LabelEncoder()
    y_sev       = severity_le.fit_transform(df['severity'])
    print(f"\nSeverity classes: {list(severity_le.classes_)}")

    Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(
        X, y_sev, test_size=0.2, random_state=42, stratify=y_sev
    )

    severity_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1,
    )
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

    # Update metadata
    meta_path = os.path.join(BASE_DIR, 'models', 'metadata.json')
    with open(meta_path, encoding='utf-8') as f:
        meta = json.load(f)
    meta['agent_macro_auc']    = round(agent_auc, 4)
    meta['severity_macro_auc'] = round(sev_auc, 4)
    meta['n_train'] = len(X_tr)
    meta['n_test']  = len(X_te)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata updated -> {meta_path}")
    print("\nTraining complete.")


if __name__ == '__main__':
    train()
