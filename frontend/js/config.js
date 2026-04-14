// config.js — shared constants used by all other modules

const Config = {

  // ── API ────────────────────────────────────────────────────────────────
  // Change this to your actual FastAPI server URL when running locally.
  // Default: http://localhost:8000 (uvicorn main)
  API_BASE: 'http://localhost:8000',

  // ── agent display ──────────────────────────────────────────────────────
  AGENT_NAMES: {
    NONE:     'No exposure detected',
    CO:       'Carbon Monoxide (CO)',
    OP:       'Nerve Agent — Organophosphate',
    PHOSGENE: 'Phosgene — eNO signal',
  },

  AGENT_COLORS: {
    NONE:     '#94a3b8',
    CO:       '#1d6fb8',
    OP:       '#dc2626',
    PHOSGENE: '#d97706',
  },

  AGENT_PILL_CLASS: {
    NONE:     'pill-none',
    CO:       'pill-chem',
    OP:       'pill-op',
    PHOSGENE: 'pill-chem',
  },

  AGENT_PILL_TEXT: {
    NONE:     'Clear',
    CO:       'Chemical',
    OP:       'Nerve agent',
    PHOSGENE: 'Chemical',
  },

  SEVERITY_LABELS: {
    0: 'No exposure',
    1: 'Mild',
    2: 'Moderate',
    3: 'Severe',
  },

  TRIAGE_COLORS: {
    Clear:     '#059669',
    Monitor:   '#d97706',
    Escalate:  '#ea580c',
    Immediate: '#dc2626',
  },

  // ── sensor definitions ─────────────────────────────────────────────────
  SENSORS: [
    { key: 'eco_ppm',  label: 'eCO',      unit: 'ppm',   ref: '<6 ppm',   warnAt: 6,  alertAt: 20, max: 80,  tag: 'Chemical',        tagClass: 'chem' },
    { key: 'eno_ppb',  label: 'eNO',      unit: 'ppb',   ref: '<25 ppb',  warnAt: 25, alertAt: 50, max: 120, tag: 'Chemical',        tagClass: 'chem' },
    { key: 'eco2_pct', label: 'eCO₂',     unit: '%',     ref: '4.0–4.5%', warnAt: 4.8, alertAt: 5.5, max: 7, tag: 'Chemical',        tagClass: 'chem' },
    { key: 'op_score', label: 'OP Score', unit: 'score', ref: '<5',       warnAt: 10, alertAt: 35, max: 100, tag: 'Organophosphate', tagClass: 'op', isOP: true },
  ],

  // ── symptom definitions ────────────────────────────────────────────────
  SYMS_OP: [
    { k: 'miosis',         label: 'Miosis',         sub: 'Pinpoint pupils',           weight: 2 },
    { k: 'salivation',     label: 'Hypersalivation', sub: 'Excessive secretions',      weight: 1 },
    { k: 'lacrimation',    label: 'Lacrimation',     sub: 'Excessive tearing',         weight: 1 },
    { k: 'bronchospasm',   label: 'Bronchospasm',    sub: 'Wheezing / airway',         weight: 2 },
    { k: 'bronchorrhea',   label: 'Bronchorrhea',    sub: 'Copious airway fluid',      weight: 2 },
    { k: 'fasciculations', label: 'Fasciculations',  sub: 'Muscle twitching',          weight: 2 },
    { k: 'muscle_weakness',label: 'Muscle weakness', sub: 'Proximal / generalized',    weight: 1 },
    { k: 'diaphoresis',    label: 'Diaphoresis',     sub: 'Profuse sweating',          weight: 1 },
  ],

  SYMS_PH: [
    { k: 'cough',          label: 'Cough',           sub: 'Dry or irritating',         weight: 1 },
    { k: 'eye_irritation', label: 'Eye irritation',  sub: 'Burning / stinging',        weight: 1 },
    { k: 'pulmonary_edema',label: 'Pulmonary edema', sub: 'Crackles / frothy sputum',  weight: 3 },
    { k: 'dyspnea',        label: 'Dyspnea',         sub: 'Difficulty breathing',      weight: 2 },
  ],

  SYMS_GEN: [
    { k: 'headache',               label: 'Headache',            sub: 'Weight ×1', weight: 1 },
    { k: 'nausea',                 label: 'Nausea',              sub: 'Weight ×1', weight: 1 },
    { k: 'vomiting',               label: 'Vomiting',            sub: 'Weight ×1', weight: 1 },
    { k: 'dizziness',              label: 'Dizziness',           sub: 'Weight ×1', weight: 1 },
    { k: 'chest_pain',             label: 'Chest pain',          sub: 'Weight ×2', weight: 2 },
    { k: 'confusion',              label: 'Confusion / AMS',     sub: 'Weight ×3', weight: 3 },
    { k: 'seizure',                label: 'Seizure',             sub: 'Weight ×4', weight: 4 },
    { k: 'loss_of_consciousness',  label: 'Loss of consciousness', sub: 'Weight ×4', weight: 4 },
  ],

  // sensor baseline for alarm score calculation (matches inference.py)
  SENSOR_BASELINES: {
    eco_ppm:  { mu: 1.26, sd: 5.0 },
    eno_ppb:  { mu: 15.0, sd: 8.0 },
    eco2_pct: { mu:  4.1, sd: 0.3 },
    op_score: { mu:  2.0, sd: 2.0 },
  },
};
