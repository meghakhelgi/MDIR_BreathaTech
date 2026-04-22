"""
BreathaTech Synthetic Data Generator v7
========================================
Monte Carlo simulation of patient presentations for:
  - CO (Carbon Monoxide)
  - OP (Organophosphate / Nerve Agent)
  - PHOSGENE
  - NONE (healthy baseline, including comorbidity mimics)

Clinical parameters grounded in:
  - CO: Hampson NB (2000); Weaver LK (2002)
  - OP: Peradeniya POP Scale; Merck Manual; Oxford QJM 2021
      - Tachycardia in 31.8-60%, bradycardia in 5.1-28% of OP cases
      - Muscarinic signs in 84%, nicotinic in 17%
  - Phosgene: Sciuto AM, PMC5457389 (2016); latent phase clinical data
  - COPD/asthma eNO baseline: Dweik RA AJRCCM 2011 (ATS eNO guidelines)

v7 realism enhancements vs v6:
  - Sensor measurement noise (multiplicative + additive) on every reading
  - Cross-sensor contamination: CO raises eNO via pulmonary inflammation;
    phosgene raises eCO via lung parenchymal damage; OP raises eCO2 via
    bronchoconstriction-driven CO2 retention
  - NONE comorbidities: COPD/asthma (elevated eNO), respiratory infection
    (elevated eNO + fever), urban ambient CO (elevated eCO), anxiety/
    hyperventilation (low eCO2) — creates realistic sensor-positive/label-
    negative cases that stress the classifier
  - Atypical presentations (10% per agent): partial or early clinical picture
  - Severity boundary blurring (15%): sensors shifted toward adjacent level,
    simulating patients caught mid-transition
  - Wider sensor SDs throughout; partial recovery cases (exposure > 60 min)

Severity grades: 1=Mild, 2=Moderate, 3=Severe
"""

import numpy as np
import pandas as pd
import random
import json
import os

np.random.seed(42)
random.seed(42)

# ── symptom synonym pools ─────────────────────────────────────────────────────
SYMPTOM_PHRASINGS = {
    "headache": [
        "headache", "head pain", "throbbing headache", "pounding in the head",
        "head is pounding", "cephalgia", "complains of headache",
        "reports significant head pain", "severe headache",
    ],
    "nausea": [
        "nausea", "nauseous", "feeling sick to stomach", "queasy",
        "reports nausea", "complains of nausea", "nauseated",
        "stomach upset with nausea", "feels like vomiting",
    ],
    "vomiting": [
        "vomiting", "emesis", "vomited x2", "actively vomiting",
        "repeated emesis", "has been vomiting", "thrown up multiple times",
        "vomitus noted", "emetic episode",
    ],
    "dizziness": [
        "dizziness", "dizzy", "lightheaded", "feeling lightheaded",
        "reports dizziness", "vertiginous", "room spinning",
        "unsteady on feet", "feels faint",
    ],
    "confusion": [
        "confusion", "confused", "altered mental status", "AMS",
        "disoriented", "not oriented to time and place",
        "difficulty answering questions", "obtunded", "encephalopathic",
        "cognitively impaired at presentation",
    ],
    "loss_of_consciousness": [
        "loss of consciousness", "LOC", "syncope", "unresponsive",
        "found unresponsive", "briefly unconscious", "passed out",
        "syncopal episode", "was unresponsive at scene",
    ],
    "chest_pain": [
        "chest pain", "chest tightness", "substernal chest pressure",
        "tight chest", "chest discomfort", "pressure in chest",
        "chest heaviness", "reports chest pain", "thoracic pain",
    ],
    "dyspnea": [
        "shortness of breath", "dyspnea", "difficulty breathing",
        "labored breathing", "SOB", "respiratory distress",
        "can't catch breath", "breathless", "air hunger",
    ],
    "diaphoresis": [
        "diaphoresis", "diaphoretic", "profuse sweating", "drenched in sweat",
        "excessive sweating", "soaked with perspiration",
        "notable diaphoresis", "heavily diaphoretic",
    ],
    "miosis": [
        "miosis", "pinpoint pupils", "bilateral miosis",
        "pupils constricted bilaterally", "markedly constricted pupils",
        "pupils 1mm bilaterally", "pinpoint pupils noted",
        "extreme pupillary constriction",
    ],
    "salivation": [
        "hypersalivation", "excessive salivation", "drooling",
        "profuse salivation", "sialorrhea", "excessive oral secretions",
        "drooling uncontrollably", "pooling of saliva",
    ],
    "lacrimation": [
        "lacrimation", "excessive tearing", "watery eyes",
        "eyes watering profusely", "marked lacrimation",
        "bilateral epiphora", "tearing from both eyes",
    ],
    "bronchospasm": [
        "bronchospasm", "wheezing", "audible wheeze",
        "bronchial constriction", "diffuse wheezing on auscultation",
        "tight airways", "wheeze noted throughout lung fields",
        "bronchorrhea and wheezing", "significant airway reactivity",
    ],
    "bronchorrhea": [
        "bronchorrhea", "excessive respiratory secretions",
        "copious airway secretions", "flooded airways",
        "significant bronchial hypersecretion", "frothy secretions from airway",
    ],
    "fasciculations": [
        "fasciculations", "muscle fasciculations", "muscle twitching",
        "involuntary muscle movements", "twitching noted in extremities",
        "visible muscle fasciculations", "myoclonic twitching",
        "fasciculating muscles noted on exam",
    ],
    "muscle_weakness": [
        "muscle weakness", "weakness", "proximal muscle weakness",
        "generalized weakness", "can't raise arms", "difficulty ambulating",
        "profound weakness", "extremity weakness noted",
        "unable to lift limbs against gravity",
    ],
    "seizure": [
        "seizure", "convulsions", "generalized tonic-clonic seizure",
        "witnessed seizure activity", "actively seizing",
        "postictal after seizure", "grand mal seizure", "convulsing",
    ],
    "cough": [
        "cough", "dry cough", "irritating cough", "persistent cough",
        "nonproductive cough", "mild cough noted", "complains of cough",
        "hacking cough", "coughing episodes",
    ],
    "eye_irritation": [
        "eye irritation", "burning eyes", "ocular irritation",
        "eyes burning", "stinging eyes", "ocular discomfort",
        "eyes feel irritated", "conjunctival irritation",
    ],
    "pulmonary_edema": [
        "pulmonary edema", "fluid in lungs", "crackles bilaterally",
        "bilateral basal crackles", "wet lung sounds",
        "rales throughout", "evidence of pulmonary edema",
        "pink frothy sputum", "flooded lung fields",
    ],
}


def pick_symptom(key, n=1):
    pool = SYMPTOM_PHRASINGS.get(key, [key])
    chosen = random.sample(pool, min(n, len(pool)))
    return "; ".join(chosen)

def maybe(p): return random.random() < p

def rng_int(lo, hi): return int(np.random.uniform(lo, hi + 1))

def _smoking_status():
    """
    Returns (is_smoker_true, is_smoker_recorded).

    In real triage settings a clinician under time pressure may not ask, or a
    patient may not disclose. ~35% of true smokers arrive with is_smoker=0 in
    the record. Physiology (exhaled CO floor) is driven by the true status;
    the model feature is driven by what was recorded — creating realistic
    CO/NONE ambiguity in undisclosed-smoker presentations.
    """
    true_smoker = maybe(0.20)
    reported    = true_smoker and maybe(0.65)   # ~35% non-disclosure rate
    return true_smoker, int(reported)

def rng_norm(mu, sd, lo=None, hi=None):
    v = np.random.normal(mu, sd)
    if lo is not None: v = max(lo, v)
    if hi is not None: v = min(hi, v)
    return round(v, 1)


# ── noise helpers ─────────────────────────────────────────────────────────────

def _noisy(val, abs_sd, lo=0, hi=None):
    """Add Gaussian measurement noise to a sensor reading."""
    v = val + np.random.normal(0, abs_sd)
    v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v

def _blur_toward(val, target, frac_lo=0.25, frac_hi=0.55):
    """Shift val a random fraction toward target (severity boundary blur)."""
    frac = np.random.uniform(frac_lo, frac_hi)
    return val + frac * (target - val)

def _physiological_baseline(is_smoker=False):
    """
    Endogenous sensor floor present in ALL patients regardless of exposure.

    eCO  — heme oxygenase produces CO during haem catabolism (~1–5 ppm
           non-smoker; ~12–20 ppm smoker). Source: Ryter SW AJP 2006.
    eNO  — constitutive eNOS/nNOS activity gives 5–25 ppb in healthy
           airways. Source: Kharitonov SA & Barnes PJ Eur Respir J 1996.
    eCO₂ — normal alveolar ventilation gives end-tidal CO₂ 3.8–4.6%.
    OP   — near-zero aptamer background; tiny cross-reactivity from dietary
           and environmental organophosphate residues (food, pesticides).
           Source: Lacasaña M Int J Environ Res Public Health 2010.
    """
    eco  = rng_norm(14.5 if is_smoker else 2.5,
                     2.2  if is_smoker else 1.5,
                     0.5,
                     22   if is_smoker else 8)
    eno  = rng_norm(15, 5.5, 5, 30)
    eco2 = rng_norm(4.1, 0.22, 3.6, 4.6)
    op   = rng_norm(1.5, 1.8, 0, 7)   # dietary/environmental cross-reactivity
    return eco, eno, eco2, op



# ── CO patient generator ──────────────────────────────────────────────────────
def make_co(severity):
    """
    CO poisoning presentation.
    Sources: Hampson NB Ann Emerg Med 2000; Weaver LK NEJM 2002.

    v7 realism:
    - eNO elevated above baseline: CO causes pulmonary inflammation,
      raising exhaled NO (but less dramatically than phosgene)
    - Wider eCO SDs — gray-zone presentations common in mild cases
    - 10% atypical: low eCO (patient partially recovered, removed from source)
    - 15% severity-blur: sensors shifted toward adjacent severity level
    - Sensor measurement noise on every reading
    - SpO2 more variable: some severe patients correctly hypoxic, some paradox
    """
    s = severity
    age  = rng_int(18, 75)
    sex  = random.choice(["M", "F"])
    is_smoker_true, is_smoker = _smoking_status()

    atypical      = maybe(0.22)   # partial recovery / brief exposure
    severity_blur = maybe(0.35)   # caught mid-transition between levels
    mopp_protected = maybe(0.15)

    hr   = rng_norm([78, 95, 115][s-1],  [14, 17, 20][s-1], 50, 180)
    sbp  = rng_norm([125, 118, 105][s-1],[15, 17, 20][s-1], 60, 200)
    dbp  = rng_norm([80, 75, 65][s-1],   [9, 11, 13][s-1],  40, 130)
    rr   = rng_norm([16, 20, 26][s-1],   [4, 5, 6][s-1],    8,  50)
    spo2 = rng_norm([97, 96, 95][s-1],   [2, 2.5, 3][s-1],  82, 100)
    temp = rng_norm(37.0, 0.5, 35.5, 38.8)

    eco_base, eno_base, eco2_base, op_base = _physiological_baseline(is_smoker_true)

    # CO exposure elevates eCO above the physiological floor
    eco_exposure = rng_norm([10, 32, 72][s-1], [12, 20, 30][s-1], 0, 292)
    if mopp_protected:
        # MOPP gear reduces absorbed CO to 20-50% of unprotected dose.
        # A true severity-3 patient in MOPP may look like severity-1 at the sensor.
        mopp_factor = rng_norm(0.32, 0.12, 0.15, 0.50)
        eco_exposure *= mopp_factor
    eco_ppm  = eco_base + eco_exposure

    # CO causes pulmonary inflammation → eNO rises above the physiological floor
    eno_inflam = rng_norm([3, 8, 17][s-1], [8, 12, 16][s-1], 0, 85)
    eno_ppb  = eno_base + eno_inflam

    # eCO₂ mildly elevated above physiological baseline
    eco2_pct = eco2_base + rng_norm([0.1, 0.4, 0.7][s-1], [0.30, 0.45, 0.60][s-1], 0, 3)

    # No OP signal — just the shared environmental/dietary floor
    op_score = op_base

    if atypical:
        # Full recovery or pre-symptomatic: sensor at baseline, vitals normalised.
        # These cases are genuinely indistinguishable from NONE at the sensor level.
        eco_ppm  = eco_base + rng_norm(0, 2.5, 0, 10)
        eno_ppb  = eno_base + rng_norm(0, 3.0, 0, 12)
        spo2     = rng_norm(98.0, 1.0, 94, 100)
        hr       = rng_norm(78, 12, 55, 105)
        sbp      = rng_norm(122, 12, 95, 155)

    if severity_blur and s > 1:
        eco_mild = eco_base + rng_norm([10, 10][s-2], [12, 12][s-2], 0, 292)
        eco_ppm  = _blur_toward(eco_ppm, eco_mild)

    # Measurement noise
    eco_ppm  = max(0, _noisy(eco_ppm,  abs_sd=6.5,  lo=0, hi=300))
    eno_ppb  = max(0, _noisy(eno_ppb,  abs_sd=7.0,  lo=0, hi=120))
    eco2_pct = max(0, _noisy(eco2_pct, abs_sd=0.28, lo=3.0, hi=8.0))
    op_score = max(0, _noisy(op_score, abs_sd=2.0,  lo=0, hi=20))

    # Time since exposure: CO accumulates HbCO over time — longer = worse.
    # ~20% of mild atypical cases: patient/clinician didn't know exposure started
    # (woke up with symptoms, came from poorly ventilated space).
    if s == 1 and atypical and maybe(0.20):
        texp = max(0, rng_norm(5, 6, 0, 20))   # unknown onset → near-zero
    else:
        texp = max(0, rng_norm([45, 90, 150][s-1], [30, 45, 60][s-1], 5, 480))

    symptoms = []
    symptom_flags = {}

    # Atypical = recovered/pre-symptomatic: sensor AND symptom picture near-NONE.
    sym_scale = 0.15 if atypical else 1.0

    if maybe([0.65, 0.90, 0.98][s-1] * sym_scale):
        symptoms.append(pick_symptom("headache")); symptom_flags["headache"] = 1
    else: symptom_flags["headache"] = 0

    if maybe([0.30, 0.65, 0.85][s-1] * sym_scale):
        symptoms.append(pick_symptom("nausea")); symptom_flags["nausea"] = 1
    else: symptom_flags["nausea"] = 0

    if maybe([0.10, 0.35, 0.65][s-1] * sym_scale):
        symptoms.append(pick_symptom("vomiting")); symptom_flags["vomiting"] = 1
    else: symptom_flags["vomiting"] = 0

    if maybe([0.40, 0.70, 0.85][s-1] * sym_scale):
        symptoms.append(pick_symptom("dizziness")); symptom_flags["dizziness"] = 1
    else: symptom_flags["dizziness"] = 0

    if maybe([0.05, 0.30, 0.80][s-1] * sym_scale):
        symptoms.append(pick_symptom("confusion")); symptom_flags["confusion"] = 1
    else: symptom_flags["confusion"] = 0

    if maybe([0.00, 0.05, 0.45][s-1] * sym_scale):
        symptoms.append(pick_symptom("loss_of_consciousness"))
        symptom_flags["loss_of_consciousness"] = 1
    else: symptom_flags["loss_of_consciousness"] = 0

    if maybe([0.10, 0.40, 0.70][s-1] * sym_scale):
        symptoms.append(pick_symptom("chest_pain")); symptom_flags["chest_pain"] = 1
    else: symptom_flags["chest_pain"] = 0

    if maybe([0.05, 0.20, 0.55][s-1] * sym_scale):
        symptoms.append(pick_symptom("dyspnea")); symptom_flags["dyspnea"] = 1
    else: symptom_flags["dyspnea"] = 0

    for k in ["miosis","salivation","lacrimation","bronchospasm","bronchorrhea",
              "fasciculations","muscle_weakness","seizure","diaphoresis",
              "cough","eye_irritation","pulmonary_edema"]:
        symptom_flags[k] = 0

    triage    = ["Monitor", "Escalate", "Immediate"][s-1]
    treatment = (
        "Remove from source; 100% O2 NRB mask; "
        + (["monitor HbCO q1h",
            "monitor ECG; consider HBO if HbCO>25% or neuro sx",
            "hyperbaric O2; cardiology consult; ICU admission"][s-1])
    )

    return dict(
        agent="CO", severity=s, triage=triage, age=age, sex=sex,
        is_smoker=int(is_smoker),
        hr=round(hr), sbp=round(sbp), dbp=round(dbp),
        rr=round(rr), spo2=round(spo2, 1), temp=round(temp, 1),
        eco_ppm=round(eco_ppm, 1), eno_ppb=round(eno_ppb, 1),
        eco2_pct=round(eco2_pct, 2), op_score=round(op_score, 1),
        time_since_exposure_min=round(texp, 1),
        symptom_text="; ".join(symptoms) if symptoms else "no complaints",
        treatment=treatment,
        **symptom_flags
    )


# ── OP / Nerve Agent patient generator ───────────────────────────────────────
def make_op(severity):
    """
    Organophosphate/nerve agent cholinergic toxidrome.
    Sources: Peradeniya POP Scale (StatPearls NBK470430);
             Oxford QJM 2021; Merck Manual.

    v7 realism:
    - OP score wider SDs — early presentations, partial decontamination,
      or sensor cross-reactivity produce more overlap with baseline
    - eNO elevated by bronchospasm-driven airway inflammation (NOT just sensor)
    - eCO₂ elevated via bronchoconstriction CO₂ retention
    - 10% atypical: incomplete SLUDGE picture (early or treated)
    - 8% early-presentation: OP score only modestly elevated (sensor catching
      before full enzyme inhibition)
    - 15% severity-blur; measurement noise on all sensors
    """
    s = severity
    age = rng_int(18, 75)
    sex = random.choice(["M", "F"])
    is_smoker_true, is_smoker = _smoking_status()

    atypical      = maybe(0.20)
    early_op      = maybe(0.20)   # very early — OP sensor catching pre-syndrome
    severity_blur = maybe(0.35)

    # Atropine pre-treatment: blocks muscarinic signs; nicotinic/CNS unaffected.
    atropine_pretreat = maybe(0.13)

    # Protective equipment reduces dermal + pulmonary OP uptake.
    mopp_protected = maybe(0.20)

    muscarinic_dominant = maybe(0.70)

    if atropine_pretreat:
        # Atropine abolishes muscarinic bradycardia → always tachycardic
        hr  = rng_norm([110, 128, 148][s-1], [16, 18, 20][s-1], 75, 185)
        sbp = rng_norm([130, 145, 158][s-1], [15, 17, 19][s-1], 85, 225)
        dbp = rng_norm([82,  92, 102][s-1],  [9, 11, 13][s-1],  50, 145)
    elif muscarinic_dominant:
        hr  = rng_norm([60, 50, 42][s-1], [12, 11, 9][s-1],  25, 90)
        sbp = rng_norm([115, 100, 80][s-1],[14, 15, 13][s-1], 50, 180)
        dbp = rng_norm([75, 65, 52][s-1], [9, 11, 9][s-1],   30, 110)
    else:
        hr  = rng_norm([95, 110, 130][s-1],[14, 15, 17][s-1], 60, 180)
        sbp = rng_norm([135, 150, 160][s-1],[15, 17, 19][s-1],90, 220)
        dbp = rng_norm([85, 95, 105][s-1], [9, 11, 13][s-1], 55, 140)

    rr   = rng_norm([18, 24, 34][s-1], [4, 6, 7][s-1],  4, 60)
    spo2 = rng_norm([96, 90, 80][s-1], [3, 5, 7][s-1],  55, 100)
    temp = rng_norm(37.1, 0.6, 35.5, 39.5)

    eco_base, eno_base, eco2_base, op_base = _physiological_baseline(is_smoker_true)

    # OP aptamer signal on top of the physiological/environmental floor.
    # Wider SDs reflect ~20-30% CV in RBC cholinesterase baseline and
    # time-to-sampling; early cases can barely clear the elevated noise floor.
    if early_op:
        # Pre-syndrome: aptamer barely above environmental floor, indistinguishable
        # from dietary organophosphate background at the sensor level.
        op_score = op_base + rng_norm(0, 3.0, 0, 10)
    else:
        op_score = op_base + rng_norm([17, 47, 81][s-1], [14, 20, 19][s-1], 0, 95)

    if mopp_protected:
        # Gear attenuates absorbed dose to 25-50% of unprotected equivalent.
        # A true severity-3 OP patient in MOPP may read like severity-1 on the sensor.
        mopp_factor = rng_norm(0.35, 0.10, 0.18, 0.55)
        op_score = op_base + (op_score - op_base) * mopp_factor
        eno_ppb_signal_scale = mopp_factor   # bronchospasm also attenuated
    else:
        eno_ppb_signal_scale = 1.0

    # eCO at physiological floor (smoker or non-smoker) — no CO exposure
    eco_ppm  = eco_base

    # eNO elevated above physiological floor: bronchospasm drives airway NO,
    # but far less than direct alveolar injury (phosgene). Keeps OP and PHOSGENE
    # clusters separated in sensor space.
    eno_signal = rng_norm([8, 16, 32][s-1], [10, 13, 18][s-1], 0, 85)
    eno_ppb  = eno_base + eno_signal * eno_ppb_signal_scale

    # eCO₂ elevated above physiological floor: CO₂ retention from bronchoconstriction
    eco2_pct = eco2_base + rng_norm([0.2, 0.7, 1.3][s-1], [0.45, 0.60, 0.75][s-1], 0, 5)

    if severity_blur and s > 1:
        op_mild  = op_base + rng_norm([17, 17][s-2], [14, 14][s-2], 0, 95)
        op_score = _blur_toward(op_score, op_mild)

    # Measurement noise
    op_score = max(0, _noisy(op_score, abs_sd=7.0,  lo=0, hi=100))
    eco_ppm  = max(0, _noisy(eco_ppm,  abs_sd=2.0,  lo=0, hi=20))
    eno_ppb  = max(0, _noisy(eno_ppb,  abs_sd=7.0,  lo=0, hi=155))
    eco2_pct = max(0, _noisy(eco2_pct, abs_sd=0.30, lo=3.0, hi=9.0))

    # Time since exposure: OP nerve agents act fast — short window for pralidoxime
    texp = max(0, rng_norm([15, 30, 55][s-1], [10, 15, 25][s-1], 2, 240))

    symptoms = []
    symptom_flags = {}

    # Two separate symptom scales:
    #   musc_scale — muscarinic signs: abolished by atropine pre-treatment
    #   nic_scale  — nicotinic/CNS signs: atropine has NO effect on these
    # Both are further attenuated by early/atypical presentation and MOPP.
    base_scale = 0.30 if (atypical or early_op) else 1.0
    mopp_sym   = mopp_factor if mopp_protected else 1.0
    musc_scale = min(0.05, base_scale) if atropine_pretreat else base_scale * mopp_sym
    nic_scale  = base_scale * mopp_sym

    if maybe([0.60, 0.85, 0.98][s-1] * musc_scale):
        symptoms.append(pick_symptom("miosis")); symptom_flags["miosis"] = 1
    else: symptom_flags["miosis"] = 0

    if maybe([0.45, 0.75, 0.95][s-1] * musc_scale):
        symptoms.append(pick_symptom("salivation")); symptom_flags["salivation"] = 1
    else: symptom_flags["salivation"] = 0

    if maybe([0.40, 0.70, 0.90][s-1] * musc_scale):
        symptoms.append(pick_symptom("lacrimation")); symptom_flags["lacrimation"] = 1
    else: symptom_flags["lacrimation"] = 0

    if maybe([0.35, 0.65, 0.90][s-1] * musc_scale):
        symptoms.append(pick_symptom("bronchospasm")); symptom_flags["bronchospasm"] = 1
    else: symptom_flags["bronchospasm"] = 0

    if maybe([0.20, 0.50, 0.80][s-1] * musc_scale):
        symptoms.append(pick_symptom("bronchorrhea")); symptom_flags["bronchorrhea"] = 1
    else: symptom_flags["bronchorrhea"] = 0

    if maybe([0.25, 0.55, 0.85][s-1] * musc_scale):
        symptoms.append(pick_symptom("diaphoresis")); symptom_flags["diaphoresis"] = 1
    else: symptom_flags["diaphoresis"] = 0

    if maybe([0.30, 0.60, 0.90][s-1] * nic_scale):
        symptoms.append(pick_symptom("nausea")); symptom_flags["nausea"] = 1
    else: symptom_flags["nausea"] = 0

    if maybe([0.20, 0.45, 0.70][s-1] * nic_scale):
        symptoms.append(pick_symptom("vomiting")); symptom_flags["vomiting"] = 1
    else: symptom_flags["vomiting"] = 0

    if maybe([0.40, 0.70, 0.92][s-1] * nic_scale):
        symptoms.append(pick_symptom("fasciculations")); symptom_flags["fasciculations"] = 1
    else: symptom_flags["fasciculations"] = 0

    if maybe([0.25, 0.55, 0.85][s-1] * nic_scale):
        symptoms.append(pick_symptom("muscle_weakness")); symptom_flags["muscle_weakness"] = 1
    else: symptom_flags["muscle_weakness"] = 0

    if maybe([0.00, 0.08, 0.50][s-1] * nic_scale):
        symptoms.append(pick_symptom("seizure")); symptom_flags["seizure"] = 1
    else: symptom_flags["seizure"] = 0

    if maybe([0.05, 0.20, 0.55][s-1] * nic_scale):
        symptoms.append(pick_symptom("confusion")); symptom_flags["confusion"] = 1
    else: symptom_flags["confusion"] = 0

    if maybe([0.00, 0.05, 0.35][s-1] * nic_scale):
        symptoms.append(pick_symptom("loss_of_consciousness"))
        symptom_flags["loss_of_consciousness"] = 1
    else: symptom_flags["loss_of_consciousness"] = 0

    if maybe([0.10, 0.25, 0.40][s-1] * nic_scale):
        symptoms.append(pick_symptom("dyspnea")); symptom_flags["dyspnea"] = 1
    else: symptom_flags["dyspnea"] = 0

    for k in ["headache","dizziness","chest_pain","cough","eye_irritation","pulmonary_edema"]:
        symptom_flags[k] = 0

    triage = ["Escalate", "Immediate", "Immediate"][s-1]
    treatment = (
        "Decontaminate; "
        + ["Atropine 2mg IV; observe for aging",
           "Atropine 4mg IV q5-10min titrate to secretions; Pralidoxime 1g IV NOW (before aging); Diazepam 5mg IV if seizing",
           "Atropine 6mg IV q5min; Pralidoxime 2g IV STAT; Diazepam 10mg IV; intubate; ICU"][s-1]
    )

    return dict(
        agent="OP", severity=s, triage=triage, age=age, sex=sex,
        is_smoker=int(is_smoker),
        hr=round(hr), sbp=round(sbp), dbp=round(dbp),
        rr=round(rr), spo2=round(spo2, 1), temp=round(temp, 1),
        eco_ppm=round(eco_ppm, 1), eno_ppb=round(eno_ppb, 1),
        eco2_pct=round(eco2_pct, 2), op_score=round(op_score, 1),
        time_since_exposure_min=round(texp, 1),
        symptom_text="; ".join(symptoms) if symptoms else "no complaints",
        treatment=treatment,
        **symptom_flags
    )


# ── Phosgene patient generator ────────────────────────────────────────────────
def make_phosgene(severity):
    """
    Phosgene (COCl2) presentation.
    Sources: Sciuto AM PMC5457389 (2016); NATO medical guidelines;
             WWI and industrial accident case series.

    v7 realism:
    - 12% very-early latent: eNO only mildly elevated (sensor catches pre-symptom)
    - Phosgene causes some eCO production via lung parenchymal oxidation
    - eNO and eCO2 wider SDs throughout
    - 10% atypical: symptoms that mimic CO (headache/dizziness dominant)
    - 15% severity-blur; measurement noise on all sensors
    """
    s = severity
    age = rng_int(18, 75)
    sex = random.choice(["M", "F"])
    is_smoker_true, is_smoker = _smoking_status()

    atypical      = maybe(0.20)
    very_early    = maybe(0.18)   # caught in first 15-30 min, eNO barely rising
    severity_blur = maybe(0.35)

    # MOPP protection: phosgene is a gas that MOPP gear effectively filters.
    mopp_protected = maybe(0.12)

    latent = (s == 1) or (s == 2 and maybe(0.55))

    if latent:
        idx = min(s - 1, 1)
        hr   = rng_norm([78, 84][idx],  [11, 13][idx], 55, 115)
        sbp  = rng_norm([122, 118][idx],[13, 15][idx], 88, 165)
        dbp  = rng_norm([80, 76][idx],  [9, 11][idx],  48, 108)
        rr   = rng_norm([15, 18][idx],  [3, 4][idx],   10, 30)
        spo2 = rng_norm([97, 95][idx],  [2, 2.5][idx], 85, 100)
    else:
        hr   = rng_norm([105, 120, 135][s-1],[15, 17, 19][s-1], 55, 180)
        sbp  = rng_norm([115, 100, 82][s-1], [15, 17, 15][s-1], 50, 180)
        dbp  = rng_norm([74, 64, 52][s-1],   [9, 11, 11][s-1],  30, 110)
        rr   = rng_norm([22, 30, 38][s-1],   [5, 6, 7][s-1],    10, 60)
        spo2 = rng_norm([94, 88, 78][s-1],   [4, 6, 8][s-1],    50, 100)

    temp = rng_norm(37.0, 0.6, 35.5, 39.2)

    eco_base, eno_base, eco2_base, op_base = _physiological_baseline(is_smoker_true)

    # eNO: phosgene-specific signal added above physiological floor.
    # Very-early cases barely clear the COPD/asthma eNO range — this is the
    # core clinical danger: sensor positive but only marginally.
    if very_early:
        # True latent phase: lung injury not yet producing measurable eNO.
        # Sensor at or below COPD baseline — clinically silent, the key danger.
        eno_signal = rng_norm(0, 4.0, 0, 15)
    else:
        # Phosgene causes direct alveolar injury → massive eNO from inflammatory
        # cascade, far stronger than bronchospasm-driven OP or blast lung eNO.
        eno_signal = rng_norm([55, 130, 250][s-1], [25, 40, 65][s-1], 0, 500)

    if mopp_protected:
        # MOPP filters phosgene effectively — absorbed dose 20-45% of unprotected.
        # A severity-3 patient in intact MOPP may present with eNO barely above
        # the COPD/infection baseline, making them look like mild phosgene or NONE.
        mopp_factor = rng_norm(0.30, 0.10, 0.15, 0.48)
        eno_signal *= mopp_factor
    else:
        mopp_factor = 1.0

    eno_ppb  = eno_base + eno_signal

    # eCO₂ elevated: dead-space physiology from alveolar flooding
    eco2_delta = rng_norm([0.7, 1.5, 2.8][s-1], [0.55, 0.85, 1.1][s-1], 0, 7)
    eco2_pct = eco2_base + eco2_delta * mopp_factor

    # Small eCO above physiological floor: phosgene reacts with lung tissue
    # producing trace oxidative CO (distinct from CO poisoning but non-zero)
    eco_ppm  = eco_base + rng_norm([0.5, 1.8, 3.5][s-1], [0.8, 1.4, 2.0][s-1], 0, 12)

    # No OP signal — just the shared environmental/dietary floor
    op_score = op_base

    if severity_blur and s > 1:
        eno_mild  = eno_base + rng_norm([55, 55][s-2], [25, 25][s-2], 0, 300)
        eno_ppb   = _blur_toward(eno_ppb, eno_mild)

    # Measurement noise
    eno_ppb  = max(0, _noisy(eno_ppb,  abs_sd=18.0, lo=0, hi=550))
    eco2_pct = max(0, _noisy(eco2_pct, abs_sd=0.32, lo=3.0, hi=12.0))
    eco_ppm  = max(0, _noisy(eco_ppm,  abs_sd=2.0,  lo=0, hi=22))
    op_score = max(0, _noisy(op_score, abs_sd=2.0,  lo=0, hi=20))

    # Time since exposure: phosgene has a long latent phase — patients arrive late
    texp = max(0, rng_norm([65, 130, 210][s-1], [35, 50, 70][s-1], 10, 600))

    symptoms = []
    symptom_flags = {}

    # very_early = true latent phase: no symptoms yet, patient feels well.
    # atypical = non-specific picture mimicking CO.
    # Both dramatically reduce the symptom picture available to the model.
    phsg_sym_scale = 0.10 if very_early else (0.35 if atypical else 1.0)

    if atypical and not very_early:
        # Mimics CO presentation — headache/dizziness dominant
        if maybe(0.70 * phsg_sym_scale): symptoms.append(pick_symptom("headache")); symptom_flags["headache"] = 1
        else: symptom_flags["headache"] = 0
        if maybe(0.55 * phsg_sym_scale): symptoms.append(pick_symptom("dizziness")); symptom_flags["dizziness"] = 1
        else: symptom_flags["dizziness"] = 0
        if maybe(0.30 * phsg_sym_scale): symptoms.append(pick_symptom("nausea")); symptom_flags["nausea"] = 1
        else: symptom_flags["nausea"] = 0
        symptom_flags.update({k: 0 for k in [
            "cough","eye_irritation","dyspnea","chest_pain","pulmonary_edema","confusion"]})
    elif very_early:
        # Truly asymptomatic — the latent phase danger. No clinical clues.
        symptom_flags.update({k: 0 for k in [
            "headache","dizziness","nausea","cough","eye_irritation",
            "dyspnea","chest_pain","pulmonary_edema","confusion"]})
    else:
        if maybe([0.15, 0.30, 0.55][s-1] if latent else [0.35, 0.60, 0.80][s-1]):
            symptoms.append(pick_symptom("cough")); symptom_flags["cough"] = 1
        else: symptom_flags["cough"] = 0

        if maybe([0.30, 0.45, 0.60][s-1]):
            symptoms.append(pick_symptom("eye_irritation")); symptom_flags["eye_irritation"] = 1
        else: symptom_flags["eye_irritation"] = 0

        if maybe([0.10, 0.25, 0.55][s-1]):
            symptoms.append(pick_symptom("nausea")); symptom_flags["nausea"] = 1
        else: symptom_flags["nausea"] = 0

        if maybe([0.05, 0.20, 0.55][s-1] if latent else [0.25, 0.65, 0.95][s-1]):
            symptoms.append(pick_symptom("dyspnea")); symptom_flags["dyspnea"] = 1
        else: symptom_flags["dyspnea"] = 0

        if maybe([0.05, 0.15, 0.40][s-1]):
            symptoms.append(pick_symptom("chest_pain")); symptom_flags["chest_pain"] = 1
        else: symptom_flags["chest_pain"] = 0

        if maybe([0.00, 0.10, 0.35][s-1]):
            symptoms.append(pick_symptom("headache")); symptom_flags["headache"] = 1
        else: symptom_flags["headache"] = 0

        if maybe([0.00, 0.05, 0.30][s-1] if latent else [0.05, 0.40, 0.80][s-1]):
            symptoms.append(pick_symptom("pulmonary_edema")); symptom_flags["pulmonary_edema"] = 1
        else: symptom_flags["pulmonary_edema"] = 0

        if maybe([0.00, 0.05, 0.20][s-1]):
            symptoms.append(pick_symptom("confusion")); symptom_flags["confusion"] = 1
        else: symptom_flags["confusion"] = 0

        symptom_flags.setdefault("dizziness", 0)

    for k in ["miosis","salivation","lacrimation","bronchospasm","bronchorrhea",
              "fasciculations","muscle_weakness","seizure","diaphoresis",
              "vomiting","loss_of_consciousness"]:
        symptom_flags.setdefault(k, 0)

    triage = ["Monitor", "Escalate", "Immediate"][s-1]
    treatment = (
        ["STRICT REST (exertion worsens edema); high-flow O2; eNO/eCO2 q1h; admit for 24h observation",
         "Strict rest; high-flow O2; NOS-2 inhibitor if available; ICU monitoring; anticipate edema",
         "ICU; mechanical ventilation likely; PEEP for pulmonary edema; vasopressors if hypotensive"][s-1]
    )

    return dict(
        agent="PHOSGENE", severity=s, triage=triage, age=age, sex=sex,
        is_smoker=int(is_smoker),
        hr=round(hr), sbp=round(sbp), dbp=round(dbp),
        rr=round(rr), spo2=round(spo2, 1), temp=round(temp, 1),
        eco_ppm=round(eco_ppm, 1), eno_ppb=round(eno_ppb, 1),
        eco2_pct=round(eco2_pct, 2), op_score=round(op_score, 1),
        time_since_exposure_min=round(texp, 1),
        symptom_text="; ".join(symptoms) if symptoms else "mild irritation only",
        treatment=treatment,
        **symptom_flags
    )


# ── Healthy baseline generator ────────────────────────────────────────────────
def make_none():
    """
    Healthy / unexposed patients.

    v7 comorbidity mimics — real-world reasons a sensor fires on a
    label-NONE patient:

      COPD / asthma (8%): eNO 25-70 ppb — overlaps with mild phosgene.
        Source: Dweik RA AJRCCM 2011; FeNO in COPD 20-40 ppb, severe asthma 50-100 ppb.

      Respiratory infection (6%): eNO 20-45 ppb + low-grade fever.
        Source: Kharitonov SA Thorax 1996.

      Urban ambient CO (9%): eCO 5-20 ppm from vehicle exhaust, indoor gas.
        Source: Hampson NB (2000): urban baseline 2-10 ppm in non-smokers.

      Anxiety / hyperventilation (5%): RR elevated, eCO2 suppressed.

      Otherwise-healthy smoker baseline already modeled via is_smoker flag.
    """
    age = rng_int(18, 75)
    sex = random.choice(["M", "F"])
    is_smoker_true, is_smoker = _smoking_status()

    # Base vitals
    hr   = rng_int(55, 95)
    sbp  = rng_int(100, 142)
    dbp  = rng_int(60, 90)
    rr   = rng_int(11, 18)
    spo2 = round(rng_norm(98.5, 1.2, 93, 100), 1)
    temp = round(rng_norm(37.0, 0.35, 36.0, 37.9), 1)

    # Healthy patients share the same physiological floor as exposed patients —
    # the comorbidities below then push individual sensors above that floor.
    eco_ppm, eno_ppb, eco2_pct, op_score = _physiological_baseline(is_smoker_true)

    # ── comorbidity modifiers ─────────────────────────────────────────────────
    # Military-context NONE patients: ~65% have at least one confounding factor.
    comorbidity = np.random.choice(
        ["none", "copd_asthma", "infection", "ambient_co", "anxiety",
         "blast_trauma"],
        p=[0.28, 0.22, 0.14, 0.18, 0.05, 0.13]
    )

    if comorbidity == "copd_asthma":
        # Severe asthma/COPD FeNO can reach 80-100 ppb, overlapping phosgene latent phase.
        eno_ppb  = rng_norm(42, 22, 15, 100)
        eco2_pct = rng_norm(4.5, 0.50, 3.8, 5.8)
        spo2     = round(rng_norm(94, 3.5, 82, 100), 1)
        rr       = rng_int(14, 28)

    elif comorbidity == "infection":
        # Respiratory infection — eNO overlaps with mild phosgene latent phase
        eno_ppb  = rng_norm(32, 14, 12, 72)
        temp     = round(rng_norm(38.3, 0.5, 37.5, 39.8), 1)
        hr       = rng_int(80, 118)
        rr       = rng_int(16, 28)

    elif comorbidity == "ambient_co":
        # Urban / indoor ambient CO — overlaps mild CO poisoning range
        eco_ppm  = rng_norm(16, 9, 3, 45)

    elif comorbidity == "anxiety":
        # Hyperventilation — suppressed eCO2 mimics early phosgene eCO2 pattern
        eco2_pct = rng_norm(3.4, 0.28, 2.9, 4.1)
        rr       = rng_int(20, 34)
        spo2     = round(rng_norm(99, 0.8, 96, 100), 1)

    elif comorbidity == "blast_trauma":
        # Blast/TBI co-injury — the most dangerous false-positive scenario.
        # Blast lung injury raises eNO indistinguishably from phosgene.
        # TBI gives confusion/LOC identical to severe OP or CO.
        # Hemorrhagic shock mimics severe OP/phosgene hemodynamics.
        # Source: DePalma RG et al. NEJM 2005; Champion HR et al. J Trauma 2003.
        eno_ppb  = eno_ppb + rng_norm(22, 12, 5, 62)   # blast lung — less than direct ALI
        eco2_pct = rng_norm(4.8, 0.5, 4.0, 6.2)         # impaired gas exchange
        eco_ppm  = eco_ppm + rng_norm(1.5, 1.2, 0, 6)   # oxidative trace
        hr       = rng_int(105, 158)                     # autonomic response
        sbp      = rng_int(62, 110)                      # hemorrhagic shock
        dbp      = rng_int(38, 68)
        rr       = rng_int(24, 42)
        spo2     = round(rng_norm(87, 6, 66, 97), 1)
        temp     = round(rng_norm(36.4, 0.8, 33.5, 38.5), 1)  # hypothermia risk

    # Measurement noise
    eco_ppm  = max(0, _noisy(eco_ppm,  abs_sd=4.0, lo=0, hi=38))
    eno_ppb  = max(0, _noisy(eno_ppb,  abs_sd=7.0, lo=0, hi=115))
    eco2_pct = max(0, _noisy(eco2_pct, abs_sd=0.28, lo=3.0, hi=7.0))
    op_score = max(0, _noisy(op_score, abs_sd=3.5, lo=0, hi=30))

    # Time since (non-)exposure.
    # Most NONE patients correctly get texp=0.
    # ~15% in mass-casualty screening: clinician enters a precautionary texp.
    # Blast trauma patients: texp may reflect time since the blast, not a chemical
    # exposure — clinician enters it as a generic "time since incident."
    if comorbidity == "blast_trauma":
        texp_none = max(0, rng_norm(20, 12, 2, 60))
    elif maybe(0.15):
        texp_none = max(0, rng_norm(25, 15, 5, 60))
    else:
        texp_none = 0

    # ── symptoms ─────────────────────────────────────────────────────────────
    all_sym_keys = [
        "headache","nausea","vomiting","dizziness","confusion",
        "loss_of_consciousness","chest_pain","dyspnea","diaphoresis",
        "miosis","salivation","lacrimation","bronchospasm","bronchorrhea",
        "fasciculations","muscle_weakness","seizure",
        "cough","eye_irritation","pulmonary_edema",
    ]
    sflag = {k: 0 for k in all_sym_keys}
    sym_text_parts = []

    if comorbidity == "blast_trauma":
        # TBI → neurological; blast lung → respiratory. Pattern overlaps with
        # severe OP (confusion, LOC, dyspnea) and severe phosgene (dyspnea,
        # chest pain, pulmonary edema). This is the worst false-positive case.
        for key, p in [("confusion", 0.72), ("loss_of_consciousness", 0.42),
                       ("dyspnea", 0.88), ("chest_pain", 0.65),
                       ("headache", 0.70), ("nausea", 0.40)]:
            if maybe(p):
                sflag[key] = 1
                sym_text_parts.append(pick_symptom(key))

    symptom_text = "; ".join(sym_text_parts) if sym_text_parts else "no complaints"

    return dict(
        agent="NONE", severity=0, triage="Clear", age=age, sex=sex,
        is_smoker=int(is_smoker),
        hr=hr, sbp=sbp, dbp=dbp, rr=rr, spo2=spo2, temp=temp,
        eco_ppm=round(eco_ppm, 1), eno_ppb=round(eno_ppb, 1),
        eco2_pct=round(eco2_pct, 2), op_score=round(op_score, 1),
        time_since_exposure_min=round(texp_none, 1),
        symptom_text=symptom_text,
        treatment="No treatment required",
        **sflag
    )


# ── Monte Carlo simulation ────────────────────────────────────────────────────
def generate(n_per_class=600, severity_dist=None):
    if severity_dist is None:
        severity_dist = [0.62, 0.27, 0.11]

    rows = []
    generators = {"CO": make_co, "OP": make_op, "PHOSGENE": make_phosgene}

    for agent, gen in generators.items():
        n_mild     = int(n_per_class * severity_dist[0])
        n_moderate = int(n_per_class * severity_dist[1])
        n_severe   = n_per_class - n_mild - n_moderate
        for sev, n in [(1, n_mild), (2, n_moderate), (3, n_severe)]:
            for _ in range(n):
                rows.append(gen(sev))

    for _ in range(n_per_class):
        rows.append(make_none())

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.insert(0, "patient_id", [f"SIM{i:05d}" for i in range(len(df))])
    return df


if __name__ == "__main__":
    df = generate(n_per_class=600)
    print(f"Generated {len(df)} patients")
    print(df["agent"].value_counts().to_string())
    print(df["triage"].value_counts().to_string())
    print("\nSensor means by agent:")
    for ag in ["NONE","CO","OP","PHOSGENE"]:
        s = df[df.agent==ag]
        print(f"  {ag:10s} eCO={s.eco_ppm.mean():.1f} eNO={s.eno_ppb.mean():.1f} "
              f"eco2={s.eco2_pct.mean():.2f} OP={s.op_score.mean():.1f} "
              f"SpO2={s.spo2.mean():.1f}")
    print("\nSample symptom text (3 per agent):")
    for ag in ["CO","OP","PHOSGENE"]:
        print(f"\n  [{ag}]")
        for txt in df[df.agent==ag]["symptom_text"].sample(3, random_state=1):
            print(f"    '{txt}'")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'backend', 'data', 'training_data.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved -> {out}")
