"""
BreathaTech Synthetic Data Generator v6
========================================
Monte Carlo simulation of patient presentations for:
  - CO (Carbon Monoxide)
  - OP (Organophosphate / Nerve Agent)
  - PHOSGENE

Clinical parameters grounded in:
  - CO: Hampson NB (2000); Weaver LK (2002)
  - OP: Peradeniya POP Scale; Merck Manual; Oxford QJM 2021
      - Tachycardia in 31.8-60%, bradycardia in 5.1-28% of OP cases
      - Muscarinic signs in 84%, nicotinic in 17%
  - Phosgene: Sciuto AM, PMC5457389 (2016); latent phase clinical data

Key design decision: symptom_text field generates NATURALISTIC free-text
symptom descriptions using synonym pools so the same clinical finding is
described in multiple ways — enabling NLP-based feature extraction and
testing classifier robustness to clinical language variation.

Severity grades: 1=Mild, 2=Moderate, 3=Severe
Treatment: encoded as a string for rule-based overlay
"""

import numpy as np
import pandas as pd
import random
import json
import os

np.random.seed(42)
random.seed(42)

# ── symptom synonym pools ─────────────────────────────────────────────────────
# Each pool contains naturalistic phrasings of the same clinical finding.
# Multiple phrasings sampled per patient → realistic variation in clinical notes.

SYMPTOM_PHRASINGS = {
    # General
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
    # OP-specific
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
    # Phosgene-specific
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
    """Randomly pick n phrasings from a symptom pool, joined with '; '."""
    pool = SYMPTOM_PHRASINGS.get(key, [key])
    chosen = random.sample(pool, min(n, len(pool)))
    return "; ".join(chosen)

def maybe(p): return random.random() < p

def rng_int(lo, hi): return int(np.random.uniform(lo, hi + 1))

def rng_norm(mu, sd, lo=None, hi=None):
    v = np.random.normal(mu, sd)
    if lo is not None: v = max(lo, v)
    if hi is not None: v = min(hi, v)
    return round(v, 1)


# ── CO patient generator ──────────────────────────────────────────────────────
def make_co(severity):
    """
    CO poisoning presentation.
    Sources: Hampson NB Ann Emerg Med 2000; Weaver LK NEJM 2002.

    Key clinical features:
    - Headache: most common symptom (64% mild, near-universal severe)
    - SpO2 paradox: pulse ox falsely normal (reads HbCO as HbO2)
    - Tachycardia: compensatory response to tissue hypoxia
    - Mental status: ranges from confusion to coma with severity
    - ECG: ischemic changes in severe cases
    """
    s = severity  # 1=Mild, 2=Moderate, 3=Severe
    age = rng_int(18, 75)
    sex = random.choice(["M", "F"])
    is_smoker = maybe(0.20)

    # Vitals — tachycardia predominates; SpO2 paradox
    hr   = rng_norm([78, 95, 115][s-1], [12, 15, 18][s-1], 50, 180)
    sbp  = rng_norm([125, 118, 105][s-1], [14, 16, 18][s-1], 60, 200)
    dbp  = rng_norm([80, 75, 65][s-1], [8, 10, 12][s-1], 40, 130)
    rr   = rng_norm([16, 20, 26][s-1], [3, 4, 5][s-1], 8, 50)
    # SpO2 paradox: pulse ox reads falsely normal even in severe CO
    spo2 = rng_norm([97, 96, 95][s-1], [1.5, 2, 2.5][s-1], 88, 100)
    temp = rng_norm(37.0, 0.4, 35.5, 38.5)

    # Breath sensors
    eco_ppm   = rng_norm([12, 35, 75][s-1], [4, 10, 18][s-1], 1, 300)
    eno_ppb   = rng_norm([16, 20, 25][s-1], [3, 4, 5][s-1], 5, 60)
    eco2_pct  = rng_norm([4.2, 4.4, 4.7][s-1], [0.2, 0.2, 0.3][s-1], 3.5, 7)
    op_score  = rng_norm(2.0, 1.0, 0, 8)  # near zero

    # Symptoms with naturalistic text
    symptoms = []
    symptom_flags = {}

    if maybe([0.65, 0.90, 0.98][s-1]):
        symptoms.append(pick_symptom("headache"))
        symptom_flags["headache"] = 1
    else: symptom_flags["headache"] = 0

    if maybe([0.30, 0.65, 0.85][s-1]):
        symptoms.append(pick_symptom("nausea"))
        symptom_flags["nausea"] = 1
    else: symptom_flags["nausea"] = 0

    if maybe([0.10, 0.35, 0.65][s-1]):
        symptoms.append(pick_symptom("vomiting"))
        symptom_flags["vomiting"] = 1
    else: symptom_flags["vomiting"] = 0

    if maybe([0.40, 0.70, 0.85][s-1]):
        symptoms.append(pick_symptom("dizziness"))
        symptom_flags["dizziness"] = 1
    else: symptom_flags["dizziness"] = 0

    if maybe([0.05, 0.30, 0.80][s-1]):
        symptoms.append(pick_symptom("confusion"))
        symptom_flags["confusion"] = 1
    else: symptom_flags["confusion"] = 0

    if maybe([0.00, 0.05, 0.45][s-1]):
        symptoms.append(pick_symptom("loss_of_consciousness"))
        symptom_flags["loss_of_consciousness"] = 1
    else: symptom_flags["loss_of_consciousness"] = 0

    if maybe([0.10, 0.40, 0.70][s-1]):
        symptoms.append(pick_symptom("chest_pain"))
        symptom_flags["chest_pain"] = 1
    else: symptom_flags["chest_pain"] = 0

    if maybe([0.05, 0.20, 0.55][s-1]):
        symptoms.append(pick_symptom("dyspnea"))
        symptom_flags["dyspnea"] = 1
    else: symptom_flags["dyspnea"] = 0

    # Zero-fill OP/phosgene flags
    for k in ["miosis","salivation","lacrimation","bronchospasm","bronchorrhea",
              "fasciculations","muscle_weakness","seizure","diaphoresis",
              "cough","eye_irritation","pulmonary_edema"]:
        symptom_flags[k] = 0

    triage = ["Monitor", "Escalate", "Immediate"][s-1]
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
        rr=round(rr), spo2=round(spo2,1), temp=round(temp,1),
        eco_ppm=round(eco_ppm,1), eno_ppb=round(eno_ppb,1),
        eco2_pct=round(eco2_pct,2), op_score=round(op_score,1),
        symptom_text="; ".join(symptoms) if symptoms else "no complaints",
        treatment=treatment,
        **symptom_flags
    )


# ── OP / Nerve Agent patient generator ───────────────────────────────────────
def make_op(severity):
    """
    Organophosphate/nerve agent cholinergic toxidrome.
    Sources: Peradeniya POP Scale (StatPearls NBK470430);
             Oxford QJM 2021 vital sign prevalence data;
             Merck Manual OP chapter.

    Key clinical notes:
    - Vital signs MIXED: tachycardia 31.8-60% vs bradycardia 5.1-28%
      (nicotinic vs muscarinic competition)
    - Muscarinic signs dominate (84% of cases): SLUDGE/DUMBELS
    - Nicotinic signs in ~17%: fasciculations, tachycardia, hypertension
    - POP scale: miosis + fasciculations + respiration + bradycardia + LOC + seizures
    - SpO2 falls due to bronchospasm (real hypoxia, NOT paradox like CO)
    - OP score elevated (aptamer signal)
    """
    s = severity
    age = rng_int(18, 75)
    sex = random.choice(["M", "F"])
    is_smoker = maybe(0.20)

    # Nicotinic vs muscarinic dominance (affects HR/BP direction)
    muscarinic_dominant = maybe(0.70)  # 70% muscarinic > nicotinic

    if muscarinic_dominant:
        # Bradycardia, hypotension
        hr  = rng_norm([60, 50, 42][s-1], [10, 10, 8][s-1], 25, 90)
        sbp = rng_norm([115, 100, 80][s-1], [12, 14, 12][s-1], 50, 180)
        dbp = rng_norm([75, 65, 52][s-1], [8, 10, 8][s-1], 30, 110)
    else:
        # Tachycardia, hypertension (nicotinic)
        hr  = rng_norm([95, 110, 130][s-1], [12, 14, 16][s-1], 60, 180)
        sbp = rng_norm([135, 150, 160][s-1], [14, 16, 18][s-1], 90, 220)
        dbp = rng_norm([85, 95, 105][s-1], [8, 10, 12][s-1], 55, 140)

    rr   = rng_norm([18, 24, 34][s-1], [3, 5, 6][s-1], 4, 60)
    spo2 = rng_norm([96, 90, 80][s-1], [2, 4, 6][s-1], 60, 100)
    temp = rng_norm(37.1, 0.5, 35.5, 39.5)

    # Breath sensors — OP score is PRIMARY discriminator
    op_score  = rng_norm([18, 48, 82][s-1], [6, 10, 10][s-1], 0, 100)
    eco_ppm   = rng_norm([1.5, 1.8, 2.2][s-1], [0.8, 1.0, 1.2][s-1], 0, 10)  # near baseline
    eno_ppb   = rng_norm([22, 38, 62][s-1], [5, 8, 10][s-1], 5, 120)
    eco2_pct  = rng_norm([4.2, 4.6, 5.2][s-1], [0.2, 0.3, 0.4][s-1], 3, 8)

    symptoms = []
    symptom_flags = {}

    # Muscarinic symptoms (SLUDGE/DUMBELS)
    if maybe([0.60, 0.85, 0.98][s-1]):
        symptoms.append(pick_symptom("miosis"))
        symptom_flags["miosis"] = 1
    else: symptom_flags["miosis"] = 0

    if maybe([0.45, 0.75, 0.95][s-1]):
        symptoms.append(pick_symptom("salivation"))
        symptom_flags["salivation"] = 1
    else: symptom_flags["salivation"] = 0

    if maybe([0.40, 0.70, 0.90][s-1]):
        symptoms.append(pick_symptom("lacrimation"))
        symptom_flags["lacrimation"] = 1
    else: symptom_flags["lacrimation"] = 0

    if maybe([0.35, 0.65, 0.90][s-1]):
        symptoms.append(pick_symptom("bronchospasm"))
        symptom_flags["bronchospasm"] = 1
    else: symptom_flags["bronchospasm"] = 0

    if maybe([0.20, 0.50, 0.80][s-1]):
        symptoms.append(pick_symptom("bronchorrhea"))
        symptom_flags["bronchorrhea"] = 1
    else: symptom_flags["bronchorrhea"] = 0

    if maybe([0.25, 0.55, 0.85][s-1]):
        symptoms.append(pick_symptom("diaphoresis"))
        symptom_flags["diaphoresis"] = 1
    else: symptom_flags["diaphoresis"] = 0

    if maybe([0.30, 0.60, 0.90][s-1]):
        symptoms.append(pick_symptom("nausea"))
        symptom_flags["nausea"] = 1
    else: symptom_flags["nausea"] = 0

    if maybe([0.20, 0.45, 0.70][s-1]):
        symptoms.append(pick_symptom("vomiting"))
        symptom_flags["vomiting"] = 1
    else: symptom_flags["vomiting"] = 0

    # Nicotinic symptoms
    if maybe([0.40, 0.70, 0.92][s-1]):
        symptoms.append(pick_symptom("fasciculations"))
        symptom_flags["fasciculations"] = 1
    else: symptom_flags["fasciculations"] = 0

    if maybe([0.25, 0.55, 0.85][s-1]):
        symptoms.append(pick_symptom("muscle_weakness"))
        symptom_flags["muscle_weakness"] = 1
    else: symptom_flags["muscle_weakness"] = 0

    if maybe([0.00, 0.08, 0.50][s-1]):
        symptoms.append(pick_symptom("seizure"))
        symptom_flags["seizure"] = 1
    else: symptom_flags["seizure"] = 0

    if maybe([0.05, 0.20, 0.55][s-1]):
        symptoms.append(pick_symptom("confusion"))
        symptom_flags["confusion"] = 1
    else: symptom_flags["confusion"] = 0

    if maybe([0.00, 0.05, 0.35][s-1]):
        symptoms.append(pick_symptom("loss_of_consciousness"))
        symptom_flags["loss_of_consciousness"] = 1
    else: symptom_flags["loss_of_consciousness"] = 0

    if maybe([0.10, 0.25, 0.40][s-1]):
        symptoms.append(pick_symptom("dyspnea"))
        symptom_flags["dyspnea"] = 1
    else: symptom_flags["dyspnea"] = 0

    # Zero-fill CO/phosgene-specific flags
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
        rr=round(rr), spo2=round(spo2,1), temp=round(temp,1),
        eco_ppm=round(eco_ppm,1), eno_ppb=round(eno_ppb,1),
        eco2_pct=round(eco2_pct,2), op_score=round(op_score,1),
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

    Key clinical notes:
    - LATENT PHASE (severity 1-2): patient may feel nearly well
      yet have elevated eNO indicating significant lung injury
    - eNO is the PRIMARY early sensor signal (rises 30-60min post-exposure)
    - eCO2 also elevated early
    - SpO2 begins to fall as edema develops
    - Pulmonary edema is the lethal endpoint — can be 4-24h post-exposure
    - Clinical severity UNDERESTIMATED by patient appearance in early phase
    """
    s = severity
    age = rng_int(18, 75)
    sex = random.choice(["M", "F"])
    is_smoker = maybe(0.20)

    # Latent vs edematous phase — severity 1-2 often still in latent phase
    latent = (s == 1) or (s == 2 and maybe(0.55))

    if latent:
        hr  = rng_norm([78, 84][s-1 if s <= 2 else 1], [10, 12][s-1 if s <= 2 else 1], 55, 110)
        sbp = rng_norm([122, 118][s-1 if s <= 2 else 1], [12, 14][s-1 if s <= 2 else 1], 90, 160)
        dbp = rng_norm([80, 76][s-1 if s <= 2 else 1], [8, 10][s-1 if s <= 2 else 1], 50, 105)
        rr  = rng_norm([15, 18][s-1 if s <= 2 else 1], [2, 3][s-1 if s <= 2 else 1], 10, 28)
        spo2= rng_norm([97, 95][s-1 if s <= 2 else 1], [1.5, 2.0][s-1 if s <= 2 else 1], 88, 100)
    else:
        # Edematous phase — deteriorating
        hr  = rng_norm([105, 120, 135][s-1], [14, 16, 18][s-1], 55, 180)
        sbp = rng_norm([115, 100, 82][s-1], [14, 16, 14][s-1], 50, 180)
        dbp = rng_norm([74, 64, 52][s-1], [8, 10, 10][s-1], 30, 110)
        rr  = rng_norm([22, 30, 38][s-1], [4, 5, 6][s-1], 10, 60)
        spo2= rng_norm([94, 88, 78][s-1], [3, 5, 7][s-1], 55, 100)

    temp = rng_norm(37.0, 0.5, 35.5, 39.0)

    # eNO is the critical early signal — elevated even in latent/asymptomatic
    eno_ppb   = rng_norm([30, 52, 82][s-1], [6, 10, 14][s-1], 5, 150)
    eco2_pct  = rng_norm([4.6, 5.0, 5.6][s-1], [0.3, 0.4, 0.5][s-1], 3.5, 9)
    eco_ppm   = rng_norm([1.4, 1.6, 1.9][s-1], [0.6, 0.8, 1.0][s-1], 0, 8)  # near baseline
    op_score  = rng_norm(2.0, 1.0, 0, 8)  # near baseline

    symptoms = []
    symptom_flags = {}

    # Early symptoms (may be minimal in latent phase!)
    if maybe([0.15, 0.30, 0.55][s-1] if latent else [0.35, 0.60, 0.80][s-1]):
        symptoms.append(pick_symptom("cough"))
        symptom_flags["cough"] = 1
    else: symptom_flags["cough"] = 0

    if maybe([0.30, 0.45, 0.60][s-1]):
        symptoms.append(pick_symptom("eye_irritation"))
        symptom_flags["eye_irritation"] = 1
    else: symptom_flags["eye_irritation"] = 0

    if maybe([0.10, 0.25, 0.55][s-1]):
        symptoms.append(pick_symptom("nausea"))
        symptom_flags["nausea"] = 1
    else: symptom_flags["nausea"] = 0

    if maybe([0.05, 0.20, 0.55][s-1] if latent else [0.25, 0.65, 0.95][s-1]):
        symptoms.append(pick_symptom("dyspnea"))
        symptom_flags["dyspnea"] = 1
    else: symptom_flags["dyspnea"] = 0

    if maybe([0.05, 0.15, 0.40][s-1]):
        symptoms.append(pick_symptom("chest_pain"))
        symptom_flags["chest_pain"] = 1
    else: symptom_flags["chest_pain"] = 0

    if maybe([0.00, 0.10, 0.35][s-1]):
        symptoms.append(pick_symptom("headache"))
        symptom_flags["headache"] = 1
    else: symptom_flags["headache"] = 0

    if maybe([0.00, 0.05, 0.30][s-1] if latent else [0.05, 0.40, 0.80][s-1]):
        symptoms.append(pick_symptom("pulmonary_edema"))
        symptom_flags["pulmonary_edema"] = 1
    else: symptom_flags["pulmonary_edema"] = 0

    if maybe([0.00, 0.05, 0.20][s-1]):
        symptoms.append(pick_symptom("confusion"))
        symptom_flags["confusion"] = 1
    else: symptom_flags["confusion"] = 0

    # Zero-fill OP-specific flags
    for k in ["miosis","salivation","lacrimation","bronchospasm","bronchorrhea",
              "fasciculations","muscle_weakness","seizure","diaphoresis",
              "dizziness","vomiting","loss_of_consciousness"]:
        symptom_flags[k] = 0

    triage = ["Monitor", "Escalate", "Immediate"][s-1]
    # Note: even Monitor-level phosgene needs strict rest — explicitly documented
    treatment = (
        ["STRICT REST (exertion worsens edema); high-flow O2; eNO/eCO2 q1h; admit for 24h observation",
         "Strict rest; high-flow O2; NOS-2 inhibitor if available; ICU monitoring; anticipate edema",
         "ICU; mechanical ventilation likely; PEEP for pulmonary edema; vasopressors if hypotensive"][s-1]
    )

    return dict(
        agent="PHOSGENE", severity=s, triage=triage, age=age, sex=sex,
        is_smoker=int(is_smoker),
        hr=round(hr), sbp=round(sbp), dbp=round(dbp),
        rr=round(rr), spo2=round(spo2,1), temp=round(temp,1),
        eco_ppm=round(eco_ppm,1), eno_ppb=round(eno_ppb,1),
        eco2_pct=round(eco2_pct,2), op_score=round(op_score,1),
        symptom_text="; ".join(symptoms) if symptoms else "mild irritation only",
        treatment=treatment,
        **symptom_flags
    )


# ── Healthy baseline generator ────────────────────────────────────────────────
def make_none():
    age = rng_int(18, 75)
    sex = random.choice(["M", "F"])
    is_smoker = maybe(0.20)
    return dict(
        agent="NONE", severity=0, triage="Clear", age=age, sex=sex,
        is_smoker=int(is_smoker),
        hr=rng_int(58, 92), sbp=rng_int(105, 138), dbp=rng_int(62, 88),
        rr=rng_int(11, 18), spo2=round(rng_norm(98.5, 1.0, 95, 100), 1),
        temp=round(rng_norm(37.0, 0.3, 36.0, 37.8), 1),
        eco_ppm=round(rng_norm(16.4 if is_smoker else 1.26, 1.5 if is_smoker else 0.5, 0.3, 25), 1),
        eno_ppb=round(rng_norm(15, 4, 5, 28), 1),
        eco2_pct=round(rng_norm(4.1, 0.2, 3.5, 4.8), 2),
        op_score=round(rng_norm(2.0, 1.0, 0, 6), 1),
        symptom_text="no complaints",
        treatment="No treatment required",
        **{k: 0 for k in [
            "headache","nausea","vomiting","dizziness","confusion",
            "loss_of_consciousness","chest_pain","dyspnea","diaphoresis",
            "miosis","salivation","lacrimation","bronchospasm","bronchorrhea",
            "fasciculations","muscle_weakness","seizure",
            "cough","eye_irritation","pulmonary_edema"]}
    )


# ── Monte Carlo simulation ────────────────────────────────────────────────────
def generate(n_per_class=600, severity_dist=None):
    """
    Generate n_per_class patients per agent per severity level.
    Severity distribution: [mild_pct, moderate_pct, severe_pct]
    Default: field distribution — mild dominant.
    """
    if severity_dist is None:
        severity_dist = [0.45, 0.35, 0.20]  # mild, moderate, severe

    rows = []
    generators = {"CO": make_co, "OP": make_op, "PHOSGENE": make_phosgene}

    for agent, gen in generators.items():
        n_mild     = int(n_per_class * severity_dist[0])
        n_moderate = int(n_per_class * severity_dist[1])
        n_severe   = n_per_class - n_mild - n_moderate
        for sev, n in [(1, n_mild), (2, n_moderate), (3, n_severe)]:
            for _ in range(n):
                rows.append(gen(sev))

    # NONE baseline
    for _ in range(n_per_class):
        rows.append(make_none())

    df = pd.DataFrame(rows)
    # Shuffle
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
    df.to_csv(out, index=False)
    print(f"\nSaved -> {out}")
