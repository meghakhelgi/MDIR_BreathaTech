// app.js — main application controller
// Loads demo cases from data/demo_cases.json (real training data rows),
// wires the form submit to the FastAPI backend, manages session history.

const App = (() => {

  let _demoCases   = {};        // loaded from demo_cases.json
  let _currentDemo = null;      // key of currently loaded demo
  let _sensors     = {};        // current sensor values (from demo or live device)
  let _sessionLog  = [];        // history strip entries
  let _apiOnline   = false;
  let _manualMode  = false;

  // ── init ────────────────────────────────────────────────────────────────
  async function init() {
    _startClock();
    Sensors.renderSignal();
    Symptoms.renderAll();
    await _loadDemoCases();
    await _checkAPI();
    _bindDemoButtons();
    Sensors.render({});
    ['banners', 'result-card', 'agent-bars'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.innerHTML = '';
    });
  }

  // ── clock ───────────────────────────────────────────────────────────────
  function _startClock() {
    const el = document.getElementById('clock');
    function tick() { if (el) el.textContent = new Date().toLocaleTimeString(); }
    tick(); setInterval(tick, 1000);
  }

  // ── load demo cases from training data JSON ─────────────────────────────
  async function _loadDemoCases() {
    try {
      const r = await fetch('data/demo_cases.json');
      _demoCases = await r.json();
      console.log('[BreathaTech] Demo cases loaded from training data:', Object.keys(_demoCases));
    } catch (e) {
      console.warn('[BreathaTech] Could not load demo_cases.json — using inline fallbacks');
      _demoCases = _fallbackDemos();
    }
  }

  // ── API health check ────────────────────────────────────────────────────
  async function _checkAPI() {
    const dot = document.getElementById('conn-dot');
    const lbl = document.getElementById('conn-lbl');
    try {
      const h = await API.health();
      _apiOnline = h.model_loaded;
      if (dot) dot.classList.toggle('offline', !_apiOnline);
      if (lbl) lbl.textContent = _apiOnline ? 'Model connected' : 'Model offline';
      _setAPIStatus(_apiOnline ? 'ok' : 'error',
        _apiOnline ? 'FastAPI model loaded — predictions will use the live model.'
                   : 'Model not loaded. Start the API server: uvicorn api:app --reload');
    } catch {
      _apiOnline = false;
      if (dot) dot.classList.add('offline');
      if (lbl) lbl.textContent = 'API offline';
      _setAPIStatus('error',
        'Cannot reach API at ' + Config.API_BASE +
        '. Start with: cd breathatech && uvicorn api:app --reload');
    }
  }

  function _setAPIStatus(cls, msg) {
    const el = document.getElementById('api-status');
    if (!el) return;
    el.className = `api-status ${cls}`;
    el.textContent = msg;
  }

  // ── demo buttons ────────────────────────────────────────────────────────
  function _bindDemoButtons() {
    document.querySelectorAll('.demo-btn[data-demo]').forEach(btn => {
      btn.addEventListener('click', () => loadDemo(btn.dataset.demo));
    });
  }

  // ── manual sensor input mode ────────────────────────────────────────────
  function toggleManualMode() {
    _manualMode = !_manualMode;
    const btn       = document.getElementById('manual-btn');
    const sensorGrid = document.getElementById('sensor-grid');
    const manualGrid = document.getElementById('manual-grid');
    if (!btn || !sensorGrid || !manualGrid) return;

    if (_manualMode) {
      // deselect demo buttons
      document.querySelectorAll('.demo-btn[data-demo]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      sensorGrid.style.display = 'none';
      manualGrid.style.display = 'grid';
      _renderManualGrid();
    } else {
      btn.classList.remove('active');
      sensorGrid.style.display = 'grid';
      manualGrid.style.display = 'none';
    }
  }

  function _renderManualGrid() {
    const grid = document.getElementById('manual-grid');
    if (!grid) return;

    grid.innerHTML = Config.SENSORS.map(s => {
      const cur = _sensors?.[s.key] ?? 0;
      return `<div class="manual-input-card">
        <span class="sensor-tag ${s.tagClass}">${s.tag}</span>
        <div class="sensor-label">${s.label}</div>
        <div class="manual-input-row">
          <input
            type="number" step="0.1" min="0"
            class="manual-val-input"
            data-key="${s.key}"
            value="${Number(cur).toFixed(s.key === 'eco2_pct' ? 2 : 1)}"
          >
          <span class="manual-unit">${s.unit}</span>
        </div>
        <div class="sensor-ref">${s.ref} &nbsp;·&nbsp; range 0–${s.max} ${s.unit}</div>
      </div>`;
    }).join('');

    grid.querySelectorAll('.manual-val-input').forEach(input => {
      input.addEventListener('input', () => {
        _sensors[input.dataset.key] = parseFloat(input.value) || 0;
        Sensors.render(_sensors);
        _predictSensorsOnly();
      });
    });
  }

  // Load a demo case — sensors only; form and ML output are not pre-filled
  function loadDemo(key) {
    const demo = _demoCases[key];
    if (!demo) return;

    // exit manual mode if active
    if (_manualMode) {
      _manualMode = false;
      const sensorGrid = document.getElementById('sensor-grid');
      const manualGrid = document.getElementById('manual-grid');
      const manualBtn  = document.getElementById('manual-btn');
      if (sensorGrid) sensorGrid.style.display = 'grid';
      if (manualGrid) manualGrid.style.display = 'none';
      if (manualBtn)  manualBtn.classList.remove('active');
    }

    _currentDemo = key;
    _sensors = { ...demo.sensors };

    // highlight active button
    document.querySelectorAll('.demo-btn').forEach(b => {
      b.classList.toggle('active', b.dataset.demo === key);
    });

    // render sensor cards only
    Sensors.render(_sensors);

    // clear full ML output — probabilities will update via auto-predict below
    ['banners', 'result-card'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.innerHTML = '';
    });

    // auto-predict with sensor values + all defaults to populate probability bars
    _predictSensorsOnly();
  }

  // Hit the API with sensor values only (no clinician input) to show early probabilities
  async function _predictSensorsOnly() {
    const barsEl = document.getElementById('agent-bars');
    if (!barsEl) return;
    if (!_apiOnline) { barsEl.innerHTML = ''; return; }

    try {
      const reading = API.buildReading(_sensors, {}, {});
      const result  = await API.predict(reading);
      Results.renderBars(result.agent_breakdown);
    } catch (e) {
      barsEl.innerHTML = '';
    }
  }

  // ── form submit → API ───────────────────────────────────────────────────
  async function submit() {
    const btn = document.getElementById('submit-btn');
    btn.disabled = true;
    btn.textContent = 'Processing…';
    _setAPIStatus('loading', 'Sending to model…');

    const vitals = {
      spo2:       _numVal('spo2'),
      hr:         _numVal('hr'),
      rr:         _numVal('rr'),
      sbp:        _numVal('sbp'),
      dbp:        _numVal('dbp'),
      is_smoker:  parseInt(document.getElementById('smoker')?.value || '0'),
      age:        _numVal('age'),
      weight_kg:  75,   // default — add weight field to form if needed
      texp:       _numVal('texp'),
    };
    const symptoms = Symptoms.getState();
    const notes    = document.getElementById('notes')?.value || '';
    const pid      = document.getElementById('pid')?.value  || '';

    try {
      const reading = API.buildReading(_sensors, vitals, symptoms);

      let result;
      if (_apiOnline) {
        // live prediction from FastAPI backend
        result = await API.predict(reading, notes, pid);
        _setAPIStatus('ok', `Prediction received — model v${result.model_version}`);
      } else {
        // API offline — show demo result with a notice
        result = _demoCases[_currentDemo] || Object.values(_demoCases)[0];
        _setAPIStatus('error',
          'API offline — showing demo result. Start: uvicorn api:app --reload');
      }

      Results.render(result, _sensors);
      _logSession(result);

    } catch (err) {
      _setAPIStatus('error', `Prediction error: ${err.message}`);
      console.error('[BreathaTech] Prediction failed:', err);
    } finally {
      btn.disabled = false;
      btn.textContent = 'Submit to ML model';
    }
  }

  // ── session history ─────────────────────────────────────────────────────
  function _logSession(result) {
    _sessionLog.unshift({
      t:      new Date().toLocaleTimeString(),
      agent:  result.agent,
      triage: result.triage,
      conf:   result.agent_confidence,
    });
    if (_sessionLog.length > 8) _sessionLog.pop();

    const strip = document.getElementById('history-strip');
    const chips = document.getElementById('history-chips');
    if (!strip || !chips) return;

    strip.style.display = 'flex';
    chips.innerHTML = _sessionLog.map(h =>
      `<div class="history-chip hc-${h.triage}">
        ${h.t} · ${h.triage} · ${h.agent} · ${Math.round(h.conf)}%
      </div>`
    ).join('');
  }

  // ── clear form ──────────────────────────────────────────────────────────
  function clearForm() {
    ['spo2','hr','rr','sbp','dbp','temp','age','texp'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.value = '';
    });
    const notes = document.getElementById('notes');
    const pid   = document.getElementById('pid');
    if (notes) notes.value = '';
    if (pid)   pid.value   = '';
    const smoker = document.getElementById('smoker');
    if (smoker) smoker.value = '0';
    Symptoms.clear();
  }

  // ── helpers ─────────────────────────────────────────────────────────────
  function _numVal(id) {
    return parseFloat(document.getElementById(id)?.value || '0') || 0;
  }

  // Inline fallback demo cases if the JSON file can't be fetched
  function _fallbackDemos() {
    return {
      co_high: {
        agent: 'CO', severity: 2, triage: 'Escalate', conf: 97.2,
        probs: { CO: 97.2, OP: 1.0, PHOSGENE: 1.0, NONE: 0.8 },
        sensors: { eco_ppm: 31.3, eno_ppb: 21.7, eco2_pct: 4.35, op_score: 2.4 },
        vitals: { spo2: 94, hr: 105, rr: 22, sbp: 112, dbp: 72, temp: 37.2, age: 45, is_smoker: 1, time_since_exposure_min: 20 },
        symptoms: { headache: 1, nausea: 1, dizziness: 1 },
        treatment: 'Remove from source. 100% O₂ NRB mask. Monitor HbCO. Consider HBO if HbCO >25% or neuro sx.',
      },
      op_severe: {
        agent: 'OP', severity: 3, triage: 'Immediate', conf: 99.5, nerve: true,
        probs: { CO: 0.2, OP: 99.5, PHOSGENE: 0.2, NONE: 0.1 },
        sensors: { eco_ppm: 2.5, eno_ppb: 64.2, eco2_pct: 5.1, op_score: 88.5 },
        vitals: { spo2: 78, hr: 40, rr: 38, sbp: 68, dbp: 44, temp: 37.8, age: 32, is_smoker: 0, time_since_exposure_min: 8 },
        symptoms: { miosis: 1, salivation: 1, lacrimation: 1, bronchospasm: 1, bronchorrhea: 1, fasciculations: 1, seizure: 1, loss_of_consciousness: 1 },
        treatment: 'NERVE AGENT — IMMEDIATE: Atropine 2–6mg IV/IM q5–10min. Pralidoxime 1–2g IV STAT — before aging. Diazepam for seizures. Decontaminate.',
      },
      phosgene_mild: {
        agent: 'PHOSGENE', severity: 1, triage: 'Monitor', conf: 88.4, phosgene: true,
        probs: { CO: 2.0, OP: 0.5, PHOSGENE: 88.4, NONE: 9.1 },
        sensors: { eco_ppm: 0.2, eno_ppb: 29.9, eco2_pct: 4.55, op_score: 1.4 },
        vitals: { spo2: 97, hr: 76, rr: 15, sbp: 124, dbp: 82, temp: 37.0, age: 34, is_smoker: 0, time_since_exposure_min: 45 },
        symptoms: { cough: 1 },
        treatment: 'STRICT REST — exertion worsens edema. High-flow O₂. Admit 24h. eNO/eCO₂ q1h. Patient may feel well but decompensate within hours.',
      },
      none: {
        agent: 'NONE', severity: 0, triage: 'Clear', conf: 99.3,
        probs: { CO: 0.3, OP: 0.2, PHOSGENE: 0.2, NONE: 99.3 },
        sensors: { eco_ppm: 1.3, eno_ppb: 8.6, eco2_pct: 4.12, op_score: 2.0 },
        vitals: { spo2: 99, hr: 68, rr: 14, sbp: 118, dbp: 76, temp: 37.0, age: 28, is_smoker: 0, time_since_exposure_min: 0 },
        symptoms: {},
        treatment: 'No treatment required. Routine monitoring.',
      },
    };
  }

  return { init, loadDemo, submit, clearForm, toggleManualMode };
})();

// boot on DOM ready
document.addEventListener('DOMContentLoaded', App.init);
