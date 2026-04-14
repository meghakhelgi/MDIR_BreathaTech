// sensors.js — renders sensor cards and computes alarm score

const Sensors = (() => {

  function _status(s, val) {
    if (s.inv)  return val <= s.alertAt ? 'alert' : val <= s.warnAt ? 'warn' : 'ok';
    if (s.isOP) return val >= s.alertAt ? 'op-hi' : val >= s.warnAt ? 'warn' : 'ok';
    return val >= s.alertAt ? 'alert' : val >= s.warnAt ? 'warn' : 'ok';
  }

  function _barColor(st) {
    return { alert: '#dc2626', warn: '#d97706', 'op-hi': '#dc2626', ok: '#1d6fb8' }[st] || '#94a3b8';
  }

  function _barPct(s, val) {
    return Math.min(100, (val / s.max) * 100).toFixed(1);
  }

  function _fmt(val, key) {
    return key === 'eco2_pct' ? Number(val).toFixed(2) : Number(val).toFixed(1);
  }

  // Render all 5 sensor cards into #sensor-grid
  function render(sensorValues) {
    const grid = document.getElementById('sensor-grid');
    if (!grid) return;

    grid.innerHTML = Config.SENSORS.map(s => {
      const val = sensorValues?.[s.key] ?? 0;
      const st  = _status(s, val);
      const pct = _barPct(s, val);
      const col = _barColor(st);

      return `<div class="sensor-card ${st}">
        <span class="sensor-tag ${s.tagClass}">${s.tag}</span>
        <div class="sensor-label">${s.label}</div>
        <div class="sensor-value">
          ${_fmt(val, s.key)}<span class="sensor-unit">${s.unit}</span>
        </div>
        <div class="sensor-ref">${s.ref}</div>
        <div class="sensor-bar">
          <div class="sensor-bar-fill" style="width:${pct}%;background:${col}"></div>
        </div>
      </div>`;
    }).join('');
  }

  // Compute alarm score — matches the breath_alarm engineered feature in inference.py
  function alarmScore(sensorValues) {
    const BL = Config.SENSOR_BASELINES;
    const scores = Object.entries(BL).map(([k, { mu, sd }]) => {
      const v = sensorValues?.[k] ?? 0;
      return Math.max(0, (v - mu) / sd);
    });
    return Math.max(...scores);
  }

  // Render signal bars in device header
  function renderSignal() {
    const el = document.getElementById('signal-bars');
    if (!el) return;
    const heights = [12, 17, 22, 28, 34];
    el.innerHTML = heights.map((h, i) =>
      `<div style="height:${h}px;opacity:${i < 4 ? 1 : .2}"></div>`
    ).join('');
  }

  return { render, alarmScore, renderSignal };
})();
