// results.js — renders the ML output panel
// Accepts either a live API response or a demo case object — same render path.

const Results = (() => {

  // Normalise either an API response or a demo case to a common shape
  function _normalise(raw) {
    // API response already has the right shape from PredictionResponse model
    if (raw.agent_confidence !== undefined) return raw;

    // Demo case — build compatible shape
    return {
      agent:             raw.agent,
      agent_name:        Config.AGENT_NAMES[raw.agent] || raw.agent,
      agent_confidence:  raw.conf ?? 95,
      agent_breakdown:   raw.probs ?? { [raw.agent]: 100 },
      triage:            raw.triage,
      triage_confidence: raw.conf ?? 95,
      immediate_score:   raw.triage === 'Immediate' ? 95 : 15,
      nerve_alert:       raw.agent === 'OP',
      phosgene_occult:   raw.phosgene ?? false,
      treatment_hint:    raw.treatment ?? '',
      severity:          raw.severity ?? 0,
      model_version:     'v1.0',
    };
  }

  function render(raw, sensorValues) {
    const r = _normalise(raw);
    const tc = Config.TRIAGE_COLORS[r.triage] || '#555';

    // banners
    let banners = '';
    if (r.nerve_alert) {
      banners += `<div class="banner banner-op">
        <span class="banner-icon">&#9888;</span>
        <div><strong>Nerve agent confirmed</strong> — administer atropine + pralidoxime IMMEDIATELY.
        Enzyme aging is time-critical. Full decontamination required.</div>
      </div>`;
    }
    if (r.phosgene_occult) {
      banners += `<div class="banner banner-ph">
        <span class="banner-icon">&#9888;</span>
        <div><strong>Phosgene occult phase</strong> — eNO elevated. Patient may feel well.
        Latent phase before pulmonary edema. Strict rest mandatory. Do NOT discharge.</div>
      </div>`;
    }
    document.getElementById('banners').innerHTML = banners;

    // result card
    const sevClass = `sev-${r.severity ?? 0}`;
    const sevLabel = Config.SEVERITY_LABELS[r.severity ?? 0] || '—';
    const pillClass = Config.AGENT_PILL_CLASS[r.agent] || 'pill-none';
    const pillText  = Config.AGENT_PILL_TEXT[r.agent] || '';

    document.getElementById('result-card').innerHTML = `
      <div class="result-card ${r.triage}">
        <div class="result-top">
          <div>
            <div class="result-triage" style="color:${tc}">${r.triage}</div>
            <div class="result-agent">
              ${r.agent_name}
              <span class="agent-type-pill ${pillClass}">${pillText}</span>
            </div>
          </div>
          <div class="result-conf">
            <div class="result-conf-num" style="color:${tc}">
              ${Number(r.agent_confidence).toFixed(1)}%
            </div>
            <div class="result-conf-lbl">confidence</div>
          </div>
        </div>
        <span class="severity-badge ${sevClass}">
          Severity ${r.severity ?? 0}: ${sevLabel}
        </span>
        <div class="treatment-box">${r.treatment_hint || r.treatment || '—'}</div>
      </div>`;

    // agent probability bars
    const breakdown = r.agent_breakdown || {};
    document.getElementById('agent-bars').innerHTML =
      ['CO', 'OP', 'PHOSGENE', 'NONE'].map(ag => {
        const pct   = Number(breakdown[ag] ?? 0);
        const color = Config.AGENT_COLORS[ag] || '#888';
        return `<div class="agent-bar-row">
          <div class="agent-bar-label">${ag === 'PHOSGENE' ? 'PHOSG.' : ag}</div>
          <div class="agent-bar-track">
            <div class="agent-bar-fill" style="width:${pct}%;background:${color}"></div>
          </div>
          <div class="agent-bar-pct" style="color:${pct > 50 ? color : '#8a96b0'}">
            ${pct.toFixed(1)}%
          </div>
        </div>`;
      }).join('');

    // stat chips
    const alarm = Sensors.alarmScore(sensorValues ?? {});
    const sevChip = document.getElementById('sev-chip');
    const alarmChip = document.getElementById('alarm-score');
    if (sevChip)   sevChip.textContent   = `${r.severity ?? 0} — ${sevLabel}`;
    if (alarmChip) alarmChip.textContent = alarm.toFixed(2);
  }

  // Render only the probability bars (used after sensor-only auto-predict)
  function renderBars(breakdown) {
    const el = document.getElementById('agent-bars');
    if (!el) return;
    el.innerHTML = ['CO', 'OP', 'PHOSGENE', 'NONE'].map(ag => {
      const pct   = Number((breakdown || {})[ag] ?? 0);
      const color = Config.AGENT_COLORS[ag] || '#888';
      return `<div class="agent-bar-row">
        <div class="agent-bar-label">${ag === 'PHOSGENE' ? 'PHOSG.' : ag}</div>
        <div class="agent-bar-track">
          <div class="agent-bar-fill" style="width:${pct}%;background:${color}"></div>
        </div>
        <div class="agent-bar-pct" style="color:${pct > 50 ? color : '#8a96b0'}">
          ${pct.toFixed(1)}%
        </div>
      </div>`;
    }).join('');
  }

  return { render, renderBars };
})();
