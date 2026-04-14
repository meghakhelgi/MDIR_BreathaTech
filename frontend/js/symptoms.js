// symptoms.js — renders a flat symptom checklist, tracks state, computes burden

const Symptoms = (() => {

  // Current on/off state — keys are symptom keys, values are 0 or 1
  let _state = {};

  // All symptoms in a single flat list
  const _ALL_SYMS = [
    ...Config.SYMS_OP,
    ...Config.SYMS_PH,
    ...Config.SYMS_GEN,
  ];

  function renderAll() {
    const el = document.getElementById('syms-all');
    if (!el) return;
    el.innerHTML = _ALL_SYMS.map(s => {
      const on = !!_state[s.k];
      return `<div class="sym${on ? ' on' : ''}"
                   onclick="Symptoms.toggle('${s.k}')">
        <div class="sym-chk">
          ${on ? `<svg width="10" height="10" viewBox="0 0 10 10" fill="none"
                    stroke="white" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="1.5,5 4,7.5 8.5,2.5"/>
                  </svg>` : ''}
        </div>
        <div>
          <div class="sym-text">${s.label}</div>
          <div class="sym-sub">${s.sub}</div>
        </div>
      </div>`;
    }).join('');
    _renderBurden();
  }

  function _renderBurden() {
    const total = _ALL_SYMS.reduce((a, s) => a + (_state[s.k] ? s.weight : 0), 0);
    const sludge = Config.SYMS_OP.filter(s => _state[s.k]).length;

    const row = document.getElementById('burden-row');
    if (!row) return;
    row.style.display = total > 0 ? 'flex' : 'none';

    const valEl = document.getElementById('burden-value');
    if (valEl) valEl.textContent = `${total}/22`;

    const pillEl = document.getElementById('burden-pills');
    if (pillEl) {
      pillEl.innerHTML = sludge > 0
        ? `<span class="burden-pill pill-sludge">${sludge} SLUDGE sign${sludge > 1 ? 's' : ''}</span>`
        : '';
    }
  }

  // Toggle one symptom and re-render
  function toggle(key) {
    _state[key] = _state[key] ? 0 : 1;
    renderAll();
  }

  // Load a full symptom state object (e.g. from a demo case)
  function load(symptomsObj) {
    _state = { ...symptomsObj };
    renderAll();
  }

  // Reset all symptoms to zero
  function clear() {
    _state = {};
    renderAll();
  }

  // Return current state as a plain object (used when building the API payload)
  function getState() {
    return { ..._state };
  }

  return { renderAll, toggle, load, clear, getState };
})();
