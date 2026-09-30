(() => {
  const caseTabs = [...document.querySelectorAll('[data-case]')];
  const stageButtons = document.querySelector('#case-stages');
  const play = document.querySelector('#case-play');
  let cases = [];
  let selectedCase = 0;
  let selectedStage = 0;
  let playback = null;

  function stopFrames() {
    window.clearInterval(playback);
    playback = null;
    play.textContent = '▶ Play frames';
  }

  function showFrame(index) {
    selectedStage = index;
    const item = cases[selectedCase];
    const frame = item.frames[index];
    const final = index === item.frames.length - 1;
    for (const mode of ['direct', 'ours']) {
      const image = document.querySelector(`#case-${mode}-image`);
      image.src = frame[mode].image;
      image.alt = `${mode === 'ours' ? 'JevVLA' : 'Direct'}: ${item.instruction}, ${frame[mode].label}`;
      document.querySelector(`#case-${mode}-step`).textContent = frame[mode].label;
      document.querySelector(`#case-${mode}-result`).textContent = final ? item[mode === 'ours' ? 'oursResult' : 'directResult'] : 'Execution in progress';
    }
    document.querySelector('#case-position').textContent = frame.phase;
    [...stageButtons.children].forEach((button, i) => button.setAttribute('aria-pressed', String(i === index)));
  }

  function chooseCase(index, focus = false) {
    stopFrames();
    selectedCase = index;
    const item = cases[index];
    caseTabs.forEach((tab, i) => {
      tab.setAttribute('aria-selected', String(i === index));
      tab.tabIndex = i === index ? 0 : -1;
      if (focus && i === index) tab.focus();
    });
    document.querySelector('#case-panel').setAttribute('aria-labelledby', `case-tab-${index}`);
    document.querySelector('#case-instruction').textContent = `“${item.instruction}”`;
    document.querySelector('#case-meta').textContent = `${item.benchmark} · seed ${item.seed}`;
    document.querySelector('#case-summary').textContent = item.summary;
    document.querySelector('#case-interface').textContent = item.interface;
    stageButtons.replaceChildren();
    item.frames.forEach((frame, i) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.setAttribute('aria-label', frame.phase);
      button.dataset.stage = String(i + 1).padStart(2, '0');
      button.addEventListener('click', () => { stopFrames(); showFrame(i); });
      stageButtons.append(button);
      for (const mode of ['direct', 'ours']) {
        const image = new Image();
        image.src = frame[mode].image;
      }
    });
    document.querySelector('#case-scores').hidden = !item.scores;
    const scores = document.querySelector('#candidate-scores');
    scores.replaceChildren();
    if (item.scores) item.scores.forEach((score, i) => {
      const cell = document.createElement('div');
      cell.className = `candidate-score${i === item.selected ? ' is-chosen' : ''}`;
      const label = document.createElement('span');
      label.textContent = `c${['₀', '₁', '₂', '₃', '₄'][i]}`;
      const value = document.createElement('strong');
      value.textContent = `${score > 0 ? '+' : ''}${score.toFixed(3)}`;
      const role = document.createElement('small');
      role.textContent = i === item.selected ? 'SELECTED' : i === item.direct ? 'DIRECT' : 'CANDIDATE';
      cell.append(label, value, role);
      scores.append(cell);
    });
    showFrame(item.frames.length - 1);
  }

  fetch('assets/cases/cases.json').then(response => {
    if (!response.ok) throw new Error('Case data unavailable');
    return response.json();
  }).then(data => {
    cases = data;
    chooseCase(0);
    caseTabs.forEach((tab, index) => {
      tab.addEventListener('click', () => chooseCase(index));
      tab.addEventListener('keydown', event => {
        let next;
        if (event.key === 'ArrowRight') next = (selectedCase + 1) % cases.length;
        if (event.key === 'ArrowLeft') next = (selectedCase + cases.length - 1) % cases.length;
        if (event.key === 'Home') next = 0;
        if (event.key === 'End') next = cases.length - 1;
        if (next !== undefined) { event.preventDefault(); chooseCase(next, true); }
      });
    });
    play.addEventListener('click', () => {
      if (playback) { stopFrames(); return; }
      showFrame(0);
      play.textContent = 'Ⅱ Pause frames';
      playback = window.setInterval(() => {
        if (selectedStage >= cases[selectedCase].frames.length - 1) { stopFrames(); return; }
        showFrame(selectedStage + 1);
      }, 1500);
    });
  }).catch(() => {
    play.disabled = true;
    caseTabs.forEach(tab => { tab.disabled = true; });
    document.querySelector('#case-position').textContent = 'Final outcome';
  });

  document.addEventListener('visibilitychange', () => { if (document.hidden) stopFrames(); });
  new IntersectionObserver(entries => {
    if (!entries[0].isIntersecting) stopFrames();
  }).observe(document.querySelector('#cases'));

  const viewer = document.querySelector('#robot-viewer');
  const loader = new IntersectionObserver(entries => {
    if (!entries.some(entry => entry.isIntersecting)) return;
    loader.disconnect();
    import('./robot-viewer.js').then(module => module.mountRobotViewer(viewer)).catch(() => {
      document.querySelector('#lab-load-status').textContent = 'The 3D scene could not load. Refresh to try again.';
      viewer.setAttribute('aria-busy', 'false');
    });
  }, { rootMargin: '500px' });
  loader.observe(viewer);
})();
