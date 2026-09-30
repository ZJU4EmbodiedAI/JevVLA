document.documentElement.classList.add('js');

const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const revealObserver = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add('is-visible');
      revealObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.08 });
document.querySelectorAll('.reveal').forEach(element => revealObserver.observe(element));

const progress = document.querySelector('.reading-progress');
const robot = document.querySelector('.robot-image');
const hero = document.querySelector('.hero');
let scrollScheduled = false;
function updateScroll() {
  const total = document.documentElement.scrollHeight - window.innerHeight;
  progress.style.transform = `scaleX(${total > 0 ? window.scrollY / total : 0})`;
  if (!reducedMotion.matches && window.innerWidth > 760 && window.scrollY < hero.offsetHeight) {
    robot.style.setProperty('--scene-shift', `${window.scrollY * 0.07}px`);
  }
  if (window.scrollY < hero.offsetHeight * 0.5) {
    document.querySelectorAll('.site-header nav a').forEach(link => link.classList.remove('active'));
  }
  scrollScheduled = false;
}
window.addEventListener('scroll', () => {
  if (!scrollScheduled) {
    requestAnimationFrame(updateScroll);
    scrollScheduled = true;
  }
}, { passive: true });
window.addEventListener('resize', updateScroll);
updateScroll();

const navObserver = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      document.querySelectorAll('.site-header nav a').forEach(link => {
        link.classList.toggle('active', link.getAttribute('href') === `#${entry.target.id}`);
      });
    }
  });
}, { rootMargin: '-20% 0px -55% 0px' });
document.querySelectorAll('section[id]').forEach(section => navObserver.observe(section));

const strength = document.querySelector('#guidance-strength');
const strengthValue = document.querySelector('#strength-value');
const state = document.querySelector('#model-state');
const tabs = [...document.querySelectorAll('[data-mode]')];
let mode = 'selection';

function bezier(t, points) {
  const u = 1 - t;
  return [0, 1].map(axis => u * u * u * points[0][axis] + 3 * u * u * t * points[1][axis] + 3 * u * t * t * points[2][axis] + t * t * t * points[3][axis]);
}

function updateIllustration() {
  const gain = Number(strength.value);
  strengthValue.value = gain.toFixed(2);
  if (mode === 'selection') {
    const energies = [1.4, 0.9, 0.15];
    const prior = [0, -0.15, -0.45];
    const logits = energies.map((energy, index) => prior[index] - gain * energy);
    const normalizer = Math.max(...logits);
    const weights = logits.map(value => Math.exp(value - normalizer));
    const total = weights.reduce((a, b) => a + b, 0);
    const selected = logits.indexOf(Math.max(...logits));
    const percentages = weights.map(value => 100 * value / total);
    const displayed = percentages.map(value => Math.floor(value));
    const order = percentages.map((value, index) => ({ index, fraction: value - displayed[index] })).sort((a, b) => b.fraction - a.fraction);
    const remainder = 100 - displayed.reduce((a, b) => a + b, 0);
    for (let index = 0; index < remainder; index++) displayed[order[index].index] += 1;
    document.querySelectorAll('.candidate-node').forEach((node, index) => {
      node.classList.toggle('is-selected', index === selected);
      node.querySelector('.probability').textContent = `${displayed[index]}%`;
      document.querySelector(`.branch-${index}`).classList.toggle('is-selected', index === selected);
    });
    document.querySelector('#selected-indicator').setAttribute('transform', `translate(0 ${118 * selected})`);
    state.textContent = `Action a${['₁', '₂', '₃'][selected]} selected`;
  } else {
    const ratio = gain / 2;
    const base = [[100, 285], [235, 95], [360, 332], [610, 220]];
    const guided = [[100, 285], [235, 95 - 30 * ratio], [360, 332 - 122 * ratio], [610, 220 - 125 * ratio]];
    document.querySelector('#guided-trajectory').setAttribute('d', `M${guided[0]}C${guided[1]} ${guided[2]} ${guided[3]}`);
    const vectors = [];
    const points = [];
    for (let index = 1; index <= 7; index++) {
      const t = index / 7;
      const start = bezier(t, base);
      const end = bezier(t, guided);
      vectors.push(`<line class="gradient-vector" x1="${start[0]}" y1="${start[1]}" x2="${end[0]}" y2="${end[1]}"/>`);
      points.push(`<circle class="guided-point" cx="${end[0]}" cy="${end[1]}" r="5"/>`);
    }
    document.querySelector('#gradient-vectors').innerHTML = vectors.join('');
    document.querySelector('#guided-points').innerHTML = points.join('');
    state.textContent = gain === 0 ? 'Original proposal' : 'Energy-guided action';
  }
}

function selectMode(nextMode, focus = false) {
  mode = nextMode;
  tabs.forEach(tab => {
    const active = tab.dataset.mode === mode;
    tab.setAttribute('aria-selected', String(active));
    tab.tabIndex = active ? 0 : -1;
    if (active && focus) tab.focus();
  });
  document.querySelector('#select-panel').hidden = mode !== 'selection';
  document.querySelector('#guide-panel').hidden = mode !== 'guidance';
  document.querySelector('#selection-copy').hidden = mode !== 'selection';
  document.querySelector('#guidance-copy').hidden = mode !== 'guidance';
  document.querySelector('#strength-label').textContent = mode === 'selection' ? 'Energy weight' : 'Guidance strength';
  updateIllustration();
}
tabs.forEach(tab => {
  tab.addEventListener('click', () => selectMode(tab.dataset.mode));
  tab.addEventListener('keydown', event => {
    if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
      event.preventDefault();
      const next = event.key === 'Home' ? 'selection' : event.key === 'End' ? 'guidance' : mode === 'selection' ? 'guidance' : 'selection';
      selectMode(next, true);
    }
  });
});
strength.addEventListener('input', updateIllustration);
updateIllustration();

const dialog = document.querySelector('#figure-dialog');
document.querySelectorAll('[data-figure]').forEach(button => {
  button.addEventListener('click', () => {
    const figure = document.querySelector('#expanded-figure');
    figure.src = button.dataset.figure;
    figure.alt = button.dataset.caption;
    document.querySelector('#figure-caption').textContent = button.dataset.caption;
    dialog.showModal();
    document.body.classList.add('dialog-open');
  });
});
document.querySelector('#close-figure').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => {
  const rect = dialog.getBoundingClientRect();
  if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
});
dialog.addEventListener('close', () => document.body.classList.remove('dialog-open'));

document.querySelector('#copy-code').addEventListener('click', async () => {
  const code = document.querySelector('#install-code').textContent;
  const feedback = document.querySelector('#copy-status');
  try {
    await navigator.clipboard.writeText(code);
    feedback.textContent = 'Copied';
  } catch {
    const range = document.createRange();
    range.selectNodeContents(document.querySelector('#install-code'));
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    feedback.textContent = 'Select & copy';
  }
  window.setTimeout(() => { feedback.textContent = ''; }, 2500);
});
