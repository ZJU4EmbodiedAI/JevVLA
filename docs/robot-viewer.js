import * as THREE from './vendor/three.js';

export function mountRobotViewer(container) {
  const canvas = document.querySelector('#robot-canvas');
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false, powerPreference: 'low-power' });
  } catch {
    document.querySelector('#lab-load-status').textContent = '3D needs WebGL in your browser. The recorded cases are available below.';
    container.setAttribute('aria-busy', 'false');
    return;
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.35;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#e9e9e5');
  scene.fog = new THREE.Fog('#e9e9e5', 8, 18);
  const camera = new THREE.PerspectiveCamera(37, 1, 0.02, 40);
  const controls = new THREE.OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.075;
  controls.enablePan = false;
  controls.enableZoom = false;
  controls.minPolarAngle = 0.08;
  controls.maxPolarAngle = Math.PI * 0.485;
  controls.rotateSpeed = 0.55;
  controls.target.set(-0.1, 0.43, 0);

  scene.add(new THREE.HemisphereLight('#ffffff', '#8e8e88', 2.7));
  const key = new THREE.DirectionalLight('#ffffff', 3.8);
  key.position.set(-3, 6, 3);
  key.castShadow = true;
  key.shadow.mapSize.set(1536, 1536);
  key.shadow.camera.left = -3;
  key.shadow.camera.right = 3;
  key.shadow.camera.top = 3;
  key.shadow.camera.bottom = -3;
  key.shadow.normalBias = 0.025;
  key.shadow.bias = -0.00015;
  scene.add(key);
  const fill = new THREE.DirectionalLight('#ffffff', 1.6);
  fill.position.set(3, 3, -4);
  scene.add(fill);

  const materials = {
    silver: new THREE.MeshStandardMaterial({ color: '#cdd0d0', roughness: 0.29, metalness: 0.65 }),
    shell: new THREE.MeshStandardMaterial({ color: '#f4f4f2', roughness: 0.36, metalness: 0.2 }),
    dark: new THREE.MeshStandardMaterial({ color: '#272a2b', roughness: 0.42, metalness: 0.55 }),
    rubber: new THREE.MeshStandardMaterial({ color: '#171919', roughness: 0.83, metalness: 0.05 }),
    red: new THREE.MeshStandardMaterial({ color: '#ee3527', roughness: 0.36, metalness: 0.12 }),
    tray: new THREE.MeshStandardMaterial({ color: '#a7aaa8', roughness: 0.58, metalness: 0.3 }),
    table: new THREE.MeshStandardMaterial({ color: '#fcfcfa', roughness: 0.88, metalness: 0.05 })
  };
  function mesh(geometry, material, parent = scene) {
    const object = new THREE.Mesh(geometry, material);
    object.castShadow = true;
    object.receiveShadow = true;
    parent.add(object);
    return object;
  }
  function box(w, h, d, material, position, parent = scene) {
    const object = mesh(new THREE.BoxGeometry(w, h, d), material, parent);
    object.position.set(...position);
    return object;
  }
  function cylinder(r1, r2, h, material, position, parent = scene) {
    const object = mesh(new THREE.CylinderGeometry(r1, r2, h, 40), material, parent);
    object.position.set(...position);
    return object;
  }
  const floor = mesh(new THREE.PlaneGeometry(200, 200), new THREE.MeshStandardMaterial({ color: '#e9e9e5', roughness: 1 }));
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -0.28;
  floor.castShadow = false;
  box(2.95, 0.16, 1.85, materials.table, [0, -0.09, 0]);
  box(2.82, 0.025, 1.72, materials.dark, [0, -0.185, 0]);
  const grid = new THREE.GridHelper(2.8, 28, '#d6d8d2', '#e4e5df');
  grid.position.y = -0.006;
  grid.scale.z = 0.63;
  grid.material.transparent = true;
  grid.material.opacity = 0.4;
  scene.add(grid);

  const base = new THREE.Vector3(-1.08, 0, -0.32);
  cylinder(0.21, 0.25, 0.07, materials.dark, [base.x, 0.03, base.z]);
  cylinder(0.17, 0.19, 0.28, materials.silver, [base.x, 0.2, base.z]);
  cylinder(0.18, 0.18, 0.045, materials.rubber, [base.x, 0.36, base.z]);
  cylinder(0.165, 0.165, 0.1, materials.shell, [base.x, 0.43, base.z]);
  cylinder(0.196, 0.196, 0.014, materials.red, [base.x, 0.075, base.z]);
  for (let i = 0; i < 4; i++) {
    const a = i * Math.PI / 2 + Math.PI / 4;
    cylinder(0.016, 0.016, 0.015, materials.silver, [base.x + Math.cos(a) * 0.2, 0.075, base.z + Math.sin(a) * 0.2]);
  }
  const shoulder = new THREE.Vector3(base.x, 0.53, base.z);
  function joint(radius) {
    const group = new THREE.Group();
    scene.add(group);
    const body = cylinder(radius, radius, radius * 1.65, materials.dark, [0, 0, 0], group);
    body.rotation.x = Math.PI / 2;
    for (const sign of [-1, 1]) {
      const cap = cylinder(radius * 0.86, radius * 0.86, 0.027, materials.silver, [0, 0, sign * radius * 0.83], group);
      cap.rotation.x = Math.PI / 2;
      const disk = cylinder(radius * 0.6, radius * 0.6, 0.03, materials.shell, [0, 0, sign * radius * 0.93], group);
      disk.rotation.x = Math.PI / 2;
    }
    return group;
  }
  const joint1 = joint(0.17);
  const joint2 = joint(0.145);
  const joint3 = joint(0.105);
  joint1.position.copy(shoulder);
  const upper = cylinder(0.092, 0.107, 1, materials.shell, [0, 0, 0]);
  const fore = cylinder(0.071, 0.086, 1, materials.shell, [0, 0, 0]);
  const upperInset = cylinder(0.033, 0.04, 1, materials.silver, [0, 0, 0]);
  const foreInset = cylinder(0.028, 0.034, 1, materials.silver, [0, 0, 0]);
  const yAxis = new THREE.Vector3(0, 1, 0);
  function linkBetween(object, from, to, shrink = 0) {
    const direction = new THREE.Vector3().subVectors(to, from);
    object.position.copy(from).add(to).multiplyScalar(0.5);
    object.quaternion.setFromUnitVectors(yAxis, direction.clone().normalize());
    object.scale.y = Math.max(0.01, direction.length() - shrink);
  }
  const gripper = new THREE.Group();
  scene.add(gripper);
  cylinder(0.068, 0.085, 0.13, materials.silver, [0, 0.03, 0], gripper);
  cylinder(0.088, 0.088, 0.045, materials.dark, [0, -0.044, 0], gripper);
  box(0.18, 0.045, 0.075, materials.dark, [0, -0.085, 0], gripper);
  const fingers = [-1, 1].map(sign => {
    const finger = new THREE.Group();
    gripper.add(finger);
    box(0.024, 0.11, 0.047, materials.silver, [0, -0.142, 0], finger);
    box(0.02, 0.035, 0.054, materials.rubber, [-sign * 0.009, -0.182, 0], finger);
    return finger;
  });

  const cubeStart = new THREE.Vector3(-0.18, 0.071, 0.45);
  const cubeGoal = new THREE.Vector3(0.78, 0.084, -0.23);
  const cube = box(0.13, 0.13, 0.13, materials.red, cubeStart.toArray());
  const tray = new THREE.Group();
  tray.position.set(cubeGoal.x, 0, cubeGoal.z);
  scene.add(tray);
  box(0.53, 0.025, 0.4, materials.tray, [0, 0.016, 0], tray);
  box(0.53, 0.07, 0.035, materials.tray, [0, 0.055, -0.2], tray);
  box(0.53, 0.07, 0.035, materials.tray, [0, 0.055, 0.2], tray);
  box(0.035, 0.07, 0.37, materials.tray, [-0.248, 0.055, 0], tray);
  box(0.035, 0.07, 0.37, materials.tray, [0.248, 0.055, 0], tray);
  const target = new THREE.Mesh(new THREE.RingGeometry(0.098, 0.104, 64), new THREE.MeshBasicMaterial({ color: '#ee3527', transparent: true, opacity: 0.7, side: THREE.DoubleSide }));
  target.rotation.x = -Math.PI / 2;
  target.position.set(cubeGoal.x, 0.032, cubeGoal.z);
  scene.add(target);

  const transferStart = new THREE.Vector3(cubeStart.x, 0.72, cubeStart.z);
  const transferGoal = new THREE.Vector3(cubeGoal.x, 0.72, cubeGoal.z);
  const preferred = new THREE.CatmullRomCurve3([transferStart, new THREE.Vector3(0.13, 0.86, 0.33), new THREE.Vector3(0.49, 0.84, -0.1), transferGoal]);
  const original = new THREE.CatmullRomCurve3([transferStart, new THREE.Vector3(0.04, 0.97, 0.65), new THREE.Vector3(0.67, 1.02, 0.24), new THREE.Vector3(0.96, 0.72, -0.02)]);
  const alternative = new THREE.CatmullRomCurve3([transferStart, new THREE.Vector3(0.28, 0.59, 0.05), new THREE.Vector3(0.65, 0.64, -0.35), new THREE.Vector3(0.6, 0.72, -0.39)]);
  function pathLine(curve, color, dashed, opacity) {
    const material = dashed ? new THREE.LineDashedMaterial({ color, dashSize: 0.035, gapSize: 0.025, transparent: true, opacity }) : new THREE.LineBasicMaterial({ color, transparent: true, opacity });
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(curve.getPoints(100)), material);
    if (dashed) line.computeLineDistances();
    scene.add(line);
    return line;
  }
  pathLine(preferred, '#e63628', false, 0.95);
  pathLine(original, '#8a8d87', true, 0.64);
  const alternateLine = pathLine(alternative, '#8a8d87', true, 0.55);
  const guideVectors = new THREE.Group();
  scene.add(guideVectors);
  for (let i = 1; i <= 6; i++) {
    const t = i / 7;
    const from = original.getPoint(t);
    const to = preferred.getPoint(t);
    const direction = to.clone().sub(from);
    const arrow = new THREE.ArrowHelper(direction.clone().normalize(), from, direction.length(), '#e63628', 0.055, 0.026);
    arrow.line.material.transparent = true;
    arrow.line.material.opacity = 0.55;
    arrow.cone.material.transparent = true;
    arrow.cone.material.opacity = 0.7;
    guideVectors.add(arrow);
  }
  guideVectors.visible = false;
  const trajectoryDot = mesh(new THREE.SphereGeometry(0.026, 20, 12), materials.red);
  trajectoryDot.position.copy(transferStart);

  const playButton = document.querySelector('#lab-play');
  const timeline = document.querySelector('#lab-timeline');
  const progress = document.querySelector('#lab-progress');
  const phaseLabel = document.querySelector('#lab-phase');
  let fraction = 0;
  let playing = false;
  let visible = false;
  let mode = 'selection';
  let previousTime = 0;
  let lastPhase = '';
  let activeFrame = 0;
  const rest = new THREE.Vector3(-0.39, 0.89, 0.1);
  const atPick = new THREE.Vector3(cubeStart.x, 0.21, cubeStart.z);
  const atPlace = new THREE.Vector3(cubeGoal.x, 0.224, cubeGoal.z);
  const smooth = value => value * value * (3 - 2 * value);
  const lerp = (a, b, t) => a.clone().lerp(b, smooth(THREE.MathUtils.clamp(t, 0, 1)));

  function poseRobot(tcp, opening) {
    const wrist = tcp.clone().add(new THREE.Vector3(0, 0.12, 0));
    const direction = wrist.clone().sub(shoulder);
    const distance = Math.min(direction.length(), 2.04);
    direction.normalize();
    const l1 = 1.08;
    const l2 = 1.02;
    const projected = (l1 * l1 - l2 * l2 + distance * distance) / (2 * distance);
    const height = Math.sqrt(Math.max(0.001, l1 * l1 - projected * projected));
    const bend = yAxis.clone().addScaledVector(direction, -yAxis.dot(direction)).normalize();
    const elbow = shoulder.clone().addScaledVector(direction, projected).addScaledVector(bend, height);
    joint2.position.copy(elbow);
    joint3.position.copy(wrist);
    const azimuth = Math.atan2(direction.x, direction.z);
    joint1.rotation.y = azimuth;
    joint2.rotation.y = azimuth;
    joint3.rotation.y = azimuth;
    linkBetween(upper, shoulder, elbow, 0.2);
    linkBetween(fore, elbow, wrist, 0.17);
    linkBetween(upperInset, shoulder, elbow, 0.15);
    upperInset.position.z += 0.083;
    linkBetween(foreInset, elbow, wrist, 0.1);
    foreInset.position.z += 0.065;
    gripper.position.copy(tcp);
    fingers[0].position.x = -(0.078 + opening * 0.052);
    fingers[1].position.x = 0.078 + opening * 0.052;
  }

  function updatePose() {
    let tcp, opening = 1, label;
    const t = fraction;
    if (t < 0.19) {
      tcp = lerp(rest, atPick, t / 0.19);
      label = 'Approach the object';
      cube.position.copy(cubeStart);
    } else if (t < 0.26) {
      tcp = atPick.clone();
      opening = 1 - (t - 0.19) / 0.07;
      label = 'Close the gripper';
      cube.position.copy(cubeStart);
    } else if (t < 0.39) {
      tcp = lerp(atPick, transferStart, (t - 0.26) / 0.13);
      opening = 0;
      cube.position.copy(tcp).add(new THREE.Vector3(0, -0.14, 0));
      label = 'Lift';
    } else if (t < 0.69) {
      const u = smooth((t - 0.39) / 0.3);
      tcp = preferred.getPoint(u);
      opening = 0;
      cube.position.copy(tcp).add(new THREE.Vector3(0, -0.14, 0));
      trajectoryDot.position.copy(tcp);
      label = mode === 'selection' ? 'Execute the selected trajectory' : 'Follow the energy-guided trajectory';
    } else if (t < 0.81) {
      tcp = lerp(transferGoal, atPlace, (t - 0.69) / 0.12);
      opening = 0;
      cube.position.copy(tcp).add(new THREE.Vector3(0, -0.14, 0));
      label = 'Lower into the tray';
    } else if (t < 0.88) {
      tcp = atPlace.clone();
      opening = (t - 0.81) / 0.07;
      cube.position.copy(cubeGoal);
      label = 'Release';
    } else {
      tcp = lerp(atPlace, transferGoal, (t - 0.88) / 0.12);
      cube.position.copy(cubeGoal);
      label = 'Placement complete';
    }
    poseRobot(tcp, opening);
    if (label !== lastPhase) { phaseLabel.textContent = label; lastPhase = label; }
    trajectoryDot.visible = t >= 0.39 && t < 0.69;
    timeline.value = String(Math.round(t * 1000));
    progress.value = `${String(Math.round(t * 100)).padStart(2, '0')}%`;
  }

  function setPlaying(value) {
    playing = value;
    document.querySelector('#lab-play-label').textContent = playing ? 'Pause movement' : fraction >= 1 ? 'Play again' : 'Play movement';
    playButton.querySelector('.play-icon').textContent = playing ? 'Ⅱ' : '▶';
    playButton.setAttribute('aria-pressed', String(playing));
    wake();
  }
  function setCamera(view) {
    const mobile = container.clientWidth < 600;
    const positions = {
      perspective: mobile ? [3.6, 2.9, 4.4] : [2.8, 2.4, 3.7],
      top: mobile ? [0, 6.6, 0.1] : [0, 5.0, 0.1],
      front: mobile ? [0.1, 2.3, 6.7] : [0.1, 1.8, 5.2]
    };
    camera.position.set(...positions[view]);
    controls.target.set(-0.1, 0.47, 0);
    controls.update();
    document.querySelectorAll('[data-camera]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.camera === view)));
    wake();
  }

  playButton.addEventListener('click', () => {
    if (fraction >= 1) fraction = 0;
    setPlaying(!playing);
  });
  document.querySelector('#lab-replay').addEventListener('click', () => { fraction = 0; updatePose(); setPlaying(true); });
  timeline.addEventListener('input', () => { fraction = Number(timeline.value) / 1000; setPlaying(false); updatePose(); wake(); });
  document.querySelectorAll('[data-camera]').forEach(button => button.addEventListener('click', () => setCamera(button.dataset.camera)));
  document.querySelectorAll('[data-lab-mode]').forEach(button => button.addEventListener('click', () => {
    mode = button.dataset.labMode;
    document.querySelectorAll('[data-lab-mode]').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
    guideVectors.visible = mode === 'guidance';
    alternateLine.visible = mode === 'selection';
    document.querySelector('.lab-legend-gray').parentElement.lastChild.textContent = mode === 'selection' ? ' Alternatives' : ' Original proposal';
    document.querySelector('#lab-mode-description').textContent = mode === 'selection' ? 'The frozen policy proposes several actions. The energy model chooses one, and the robot carries out the selected grasp and placement.' : 'The energy gradient supplies local correction directions. The red path illustrates the adjusted motion; arrows show how the proposal moves toward it.';
    fraction = 0;
    lastPhase = '';
    updatePose();
    setPlaying(true);
  }));

  let dampingFrames = 0;
  controls.addEventListener('change', () => { dampingFrames = 25; wake(); });
  function wake() {
    if (!activeFrame && visible && !document.hidden) activeFrame = requestAnimationFrame(animate);
  }
  function animate(time) {
    activeFrame = 0;
    const delta = previousTime ? Math.min((time - previousTime) / 1000, 0.05) : 0;
    previousTime = time;
    if (playing) {
      fraction = Math.min(1, fraction + delta / 11);
      updatePose();
      if (fraction >= 1) setPlaying(false);
    }
    controls.update();
    renderer.render(scene, camera);
    dampingFrames = Math.max(0, dampingFrames - 1);
    if (playing || dampingFrames > 0) wake();
  }
  const observer = new IntersectionObserver(entries => {
    visible = entries[0].isIntersecting;
    if (visible) { previousTime = 0; wake(); }
    else { cancelAnimationFrame(activeFrame); activeFrame = 0; }
  }, { threshold: 0.05 });
  observer.observe(container);
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) { cancelAnimationFrame(activeFrame); activeFrame = 0; }
    else { previousTime = 0; wake(); }
  });
  new ResizeObserver(() => {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight, false);
    wake();
  }).observe(container);
  setCamera('perspective');
  updatePose();
  renderer.setSize(container.clientWidth, container.clientHeight, false);
  camera.aspect = container.clientWidth / container.clientHeight;
  camera.updateProjectionMatrix();
  renderer.render(scene, camera);
  document.querySelector('#lab-loading').hidden = true;
  container.setAttribute('aria-busy', 'false');
  container.dataset.ready = 'true';
  [playButton, timeline, document.querySelector('#lab-replay')].forEach(control => { control.disabled = false; });
}
