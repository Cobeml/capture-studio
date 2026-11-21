// Capture Studio demo viewer: video -> point cloud -> BIM -> agentic design. Reads everything from the MCP server's state.
import * as THREE from 'three';
import { OrbitControls } from '/web/vendor/OrbitControls.js';

const $ = (s) => document.querySelector(s);
const ACCENT = 0xff8a3d, BAD = 0xff5a5f;
const USE_COLOR = { living: 0x4c9aff, bedroom: 0x9b7bff, office: 0x3ddc97, study: 0x3ddc97, dining: 0x4c9aff,
  kitchen: 0xffc857, bathroom: 0x5ad1e6, store: 0x8a93a0, corridor: 0x8a93a0 };

// ---------------------------------------------------------------- three setup (Z up, metres)
const renderer = new THREE.WebGLRenderer({ canvas: $('#gl'), antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.localClippingEnabled = true;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0d10);
const camera = new THREE.PerspectiveCamera(42, 1, 0.05, 400);
camera.up.set(0, 0, 1);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.maxPolarAngle = Math.PI * 0.49; controls.autoRotateSpeed = 0.9;
scene.add(new THREE.HemisphereLight(0xffffff, 0x30343c, 1.5));
const sun = new THREE.DirectionalLight(0xffffff, 1.6); sun.position.set(6, -9, 14); scene.add(sun);
const grid = new THREE.GridHelper(60, 60, 0x1c2128, 0x151a20); grid.rotation.x = Math.PI / 2; grid.position.z = -0.32; scene.add(grid);
const clipCeiling = new THREE.Plane(new THREE.Vector3(0, 0, -1), 2.6);

const world = new THREE.Group(); scene.add(world);        // BIM elements
const overlay = new THREE.Group(); scene.add(overlay);    // egress paths etc.
let cloud = null, meta = null, camMarker = null, pathLine = null;
let center = new THREE.Vector3(4, 3, 1), radius = 9;

function resize() { renderer.setSize(innerWidth, innerHeight, false); camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); }
addEventListener('resize', resize); resize();

function frameCamera(az = -0.9, el = 0.82, k = 1.55, instant = false) {
  const d = radius * k;
  const to = new THREE.Vector3(center.x + d * Math.cos(el) * Math.cos(az), center.y + d * Math.cos(el) * Math.sin(az), center.z + d * Math.sin(el));
  camGoal = { pos: to, t: instant ? 1 : 0, from: camera.position.clone() };
  controls.target.copy(center);
  if (instant) camera.position.copy(to);
}
let camGoal = null;

// ---------------------------------------------------------------- point cloud
async function loadCloud() {
  const m = await fetch('/data/scan/meta.json'); if (!m.ok) return;
  meta = await m.json();
  const buf = await (await fetch('/data/scan/cloud.bin?v=' + meta.n)).arrayBuffer();
  const n = meta.n, pos = new Float32Array(buf, 0, n * 3), col = new Uint8Array(buf, n * 12, n * 3);
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('color', new THREE.BufferAttribute(col, 3, true));
  cloud = new THREE.Points(g, new THREE.PointsMaterial({ size: 0.028, vertexColors: true, transparent: true, clippingPlanes: [clipCeiling], depthWrite: true }));
  cloud.visible = false; scene.add(cloud);
  center.set(...meta.center); radius = meta.radius;
  clipCeiling.constant = meta.ceiling_height - 0.2;
  // moving phone marker + path
  camMarker = new THREE.Mesh(new THREE.ConeGeometry(0.11, 0.26, 4), new THREE.MeshBasicMaterial({ color: ACCENT }));
  camMarker.visible = false; scene.add(camMarker);
  const pg = new THREE.BufferGeometry().setFromPoints(meta.cameras.map((c) => new THREE.Vector3(c[0], c[1], c[2])));
  pathLine = new THREE.Line(pg, new THREE.LineBasicMaterial({ color: ACCENT, transparent: true, opacity: 0.8 }));
  pathLine.visible = false; scene.add(pathLine);
  frameCamera(-0.9, 0.95, 2.5, true);
}

function revealTo(frame) {
  if (!cloud) return;
  const f = Math.max(0, Math.min(meta.frames, Math.floor(frame)));
  cloud.geometry.setDrawRange(0, meta.frame_offsets[f]);
  pathLine.geometry.setDrawRange(0, Math.max(2, f));
  const c = meta.cameras[Math.min(f, meta.frames - 1)];
  camMarker.position.set(c[0], c[1], c[2]);
  camMarker.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3(c[3], c[4], c[5]).normalize());
}

// ---------------------------------------------------------------- BIM elements
const groups = new Map();   // id -> { group, key, born, el }
const edgeMat = (c, o = 0.9) => new THREE.LineBasicMaterial({ color: c, transparent: true, opacity: o });

function buildElement(e) {
  const g = new THREE.Group();
  g.position.set(...e.pose.pos); g.rotation.z = e.pose.yaw;
  const proposed = e.provenance === 'proposed';
  if (e.kind === 'space') {
    const shape = new THREE.Shape(e.polygon.map((p) => new THREE.Vector2(p[0], p[1])));
    const mesh = new THREE.Mesh(new THREE.ShapeGeometry(shape), new THREE.MeshBasicMaterial({ color: USE_COLOR[e.use] ?? 0x8a93a0, transparent: true, opacity: 0.26, depthWrite: false, side: THREE.DoubleSide }));
    mesh.position.z = 0.015; g.add(mesh);
    const pts = e.polygon.map((p) => new THREE.Vector3(p[0], p[1], 0.02)); pts.push(pts[0]);
    g.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), edgeMat(USE_COLOR[e.use] ?? 0x8a93a0, 0.9)));
    return g;
  }
  for (const b of e.boxes) {
    const geo = new THREE.BoxGeometry(...b.s);
    let mat;
    if (e.kind === 'window') mat = new THREE.MeshStandardMaterial({ color: 0x9fd3e6, transparent: true, opacity: 0.35, roughness: 0.1 });
    else if (e.kind === 'door') mat = new THREE.MeshStandardMaterial({ color: proposed ? ACCENT : 0x9a7b5a, transparent: true, opacity: 0.55 });
    else if (e.kind === 'slab') mat = new THREE.MeshStandardMaterial({ color: 0x3a3f47, roughness: 0.95 });
    else mat = new THREE.MeshStandardMaterial({ color: proposed ? 0xffb37e : 0xd9dde2, roughness: 0.85, transparent: !proposed, opacity: proposed ? 1 : 0.78,
      emissive: proposed ? ACCENT : 0x000000, emissiveIntensity: proposed ? 0.18 : 0 });
    const mesh = new THREE.Mesh(geo, mat); mesh.position.set(...b.c); mesh.userData.id = e.id; g.add(mesh);
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geo), edgeMat(proposed ? ACCENT : 0x6b7480, proposed ? 1 : 0.55));
    edges.position.copy(mesh.position); g.add(edges);
  }
  return g;
}

function applyModel(model, { animate = false, stagger = 0, flag = [], flagColor = ACCENT } = {}) {
  const now = performance.now(); let i = 0;
  const seen = new Set();
  for (const e of Object.values(model.elements)) {
    seen.add(e.id);
    const key = JSON.stringify(e), old = groups.get(e.id);
    if (old && old.key === key) { setFlag(old, flag.includes(e.id) ? flagColor : null); continue; }
    if (old) world.remove(old.group);
    const group = buildElement(e);
    const rec = { group, key, el: e, born: animate ? now + (stagger ? i++ * stagger : 0) : 0, base: group.position.clone(), baseQ: group.quaternion.clone() };
    if (animate && e.kind !== 'space') group.scale.z = 0.001;
    world.add(group); groups.set(e.id, rec);
    setFlag(rec, flag.includes(e.id) ? flagColor : null);
  }
  for (const [id, rec] of groups) if (!seen.has(id)) { world.remove(rec.group); groups.delete(id); }
  clipCeiling.constant = (model.level.ceiling_height ?? 2.9) - 0.2;
  $('#rev').textContent = model.revision ? 'R' + model.revision : '';
  buildLabels(model);
}

function setFlag(rec, color) {
  rec.group.traverse((o) => {
    if (o.isLineSegments) { o.material.color.setHex(color ?? (rec.el.provenance === 'proposed' ? ACCENT : 0x6b7480)); o.material.opacity = color ? 1 : (rec.el.provenance === 'proposed' ? 1 : 0.55); }
    if (o.isMesh && o.material.emissive && rec.el.kind !== 'slab') { o.material.emissive.setHex(color ?? (rec.el.provenance === 'proposed' ? ACCENT : 0)); o.material.emissiveIntensity = color === BAD ? 0.7 : (rec.el.provenance === 'proposed' ? 0.18 : 0); }
  });
}

let labels = [];
function buildLabels(model) {
  $('#labels').innerHTML = ''; labels = [];
  for (const e of Object.values(model.elements)) if (e.kind === 'space') {
    const cx = e.polygon.reduce((a, p) => a + p[0], 0) / e.polygon.length, cy = e.polygon.reduce((a, p) => a + p[1], 0) / e.polygon.length;
    let a = 0; e.polygon.forEach((p, i) => { const q = e.polygon[(i + 1) % e.polygon.length]; a += p[0] * q[1] - q[0] * p[1]; });
    const d = document.createElement('div'); d.className = 'label'; d.innerHTML = `${e.name}<small>${Math.abs(a / 2).toFixed(1)} m²</small>`;
    $('#labels').appendChild(d); labels.push({ d, v: new THREE.Vector3(cx, cy, 0.05) });
  }
}

function drawEgress(v) {
  overlay.clear();
  if (!v?.egress?.paths) return;
  for (const p of Object.values(v.egress.paths)) {
    if (!p) continue;
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(p.points.map((q) => new THREE.Vector3(q[0], q[1], 0.06))),
      new THREE.LineDashedMaterial({ color: 0x3ddc97, dashSize: 0.18, gapSize: 0.12 }));
    line.computeLineDistances(); overlay.add(line);
  }
}

// ---------------------------------------------------------------- UI: pills, side cards, caption
function pills(v) {
  const set = (id, ok, hold) => { $(id).className = 'pill ' + (ok == null ? '' : ok ? 'ok' : hold ? 'hold' : 'bad'); };
  set('#p-code', v?.code_ok); set('#p-physics', v?.physics_ok);
  set('#p-safety', v?.safety_ok, v && !v.safety_ok && v.counts?.hold > 0 && true);
  set('#p-wishes', v ? v.desires_ok : null, true);
}

function side(S, live) {
  let h = '';
  if (S.requirements.length) h += `<div class="card"><h4>Wishes</h4>${S.requirements.map((r) => `<div class="row ${r.status}"><b></b>${esc(r.text)}</div>`).join('')}</div>`;
  const iss = live ? (S.validation?.issues ?? []) : [];
  const open = iss.filter((i) => i.severity !== 'warn'), warn = iss.filter((i) => i.severity === 'warn');
  if (open.length) h += `<div class="card"><h4>To resolve</h4>${open.slice(0, 6).map((i) => `<div class="row ${i.severity}"><b></b>${esc(i.rule)}<small>${esc(i.category)}</small></div>`).join('')}</div>`;
  if (live && S.next_steps.length) h += `<div class="card"><h4>Next steps</h4><ol class="steps">${S.next_steps.map((s) => `<li>${esc(s.title)} <small>· ${esc(s.who)}</small></li>`).join('')}</ol></div>`;
  else if (warn.length) h += `<div class="card"><h4>On site</h4>${warn.slice(0, 4).map((i) => `<div class="row warn"><b></b>${esc(i.rule)}</div>`).join('')}</div>`;
  $('#side').innerHTML = h;
  const q = S.questions.find((x) => x.answer == null);
  $('#q').style.display = q ? 'block' : 'none';
  if (q && $('#q').dataset.id !== q.id) {
    $('#q').dataset.id = q.id; $('#q p').textContent = q.question; $('#q div').innerHTML = '';
    for (const o of q.options) { const b = document.createElement('button'); b.textContent = o; b.onclick = () => post('/api/answer', { id: q.id, answer: o }); $('#q div').appendChild(b); }
  }
}
const esc = (s) => String(s ?? '').replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
const post = (u, b) => fetch(u, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(b) }).then(poll);
const caption = (t) => { $('#caption').textContent = t || ''; };

// ---------------------------------------------------------------- stages + replay
let S = null, stage = 'scan', queue = [], shown = 0, busyUntil = 0, anim = null, autoplay = false, stageAt = 0, selected = null;
const video = $('#video');

function setStage(s) {
  stage = s; stageAt = performance.now(); document.body.dataset.stage = s;
  document.querySelectorAll('#steps button[data-s]').forEach((b) => b.classList.toggle('on', b.dataset.s === s));
  controls.autoRotate = s === 'cloud' || s === 'bim';
  if (cloud) { cloud.visible = s !== 'scan'; cloud.material.opacity = s === 'cloud' ? 1 : s === 'bim' ? 0.32 : 0.07; camMarker.visible = pathLine.visible = s === 'cloud'; }
  world.visible = s === 'bim' || s === 'design'; overlay.visible = s === 'design'; $('#labels').style.display = s === 'design' ? '' : 'none';
  anim = null;
  if (s === 'scan') { video.loop = true; video.playbackRate = 1; video.currentTime = 0; video.play().catch(() => {}); caption('Phone video of the site'); $('#stat').textContent = ''; }
  if (s === 'cloud') { video.loop = false; video.currentTime = 0; video.playbackRate = 3; video.play().catch(() => {}); caption('LingBot-Map → point cloud'); frameCamera(-0.9, 0.95, 2.5); }
  if (s === 'bim') {
    video.pause(); revealTo(1e9); groups.forEach((r) => world.remove(r.group)); groups.clear();
    const h0 = S?.history?.[0];
    if (h0) { applyModel(h0.model, { animate: true, stagger: 90 }); const c = count(h0.model);
      caption('Point cloud → as-built BIM'); $('#stat').textContent = `${c.wall} walls · ${c.door} doors · ${c.window} windows · scan fit P50 ${h0.model.scan?.fit_p50_mm ?? '–'} mm`; }
    else caption('No BIM yet — run fit_bim');
    frameCamera(-0.9, 1.0, 2.4);
  }
  if (s === 'design') {
    video.pause(); revealTo(1e9); controls.autoRotate = false; $('#stat').textContent = '';
    if (S?.history?.length) { applyModel(S.history[0].model); queue = S.history.slice(1); shown = S.history.length; busyUntil = performance.now() + 700; pills(null); side(S, false); caption('Agent designs with you'); }
    frameCamera(-1.15, 1.08, 2.3);
  }
}
const count = (m) => Object.values(m.elements).reduce((a, e) => (a[e.kind] = (a[e.kind] ?? 0) + 1, a), { wall: 0, door: 0, window: 0 });

function stepReplay(now) {
  if (stage !== 'design' || !S || now < busyUntil) return;
  if (queue.length) {
    const ev = queue.shift();
    const failing = ev.action === 'validate' && ev.validation && !(ev.validation.code_ok && ev.validation.safety_ok && ev.validation.physics_ok);
    applyModel(ev.model, { animate: true, flag: ev.focus, flagColor: failing ? BAD : ACCENT });
    caption(ev.caption); pills(ev.validation);
    busyUntil = now + (ev.action === 'validate' ? 1700 : 1150);
    if (ev.animation?.frames?.length) { anim = { frames: ev.animation.frames, fps: ev.animation.fps, t0: now + 500 }; busyUntil = now + 500 + ev.animation.frames.length * 1000 / ev.animation.fps + 900; }
    if (!queue.length) busyUntil += 300;
    return;
  }
  if (anim) { anim = null; groups.forEach((r) => { r.group.position.copy(r.base); r.group.quaternion.copy(r.baseQ); }); }
  if (S._liveShown !== S.seq) {       // settled on the live model
    S._liveShown = S.seq;
    const bad = (S.validation?.issues ?? []).filter((i) => i.severity === 'fail' && i.element).map((i) => i.element);
    applyModel(S.model, { animate: true, flag: bad, flagColor: BAD });
    pills(S.validation); side(S, true); drawEgress(S.validation);
    caption(S.agent_note || S.history.at(-1)?.caption || '');
  }
}

async function poll() {
  try {
    const r = await fetch('/api/state?since=' + (S?.seq ?? -1));
    if (r.status !== 200) return;
    const prev = S; S = await r.json();
    if (stage === 'design') {
      if (S.history.length > shown) queue.push(...S.history.slice(shown));
      else if (S.history.length < shown) { queue = []; groups.forEach((g) => world.remove(g.group)); groups.clear(); }
      shown = S.history.length; side(S, !queue.length);
    } else if (!prev && S.stage === 'design' && location.hash === '#design') setStage('design');
  } catch { /* server restarting */ }
}

// ---------------------------------------------------------------- interaction
document.querySelectorAll('#steps button[data-s]').forEach((b) => (b.onclick = () => { autoplay = false; setStage(b.dataset.s); }));
$('#play').onclick = () => { autoplay = true; setStage('scan'); };
$('#say form').onsubmit = (ev) => {
  ev.preventDefault(); const i = $('#say input'); if (!i.value.trim()) return;
  post('/api/message', { text: i.value.trim(), selected }); i.value = ''; $('#sent').textContent = 'Sent to the agent';
  setTimeout(() => ($('#sent').textContent = selected ? 'Selected: ' + groups.get(selected)?.el.name : ''), 2500);
};
const ray = new THREE.Raycaster(); let down = null;
renderer.domElement.addEventListener('pointerdown', (e) => (down = [e.clientX, e.clientY]));
renderer.domElement.addEventListener('pointerup', (e) => {
  if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4 || stage !== 'design') return;
  ray.setFromCamera(new THREE.Vector2((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1), camera);
  const hit = ray.intersectObjects(world.children, true).find((h) => h.object.userData.id && groups.get(h.object.userData.id)?.el.kind !== 'slab');
  if (selected && groups.get(selected)) setFlag(groups.get(selected), null);
  selected = hit ? hit.object.userData.id : null;
  if (selected) setFlag(groups.get(selected), 0xffffff);
  $('#sent').textContent = selected ? 'Selected: ' + groups.get(selected).el.name : '';
});

// ---------------------------------------------------------------- loop
const v3 = new THREE.Vector3();
function tick(now) {
  requestAnimationFrame(tick);
  if (camGoal && camGoal.t < 1) { camGoal.t = Math.min(1, camGoal.t + 0.025); const k = 1 - Math.pow(1 - camGoal.t, 3); camera.position.lerpVectors(camGoal.from, camGoal.pos, k); }
  if (stage === 'cloud' && meta) revealTo(video.ended ? 1e9 : video.currentTime * meta.fps);
  for (const r of groups.values()) if (r.born) {
    const k = Math.min(1, Math.max(0, (now - r.born) / 550)); r.group.scale.z = Math.max(0.001, 1 - Math.pow(1 - k, 3)); if (k >= 1) r.born = 0;
  }
  if (anim && now >= anim.t0) {
    const f = anim.frames[Math.min(anim.frames.length - 1, Math.floor((now - anim.t0) / 1000 * anim.fps))];
    for (const [id, p] of Object.entries(f)) { const r = groups.get(id); if (r) { r.group.position.set(p[0], p[1], p[2]); r.group.quaternion.set(p[4], p[5], p[6], p[3]); } }
  }
  stepReplay(now);
  if (autoplay) {
    const dt = now - stageAt;
    if (stage === 'scan' && dt > 4500) setStage('cloud');
    else if (stage === 'cloud' && (video.ended || dt > 16000) && dt > 3000) { if (dt > (video.ended ? 0 : 16000) && !tick.hold) { tick.hold = now; } if (tick.hold && now - tick.hold > 2200) { tick.hold = 0; setStage('bim'); } }
    else if (stage === 'bim' && dt > 5500) { autoplay = false; setStage('design'); }
  }
  controls.update(); renderer.render(scene, camera);
  if (stage === 'design') for (const l of labels) { v3.copy(l.v).project(camera); l.d.style.left = ((v3.x + 1) / 2 * innerWidth) + 'px'; l.d.style.top = ((1 - v3.y) / 2 * innerHeight) + 'px'; }
}

await loadCloud();
await poll();
const start = location.hash.slice(1);
setStage(['scan', 'cloud', 'bim', 'design'].includes(start) ? start : 'scan');
setInterval(poll, 800);
requestAnimationFrame(tick);
