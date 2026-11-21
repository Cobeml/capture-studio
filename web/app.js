// Capture Studio demo viewer: video -> point cloud -> BIM -> agentic design. Reads everything from the MCP server's state.
// Every animation is a function of clock(). With ?capture=1 the clock is virtual and advanced by window.__demo.frame(ms)
// (scripts/record_demo.py steps it at 30 fps and screenshots each frame), otherwise it is performance.now().
import * as THREE from 'three';
import { OrbitControls } from '/web/vendor/OrbitControls.js';

const $ = (s) => document.querySelector(s);
const PARAMS = new URLSearchParams(location.search);
const CAPTURE = PARAMS.has('capture');
let vnow = 0;
const clock = () => (CAPTURE ? vnow : performance.now());
const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const ease = (x) => { x = clamp(x); return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2; };
const easeOut = (x) => 1 - Math.pow(1 - clamp(x), 3);
const easeBack = (x) => { x = clamp(x); const c = 1.5; return 1 + (c + 1) * Math.pow(x - 1, 3) + c * Math.pow(x - 1, 2); };
const span = (t, a, b) => clamp((t - a) / (b - a));
const fmt = (n) => Math.round(n).toLocaleString('en-US');

const ACCENT = 0xff8a3d, BAD = 0xff5a5f, OK = 0x3ddc97;
const USE_COLOR = { living: 0x4c9aff, bedroom: 0x9b7bff, office: 0x3ddc97, study: 0x3ddc97, dining: 0x4c9aff,
  kitchen: 0xffc857, bathroom: 0x5ad1e6, store: 0x8a93a0, corridor: 0x8a93a0, workshop: 0xffc857, production: 0xffc857,
  storage: 0x8a93a0, warehouse: 0x8a93a0 };
const SCAN_MS = 6000, CLOUD_TARGET_MS = 14000, CLOUD_HOLD_MS = 2800, BIM_MS = 11000;
const CUT = 1.2, CUT_TALL = 2.2;               // design-stage cutaway heights: walls/openings, equipment/other
const CUT_KINDS = new Set(['wall', 'door', 'window', 'column', 'opening']);

// ---------------------------------------------------------------- three setup (Z up, metres)
const renderer = new THREE.WebGLRenderer({ canvas: $('#gl'), antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(CAPTURE ? 1 : Math.min(devicePixelRatio, 2));
renderer.localClippingEnabled = true;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0d10);
const camera = new THREE.PerspectiveCamera(40, 1, 0.05, 2000);
camera.up.set(0, 0, 1);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.maxPolarAngle = Math.PI * 0.49;
let manual = false;
controls.addEventListener('start', () => (manual = true));
const hemi = new THREE.HemisphereLight(0xffffff, 0x30343c, 1.6); scene.add(hemi);
const sun = new THREE.DirectionalLight(0xffffff, 1.7); sun.position.set(6, -9, 14); scene.add(sun); scene.add(sun.target);
renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
sun.castShadow = true; sun.shadow.mapSize.set(4096, 4096); sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.03;
const fill = new THREE.DirectionalLight(0x9fb4d0, 0.5); fill.position.set(-8, 7, 5); scene.add(fill);
let grid = null;

const world = new THREE.Group(); scene.add(world);        // BIM elements
const overlay = new THREE.Group(); scene.add(overlay);    // egress paths etc.
const NOCLIP = () => new THREE.Plane(new THREE.Vector3(0, 0, 1), 1e6);
const wipePlane = NOCLIP();                                // model side of the scan|model compare wipe
let cloud = null, cloudU = null, meta = null, frustum = null, pathLine = null;
const sceneC = new THREE.Vector3(4, 3, 1); let sceneR = 9, ceiling = 2.9, cloudC = null, cloudR = null;

function resize() {
  renderer.setSize(innerWidth, innerHeight, false); camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
  updateScale();
}
function updateScale() { if (cloudU) cloudU.uScale.value = renderer.domElement.height / 2 / Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2); }
addEventListener('resize', resize); resize();

// ---------------------------------------------------------------- camera director (target + azimuth/elevation/distance)
const cam = { target: new THREE.Vector3(), az: -0.9, el: 0.9, dist: 20 };
let camFrom = null, camAt = 0, camShot = null, focus = { c: null, w: 0, goal: 0 };
const fitDist = (R) => { const v = THREE.MathUtils.degToRad(camera.fov) / 2, h = Math.atan(Math.tan(v) * camera.aspect); return R / Math.sin(Math.min(v, h)); };
const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));

function shot(t) {          // camera for the current stage at stage time t (ms)
  const s = t / 1000;
  if (stage === 'scan' || stage === 'cloud') {
    const C = cloudC ?? sceneC, R = cloudR ?? sceneR;
    return { target: C, az: -0.9 + (stage === 'cloud' ? s * 0.055 : 0), el: 0.92 - (stage === 'cloud' ? Math.min(s, 16) * 0.012 : 0), dist: fitDist(R) * (1.0 - (stage === 'cloud' ? Math.min(s, 16) * 0.006 : 0)) };
  }
  if (stage === 'bim') return { target: sceneC, az: camShot.az0 + s * 0.05, el: 0.8, dist: fitDist(sceneR) * 0.95 };
  // design: higher, slow drift, gentle push towards what the agent just touched
  const w = focus.w, tgt = sceneC.clone();
  if (focus.c && w > 0) tgt.lerp(focus.c, 0.4 * w);
  return { target: tgt, az: camShot.az0 + s * 0.012, el: 0.95, dist: fitDist(sceneR) * (0.98 - 0.12 * w) };
}

function camStage() { camFrom = { target: cam.target.clone(), az: cam.az, el: cam.el, dist: cam.dist }; camAt = clock(); camShot = { az0: cam.az }; manual = false; }

function updateCamera(now) {
  if (manual) { controls.update(); return; }
  const g = shot(now - stageAt);
  const k = camFrom ? ease((now - camAt) / 1900) : 1;
  if (camFrom) {
    cam.target.lerpVectors(camFrom.target, g.target, k);
    cam.az = camFrom.az + wrap(g.az - camFrom.az) * k; cam.el = camFrom.el + (g.el - camFrom.el) * k; cam.dist = camFrom.dist + (g.dist - camFrom.dist) * k;
    if (k >= 1) camFrom = null;
  } else { cam.target.copy(g.target); cam.az = g.az; cam.el = g.el; cam.dist = g.dist; }
  const { target: c, az, el, dist: d } = cam;
  camera.position.set(c.x + d * Math.cos(el) * Math.cos(az), c.y + d * Math.cos(el) * Math.sin(az), c.z + d * Math.sin(el));
  camera.lookAt(c); controls.target.copy(c);
}

// ---------------------------------------------------------------- point cloud (custom shader: capture-order reveal + fresh-point glow)
const CLOUD_VS = `
#include <common>
#include <clipping_planes_pars_vertex>
uniform float uReveal, uLo, uFresh, uSize, uScale, uMinPx, uMaxPx; uniform vec3 uHi; uniform vec4 uWipe;
varying vec3 vColor; varying float vSide;
void main() {
  float id = float(gl_VertexID);
  if (id >= uReveal || id < uLo) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; return; }
  vec4 wp = modelMatrix * vec4(position, 1.0);
  vec4 mvPosition = viewMatrix * wp;
  float fresh = 1.0 - smoothstep(0.0, 1.0, (uReveal - id) / max(uFresh, 1.0));
  vColor = mix(color, uHi, fresh * 0.8);
  vSide = dot(uWipe.xyz, wp.xyz) + uWipe.w;
  gl_PointSize = clamp(uSize * uScale / -mvPosition.z * (1.0 + fresh * 1.6), uMinPx, uMaxPx);
  gl_Position = projectionMatrix * mvPosition;
  #include <clipping_planes_vertex>
}`;
const CLOUD_FS = `
#include <clipping_planes_pars_fragment>
uniform float uOpScan, uOpModel, uRound; varying vec3 vColor; varying float vSide;
void main() {
  #include <clipping_planes_fragment>
  vec2 c = gl_PointCoord - 0.5; if (uRound > 0.5 && dot(c, c) > 0.25) discard;
  gl_FragColor = vec4(vColor, vSide < 0.0 ? uOpScan : uOpModel);
}`;
const clipCeiling = new THREE.Plane(new THREE.Vector3(0, 0, -1), 2.7);

async function loadCloud() {
  const m = await fetch('/data/scan/meta.json'); if (!m.ok) return;
  meta = await m.json();
  const buf = await (await fetch('/data/scan/cloud.bin?v=' + meta.n)).arrayBuffer();
  const n = meta.n, pos = new Float32Array(buf, 0, n * 3), col = new Uint8Array(buf, n * 12, n * 3);
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('color', new THREE.BufferAttribute(col, 3, true));
  cloudU = { uReveal: { value: n }, uLo: { value: 0 }, uFresh: { value: n / Math.max(1, meta.frames) }, uSize: { value: (meta.voxel_m ?? 0.02) * 1.35 },
    uScale: { value: 500 }, uHi: { value: new THREE.Color(1.0, 0.62, 0.32) }, uWipe: { value: new THREE.Vector4(0, 0, 0, 1) },
    uOpScan: { value: 1 }, uOpModel: { value: 1 }, uMinPx: { value: 1.5 }, uMaxPx: { value: 64 }, uRound: { value: 1 } };
  updateScale();
  cloud = new THREE.Points(g, new THREE.ShaderMaterial({ uniforms: cloudU, vertexShader: CLOUD_VS, fragmentShader: CLOUD_FS, vertexColors: true,
    transparent: true, depthWrite: true, clipping: true, clippingPlanes: [clipCeiling] }));
  cloud.frustumCulled = false; cloud.visible = false; scene.add(cloud);
  ceiling = meta.ceiling_height ?? ceiling;
  cloudC = new THREE.Vector3(meta.center[0], meta.center[1], Math.min(meta.center[2], ceiling * 0.35));
  cloudR = Math.hypot(meta.radius, ceiling / 2);
  sceneC.copy(cloudC); sceneR = cloudR;
  clipCeiling.constant = ceiling - 0.2;
  // phone frustum + captured path
  const sz = clamp(meta.radius * 0.085, 0.3, 1.4), w = sz * 0.75, h = sz * 0.45;
  const P = [[0, 0, 0], [-w, sz, -h], [w, sz, -h], [w, sz, h], [-w, sz, h]].map((p) => new THREE.Vector3(...p));
  const segs = [0, 1, 0, 2, 0, 3, 0, 4, 1, 2, 2, 3, 3, 4, 4, 1].map((i) => P[i]);
  frustum = new THREE.Group();
  frustum.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(segs), new THREE.LineBasicMaterial({ color: ACCENT })));
  const face = new THREE.BufferGeometry().setFromPoints([P[1], P[2], P[3], P[1], P[3], P[4]]);
  frustum.add(new THREE.Mesh(face, new THREE.MeshBasicMaterial({ color: ACCENT, transparent: true, opacity: 0.22, side: THREE.DoubleSide, depthWrite: false })));
  const glow = new THREE.Mesh(new THREE.SphereGeometry(sz * 0.12, 16, 12), new THREE.MeshBasicMaterial({ color: 0xffffff }));
  frustum.add(glow);
  frustum.visible = false; scene.add(frustum);
  pathLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: ACCENT, transparent: true, opacity: 0.85 }));
  pathLine.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array((meta.frames + 1) * 3), 3));
  pathLine.visible = false; pathLine.frustumCulled = false; scene.add(pathLine);
  buildGrid();
}

function buildGrid() {
  if (grid) scene.remove(grid);
  const size = Math.max(20, Math.ceil(sceneR * 6 / 5) * 5);
  grid = new THREE.GridHelper(size, size, 0x1c2128, 0x14181e); grid.rotation.x = Math.PI / 2;
  grid.position.set(Math.round(sceneC.x), Math.round(sceneC.y), -0.3); scene.add(grid);
}

const camAtFrame = (f) => {     // interpolated phone pose at fractional frame f
  const C = meta.cameras, i = clamp(Math.floor(f), 0, C.length - 1), j = Math.min(i + 1, C.length - 1), k = clamp(f - i);
  const a = C[i], b = C[j];
  return { p: new THREE.Vector3(a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k),
    f: new THREE.Vector3(a[3] + (b[3] - a[3]) * k, a[4] + (b[4] - a[4]) * k, a[5] + (b[5] - a[5]) * k).normalize() };
};

function revealTo(frame) {      // fractional frame: points revealed in capture order, synced to the inset video
  if (!cloud) return 0;
  const F = meta.frames, O = meta.frame_offsets, f = clamp(frame, 0, F);
  const i = Math.min(Math.floor(f), F - 1), k = f >= F ? 1 : f - i;
  const o1 = O[i + 1] ?? meta.n, n = f >= F ? meta.n : O[i] + (o1 - O[i]) * k;
  cloudU.uReveal.value = n;
  cloudU.uFresh.value = Math.max(2000, (o1 - O[i]) * 0.9);
  cloud.geometry.setDrawRange(0, Math.ceil(n));
  // path up to the current pose + frustum
  const now = camAtFrame(Math.min(f, F - 1)), arr = pathLine.geometry.attributes.position.array;
  const m = Math.min(Math.floor(f), F - 1);
  for (let j = 0; j <= m; j++) { const c = meta.cameras[j]; arr.set([c[0], c[1], c[2]], j * 3); }
  arr.set([now.p.x, now.p.y, now.p.z], (m + 1) * 3);
  pathLine.geometry.attributes.position.needsUpdate = true; pathLine.geometry.setDrawRange(0, m + 2);
  const r = new THREE.Vector3().crossVectors(now.f, new THREE.Vector3(0, 0, 1));
  if (r.lengthSq() < 1e-6) r.set(1, 0, 0); r.normalize();
  const u = new THREE.Vector3().crossVectors(r, now.f).normalize();
  frustum.position.copy(now.p); frustum.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(r, now.f, u));
  return n;
}

// ---------------------------------------------------------------- BIM elements
const groups = new Map();   // id -> rec
const edgeMat = (c, o = 0.9) => new THREE.LineBasicMaterial({ color: c, transparent: true, opacity: o, clippingPlanes: [wipePlane] });
const hex = (s, d) => { try { return s ? new THREE.Color(s) : new THREE.Color(d); } catch { return new THREE.Color(d); } };

function elementMaterial(e) {
  const proposed = e.provenance === 'proposed', c = hex(e.color, 0x9aa0a6), clip = [wipePlane];
  if (e.kind === 'window') return new THREE.MeshStandardMaterial({ color: proposed ? 0xffc9a0 : 0x9fd3e6, transparent: true, opacity: 0.38, roughness: 0.1, depthWrite: false, clippingPlanes: clip });
  if (e.kind === 'door') return new THREE.MeshStandardMaterial({ color: proposed ? ACCENT : c, transparent: true, opacity: 0.7, roughness: 0.7, clippingPlanes: clip });
  if (e.kind === 'slab') return new THREE.MeshStandardMaterial({ color: c.multiplyScalar(0.42), roughness: 0.95, clippingPlanes: clip });
  if (proposed) return new THREE.MeshStandardMaterial({ color: 0xffb37e, roughness: 0.75, emissive: ACCENT, emissiveIntensity: 0.2, clippingPlanes: clip });
  return new THREE.MeshStandardMaterial({ color: c, roughness: 0.85, metalness: e.kind === 'equipment' ? 0.15 : 0, emissive: 0x000000, clippingPlanes: clip });
}

function buildElement(e) {
  const g = new THREE.Group();
  g.position.set(...e.pose.pos); g.rotation.z = e.pose.yaw;
  const proposed = e.provenance === 'proposed';
  const parts = [];
  if (e.kind === 'space' && e.polygon?.length > 2) {
    const col = USE_COLOR[e.use] ?? 0x8a93a0;
    const shape = new THREE.Shape(e.polygon.map((p) => new THREE.Vector2(p[0], p[1])));
    const mat = new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.3, depthWrite: false, side: THREE.DoubleSide, clippingPlanes: [wipePlane] });
    const mesh = new THREE.Mesh(new THREE.ShapeGeometry(shape), mat); mesh.position.z = 0.02; mesh.userData.id = e.id; mesh.renderOrder = 1; g.add(mesh);
    const pts = e.polygon.map((p) => new THREE.Vector3(p[0], p[1], 0.03)); pts.push(pts[0]);
    g.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), edgeMat(col, 0.95)));
    return { g, parts, fades: [[mat, 0.3]] };
  }
  const cutH = CUT_KINDS.has(e.kind) ? CUT : CUT_TALL;
  for (const b of e.boxes ?? []) {
    const geo = new THREE.BoxGeometry(...b.s), bg = new THREE.Group();
    bg.position.set(b.c[0], b.c[1], b.c[2]);
    const mesh = new THREE.Mesh(geo, elementMaterial(e)); mesh.userData.id = e.id; mesh.castShadow = e.kind !== 'slab' && e.kind !== 'window'; mesh.receiveShadow = true; bg.add(mesh);
    if (e.kind !== 'slab') bg.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo), edgeMat(proposed ? ACCENT : 0x2b3038, proposed ? 1 : 0.6)));
    g.add(bg);
    parts.push({ bg, bottom: b.c[2] - b.s[2] / 2, h: b.s[2], z0: e.pose.pos[2], cut: proposed || e.kind === 'slab' ? null : cutH });
  }
  return { g, parts, fades: [] };
}

function elementBox(e, box = new THREE.Box3()) {
  const [x, y, z] = e.pose.pos, c = Math.cos(e.pose.yaw), s = Math.sin(e.pose.yaw);
  if (e.kind === 'space' && e.polygon) { for (const p of e.polygon) box.expandByPoint(new THREE.Vector3(x + c * p[0] - s * p[1], y + s * p[0] + c * p[1], z)); return box; }
  for (const b of e.boxes ?? []) for (const dx of [-1, 1]) for (const dy of [-1, 1]) for (const dz of [-1, 1]) {
    const lx = b.c[0] + dx * b.s[0] / 2, ly = b.c[1] + dy * b.s[1] / 2;
    box.expandByPoint(new THREE.Vector3(x + c * lx - s * ly, y + s * lx + c * ly, z + b.c[2] + dz * b.s[2] / 2));
  }
  return box;
}

function frameModel(model) {   // scene bounds from slab/walls/equipment, else the cloud
  const box = new THREE.Box3();
  for (const e of Object.values(model.elements)) if (e.kind !== 'space') elementBox(e, box);
  if (box.isEmpty()) return;
  box.getCenter(sceneC); const sz = box.getSize(new THREE.Vector3());
  sceneC.z = Math.min(sceneC.z, ceiling * 0.35);
  sceneR = Math.max(1.5, Math.hypot(sz.x, sz.y) / 2, sz.z / 2) * 1.02;
  placeSun();
  ceiling = model.level?.ceiling_height ?? ceiling; clipCeiling.constant = ceiling - 0.2;
  buildGrid();
}

function applyModel(model, { animate = false, stagger = 0, flag = [], flagColor = ACCENT, order = null } = {}) {
  const now = clock(); const seen = new Set(); let i = 0;
  const els = Object.values(model.elements);
  if (order) els.sort(order);
  const fresh = els.filter((e) => { const old = groups.get(e.id); return !old || old.key !== JSON.stringify(e); }).length;
  for (const e of els) {
    seen.add(e.id);
    const key = JSON.stringify(e), old = groups.get(e.id);
    if (old && old.key === key) { setFlag(old, flag.includes(e.id) ? flagColor : null); continue; }
    if (old) world.remove(old.group);
    const { g, parts, fades } = buildElement(e);
    const delay = animate && stagger ? (i++ / Math.max(1, fresh - 1)) * stagger : 0;
    const rec = { group: g, parts, fades, key, el: e, born: animate ? now + delay : 0, pop: animate && e.provenance === 'proposed',
      base: g.position.clone(), baseQ: g.quaternion.clone() };
    world.add(g); groups.set(e.id, rec);
    setFlag(rec, flag.includes(e.id) ? flagColor : null);
    updateElement(rec, now);
  }
  for (const [id, rec] of groups) if (!seen.has(id)) { world.remove(rec.group); groups.delete(id); }
  $('#rev').textContent = model.revision ? 'R' + model.revision : '';
  buildLabels(model);
}

let cutNow = Infinity;
function updateElement(r, now) {
  let k = 1;
  if (r.born) { k = clamp((now - r.born) / (r.pop ? 650 : 750)); if (now >= r.born + 2000) r.born = 0; }
  const rise = r.born ? (r.pop ? easeBack(k) : easeOut(k)) : 1;
  for (const [m, o] of r.fades) m.opacity = o * clamp(k);
  for (const p of r.parts) {
    let h = p.h;
    if (p.cut != null && cutNow < 1e5) h = clamp(Math.max(cutNow, p.cut) - p.z0 - p.bottom, 0, p.h);
    h *= Math.max(0.001, rise);
    p.bg.visible = h > 0.002;
    p.bg.scale.z = Math.max(0.001, h / p.h); p.bg.position.z = p.bottom + h / 2;
  }
}

function setFlag(rec, color) {
  const proposed = rec.el.provenance === 'proposed', eDef = follow ? 0xe6ebf0 : 0x2b3038, eOp = follow ? 0.3 : 0.6;
  rec.group.traverse((o) => {
    if (o.userData.ghost) return;
    if (o.isLineSegments) { o.material.color.setHex(color ?? (proposed ? ACCENT : eDef)); o.material.opacity = color ? 1 : (proposed ? 1 : eOp); }
    if (o.isMesh && o.material.emissive && rec.el.kind !== 'slab') {
      o.material.emissive.setHex(color ?? (proposed ? ACCENT : 0)); o.material.emissiveIntensity = color === BAD ? 0.75 : color ? 0.45 : (proposed ? 0.2 : 0);
    }
  });
}

let labels = [], labelKey = '';
function buildLabels(model) {
  const sp = Object.values(model.elements).filter((e) => e.kind === 'space' && e.polygon?.length > 2);
  const key = JSON.stringify(sp.map((e) => [e.id, e.name, e.polygon]));
  if (key === labelKey) return; labelKey = key;
  $('#labels').innerHTML = ''; labels = [];
  for (const e of sp) {
    const cx = e.polygon.reduce((a, p) => a + p[0], 0) / e.polygon.length, cy = e.polygon.reduce((a, p) => a + p[1], 0) / e.polygon.length;
    let a = 0; e.polygon.forEach((p, i) => { const q = e.polygon[(i + 1) % e.polygon.length]; a += p[0] * q[1] - q[0] * p[1]; });
    const d = document.createElement('div'); d.className = 'label'; d.innerHTML = `${esc(e.name)}<small>${Math.abs(a / 2).toFixed(1)} m²</small>`;
    $('#labels').appendChild(d); labels.push({ d, v: new THREE.Vector3(cx + e.pose.pos[0], cy + e.pose.pos[1], 0.05) });
  }
}

let egressKey = '', egressAt = 0;
function drawEgress(v) {
  const key = JSON.stringify(v?.egress?.paths ?? null);
  if (key === egressKey) return; egressKey = key; egressAt = clock();
  overlay.clear();
  if (!v?.egress?.paths) return;
  for (const p of Object.values(v.egress.paths)) {
    if (!p?.points?.length) continue;
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(p.points.map((q) => new THREE.Vector3(q[0], q[1], 0.08))),
      new THREE.LineDashedMaterial({ color: OK, dashSize: 0.2, gapSize: 0.12 }));
    line.computeLineDistances(); line.geometry.setDrawRange(0, 0); line.userData.n = p.points.length; overlay.add(line);
    const end = p.points.at(-1);
    const dot = new THREE.Mesh(new THREE.CircleGeometry(0.14, 24), new THREE.MeshBasicMaterial({ color: OK, transparent: true, opacity: 0 }));
    dot.position.set(end[0], end[1], 0.09); dot.userData.dot = true; overlay.add(dot);
  }
}
function updateEgress(now) {
  const k = easeOut((now - egressAt) / 1600);
  for (const o of overlay.children) {
    if (o.userData.n) o.geometry.setDrawRange(0, Math.ceil(o.userData.n * k));
    if (o.userData.dot) o.material.opacity = span(now - egressAt, 1300, 1700);
  }
}

// ---------------------------------------------------------------- UI: pills, side cards, caption
function pills(v) {
  const set = (id, ok, hold) => {
    const c = 'pill ' + (ok == null ? '' : ok ? 'ok' : hold ? 'hold' : 'bad'), el = $(id);
    if (el.dataset.c !== c) { el.dataset.c = c; el.className = c; if (ok != null) { void el.offsetWidth; el.classList.add('flip'); } }
  };
  set('#p-code', v?.code_ok); set('#p-physics', v?.physics_ok);
  set('#p-safety', v?.safety_ok, v && !v.safety_ok && v.counts?.hold > 0);
  set('#p-wishes', v ? v.desires_ok : null, true);
}

function side(S, live) {
  let h = '';
  if (S.requirements.length) h += `<div class="card"><h4>Wishes</h4>${S.requirements.slice(0, 5).map((r) => `<div class="row ${r.status}"><b></b>${esc(r.text)}</div>`).join('')}</div>`;
  const iss = live ? (S.validation?.issues ?? []) : [];
  const open = iss.filter((i) => i.severity !== 'warn');
  if (open.length) h += `<div class="card"><h4>To resolve</h4>${open.slice(0, 4).map((i) => `<div class="row ${i.severity}"><b></b>${esc(i.rule)}<small>${esc(i.category)}</small></div>`).join('')}</div>`;
  if (live && S.next_steps.length) h += `<div class="card"><h4>Next steps</h4><ol class="steps">${S.next_steps.slice(0, 5).map((s) => `<li>${esc(s.title)} <small>· ${esc(s.who)}</small></li>`).join('')}</ol></div>`;
  if ($('#side').dataset.h !== h) { $('#side').dataset.h = h; $('#side').innerHTML = h; }
  const q = S.questions.find((x) => x.answer == null);
  $('#q').classList.toggle('show', !!q);
  if (!q) $('#q').dataset.id = '';
  if (q && $('#q').dataset.id !== q.id) {
    $('#q').dataset.id = q.id; $('#q p').textContent = q.question; $('#q div').innerHTML = '';
    for (const o of q.options) {
      const b = document.createElement('button'); b.textContent = o;
      b.onclick = () => { b.classList.add('hit'); toast('Answered: ' + o, true); post('/api/answer', { id: q.id, answer: o }); };
      $('#q div').appendChild(b);
    }
  }
}
const esc = (s) => String(s ?? '').replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
const post = (u, b) => fetch(u, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(b) }).then(poll);
const TAGS = { scan: '01 Video', cloud: '02 Point cloud', bim: '03 BIM', design: '04 Agentic design' };
function caption(t) {
  $('#tag').textContent = TAGS[stage];
  const s = $('#caption span'); if (s.textContent === (t || '')) return;
  s.textContent = t || ''; s.classList.remove('fresh'); void s.offsetWidth; s.classList.add('fresh');
}
let toastUntil = 0;
function toast(t, ok = false) { $('#sent').textContent = t; $('#sent').className = ok ? 'ok' : ''; toastUntil = clock() + 2600; }

// ---------------------------------------------------------------- stages + replay
let S = null, stage = 'scan', queue = [], shown = 0, busyUntil = 0, anim = null, autoplay = false, stageAt = 0, selected = null;
let revealDoneAt = 0, liveAt = 0, cloudRate = 1, base = null, baseIdx = 0, evMs = 1100;
const video = $('#video');
const vdur = () => (isFinite(video.duration) && video.duration > 0 ? video.duration : meta ? meta.frames / meta.fps : 10);

function baseline() {      // the as-built model the design starts from: last fit_bim event (else first event, else live model)
  const H = S?.history ?? [];
  let i = -1; H.forEach((h, j) => { if (h.action === 'fit_bim') i = j; });
  if (i < 0 && H.length) i = 0;
  baseIdx = i; return i >= 0 ? H[i].model : S?.model;
}
const KIND_RANK = { slab: 0, wall: 1, column: 2, door: 3, window: 3, opening: 3, equipment: 4 };
const count = (m) => Object.values(m.elements).reduce((a, e) => (a[e.kind] = (a[e.kind] ?? 0) + 1, a), {});
const plural = (n, w) => `${n} ${w}${n === 1 ? '' : w.endsWith('h') ? 'es' : 's'}`;

function setStage(s) {
  if (follow) exitFollow();
  stage = s; stageAt = clock(); document.body.dataset.stage = s;
  const order = ['scan', 'cloud', 'bim', 'design'];
  document.querySelectorAll('#steps button[data-s]').forEach((b) => { b.classList.toggle('on', b.dataset.s === s); b.classList.toggle('done', order.indexOf(b.dataset.s) < order.indexOf(s)); });
  if (cloud) { cloud.visible = s !== 'scan'; frustum.visible = pathLine.visible = s === 'cloud'; }
  world.visible = s === 'bim' || s === 'design'; overlay.visible = s === 'design'; $('#labels').style.display = s === 'design' ? '' : 'none';
  anim = null; revealDoneAt = 0; liveAt = 0; cutNow = Infinity; wipePlane.copy(NOCLIP());
  $('#stat').innerHTML = '';
  base = baseline();
  if (base) frameModel(base);
  else if (cloudC) { sceneC.copy(cloudC); sceneR = cloudR; }
  if (s === 'scan') { video.loop = true; video.playbackRate = 1; video.currentTime = 0; if (!CAPTURE) video.play().catch(() => {}); caption('Phone walkthrough of the floor'); }
  if (s === 'cloud') {
    cloudRate = Math.max(1, vdur() / (CLOUD_TARGET_MS / 1000));
    video.loop = false; video.currentTime = 0; video.playbackRate = Math.min(16, cloudRate); if (!CAPTURE) video.play().catch(() => {});
    caption(meta ? 'LingBot-Map reconstructs the scan' : 'No point cloud yet — run reconstruct_scan');
    if (cloud) revealTo(0);
  }
  if (s === 'bim') {
    video.pause(); if (cloud) revealTo(1e9);
    groups.forEach((r) => world.remove(r.group)); groups.clear(); labelKey = '';
    if (base) {
      const ax = Math.cos(cam.az + Math.PI / 2), ay = Math.sin(cam.az + Math.PI / 2);
      const along = (e) => (e.pose.pos[0] * ax + e.pose.pos[1] * ay);
      applyModel(base, { animate: true, stagger: 2600, order: (a, b) => (KIND_RANK[a.kind] ?? 5) - (KIND_RANK[b.kind] ?? 5) || along(a) - along(b) });
      groups.forEach((r) => (r.born += 700));
      const c = count(base), bits = [];
      if (c.wall) bits.push(plural(c.wall, 'wall')); if (c.door) bits.push(plural(c.door, 'door')); if (c.window) bits.push(plural(c.window, 'window'));
      if (c.equipment) bits.push(`${c.equipment} equipment`); if (c.column) bits.push(plural(c.column, 'column'));
      caption('Point cloud → as-built model');
      $('#stat').innerHTML = bits.join(' · ') + (base.scan?.fit_p50_mm != null ? ` · fit <b>${base.scan.fit_p50_mm} mm</b> P50` : '');
    } else caption('No BIM yet — run fit_bim');
  }
  if (s === 'design') {
    video.pause(); if (cloud) revealTo(1e9);
    if (base) applyModel(base);
    queue = S?.history ? S.history.slice(baseIdx + 1) : []; shown = S?.history?.length ?? 0;
    evMs = clamp(26000 / Math.max(1, queue.length), 700, 1400);
    busyUntil = stageAt + 1700; pills(null); if (S) side(S, false); S && (S._liveShown = null); egressKey = ''; overlay.clear();
    focus = { c: null, w: 0, goal: 0 };
    caption('The agent redesigns it with you');
  }
  camStage();
}

function movedNames(a) {   // elements that actually move in a physics clip
  const f0 = a.frames[0], f1 = a.frames.at(-1), out = [];
  for (const id of Object.keys(f1)) {
    const p = f0[id], q = f1[id]; if (!p) continue;
    const d = Math.hypot(q[0] - p[0], q[1] - p[1], q[2] - p[2]), dq = Math.abs(p[3] * q[3] + p[4] * q[4] + p[5] * q[5] + p[6] * q[6]);
    if (d > 0.05 || dq < 0.996) out.push(groups.get(id)?.el.name ?? id);
  }
  return out;
}

function focusOn(ids) {
  const box = new THREE.Box3();
  for (const id of ids) { const r = groups.get(id); if (r && r.el.kind !== 'slab') elementBox(r.el, box); }
  if (box.isEmpty()) { focus.goal = 0; return; }
  focus.c = box.getCenter(new THREE.Vector3()); focus.c.z = Math.min(focus.c.z, sceneC.z); focus.goal = 1;
}

function stepReplay(now) {
  if (stage !== 'design' || !S || now < busyUntil) return;
  if (anim) { anim = null; groups.forEach((r) => { r.group.position.copy(r.base); r.group.quaternion.copy(r.baseQ); }); }
  if (queue.length) {
    const ev = queue.shift();
    const failing = ev.action === 'validate' && ev.validation && !(ev.validation.code_ok && ev.validation.safety_ok && ev.validation.physics_ok !== false);
    applyModel(ev.model, { animate: true, flag: ev.focus ?? [], flagColor: failing ? BAD : ACCENT });
    pills(ev.validation); focusOn(ev.focus ?? []);
    caption(ev.caption);
    busyUntil = now + evMs + (ev.action === 'validate' ? 500 : 0);
    if (ev.animation?.frames?.length) {
      const names = movedNames(ev.animation);
      if (names.length) caption(`MuJoCo stability test · ${names.slice(0, 2).join(', ')} ${names.length > 1 ? 'fall' : 'falls'}`);
      anim = { frames: ev.animation.frames, fps: (ev.animation.fps || 25) * 0.6, t0: now + 450 };   // 0.6x slow motion
      busyUntil = anim.t0 + ev.animation.frames.length * 1000 / anim.fps + 1900;
    }
    if (!queue.length) busyUntil += 300;
    return;
  }
  if (S._liveShown !== S.seq) {       // settled on the live model
    const first = S._liveShown == null;
    S._liveShown = S.seq;
    const bad = (S.validation?.issues ?? []).filter((i) => i.severity === 'fail' && i.element).map((i) => i.element);
    applyModel(S.model, { animate: true, flag: bad, flagColor: BAD });
    pills(S.validation); side(S, true); drawEgress(S.validation);
    if (first) { caption(S.agent_note || liveSummary(S.validation) || S.history.at(-1)?.caption || ''); focus.goal = 0; liveAt = now; }
  }
}

async function poll() {
  try {
    const r = await fetch('/api/state?since=' + (S?.seq ?? -1));
    if (r.status !== 200) return;
    const prev = S; S = await r.json();
    if (prev) S._liveShown = prev._liveShown;
    if (stage === 'design') {
      if (S.history.length > shown) { queue.push(...S.history.slice(shown)); S._liveShown = null; liveAt = 0; }
      else if (S.history.length < shown) { queue = []; groups.forEach((g) => world.remove(g.group)); groups.clear(); S._liveShown = null; }
      shown = S.history.length; side(S, !queue.length);
    } else if (!prev && S.stage === 'design' && location.hash === '#design') setStage('design');
  } catch { /* server restarting */ }
}

function liveSummary(v) {
  if (!v?.counts) return '';
  const { fail = 0, hold = 0 } = v.counts;
  return fail || hold ? [fail && `${fail} to fix`, hold && `${hold} to sign off`].filter(Boolean).join(' · ') : 'All checks pass';
}

function status() {
  const t = clock() - stageAt;
  const done = stage === 'scan' ? t >= SCAN_MS
    : stage === 'cloud' ? (!meta ? t > 3000 : revealDoneAt > 0 && clock() - revealDoneAt >= CLOUD_HOLD_MS)
    : stage === 'bim' ? t >= BIM_MS
    : (liveAt > 0 && clock() - liveAt >= 2500) || (!S?.history?.length && t > 3000);
  return { stage, t, done, queue: queue.length, question: !!S?.questions?.find((q) => q.answer == null) };
}

// ---------------------------------------------------------------- per-stage animation
const wipeLabel = $('#wipe');
function setWipe(x) {      // NDC x of the divider: scan to the left, model to the right; null = off
  if (x == null) { wipePlane.copy(NOCLIP()); if (cloudU) cloudU.uWipe.value.set(0, 0, 0, 1); wipeLabel.style.opacity = 0; return; }
  camera.updateMatrixWorld();
  const eye = camera.position, ray = (sx) => new THREE.Vector3(sx, 0, 0.5).unproject(camera).sub(eye).normalize();
  const up = new THREE.Vector3(0, 1, 0).applyQuaternion(camera.quaternion);
  const n = new THREE.Vector3().crossVectors(up, ray(x)).normalize();
  if (n.dot(ray(x + 0.2)) < 0) n.negate();
  wipePlane.setFromNormalAndCoplanarPoint(n, eye);
  if (cloudU) cloudU.uWipe.value.set(n.x, n.y, n.z, wipePlane.constant);
  wipeLabel.style.left = ((x + 1) / 2 * innerWidth) + 'px';
  wipeLabel.style.opacity = clamp(Math.min(x + 1, 1 - x) * 6);
}

function animateStage(now) {
  const t = now - stageAt;
  if (stage === 'cloud' && meta) {
    const vt = Math.min(vdur(), t / 1000 * cloudRate);
    const n = revealTo(vt / vdur() * meta.frames * 1.0001);
    if (!CAPTURE && !video.paused && Math.abs(video.currentTime - vt) > 0.4) video.currentTime = vt;
    if (n >= meta.n && !revealDoneAt) revealDoneAt = now;
    const fr = revealDoneAt ? 1 - span(now - revealDoneAt, 300, 1300) : 1;
    frustum.scale.setScalar(Math.max(0.001, fr)); frustum.visible = fr > 0.01;
    $('#stat').innerHTML = `<b>${fmt(n)}</b> points · frame ${Math.min(meta.frames, Math.floor(vt * meta.fps) + 1)} / ${meta.frames}`;
    if (revealDoneAt && now - revealDoneAt > 600) caption('Coloured point cloud · gravity-aligned, metric');
    cloudU.uOpModel.value = cloudU.uOpScan.value = 1;
  }
  if (stage === 'bim') {
    if (cloudU) { cloudU.uOpModel.value = 1 - 0.8 * ease(span(t, 900, 3800)); cloudU.uOpScan.value = 1; cloudU.uReveal.value = meta.n; cloudU.uFresh.value = 1; }
    // compare wipe: sweep in from the left, rock, then hand the frame back to the model
    let w = null;
    if (t > 4600 && t < 10200) {
      w = t < 6200 ? -1.05 + 1.4 * ease(span(t, 4600, 6200)) : t < 7800 ? 0.35 - 0.7 * ease(span(t, 6200, 7800)) : t < 8800 ? -0.35 + 0.35 * ease(span(t, 7800, 8800)) : -1.05 * ease(span(t, 9200, 10200));
      if (t > 6000 && t < 6100) caption('Scan vs. rebuilt model');
    }
    setWipe(w);
  } else setWipe(null);
  if (stage === 'design') {
    const k = ease(span(t, 200, 1600));
    cutNow = k >= 1 ? CUT : (ceiling + 0.5) + (CUT - ceiling - 0.5) * k;
    clipCeiling.constant = Math.max(cutNow, 0.3) + 0.02;
    if (cloudU) { cloudU.uOpModel.value = 1 - 0.85 * ease(span(t, 0, 1200)); cloudU.uFresh.value = 1; }
    focus.w += (focus.goal - focus.w) * 0.03;
    updateEgress(now);
  } else if (stage !== 'scan') clipCeiling.constant = ceiling - 0.2;
  for (const r of groups.values()) updateElement(r, now);
  if (anim && now >= anim.t0) {
    const f = anim.frames[Math.min(anim.frames.length - 1, Math.floor((now - anim.t0) / 1000 * anim.fps))];
    for (const [id, p] of Object.entries(f)) { const r = groups.get(id); if (r) { r.group.position.set(p[0], p[1], p[2]); r.group.quaternion.set(p[4], p[5], p[6], p[3]); } }
  }
  if (toastUntil && now > toastUntil) { toastUntil = 0; $('#sent').textContent = selected ? 'Selected: ' + groups.get(selected)?.el.name : ''; $('#sent').className = ''; }
}

// ---------------------------------------------------------------- interaction
document.querySelectorAll('#steps button[data-s]').forEach((b) => (b.onclick = () => { autoplay = false; setStage(b.dataset.s); }));
$('#play').onclick = () => { autoplay = true; setStage('scan'); };
$('#say form').onsubmit = (ev) => {
  ev.preventDefault(); const i = $('#say input'); if (!i.value.trim()) return;
  post('/api/message', { text: i.value.trim(), selected }); i.value = ''; toast('Sent to the agent ✓', true); i.blur();
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


// ---------------------------------------------------------------- follow mode: first-person along the capture path
// The camera sits on the (Catmull-Rom interpolated) LingBot-Map pose for video time t, with the video's horizontal FOV.
// Layers: video | cloud | bim (as-built, last fit_bim) | design (live model) | cloud+bim | cloud+design.
let follow = null, track = null;
const FOLLOW_LAYERS = ['video', 'cloud', 'bim', 'design', 'cloud+bim', 'cloud+design'];
const BG = new THREE.Color(0x0b0d10), BG_FOLLOW = new THREE.Color(0x15191e), FOG = new THREE.Fog(0x15191e, 9, 42);
const floorGrid = new THREE.Group(); floorGrid.visible = false; scene.add(floorGrid);

function frameDt() {        // seconds of source video per reconstruction frame
  const p = parseFloat(PARAMS.get('frame_dt')); if (p > 0) return p;
  if (meta?.frame_dt) return meta.frame_dt;
  const nominal = 1 / (meta?.fps || 2), d = video.duration;
  // frames were sampled every round(src_fps / fps) source frames over the whole clip: duration / frames is closer than 1/fps
  if (meta && isFinite(d) && d > 0 && Math.abs(d / meta.frames - nominal) / nominal < 0.03) return d / meta.frames;
  return nominal;
}

function buildTrack() {
  const sig = parseFloat(PARAMS.get('smooth') || '0');
  const raw = meta.cameras.map((c) => ({ p: new THREE.Vector3(c[0], c[1], c[2]), f: new THREE.Vector3(c[3], c[4], c[5]).normalize(),
    u: c.length >= 9 ? new THREE.Vector3(c[6], c[7], c[8]).normalize() : new THREE.Vector3(0, 0, 1) }));
  if (!(sig > 0)) return raw;
  const R = Math.ceil(sig * 3);
  return raw.map((_, i) => {
    const o = { p: new THREE.Vector3(), f: new THREE.Vector3(), u: new THREE.Vector3() }; let w = 0;
    for (let j = -R; j <= R; j++) { const r = raw[clamp(i + j, 0, raw.length - 1)], k = Math.exp(-j * j / (2 * sig * sig)); o.p.addScaledVector(r.p, k); o.f.addScaledVector(r.f, k); o.u.addScaledVector(r.u, k); w += k; }
    o.p.divideScalar(w); o.f.normalize(); o.u.normalize(); return o;
  });
}
const crm = (a, b, c, d, t, out) => { const t2 = t * t, t3 = t2 * t;
  return out.set(0, 0, 0).addScaledVector(a, -0.5 * t3 + t2 - 0.5 * t).addScaledVector(b, 1.5 * t3 - 2.5 * t2 + 1).addScaledVector(c, -1.5 * t3 + 2 * t2 + 0.5 * t).addScaledVector(d, 0.5 * t3 - 0.5 * t2); };
const _fw = new THREE.Vector3(), _up = new THREE.Vector3(), _rt = new THREE.Vector3(), _bk = new THREE.Vector3(), _m4 = new THREE.Matrix4();
function applyPose(t) {
  const T = track, n = T.length; if (n < 2) return;
  const f = clamp(t / frameDt(), 0, n - 1), i = Math.min(Math.floor(f), n - 2), k = f - i, g = (j) => T[clamp(j, 0, n - 1)];
  crm(g(i - 1).p, g(i).p, g(i + 1).p, g(i + 2).p, k, camera.position);
  crm(g(i - 1).f, g(i).f, g(i + 1).f, g(i + 2).f, k, _fw).normalize();
  crm(g(i - 1).u, g(i).u, g(i + 1).u, g(i + 2).u, k, _up).normalize();
  _rt.crossVectors(_fw, _up).normalize(); _up.crossVectors(_rt, _fw).normalize();
  _m4.makeBasis(_rt, _up, _bk.copy(_fw).negate()); camera.quaternion.setFromRotationMatrix(_m4);
}
function followFov() {       // vertical FOV at the output aspect that gives the video's horizontal FOV
  const th = Math.tan(THREE.MathUtils.degToRad(meta.fov_y_deg ?? 55) / 2) * (meta.aspect ?? 16 / 9) * (parseFloat(PARAMS.get('fov_scale')) || 1);
  return THREE.MathUtils.radToDeg(2 * Math.atan(th / camera.aspect));
}

function setModelOpacity(op) {
  world.traverse((o) => {
    if (!o.material || !o.material.isMaterial || o.userData.ghost) return;
    const m = o.material, u = o.userData;
    if (u.op0 == null) { u.op0 = m.opacity; u.tr0 = m.transparent; u.dw0 = m.depthWrite; }
    const tr = op < 1 || u.tr0;
    if (m.transparent !== tr) { m.transparent = tr; m.needsUpdate = true; }
    m.opacity = u.op0 * op; m.depthWrite = op < 1 ? false : u.dw0;
  });
}

function enterFollow(layer) {
  if (!meta?.cameras?.length) return;
  if (!FOLLOW_LAYERS.includes(layer)) layer = 'cloud';
  if (!track) track = buildTrack();
  if (!follow) { follow = { t: CAPTURE ? 0 : video.currentTime, layer }; if (!CAPTURE) { video.loop = true; video.playbackRate = 1; video.play().catch(() => {}); } }
  follow.layer = layer; document.body.dataset.follow = layer;
  camera.fov = followFov(); camera.near = 0.03; camera.updateProjectionMatrix(); updateScale();
  const hasCloud = layer === 'cloud' || layer.startsWith('cloud+'), which = layer.includes('bim') ? 'bim' : layer.includes('design') ? 'design' : null;
  anim = null; cutNow = Infinity; setWipe(null); clipCeiling.constant = 1e6; overlay.visible = false; $('#labels').style.display = 'none';
  if (cloud) {
    cloud.visible = hasCloud; frustum.visible = pathLine.visible = false; cloud.geometry.setDrawRange(0, meta.n);
    const u = cloudU; u.uReveal.value = meta.n; u.uFresh.value = 1; u.uOpModel.value = u.uOpScan.value = 1;
    u.uSize.value = (meta.voxel_m ?? 0.02) * 1.3; u.uMinPx.value = 1.5; u.uMaxPx.value = 13; u.uRound.value = 0;
  }
  world.visible = !!which;
  if (which) {
    base = baseline();
    const m = which === 'bim' ? base : S?.model;
    if (m) applyModel(m);
    groups.forEach((r) => { r.born = 0; r.group.position.copy(r.base); r.group.quaternion.copy(r.baseQ); updateElement(r, clock()); });
    groups.forEach((r) => { setFlag(r, null); r.near = null; });
    setModelOpacity(hasCloud ? 0.45 : 1);
    if (which === 'design') groups.forEach(addGhost); else dropGhosts();     // proposed work stays visible through scanned equipment

  }
  hemi.intensity = which && !hasCloud ? 2.3 : 1.6;
  if (grid) grid.visible = false;
  floorGrid.visible = !!which;
  scene.fog = which && !hasCloud ? FOG : null;
  scene.background = which && !hasCloud ? BG_FOLLOW : BG;
}

function addGhost(r) {
  if (r.ghost || r.el.provenance !== 'proposed' || r.el.kind === 'space') return;
  r.ghost = [];
  for (const p of r.parts) {
    const mesh = p.bg.children.find((o) => o.isMesh); if (!mesh) continue;
    const fillG = new THREE.Mesh(mesh.geometry, new THREE.MeshBasicMaterial({ color: ACCENT, transparent: true, opacity: 0.22, depthTest: false, depthWrite: false, fog: false }));
    const edgeG = new THREE.LineSegments(new THREE.EdgesGeometry(mesh.geometry), new THREE.LineBasicMaterial({ color: ACCENT, transparent: true, opacity: 0.85, depthTest: false, fog: false }));
    fillG.renderOrder = edgeG.renderOrder = 20; fillG.userData.ghost = edgeG.userData.ghost = true;
    p.bg.add(fillG, edgeG); r.ghost.push(fillG, edgeG);
  }
}
function dropGhosts() { groups.forEach((r) => { r.ghost?.forEach((o) => o.parent?.remove(o)); r.ghost = null; }); }

function followWin() { const w = PARAMS.get('win'); return w == null ? 12 : parseFloat(w) || 0; }   // frames either side; 0 = whole cloud
function nearFade(r, k) {
  const g = follow?.layer.startsWith('cloud+') ? 0.45 : 1;
  r.group.traverse((o) => {
    const m = o.material; if (!m?.isMaterial || o.userData.ghost) return; const u = o.userData; if (u.op0 == null) { u.op0 = m.opacity; u.tr0 = m.transparent; u.dw0 = m.depthWrite; }
    const op = g * k, tr = op < 1 || u.tr0; if (m.transparent !== tr) { m.transparent = tr; m.needsUpdate = true; }
    m.opacity = u.op0 * op; m.depthWrite = op < 1 ? false : u.dw0;
  });
}

function exitFollow() {
  follow = null; delete document.body.dataset.follow;
  camera.fov = 40; camera.near = 0.05; camera.updateProjectionMatrix(); updateScale();
  scene.fog = null; scene.background = BG; floorGrid.visible = false; if (grid) grid.visible = true;
  setModelOpacity(1); hemi.intensity = 1.6;
  dropGhosts();
  groups.forEach((r) => { r.group.visible = true; r.near = null; setFlag(r, null); });
  if (cloudU) cloudU.uLo.value = 0;
  if (cloudU) { cloudU.uSize.value = (meta.voxel_m ?? 0.02) * 1.35; cloudU.uMinPx.value = 1.5; cloudU.uMaxPx.value = 64; cloudU.uRound.value = 1; }
  manual = false;
}

function followTick() {
  if (!CAPTURE) follow.t = video.currentTime;
  if (follow.layer === 'video') return;
  applyPose(follow.t);
  if (cloud && cloud.visible) {           // temporal window: only points from frames near now, so drift ghosts from other passes stay out
    const W = followWin(), f = follow.t / frameDt(), O = meta.frame_offsets, F = meta.frames;
    if (W > 0) { const a = clamp(Math.floor(f - W), 0, F), b = clamp(Math.ceil(f + W) + 1, 0, F); cloudU.uLo.value = O[a]; cloudU.uReveal.value = O[b] ?? meta.n; }
    else { cloudU.uLo.value = 0; cloudU.uReveal.value = meta.n; }
  }
  for (const r of groups.values()) {      // never render from inside a box: hide elements that contain the eye
    if (r.el.kind === 'slab' || r.el.kind === 'space') continue;
    r.box ??= elementBox(r.el);
    const k = clamp((r.box.distanceToPoint(camera.position) - 0.35) / 1.0);   // fade boxes the eye is in or right against
    r.group.visible = k > 0.02;
    if (r.near !== k) { r.near = k; nearFade(r, k); }
  }
  renderer.render(scene, camera);
}

addEventListener('keydown', (e) => {      // f: follow mode, 1-6: layer, Esc: back to the stages
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'f' && !follow) enterFollow('cloud');
  else if (follow && '123456'.includes(e.key) && e.key.length === 1) enterFollow(FOLLOW_LAYERS['123456'.indexOf(e.key)]);
  else if (e.key === 'Escape' && follow) setStage(stage);
});

function placeSun() {
  sun.position.copy(sceneC).addScaledVector(new THREE.Vector3(6, -9, 14).normalize(), sceneR * 2.2);
  sun.target.position.copy(sceneC);
  const c = sun.shadow.camera, r = sceneR * 1.15;
  c.left = -r; c.right = r; c.top = r; c.bottom = -r; c.near = 0.1; c.far = sceneR * 5; c.updateProjectionMatrix();
  floorGrid.clear();
  const n = Math.ceil(sceneR * 2.4), g = new THREE.GridHelper(n, n, 0x000000, 0x000000);
  g.material.transparent = true; g.material.opacity = 0.22; g.material.depthWrite = false;
  g.rotation.x = Math.PI / 2; g.position.set(Math.round(sceneC.x), Math.round(sceneC.y), 0.006); floorGrid.add(g);
}

// ---------------------------------------------------------------- loop
const v3 = new THREE.Vector3();
let holdAt = 0;
function tick(now) {
  if (follow) { followTick(); return; }
  animateStage(now);
  stepReplay(now);
  if (autoplay) {
    if (status().done) { holdAt ||= now; if (now - holdAt > 900) { holdAt = 0; const nx = { scan: 'cloud', cloud: 'bim', bim: 'design' }[stage]; if (nx) setStage(nx); else autoplay = false; } }
  }
  updateCamera(now);
  renderer.render(scene, camera);
  if (stage === 'design') for (const l of labels) {
    v3.copy(l.v).project(camera); l.d.style.left = ((v3.x + 1) / 2 * innerWidth) + 'px'; l.d.style.top = ((1 - v3.y) / 2 * innerHeight) + 'px';
  }
}

// ---------------------------------------------------------------- capture hooks (scripts/record_demo.py)
const animStart = new WeakMap();
function syncCssAnimations() {   // CSS transitions/keyframes follow the virtual clock
  for (const a of document.getAnimations()) {
    let s = animStart.get(a); if (s == null) { s = vnow; animStart.set(a, s); a.pause(); }
    a.currentTime = vnow - s;
  }
}
const seekVideo = (t) => new Promise((res) => {
  if (Math.abs(video.currentTime - t) < 0.004) return res();
  const done = () => { video.removeEventListener('seeked', done); clearTimeout(to); res(); };
  const to = setTimeout(done, 3000);
  video.addEventListener('seeked', done); video.currentTime = t;
});
let cursor = { x: innerWidth * 0.62, y: innerHeight * 0.55, fx: 0, fy: 0, tx: 0, ty: 0, t0: 0, ms: 1 };
window.__demo = {
  capture: CAPTURE,
  status,
  go: (s) => { autoplay = false; setStage(s); return status(); },
  poll: () => poll().then(status),
  async frame(ms) {            // advance the virtual clock to ms, render, line the video up, settle CSS
    vnow = ms;
    const t = vnow - stageAt;
    if (stage === 'scan') await seekVideo((t / 1000) % vdur());
    else if (stage === 'cloud') await seekVideo(Math.min(vdur() - 0.04, t / 1000 * cloudRate));
    tick(vnow);
    const c = cursor, k = ease((vnow - c.t0) / c.ms);
    c.x = c.fx + (c.tx - c.fx) * k; c.y = c.fy + (c.ty - c.fy) * k;
    $('#cursor').style.left = c.x + 'px'; $('#cursor').style.top = c.y + 'px';
    syncCssAnimations();
    return status();
  },
  cursor(x, y, ms = 800, show = true) { Object.assign(cursor, { fx: cursor.x, fy: cursor.y, tx: x, ty: y, t0: vnow, ms: Math.max(1, ms) }); $('#cursor').style.display = show ? 'block' : 'none'; },
  click() { const c = $('#cursor'); c.classList.remove('click'); void c.offsetWidth; c.classList.add('click'); },
  async follow(layer, t) {   // first-person render of one layer at video time t (s); the video layer seeks the <video>
    if (!follow || follow.layer !== layer) enterFollow(layer);
    follow.t = t;
    if (layer === 'video') await seekVideo(Math.min(vdur() - 0.02, t));
    followTick();
    return { layer, t, frame: t / frameDt(), fov: camera.fov };
  },
  followInfo: () => ({ frameDt: frameDt(), fov: camera.fov, frames: meta?.frames, duration: vdur() }),
};

if (!CAPTURE) addEventListener('error', () => {});
await loadCloud();
await new Promise((r) => (video.readyState >= 1 ? r() : (video.addEventListener('loadedmetadata', r, { once: true }), video.addEventListener('error', r, { once: true }), setTimeout(r, 4000))));
await poll();
const start = location.hash.slice(1);
setStage(['scan', 'cloud', 'bim', 'design'].includes(start) ? start : 'scan');
if (PARAMS.get('follow') && meta) enterFollow(PARAMS.get('follow'));
if (CAPTURE) { video.pause(); window.__demo.ready = true; }
else {
  setInterval(poll, 800);
  const loop = (now) => { requestAnimationFrame(loop); tick(now); };
  requestAnimationFrame(loop);
}
