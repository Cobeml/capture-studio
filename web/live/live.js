// Capture Studio · live demo console.
// Six presenter stages over the running MCP server's HTTP API (same origin, read: /api/state, /data/...;
// write: /api/message, /api/answer, and MCP tools/call on /mcp). Nothing is simulated: every number comes
// from /api/state, /data/scan/meta.json, /data/scan/objects.json or a tool result. The 3D scene is adapted
// from web/app.js (point-cloud shader, BIM element builder, history replay, MuJoCo clip playback, egress).
import * as THREE from 'three';
import { OrbitControls } from '/web/vendor/OrbitControls.js';
import { GLTFLoader } from './vendor/loaders/GLTFLoader.js';

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const ease = (x) => { x = clamp(x); return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2; };
const easeOut = (x) => 1 - Math.pow(1 - clamp(x), 3);
const easeBack = (x) => { x = clamp(x); const c = 1.5; return 1 + (c + 1) * Math.pow(x - 1, 3) + c * Math.pow(x - 1, 2); };
const span = (t, a, b) => clamp((t - a) / (b - a));
const fmt = (n) => Math.round(n).toLocaleString('en-US');
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const now = () => performance.now();
const rad = THREE.MathUtils.degToRad;

// Teammate palette: lime accent, mint "editable object", cyan icon colour.
const LIME = 0xd6f869, MINT = 0xabddbc, CYAN = 0x67c8dc, BAD = 0xff6f61, HOLD = 0xf4bf4f, SELC = 0xffffff, HOVC = 0x8fd3e2;
const USE_COLOR = { work_cell: MINT, assembly: MINT, storage: 0x8a8f86, aisle: HOLD, forklift_aisle: HOLD, office: CYAN, exit_access: CYAN,
  living: CYAN, bedroom: 0xb9a6ff, kitchen: HOLD, bathroom: CYAN, corridor: 0x8a8f86 };
const GPU = 'RTX 5070';   // the machine the precomputed LingBot-Map run used (nvidia-smi on the demo box)
const AGENT_CMD = 'source .venv/bin/activate && python scripts/agent_gemini.py --watch --route auto';
const DEFAULT_BRIEF = 'Add a CNC machining cell with a robot loader in the empty west bay, plus a tool rack. Keep a forklift aisle to the entrance.';
const HERO_PROMPT = 'Add a bar feeder on the left and an interlocked guard at the front.';

const STAGES = ['capture', 'cloud', 'bim', 'explore', 'design', 'deliver'];
const COPY = {
  capture: { nav: 'Capture', title: 'Phone <em>capture</em>', lede: 'One walkthrough video of a machine shop, shot on a phone. No scanner, no markers.' },
  cloud: { nav: 'Reconstruct', title: 'Point <em>cloud</em>', lede: 'LingBot-Map turns the video into a coloured, metric point cloud, frame by frame.' },
  bim: { nav: 'Rebuild', title: 'As-built <em>model</em>', lede: 'Slab, walls and scan-coloured equipment, fitted to the cloud and measured against it.' },
  explore: { nav: 'Understand', title: 'Every point, <em>known.</em>', lede: 'Each point belongs to an object. Hover to see what it is; open a machine to inspect its generated 3D model.' },
  design: { nav: 'Design', title: 'Design with the <em>agent.</em>', lede: 'Gemini edits the floor through our MCP server. Every change is checked for code, safety and physics.' },
  deliver: { nav: 'Deliver', title: 'Deliver the <em>package.</em>', lede: 'IFC, GLB and MuJoCo files, quantities, and the next steps on site.' },
};
const DEFAULT_VIEW = { bim: 'compare', explore: 'cloud', design: 'model', deliver: 'model' };

// ---------------------------------------------------------------- three setup (Z up, metres)
const vp = $('#viewport');
const renderer = new THREE.WebGLRenderer({ canvas: $('#gl'), antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setClearColor(0x000000, 0);
renderer.localClippingEnabled = true;
renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, 1, 0.05, 2000);
camera.up.set(0, 0, 1);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.maxPolarAngle = Math.PI * 0.49;
let manual = false;
controls.addEventListener('start', () => (manual = true));
const hemi = new THREE.HemisphereLight(0xffffff, 0x2c3330, 1.6); scene.add(hemi);
const sun = new THREE.DirectionalLight(0xfffaf0, 1.7); sun.position.set(6, -9, 14); scene.add(sun); scene.add(sun.target);
sun.castShadow = true; sun.shadow.mapSize.set(2048, 2048); sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.03;
const fill = new THREE.DirectionalLight(0xb7d4c4, 0.5); fill.position.set(-8, 7, 5); scene.add(fill);
let grid = null;
const world = new THREE.Group(); scene.add(world);
const overlay = new THREE.Group(); scene.add(overlay);
const NOCLIP = () => new THREE.Plane(new THREE.Vector3(0, 0, 1), 1e6);
const wipePlane = NOCLIP();
let cloud = null, cloudU = null, meta = null, frustum = null, pathLine = null, pickPoints = null;
const sceneC = new THREE.Vector3(4, 3, 1); let sceneR = 9, ceiling = 2.9, cloudC = null, cloudR = null;
let W = 1, H = 1;

// Segmented objects (optional; contract: /data/scan/objects.json, objects.glb, cloud_labels.bin). Missing files = boxes, no labels.
const OBJ = { list: [], byKey: new Map(), byIndex: new Map(), meshes: new Map(), labels: null, up: 'Y', maxIndex: 0 };
let SEL = [], HOVER = null, VIEW = 'model', SHOW_LABELS = true, SEGMENTS = false, HERO = null;   // explore state

function resize() {
  W = Math.max(1, vp.clientWidth); H = Math.max(1, vp.clientHeight);
  renderer.setSize(W, H, false); camera.aspect = W / H; camera.updateProjectionMatrix();
  if (cloudU) cloudU.uScale.value = renderer.domElement.height / 2 / Math.tan(rad(camera.fov) / 2);
}
new ResizeObserver(resize).observe(vp); resize();

// ---------------------------------------------------------------- camera director
const cam = { target: new THREE.Vector3(), az: -0.9, el: 0.9, dist: 20 };
let camFrom = null, camAt = 0, camAz0 = -0.9, focus = { c: null, w: 0, goal: 0, r: 0 };
const fitDist = (R) => { const v = rad(camera.fov) / 2, h = Math.atan(Math.tan(v) * camera.aspect); return R / Math.sin(Math.min(v, h)); };
const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));
function shot(t) {
  const s = t / 1000;
  if (stage === 'capture' || stage === 'cloud') {
    const C = cloudC ?? sceneC, R = cloudR ?? sceneR, k = Math.min(s, 16);
    return { target: C, az: -0.9 + (stage === 'cloud' ? s * 0.05 : 0), el: 0.92 - (stage === 'cloud' ? k * 0.012 : 0), dist: fitDist(R) * (0.88 - (stage === 'cloud' ? k * 0.006 : 0)) };
  }
  if (stage === 'bim') return { target: sceneC, az: camAz0 + s * 0.04, el: 0.8, dist: fitDist(sceneR) * 0.82 };
  const w = focus.w, tgt = sceneC.clone();
  if (focus.c && w > 0) tgt.lerp(focus.c, (HERO ? 0.9 : 0.4) * w);
  const d0 = fitDist(sceneR) * 0.84, d1 = HERO && focus.r ? Math.max(fitDist(focus.r) * 1.6, 5) : d0 * 0.86;
  return { target: tgt, az: camAz0 + s * (stage === 'deliver' ? 0.03 : 0.008), el: stage === 'explore' ? 0.82 : 0.98, dist: d0 + (d1 - d0) * w };
}
function camStage() { camFrom = { target: cam.target.clone(), az: cam.az, el: cam.el, dist: cam.dist }; camAt = now(); camAz0 = cam.az; manual = false; }
function updateCamera(t) {
  if (manual) { controls.update(); return; }
  const g = shot(t - stageAt), k = camFrom ? ease((t - camAt) / 1800) : 1;
  if (camFrom) {
    cam.target.lerpVectors(camFrom.target, g.target, k);
    cam.az = camFrom.az + wrap(g.az - camFrom.az) * k; cam.el = camFrom.el + (g.el - camFrom.el) * k; cam.dist = camFrom.dist + (g.dist - camFrom.dist) * k;
    if (k >= 1) camFrom = null;
  } else { cam.target.copy(g.target); cam.az = g.az; cam.el = g.el; cam.dist = g.dist; }
  const { target: c, az, el, dist: d } = cam;
  camera.position.set(c.x + d * Math.cos(el) * Math.cos(az), c.y + d * Math.cos(el) * Math.sin(az), c.z + d * Math.sin(el));
  camera.lookAt(c); controls.target.copy(c);
}

// ---------------------------------------------------------------- point cloud
// Capture-order reveal + fresh-point glow (from web/app.js), plus: per-point object index (cloud_labels.bin) driving
// hover/selection highlight, segment tint, and a per-object rigid transform table (uMove) so moved machines' points
// glide to their new pose. Row 0 of uMove: (origin x, origin y, current x, current y); row 1: (d yaw, visible).
const CLOUD_VS = `
#include <common>
#include <clipping_planes_pars_vertex>
uniform float uReveal, uFresh, uSize, uScale, uHover, uSeg, uHide, uDim; uniform float uSel[8]; uniform vec3 uHi; uniform vec4 uWipe;
uniform sampler2D uMove;
attribute float aLabel;
varying vec3 vColor; varying float vSide; varying float vLabel;
vec3 hue(float k) { return 0.55 + 0.45 * cos(6.2832 * (fract(k * 0.618034) + vec3(0.0, 0.33, 0.67))); }
void main() {
  float id = float(gl_VertexID);
  int k = int(aLabel + 0.5);
  vec3 p = position;
  float vis = 1.0;
  if (k > 0) {
    vec4 a = texelFetch(uMove, ivec2(k, 0), 0);
    vec4 b = texelFetch(uMove, ivec2(k, 1), 0);
    float c = cos(b.x), s = sin(b.x); vec2 d = p.xy - a.xy;
    p.xy = vec2(c * d.x - s * d.y, s * d.x + c * d.y) + a.zw;
    vis = b.y;
  }
  if (id >= uReveal || vis < 0.5 || (uHide > 0.5 && abs(aLabel - uHide) < 0.5)) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; return; }
  vec4 wp = modelMatrix * vec4(p, 1.0);
  vec4 mvPosition = viewMatrix * wp;
  float fresh = 1.0 - smoothstep(0.0, 1.0, (uReveal - id) / max(uFresh, 1.0));
  vColor = mix(color, uHi, fresh * 0.75);
  bool sel = false; for (int i = 0; i < 8; i++) if (k > 0 && abs(uSel[i] - aLabel) < 0.5) sel = true;
  bool hov = k > 0 && abs(uHover - aLabel) < 0.5;
  if (k > 0 && uSeg > 0.5) vColor = mix(vColor, hue(aLabel), 0.5);
  if (sel) vColor = mix(vColor, uHi, 0.72); else if (hov) vColor = mix(vColor, vec3(1.0), 0.5);
  else if (uDim > 0.5) vColor *= 0.45;
  vSide = dot(uWipe.xyz, wp.xyz) + uWipe.w; vLabel = aLabel;
  gl_PointSize = clamp(uSize * uScale / -mvPosition.z * (1.0 + fresh * 1.6 + (hov || sel ? 0.35 : 0.0)), 1.5, 64.0);
  gl_Position = projectionMatrix * mvPosition;
  #include <clipping_planes_vertex>
}`;
const CLOUD_FS = `
#include <clipping_planes_pars_fragment>
uniform float uOpScan, uOpModel; varying vec3 vColor; varying float vSide; varying float vLabel;
void main() {
  #include <clipping_planes_fragment>
  vec2 c = gl_PointCoord - 0.5; if (dot(c, c) > 0.25) discard;
#ifdef PICK
  gl_FragColor = vec4(mod(vLabel, 256.0) / 255.0, floor(vLabel / 256.0) / 255.0, 1.0, 1.0);
#else
  gl_FragColor = vec4(vColor, vSide < 0.0 ? uOpScan : uOpModel);
#endif
}`;
const clipCeiling = new THREE.Plane(new THREE.Vector3(0, 0, -1), 2.7);
let moveTex = null;

async function loadCloud() {
  const m = await fetch('/data/scan/meta.json'); if (!m.ok) return;
  meta = await m.json();
  const buf = await (await fetch('/data/scan/cloud.bin?v=' + meta.n)).arrayBuffer();
  const n = meta.n, pos = new Float32Array(buf, 0, n * 3), col = new Uint8Array(buf, n * 12, n * 3);
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('color', new THREE.BufferAttribute(col, 3, true));
  g.setAttribute('aLabel', new THREE.BufferAttribute(new Uint16Array(n), 1));
  moveTex = makeMoveTex(1);
  cloudU = { uReveal: { value: n }, uFresh: { value: 1 }, uSize: { value: (meta.voxel_m ?? 0.02) * 1.35 }, uScale: { value: 500 },
    uHover: { value: 0 }, uSeg: { value: 0 }, uHide: { value: 0 }, uDim: { value: 0 }, uSel: { value: new Array(8).fill(0) }, uMove: { value: moveTex.tex },
    uHi: { value: new THREE.Color(LIME) }, uWipe: { value: new THREE.Vector4(0, 0, 0, 1) }, uOpScan: { value: 1 }, uOpModel: { value: 1 } };
  const mat = (pick) => new THREE.ShaderMaterial({ uniforms: cloudU, vertexShader: CLOUD_VS, fragmentShader: CLOUD_FS, vertexColors: true,
    transparent: !pick, depthWrite: true, clipping: true, clippingPlanes: [clipCeiling], defines: pick ? { PICK: 1 } : {}, blending: pick ? THREE.NoBlending : THREE.NormalBlending });
  cloud = new THREE.Points(g, mat(false));
  cloud.frustumCulled = false; cloud.visible = false; scene.add(cloud);
  pickPoints = new THREE.Points(g, mat(true)); pickPoints.frustumCulled = false; pickScene.add(pickPoints);
  ceiling = meta.ceiling_height ?? ceiling;
  cloudC = new THREE.Vector3(meta.center[0], meta.center[1], Math.min(meta.center[2] ?? 1, ceiling * 0.35));
  cloudR = Math.hypot(meta.radius, ceiling / 2);
  sceneC.copy(cloudC); sceneR = cloudR; clipCeiling.constant = ceiling - 0.2;
  const sz = clamp(meta.radius * 0.085, 0.3, 1.4), w = sz * 0.75, h = sz * 0.45;
  const P = [[0, 0, 0], [-w, sz, -h], [w, sz, -h], [w, sz, h], [-w, sz, h]].map((p) => new THREE.Vector3(...p));
  frustum = new THREE.Group();
  frustum.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints([0, 1, 0, 2, 0, 3, 0, 4, 1, 2, 2, 3, 3, 4, 4, 1].map((i) => P[i])), new THREE.LineBasicMaterial({ color: LIME })));
  frustum.add(new THREE.Mesh(new THREE.BufferGeometry().setFromPoints([P[1], P[2], P[3], P[1], P[3], P[4]]),
    new THREE.MeshBasicMaterial({ color: LIME, transparent: true, opacity: 0.2, side: THREE.DoubleSide, depthWrite: false })));
  frustum.add(new THREE.Mesh(new THREE.SphereGeometry(sz * 0.12, 16, 12), new THREE.MeshBasicMaterial({ color: 0xffffff })));
  frustum.visible = false; scene.add(frustum);
  pathLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: LIME, transparent: true, opacity: 0.85 }));
  pathLine.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array((meta.frames + 1) * 3), 3));
  pathLine.visible = false; pathLine.frustumCulled = false; scene.add(pathLine);
  buildGrid();
}
function makeMoveTex(n) {
  const data = new Float32Array(n * 2 * 4);
  for (let k = 0; k < n; k++) data[(n + k) * 4 + 1] = 1;     // row 1: visible
  const tex = new THREE.DataTexture(data, n, 2, THREE.RGBAFormat, THREE.FloatType);
  tex.minFilter = tex.magFilter = THREE.NearestFilter; tex.needsUpdate = true;
  return { tex, data, n };
}

async function loadObjects() {
  try {
    const r = await fetch('/data/scan/objects.json'); if (!r.ok) return;
    const j = await r.json();
    OBJ.list = j.objects ?? []; OBJ.up = String(j.glb_up ?? 'Y').toUpperCase();
    for (const o of OBJ.list) { OBJ.byKey.set(o.key, o); OBJ.byIndex.set(o.index, o); OBJ.maxIndex = Math.max(OBJ.maxIndex, o.index); }
  } catch { return; }
  if (cloudU) { moveTex = makeMoveTex(OBJ.maxIndex + 1); cloudU.uMove.value = moveTex.tex; MOVE.cur = new Map(); }
  const jobs = [fetch('/data/scan/cloud_labels.bin').then(async (r) => {
    if (!r.ok || !meta) return;
    const buf = await r.arrayBuffer(); if (buf.byteLength !== meta.n * 2) return;   // must match cloud.bin point order
    OBJ.labels = new Uint16Array(buf);
    if (cloud) cloud.geometry.setAttribute('aLabel', new THREE.BufferAttribute(OBJ.labels, 1));
  }).catch(() => {})];
  if (OBJ.list.some((o) => o.mesh)) jobs.push(new GLTFLoader().loadAsync('/data/scan/objects.glb').then((gltf) => {
    gltf.scene.updateMatrixWorld(true);
    for (const o of OBJ.list) {
      if (!o.mesh) continue;
      const node = gltf.scene.getObjectByName(o.mesh); if (!node) continue;
      const meshes = []; node.traverse((m) => { if (m.isMesh) meshes.push({ geo: m.geometry, matrix: m.matrixWorld.clone(), material: m.material }); });
      if (meshes.length) OBJ.meshes.set(o.mesh, meshes);
    }
  }).catch(() => {}));
  await Promise.all(jobs);
}
const Y2Z = new THREE.Matrix4().makeRotationX(Math.PI / 2);   // glTF Y-up -> our Z-up: (x, y, z) -> (x, -z, y)
const objOf = (e) => (e?.object_key ? OBJ.byKey.get(e.object_key) : null);
function elementMeshes(e) {   // hero objects with a generated 3D model; everything else stays a box
  if (e.provenance !== 'scanned') return null;
  const key = e.mesh ?? objOf(e)?.mesh;
  return key ? OBJ.meshes.get(key) ?? null : null;
}
const isHero = (e) => !!(e && elementMeshes(e));

function buildGrid() {
  if (grid) scene.remove(grid);
  const size = Math.max(20, Math.ceil(sceneR * 6 / 5) * 5);
  grid = new THREE.GridHelper(size, size, 0x2a352d, 0x1b231d); grid.rotation.x = Math.PI / 2;
  grid.position.set(Math.round(sceneC.x), Math.round(sceneC.y), -0.3); scene.add(grid);
}
const camAtFrame = (f) => {
  const C = meta.cameras, i = clamp(Math.floor(f), 0, C.length - 1), j = Math.min(i + 1, C.length - 1), k = clamp(f - i), a = C[i], b = C[j];
  return { p: new THREE.Vector3(a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k),
    f: new THREE.Vector3(a[3] + (b[3] - a[3]) * k, a[4] + (b[4] - a[4]) * k, a[5] + (b[5] - a[5]) * k).normalize() };
};
function revealTo(frame) {     // fractional frame -> points revealed in capture order (meta.frame_offsets)
  if (!cloud) return 0;
  const F = meta.frames, O = meta.frame_offsets, f = clamp(frame, 0, F);
  const i = Math.min(Math.floor(f), F - 1), k = f >= F ? 1 : f - i;
  const o1 = O[i + 1] ?? meta.n, n = f >= F ? meta.n : O[i] + (o1 - O[i]) * k;
  cloudU.uReveal.value = n; cloudU.uFresh.value = Math.max(2000, (o1 - O[i]) * 0.9);
  cloud.geometry.setDrawRange(0, Math.ceil(n));
  const cur = camAtFrame(Math.min(f, F - 1)), arr = pathLine.geometry.attributes.position.array, m = Math.min(Math.floor(f), F - 1);
  for (let j = 0; j <= m; j++) { const c = meta.cameras[j]; arr.set([c[0], c[1], c[2]], j * 3); }
  arr.set([cur.p.x, cur.p.y, cur.p.z], (m + 1) * 3);
  pathLine.geometry.attributes.position.needsUpdate = true; pathLine.geometry.setDrawRange(0, m + 2);
  const r = new THREE.Vector3().crossVectors(cur.f, new THREE.Vector3(0, 0, 1)); if (r.lengthSq() < 1e-6) r.set(1, 0, 0); r.normalize();
  const u = new THREE.Vector3().crossVectors(r, cur.f).normalize();
  frustum.position.copy(cur.p); frustum.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(r, cur.f, u));
  return n;
}

// ---------------------------------------------------------------- moving points: per-object transform table, animated
const MOVE = { cur: new Map(), from: null, to: null, t0: 0, ms: 1100 };
function poseTargets(model) {   // object index -> {ox, oy, cx, cy, dth, vis}
  const out = new Map(); if (!model || !OBJ.list.length) return out;
  const baseKeys = new Set(Object.values(baselineOf(S)?.elements ?? {}).map((e) => e.object_key).filter(Boolean));
  const present = new Set();
  for (const e of Object.values(model.elements)) {
    const o = objOf(e); if (!o) continue; present.add(o.key);
    const org = e.origin ?? { pos: e.pos, yaw_deg: e.yaw_deg ?? 0 };
    out.set(o.index, { ox: org.pos[0], oy: org.pos[1], cx: e.pos?.[0] ?? org.pos[0], cy: e.pos?.[1] ?? org.pos[1], dth: rad((e.yaw_deg ?? 0) - (org.yaw_deg ?? 0)), vis: 1 });
  }
  for (const o of OBJ.list) if (!present.has(o.key) && baseKeys.has(o.key)) out.set(o.index, { ox: 0, oy: 0, cx: 0, cy: 0, dth: 0, vis: 0 });   // removed by the design
  return out;
}
function setCloudPoses(model, animate = true) {
  if (!moveTex || moveTex.n < 2) return;
  const to = poseTargets(model), from = new Map();
  for (const [k, p] of to) from.set(k, MOVE.cur.get(k) ?? { ...p, cx: p.ox, cy: p.oy, dth: 0 });
  for (const [k, p] of MOVE.cur) if (!to.has(k)) { to.set(k, { ...p, cx: p.ox, cy: p.oy, dth: 0, vis: 1 }); from.set(k, p); }
  MOVE.from = from; MOVE.to = to; MOVE.t0 = animate ? now() : -1e9;
  stepCloudPoses(now());
}
function stepCloudPoses(t) {
  if (!MOVE.to) return;
  const k = ease((t - MOVE.t0) / MOVE.ms), d = moveTex.data, n = moveTex.n;
  for (const [i, b] of MOVE.to) {
    if (i >= n) continue;
    const a = MOVE.from.get(i) ?? b;
    const p = { ox: b.ox, oy: b.oy, cx: a.cx + (b.cx - a.cx) * k, cy: a.cy + (b.cy - a.cy) * k, dth: a.dth + (b.dth - a.dth) * k, vis: k >= 1 ? b.vis : (b.vis || a.vis) };
    MOVE.cur.set(i, p);
    d.set([p.ox, p.oy, p.cx, p.cy], i * 4); d[(n + i) * 4] = p.dth; d[(n + i) * 4 + 1] = p.vis;
  }
  moveTex.tex.needsUpdate = true;
  if (k >= 1) MOVE.to = null;
}

// ---------------------------------------------------------------- BIM elements
const groups = new Map();
const CUT = 1.2, CUT_TALL = 2.2, CUT_KINDS = new Set(['wall', 'door', 'window', 'column', 'opening']);
const edgeMat = (c, o = 0.9) => new THREE.LineBasicMaterial({ color: c, transparent: true, opacity: o, clippingPlanes: [wipePlane] });
const hex = (s, d) => { try { return s ? new THREE.Color(s) : new THREE.Color(d); } catch { return new THREE.Color(d); } };
function elementMaterial(e) {
  const proposed = e.provenance === 'proposed', c = hex(e.color, 0x9aa0a6), clip = [wipePlane];
  if (e.kind === 'window') return new THREE.MeshStandardMaterial({ color: proposed ? 0xe4f6c4 : 0x9fd3e6, transparent: true, opacity: 0.38, roughness: 0.1, depthWrite: false, clippingPlanes: clip });
  if (e.kind === 'door') return new THREE.MeshStandardMaterial({ color: proposed ? LIME : c, transparent: true, opacity: 0.75, roughness: 0.7, emissive: 0, clippingPlanes: clip });
  if (e.kind === 'slab') return new THREE.MeshStandardMaterial({ color: c.multiplyScalar(0.42), roughness: 0.95, clippingPlanes: clip });
  if (proposed) return new THREE.MeshStandardMaterial({ color: MINT, roughness: 0.7, emissive: LIME, emissiveIntensity: 0.12, clippingPlanes: clip });
  return new THREE.MeshStandardMaterial({ color: c, roughness: 0.85, metalness: e.kind === 'equipment' ? 0.15 : 0, emissive: 0, clippingPlanes: clip });
}
function partMesh(b, e, clip) {    // add-on from add_part: {c, s, shape, color} in the element's local frame
  const geo = b.shape === 'cylinder' ? new THREE.CylinderGeometry(Math.min(b.s[0], b.s[1]) / 2, Math.min(b.s[0], b.s[1]) / 2, b.s[2], 28).rotateX(Math.PI / 2) : new THREE.BoxGeometry(...b.s);
  const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: hex(b.color, 0xff8a3d), roughness: 0.55, metalness: 0.2, emissive: 0, clippingPlanes: clip }));
  mesh.userData.id = e.id; mesh.userData.part = b.part; mesh.castShadow = mesh.receiveShadow = true;
  const bg = new THREE.Group(); bg.position.set(...b.c); bg.add(mesh);
  bg.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo, 30), new THREE.LineBasicMaterial({ color: LIME, transparent: true, opacity: 0.9, clippingPlanes: clip })));
  return bg;
}
function buildElement(e, clip = [wipePlane]) {
  const g = new THREE.Group(); g.position.set(...e.pose.pos); g.rotation.z = e.pose.yaw;
  const proposed = e.provenance === 'proposed', parts = [];
  if (e.kind === 'space' && e.polygon?.length > 2) {
    const col = USE_COLOR[e.use] ?? 0x8a8f86;
    const mat = new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.26, depthWrite: false, side: THREE.DoubleSide, clippingPlanes: clip });
    const mesh = new THREE.Mesh(new THREE.ShapeGeometry(new THREE.Shape(e.polygon.map((p) => new THREE.Vector2(p[0], p[1])))), mat);
    mesh.position.z = 0.02; mesh.renderOrder = 1; mesh.userData.id = e.id; g.add(mesh);
    const pts = e.polygon.map((p) => new THREE.Vector3(p[0], p[1], 0.03)); pts.push(pts[0]);
    g.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), edgeMat(col, 0.95)));
    return { g, parts, fades: [[mat, 0.26]] };
  }
  const cutH = CUT_KINDS.has(e.kind) ? CUT : CUT_TALL, shapes = elementMeshes(e);
  const boxes = e.boxes ?? [];
  if (shapes) {   // generated 3D model, in the object's local frame, parented under the element pose so it moves with edits
    const bg = new THREE.Group();
    for (const s of shapes) {
      const mat = s.material.clone(); mat.clippingPlanes = clip; if (mat.emissive) mat.emissive.setHex(0);
      const m = new THREE.Mesh(s.geo, mat); m.matrixAutoUpdate = false;
      m.matrix.copy(OBJ.up === 'Y' ? Y2Z.clone().multiply(s.matrix) : s.matrix);
      m.castShadow = m.receiveShadow = true; m.userData.id = e.id; bg.add(m);
    }
    g.add(bg);
    const main = boxes[0];
    parts.push({ bg, mesh: true, bottom: 0, h: main ? main.c[2] + main.s[2] / 2 : (e.size?.[2] ?? 1), z0: e.pose.pos[2], cut: null });
  }
  boxes.forEach((b, i) => {
    if (b.part) { const bg = partMesh(b, e, clip); g.add(bg); parts.push({ bg, part: b.part, bottom: b.c[2] - b.s[2] / 2, h: b.s[2], z0: e.pose.pos[2], cut: null }); return; }
    if (shapes && i === 0) return;
    const geo = new THREE.BoxGeometry(...b.s), bg = new THREE.Group(); bg.position.set(...b.c);
    const mesh = new THREE.Mesh(geo, elementMaterial(e)); mesh.userData.id = e.id; mesh.castShadow = e.kind !== 'slab' && e.kind !== 'window'; mesh.receiveShadow = true; bg.add(mesh);
    if (e.kind !== 'slab') bg.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo), edgeMat(proposed ? LIME : 0x1e231f, proposed ? 1 : 0.6)));
    g.add(bg);
    parts.push({ bg, bottom: b.c[2] - b.s[2] / 2, h: b.s[2], z0: e.pose.pos[2], cut: proposed || e.kind === 'slab' ? null : cutH });
  });
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
function frameModel(model) {
  const box = new THREE.Box3();
  for (const e of Object.values(model.elements)) if (e.kind !== 'space') elementBox(e, box);
  if (box.isEmpty()) return;
  box.getCenter(sceneC); const sz = box.getSize(new THREE.Vector3());
  sceneC.z = Math.min(sceneC.z, ceiling * 0.35);
  sceneR = Math.max(1.5, Math.hypot(sz.x, sz.y) / 2, sz.z / 2);
  ceiling = model.level?.ceiling_height ?? ceiling;
  placeSun(); buildGrid();
}
function placeSun() {
  sun.position.copy(sceneC).addScaledVector(new THREE.Vector3(6, -9, 14).normalize(), sceneR * 2.2); sun.target.position.copy(sceneC);
  const c = sun.shadow.camera, r = sceneR * 1.15; c.left = -r; c.right = r; c.top = r; c.bottom = -r; c.near = 0.1; c.far = sceneR * 5; c.updateProjectionMatrix();
}
function clearWorld() { groups.forEach((r) => world.remove(r.group)); groups.clear(); labelKey = ''; }
const shapeKey = (e) => JSON.stringify({ ...e, pose: null, pos: null, yaw_deg: null, origin: null, relocated: null, anchored: null, boxes: (e.boxes ?? []).filter((b) => !b.part), parts: null });
let shownModel = null;
function applyModel(model, { animate = false, stagger = 0, flag = {}, order = null, cloudAnim = true } = {}) {
  const t = now(), seen = new Set(); let i = 0;
  const els = Object.values(model.elements); if (order) els.sort(order);
  const fresh = els.filter((e) => groups.get(e.id)?.key !== JSON.stringify(e)).length;
  for (const e of els) {
    seen.add(e.id);
    const key = JSON.stringify(e), old = groups.get(e.id);
    if (old && old.key === key) { old.flag = flag[e.id] ?? null; setFlag(old, flagOf(old)); continue; }
    if (old) world.remove(old.group);
    const { g, parts, fades } = buildElement(e);
    const delay = animate && stagger ? (i++ / Math.max(1, fresh - 1)) * stagger : 0;
    const same = old && shapeKey(old.el) === shapeKey(e);           // moved and/or parts added: animate those, not a full re-rise
    const rec = { group: g, parts, fades, key, el: e, born: animate && !same ? t + delay : 0, pop: animate && e.provenance === 'proposed',
      base: g.position.clone(), baseQ: g.quaternion.clone(), yaw: g.rotation.z, flag: flag[e.id] ?? null };
    if (same) {
      const oldParts = new Set((old.el.boxes ?? []).map((b) => b.part).filter(Boolean));
      if (animate) for (const p of parts) if (p.part && !oldParts.has(p.part)) p.born = t;
      if (old.group.position.distanceTo(g.position) > 0.01 || Math.abs(wrap(old.group.rotation.z - g.rotation.z)) > 0.01) {
        rec.move = { from: old.group.position.clone(), yaw0: old.group.rotation.z, t0: animate ? t : -1e9 };
      }
    }
    world.add(g); groups.set(e.id, rec); setFlag(rec, flagOf(rec)); updateElement(rec, t);
  }
  for (const [id, rec] of groups) if (!seen.has(id)) { world.remove(rec.group); groups.delete(id); }
  shownModel = model;
  setCloudPoses(model, cloudAnim && animate);
  buildLabels(model);
  if (HERO && !groups.has(HERO)) closeHero();
}
let cutNow = Infinity;
function updateElement(r, t) {
  let k = 1;
  if (r.born) { k = clamp((t - r.born) / (r.pop ? 650 : 750)); if (t >= r.born + 2000) r.born = 0; }
  const rise = r.born ? (r.pop ? easeBack(k) : easeOut(k)) : 1;
  for (const [m, o] of r.fades) m.opacity = o * clamp(k);
  for (const p of r.parts) {
    let h = p.h;
    if (p.cut != null && cutNow < 1e5) h = clamp(Math.max(cutNow, p.cut) - p.z0 - p.bottom, 0, p.h);
    const pr = p.born ? easeBack((t - p.born) / 650) : 1; if (p.born && t > p.born + 1500) p.born = 0;
    h *= Math.max(0.001, rise * pr);
    p.bg.visible = h > 0.002; p.bg.scale.z = Math.max(0.001, h / p.h); p.bg.position.z = p.mesh ? 0 : p.bottom + h / 2;
  }
  if (r.move && !D.anim) {
    const m = ease((t - r.move.t0) / MOVE.ms);
    r.group.position.lerpVectors(r.move.from, r.base, m); r.group.rotation.z = r.move.yaw0 + wrap(r.yaw - r.move.yaw0) * m;
    if (m >= 1) { r.group.position.copy(r.base); r.group.quaternion.copy(r.baseQ); r.move = null; }
  }
}
const flagOf = (rec) => (rec.el.id === HERO ? rec.flag : SEL.includes(rec.el.id) ? SELC : HOVER === rec.el.id ? HOVC : rec.flag);
function refreshFlags() { groups.forEach((r) => setFlag(r, flagOf(r))); }
function setFlag(rec, color) {
  const proposed = rec.el.provenance === 'proposed';
  rec.group.traverse((o) => {
    if (o.isLineSegments && !o.parent?.children.some((c) => c.userData.part)) { o.material.color.setHex(color ?? (proposed ? LIME : 0x1e231f)); o.material.opacity = color ? 1 : (proposed ? 1 : 0.6); }
    if (o.isMesh && o.material.emissive && rec.el.kind !== 'slab' && !o.userData.part) {
      o.material.emissive.setHex(color ?? (proposed ? LIME : 0));
      o.material.emissiveIntensity = color === BAD ? 0.8 : color === SELC ? 0.32 : color === HOVC ? 0.3 : color ? 0.5 : (proposed ? 0.12 : 0);
    }
  });
}
let labels = [], labelKey = '';
function buildLabels(model) {   // zone labels (spaces)
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
  if (key === egressKey) return; egressKey = key; egressAt = now(); overlay.clear();
  for (const p of Object.values(v?.egress?.paths ?? {})) {
    if (!p?.points?.length) continue;
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(p.points.map((q) => new THREE.Vector3(q[0], q[1], 0.08))),
      new THREE.LineDashedMaterial({ color: CYAN, dashSize: 0.22, gapSize: 0.12 }));
    line.computeLineDistances(); line.geometry.setDrawRange(0, 0); line.userData.n = p.points.length; overlay.add(line);
    const end = p.points.at(-1);
    const dot = new THREE.Mesh(new THREE.CircleGeometry(0.16, 24), new THREE.MeshBasicMaterial({ color: CYAN, transparent: true, opacity: 0 }));
    dot.position.set(end[0], end[1], 0.09); dot.userData.dot = true; overlay.add(dot);
  }
}
function updateEgress(t) {
  const k = easeOut((t - egressAt) / 1600);
  for (const o of overlay.children) {
    if (o.userData.n) o.geometry.setDrawRange(0, Math.ceil(o.userData.n * k));
    if (o.userData.dot) o.material.opacity = span(t - egressAt, 1300, 1700);
  }
}
function focusOn(ids) {
  const box = new THREE.Box3();
  for (const id of ids ?? []) { const r = groups.get(id); if (r && r.el.kind !== 'slab') elementBox(r.el, box); }
  if (box.isEmpty()) { focus.goal = 0; return; }
  focus.c = box.getCenter(new THREE.Vector3()); focus.r = box.getSize(new THREE.Vector3()).length() / 2; focus.c.z = Math.min(focus.c.z, sceneC.z); focus.goal = 1;
}
function movedNames(a) {
  const f0 = a.frames[0], f1 = a.frames.at(-1), out = [];
  for (const id of Object.keys(f1)) {
    const p = f0[id], q = f1[id]; if (!p) continue;
    const d = Math.hypot(q[0] - p[0], q[1] - p[1], q[2] - p[2]), dq = Math.abs(p[3] * q[3] + p[4] * q[4] + p[5] * q[5] + p[6] * q[6]);
    if (d > 0.05 || dq < 0.996) out.push(nameOf(groups.get(id)?.el) ?? id);
  }
  return out;
}

// ---------------------------------------------------------------- compare wipe (scan | model), draggable
let wipeX = null, wipeUser = false;
function setWipe(x) {
  wipeX = x;
  if (x == null) { wipePlane.copy(NOCLIP()); if (cloudU) cloudU.uWipe.value.set(0, 0, 0, 1); $('#wipe').classList.remove('on'); return; }
  camera.updateMatrixWorld();
  const eye = camera.position, ray = (sx) => new THREE.Vector3(sx, 0, 0.5).unproject(camera).sub(eye).normalize();
  const up = new THREE.Vector3(0, 1, 0).applyQuaternion(camera.quaternion);
  const n = new THREE.Vector3().crossVectors(up, ray(x)).normalize();
  if (n.dot(ray(x + 0.2)) < 0) n.negate();
  wipePlane.setFromNormalAndCoplanarPoint(n, eye);
  if (cloudU) cloudU.uWipe.value.set(n.x, n.y, n.z, wipePlane.constant);
  $('#wipe').style.left = ((x + 1) / 2 * W) + 'px';
  $('#wipe').classList.toggle('on', x > -0.98 && x < 0.98);
}
$('#wipe-knob').addEventListener('pointerdown', (e) => {
  e.preventDefault(); e.stopPropagation(); wipeUser = true; $('#wipe-knob').setPointerCapture(e.pointerId);
  const move = (ev) => { const r = vp.getBoundingClientRect(); wipeX = clamp(((ev.clientX - r.left) / r.width) * 2 - 1, -0.95, 0.95); };
  const upH = () => { $('#wipe-knob').removeEventListener('pointermove', move); $('#wipe-knob').removeEventListener('pointerup', upH); };
  $('#wipe-knob').addEventListener('pointermove', move); $('#wipe-knob').addEventListener('pointerup', upH);
});

// ---------------------------------------------------------------- explore: GPU point picking, hover, selection, info, labels
const pickScene = new THREE.Scene();
const PK = 9, pickRT = new THREE.WebGLRenderTarget(PK, PK);
const pickBuf = new Uint8Array(PK * PK * 4);
function gpuPickLabel(cx, cy) {    // object index under the cursor (0 = unlabelled point or nothing); one 1x1 render
  if (!pickPoints || !OBJ.labels) return 0;
  const pr = renderer.getPixelRatio();
  camera.setViewOffset(W * pr, H * pr, cx * pr - (PK - 1) / 2, cy * pr - (PK - 1) / 2, PK, PK);
  const cc = renderer.getClearColor(new THREE.Color()), ca = renderer.getClearAlpha();
  renderer.setRenderTarget(pickRT); renderer.setClearColor(0x000000, 0); renderer.clear();
  renderer.render(pickScene, camera);
  renderer.readRenderTargetPixels(pickRT, 0, 0, PK, PK, pickBuf);
  renderer.setRenderTarget(null); renderer.setClearColor(cc, ca); camera.clearViewOffset();
  let best = 0, bd = 1e9; const m = (PK - 1) / 2;     // labelled pixel nearest the cursor within a small window
  for (let y = 0; y < PK; y++) for (let x = 0; x < PK; x++) {
    const i = (y * PK + x) * 4; if (pickBuf[i + 2] < 128) continue;
    const k = pickBuf[i] + pickBuf[i + 1] * 256, d = (x - m) ** 2 + (y - m) ** 2;
    if (k && d < bd) { bd = d; best = k; }
  }
  return best;
}
const raycaster = new THREE.Raycaster();
function rayPick(cx, cy, onlyIds = null) {
  raycaster.setFromCamera(new THREE.Vector2((cx / W) * 2 - 1, -(cy / H) * 2 + 1), camera);
  const roots = onlyIds ? onlyIds.map((id) => groups.get(id)?.group).filter(Boolean) : world.children;
  const hit = raycaster.intersectObjects(roots, true).find((h) => {
    if (!h.object.isMesh || !h.object.userData.id) return false;
    const el = groups.get(h.object.userData.id)?.el; if (!el || el.kind === 'slab') return false;
    let o = h.object; while (o) { if (!o.visible) return false; o = o.parent; }   // ignore cut-away / hidden parts
    return true;
  });
  return hit ? hit.object.userData.id : null;
}
const pickModel = () => shownModel ?? baselineOf(S) ?? baselineOf(REC);
function elementForObject(o) {
  const m = pickModel(); if (!m || !o) return null;
  return Object.values(m.elements).find((e) => e.object_key === o.key)?.id ?? null;
}
function pick(cx, cy) {   // -> element id, or 'obj:<key>' for a segmented object with no element
  const cloudView = stage === 'cloud' || VIEW === 'cloud';
  if (cloudView) {
    if (HERO) { const h = rayPick(cx, cy, [HERO]); if (h) return h; }
    if (OBJ.labels) { const k = gpuPickLabel(cx, cy); if (!k) return null; const o = OBJ.byIndex.get(k); return elementForObject(o) ?? (o ? 'obj:' + o.key : null); }
    return stage === 'cloud' ? null : rayPick(cx, cy);     // no labels file: fall back to the fitted boxes
  }
  return rayPick(cx, cy);
}
const pickEnabled = () => ['bim', 'explore', 'design', 'deliver'].includes(stage) || (stage === 'cloud' && revealDoneAt > 0);

function infoOf(id) {
  if (!id) return null;
  if (id.startsWith('obj:')) { const o = OBJ.byKey.get(id.slice(4)); return o && { id, o, e: null, label: o.label, fn: o.function, cat: o.category, size: o.size, conf: o.confidence, src: o.label_source, prov: 'scanned' }; }
  const e = groups.get(id)?.el ?? Object.values(pickModel()?.elements ?? {}).find((x) => x.id === id); if (!e) return null;
  const o = objOf(e);
  let size = e.size;
  if (!size && e.boxes?.length) { const b = new THREE.Box3(); for (const x of e.boxes) { b.expandByPoint(new THREE.Vector3(x.c[0] - x.s[0] / 2, x.c[1] - x.s[1] / 2, x.c[2] - x.s[2] / 2)); b.expandByPoint(new THREE.Vector3(x.c[0] + x.s[0] / 2, x.c[1] + x.s[1] / 2, x.c[2] + x.s[2] / 2)); } const s = b.getSize(new THREE.Vector3()); size = [s.x, s.y, s.z]; }
  return { id, e, o, label: e.label ?? o?.label ?? e.name, fn: e.function ?? o?.function, cat: e.category ?? o?.category ?? e.kind, size, conf: e.confidence ?? o?.confidence,
    src: o?.label_source, prov: e.provenance, ifc: e.ifc_class, hero: isHero(e), parts: (e.boxes ?? []).filter((b) => b.part), moved: !!e.relocated };
}
const nameOf = (e) => (e ? e.label ?? objOf(e)?.label ?? e.name : null);
const shortName = (s) => (s && s.length > 28 ? s.slice(0, 26) + '…' : s);

function setHover(id) {
  if (HOVER === id) return;
  HOVER = id; refreshFlags();
  const k = id ? labelIndexOf(id) : 0;
  if (cloudU) { cloudU.uHover.value = k; cloudU.uDim.value = k && (VIEW === 'cloud' || stage === 'cloud') ? 1 : 0; }
  vp.style.cursor = id ? 'pointer' : '';
}
function labelIndexOf(id) {
  if (!id) return 0;
  if (id.startsWith('obj:')) return OBJ.byKey.get(id.slice(4))?.index ?? 0;
  return objOf(groups.get(id)?.el ?? Object.values(pickModel()?.elements ?? {}).find((x) => x.id === id))?.index ?? 0;
}
function setSelection(ids) {
  SEL = ids.slice(0, 8); refreshFlags();
  if (cloudU) { const a = new Array(8).fill(0); SEL.forEach((id, i) => (a[i] = labelIndexOf(id))); cloudU.uSel.value = a; }
  renderInfo(); renderMentions(); renderExplorePanel();
}
function renderInfo() {
  const box = $('#info'), id = SEL.at(-1), inf = infoOf(id);
  if (!inf || HERO || !$('#signoff').hidden) { box.hidden = true; return; }
  box.hidden = false;
  const sz = inf.size ? `${inf.size[0].toFixed(2)} × ${inf.size[1].toFixed(2)} m` : '–', ht = inf.size?.[2] != null ? `${inf.size[2].toFixed(2)} m` : '–';
  const more = SEL.length > 1 ? `<div class="info-more">${SEL.length} selected · ${SEL.slice(0, -1).map((x) => esc(shortName(infoOf(x)?.label))).join(', ')}</div>` : '';
  box.innerHTML = `<button class="info-x" aria-label="Clear selection">×</button>
    <div class="info-kicker"><span class="chip">${esc(inf.cat ?? '')}</span><span class="chip ${inf.prov === 'proposed' ? 'lime' : ''}">${esc(inf.prov ?? '')}</span>${inf.moved ? '<span class="chip lime">moved</span>' : ''}${inf.hero ? '<span class="chip hero">3D model</span>' : ''}</div>
    <h3>${esc(inf.label)}</h3>
    ${inf.fn ? `<p>${esc(inf.fn)}</p>` : ''}
    <dl><dt>footprint</dt><dd>${sz}</dd><dt>height</dt><dd>${ht}</dd>
    ${inf.conf != null ? `<dt>label confidence</dt><dd>${Math.round(inf.conf * 100)} %${inf.src ? ' · ' + esc(inf.src) : ''}</dd>` : ''}
    ${inf.ifc ? `<dt>IFC class</dt><dd>${esc(inf.ifc)}</dd>` : ''}${inf.e?.fit ? `<dt>scan fit</dt><dd>P50 ${inf.e.fit.p50} mm</dd>` : ''}
    <dt>id</dt><dd>${esc(inf.e?.id ?? inf.o?.key ?? id)}</dd></dl>
    ${inf.parts?.length ? `<div class="info-more">Added parts: ${inf.parts.map((p) => esc(p.name)).join(', ')}</div>` : ''}
    ${more}
    <div class="info-actions">${inf.hero ? '<button class="button primary small" data-act="hero">Open 3D model</button>' : ''}${inf.e && stage !== 'design' ? '<button class="button secondary small" data-act="ask">Ask the agent →</button>' : ''}</div>`;
  box.querySelector('.info-x').onclick = () => setSelection([]);
  box.querySelector('[data-act="hero"]')?.addEventListener('click', () => openHero(inf.id));
  box.querySelector('[data-act="ask"]')?.addEventListener('click', () => setStage('design'));
}

// floating object labels, collision-managed (selected and hovered always win, then the largest objects)
const olabels = new Map(); let olayoutAt = 0;
const _v = new THREE.Vector3();
function objectLabelCandidates() {
  const m = shownModel ?? pickModel(); if (!m) return [];
  return Object.values(m.elements).filter((e) => ['equipment', 'fixture'].includes(e.kind) && (e.label || objOf(e) || e.provenance === 'proposed' || SEL.includes(e.id) || HOVER === e.id));
}
function layoutObjectLabels(t) {
  const host = $('#olabels'), show = SHOW_LABELS && ['explore', 'design', 'deliver', 'bim'].includes(stage) || (stage === 'cloud' && revealDoneAt && SHOW_LABELS);
  if (!show) { host.style.display = 'none'; return; } host.style.display = '';
  const cands = objectLabelCandidates(), seen = new Set();
  for (const e of cands) {
    seen.add(e.id);
    let L = olabels.get(e.id);
    if (!L) { L = { d: document.createElement('div') }; L.d.className = 'olabel'; host.appendChild(L.d); olabels.set(e.id, L); }
    const rec = groups.get(e.id), pos = rec ? rec.group.position : new THREE.Vector3(...e.pose.pos);
    const h = (e.size?.[2] ?? 1.5) + 0.25;
    _v.set(pos.x, pos.y, pos.z + h).project(camera);
    L.x = (_v.x + 1) / 2 * W; L.y = (1 - _v.y) / 2 * H; L.front = _v.z < 1;
    const sel = SEL.includes(e.id), hov = HOVER === e.id, txt = shortName(nameOf(e)), fn = (sel || hov) && (e.function ?? objOf(e)?.function);
    const html = `${esc(txt)}${fn ? `<small>${esc(fn.length > 70 ? fn.slice(0, 68) + '…' : fn)}</small>` : ''}`;
    if (L.html !== html) { L.html = html; L.d.innerHTML = html; L.w = 0; }
    L.prio = (sel ? 1e6 : 0) + (hov ? 5e5 : 0) + (e.label || objOf(e) ? 1e3 : 0) + (e.size ? e.size[0] * e.size[1] : 1);
    L.d.classList.toggle('on', sel || hov); L.d.style.left = L.x + 'px'; L.d.style.top = L.y + 'px';
  }
  for (const [id, L] of olabels) if (!seen.has(id)) { L.d.remove(); olabels.delete(id); }
  if (t - olayoutAt < 120) return; olayoutAt = t;
  const maxN = W > 1100 ? 14 : 10, placed = [];
  const list = [...olabels.values()].sort((a, b) => b.prio - a.prio);
  for (const L of list) {
    if (!L.w) { L.w = L.d.offsetWidth; L.h = L.d.offsetHeight; }
    const r = { x0: L.x - L.w / 2 - 4, x1: L.x + L.w / 2 + 4, y0: L.y - L.h - 4, y1: L.y + 4 };
    const inView = L.front && L.x > 0 && L.x < W && L.y > 20 && L.y < H;
    const ok = inView && (L.prio >= 5e5 || (placed.length < maxN && !placed.some((p) => r.x0 < p.x1 && r.x1 > p.x0 && r.y0 < p.y1 && r.y1 > p.y0)));
    if (ok) placed.push(r);
    L.d.classList.toggle('hide', !ok);
  }
}

// pointer handling on the 3D view
let down = null, hoverReq = null, lastHoverAt = 0;
renderer.domElement.addEventListener('pointerdown', (e) => (down = [e.clientX, e.clientY]));
renderer.domElement.addEventListener('pointerleave', () => { setHover(null); $('#tip').hidden = true; });
renderer.domElement.addEventListener('pointermove', (e) => { hoverReq = e; });
function doHover(t) {
  if (!hoverReq || t - lastHoverAt < 60) return;
  const e = hoverReq; hoverReq = null; lastHoverAt = t;
  if (!pickEnabled() || e.buttons) { setHover(null); $('#tip').hidden = true; return; }
  const r = vp.getBoundingClientRect(), cx = e.clientX - r.left, cy = e.clientY - r.top;
  const id = pick(cx, cy); setHover(id);
  const inf = infoOf(id), tip = $('#tip');
  const L = olabels.get(id);
  if (!inf || (L && !L.d.classList.contains('hide'))) { tip.hidden = true; return; }   // its floating label already says it
  tip.hidden = false; tip.innerHTML = `<b>${esc(shortName(inf.label))}</b>${inf.hero ? '<span class="chip hero">3D</span>' : ''}<small>${esc(inf.cat ?? '')}${inf.e ? '' : ''}</small>`;
  tip.style.left = Math.min(cx + 16, W - 260) + 'px'; tip.style.top = Math.max(cy - 12, 8) + 'px';
}
renderer.domElement.addEventListener('pointerup', (e) => {
  if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4 || !pickEnabled()) return;
  if (stage === 'bim' && !wipeUser) { wipeUser = true; wipeX = null; }
  const r = vp.getBoundingClientRect(), id = pick(e.clientX - r.left, e.clientY - r.top);
  const multi = e.shiftKey || e.ctrlKey || e.metaKey;
  if (!id) { if (!multi) setSelection([]); return; }
  if (multi) setSelection(SEL.includes(id) ? SEL.filter((x) => x !== id) : [...SEL, id]);
  else {
    setSelection([id]);
    if (stage === 'explore' && isHero(groups.get(id)?.el)) openHero(id);
  }
});

// view switch (cloud / blend / model), segments, labels
function setView(v) {
  VIEW = v; $$('#viewsw [data-v]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.v === v)));
  if (stage === 'bim') { wipeUser = true; wipeX = null; }
  setHover(null);
}
$$('#viewsw [data-v]').forEach((b) => (b.onclick = () => setView(b.dataset.v)));
$('#sw-seg').onclick = () => { SEGMENTS = !SEGMENTS; $('#sw-seg').setAttribute('aria-pressed', String(SEGMENTS)); };
$('#sw-labels').onclick = () => { SHOW_LABELS = !SHOW_LABELS; $('#sw-labels').setAttribute('aria-pressed', String(SHOW_LABELS)); };

// ---------------------------------------------------------------- hero inspector: one machine, its generated 3D model, its own orbit
const hero = { renderer: null, scene: null, camera: null, controls: null, group: null, key: '' };
function openHero(id) {
  const e = groups.get(id)?.el; if (!isHero(e)) return;
  HERO = id; setSelection([id]);
  document.body.classList.add('hero-open');
  if (!SEL.includes(id)) setSelection([id]);
  if (VIEW === 'model' && stage === 'explore') setView('cloud');
  if (cloudU) cloudU.uHide.value = labelIndexOf(id);
  focusOn([id]); manual = false; camFrom = { target: cam.target.clone(), az: cam.az, el: cam.el, dist: cam.dist }; camAt = now();
  if (!hero.renderer) {
    hero.renderer = new THREE.WebGLRenderer({ canvas: $('#hero-gl'), antialias: true, alpha: true });
    hero.renderer.setPixelRatio(Math.min(devicePixelRatio, 2)); hero.renderer.setClearColor(0, 0);
    hero.scene = new THREE.Scene();
    hero.scene.add(new THREE.HemisphereLight(0xffffff, 0x2c3330, 1.8));
    const d = new THREE.DirectionalLight(0xffffff, 1.6); d.position.set(4, -6, 8); hero.scene.add(d);
    const g = new THREE.GridHelper(10, 20, 0x2a352d, 0x1b231d); g.rotation.x = Math.PI / 2; hero.scene.add(g);
    hero.camera = new THREE.PerspectiveCamera(35, 1, 0.02, 200); hero.camera.up.set(0, 0, 1);
    hero.controls = new OrbitControls(hero.camera, hero.renderer.domElement);
    hero.controls.enableDamping = true; hero.controls.autoRotate = true; hero.controls.autoRotateSpeed = 1.4;
    hero.controls.addEventListener('start', () => (hero.controls.autoRotate = false));
  }
  hero.key = ''; renderHero(true);
  $('#hero-prompt').value = HERO_PROMPT;
}
function renderHero(fit = false) {
  const e = groups.get(HERO)?.el; if (!e) return closeHero();
  const key = JSON.stringify(e.boxes) + (e.label ?? '');
  if (key !== hero.key) {
    hero.key = key;
    if (hero.group) hero.scene.remove(hero.group);
    const { g } = buildElement({ ...e, pose: { pos: [0, 0, 0], yaw: 0 } }, []);
    hero.group = g; hero.scene.add(g);
    if (fit) {
      const box = new THREE.Box3().setFromObject(g), c = box.getCenter(new THREE.Vector3()), r = box.getSize(new THREE.Vector3()).length() / 2;
      hero.controls.target.copy(c); hero.camera.position.set(c.x + r * 1.9, c.y - r * 2.3, c.z + r * 1.3);
    }
    const inf = infoOf(HERO);
    $('#hero-title').textContent = inf.label;
    $('#hero-fn').textContent = inf.fn ?? '';
    $('#hero-meta').innerHTML = `<span class="chip">${esc(inf.cat ?? '')}</span><span class="chip hero">generated 3D model</span>${inf.size ? `<span class="chip">${inf.size[0].toFixed(1)} × ${inf.size[1].toFixed(1)} × ${inf.size[2].toFixed(1)} m</span>` : ''}${inf.conf != null ? `<span class="chip">${Math.round(inf.conf * 100)} % label confidence</span>` : ''}`;
    $('#hero-parts').innerHTML = inf.parts.length ? inf.parts.map((p) => `<li><i style="background:${esc(p.color ?? '#ff8a3d')}"></i>${esc(p.name)}<small>${p.shape ?? 'box'} · ${p.s.map((v) => v.toFixed(2)).join(' × ')} m</small></li>`).join('') : '<li class="empty">No added parts yet. Ask the agent below.</li>';
  }
}
function closeHero() {
  HERO = null; document.body.classList.remove('hero-open');
  if (cloudU) cloudU.uHide.value = 0; focus.goal = 0; renderInfo();
}
function heroTick() {
  if (!HERO || !hero.renderer) return;
  const c = $('#hero-gl'), w = c.clientWidth, h = c.clientHeight;
  if (c.width !== Math.round(w * hero.renderer.getPixelRatio())) { hero.renderer.setSize(w, h, false); hero.camera.aspect = w / h; hero.camera.updateProjectionMatrix(); }
  renderHero(); hero.controls.update(); hero.renderer.render(hero.scene, hero.camera);
}
$('#hero-close').onclick = closeHero;
$('#hero-form').onsubmit = (ev) => { ev.preventDefault(); sendMessage($('#hero-prompt').value, [HERO], 'hero'); };

// ---------------------------------------------------------------- server I/O
let S = null, REC = null;
async function getRecorded() {
  if (!REC) REC = await (await fetch('recorded-session.json')).json();
  return REC;
}
async function post(url, body) {
  const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}
let rpcId = 0;
async function mcp(name, args = {}) {   // stateless streamable-HTTP JSON-RPC, JSON responses
  const r = await fetch('/mcp', { method: 'POST', headers: { 'Content-Type': 'application/json', Accept: 'application/json, text/event-stream' },
    body: JSON.stringify({ jsonrpc: '2.0', id: ++rpcId, method: 'tools/call', params: { name, arguments: args } }) });
  const j = await r.json();
  if (j.error) throw new Error(j.error.message);
  const text = j.result?.content?.find((c) => c.type === 'text')?.text ?? '{}';
  let out; try { out = JSON.parse(text); } catch { out = { text }; }
  if (j.result?.isError) throw new Error(out.text ?? text);
  if (out?.error) throw new Error(out.error);
  return out;
}

const agent = { key: null, at: 0, briefAt: 0, sentText: null, sentId: null, pickedUp: false };
const agentKey = (s) => JSON.stringify([s.history.length, s.history.at(-1)?.seq, s.agent_note, s.questions.length, s.requirements.length,
  s.next_steps.length, s.messages.filter((m) => m.read).length, s.questions.filter((q) => q.answer != null).length]);
let polling = false;
async function poll(force = false) {
  if (polling && !force) return; polling = true;
  try {
    const r = await fetch('/api/state' + (S && !force ? '?since=' + S.seq : ''));
    if (r.status !== 200) return;
    const prev = S; S = await r.json();
    const k = agentKey(S);
    if (agent.key != null && k !== agent.key && !D.loading) agent.at = now();
    agent.key = k;
    if (agent.sentText && !agent.pickedUp) {
      const m = S.messages.find((x) => x.text === agent.sentText && (!agent.sentId || x.id === agent.sentId));
      if (m) agent.sentId = m.id;
      if (m?.read) { agent.pickedUp = true; toast('The agent picked up your request'); }
    }
    onState(prev);
  } catch { /* server restarting */ } finally { polling = false; }
}
async function selfWrite(fn) {   // our own writes must not look like agent activity
  const out = await fn();
  await poll(true); agent.key = S ? agentKey(S) : agent.key;
  return out;
}

// ---------------------------------------------------------------- UI helpers
let toastT = 0;
function toast(t, bad = false) { const el = $('#toast'); el.textContent = t; el.className = 'toast on' + (bad ? ' bad' : ''); clearTimeout(toastT); toastT = setTimeout(() => (el.className = 'toast'), 3200); }
function caption(tag, t) {
  $('#vp-tag').textContent = tag;
  const s = $('#vp-caption span'); if (s.textContent === (t || '')) return;
  s.textContent = t || ''; s.classList.remove('fresh'); void s.offsetWidth; s.classList.add('fresh');
}
function setPill(el, ok, hold) {
  const c = 'pill ' + (ok == null ? '' : ok ? 'ok' : hold ? 'hold' : 'bad');
  if (el.dataset.c !== c) { el.dataset.c = c; el.className = c; if (ok != null) { void el.offsetWidth; el.classList.add('flip'); } }
}
function pills(v, root = document) {
  setPill(root.querySelector('#p-code, .p-code'), v?.code_ok);
  setPill(root.querySelector('#p-safety, .p-safety'), v?.safety_ok, v && !v.safety_ok && v.counts?.hold > 0);
  setPill(root.querySelector('#p-physics, .p-physics'), v?.physics_ok);
  setPill(root.querySelector('#p-wishes, .p-wishes'), v ? v.desires_ok : null, true);
}
const liveSummary = (v) => {
  if (!v?.counts) return '';
  const { fail = 0, hold = 0 } = v.counts;
  return fail || hold ? [fail && `${fail} to fix`, hold && `${hold} to sign off`].filter(Boolean).join(' · ') : 'All checks pass';
};
const fact = (big, label, cls = '') => `<div class="fact ${cls}"><b>${big}</b><span>${label}</span></div>`;
const count = (m) => Object.values(m.elements).reduce((a, e) => ((a[e.kind] = (a[e.kind] ?? 0) + 1), a), {});
const baselineIdx = (H) => { let i = -1; H.forEach((h, j) => { if (h.action === 'fit_bim') i = j; }); return i < 0 && H.length ? 0 : i; };
const baselineOf = (st) => { const H = st?.history ?? []; const i = baselineIdx(H); return i >= 0 ? H[i].model : st?.model; };
const hasDesign = (st) => !!st && st.history.length - 1 > baselineIdx(st.history);
const mmss = (s) => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, '0')}`;

// ---------------------------------------------------------------- stages
let stage = 'capture', stageAt = 0, cloudRate = 1, revealDoneAt = 0;
const video = $('#video');
const vdur = () => (isFinite(video.duration) && video.duration > 0 ? video.duration : meta ? meta.frames / meta.fps : 10);

function setStage(s, { push = true } = {}) {
  const prevStage = stage;
  stage = s; stageAt = now(); document.body.dataset.stage = s;
  const i = STAGES.indexOf(s), N = STAGES.length;
  $$('#stage-nav a').forEach((a, j) => { a.dataset.n = '0' + (j + 1); if (j === i) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current'); a.classList.toggle('done', j < i); });
  $('#title').innerHTML = `${COPY[s].title}<sup>0${i + 1}</sup>`; $('#lede').textContent = COPY[s].lede;
  for (const el of [$('#title'), $('#lede')]) { el.classList.remove('fresh'); void el.offsetWidth; el.classList.add('fresh'); }
  $('#pos').textContent = '0' + (i + 1); $('#pos-n').textContent = '0' + N;
  $('#btn-back').disabled = i === 0;
  $('#btn-next').textContent = i < N - 1 ? `${COPY[STAGES[i + 1]].nav} →` : 'Back to start ↺';
  if (push) history.replaceState(null, '', '#' + s);
  if (HERO && s !== 'explore' && s !== 'design') closeHero();
  if (s !== prevStage || s === 'explore') setView(s === 'explore' && !OBJ.labels ? 'model' : DEFAULT_VIEW[s] ?? 'model');
  // scene layers
  if (cloud) { cloud.visible = s !== 'capture'; frustum.visible = pathLine.visible = s === 'cloud'; }
  world.visible = ['bim', 'explore', 'design', 'deliver'].includes(s); overlay.visible = s === 'design' || s === 'deliver';
  cutNow = Infinity; wipeUser = false; setWipe(null); revealDoneAt = 0; focus = { c: null, w: 0, goal: 0, r: 0 };
  $('#vp-note').textContent = ''; $('#signoff').hidden = true; D.anim = null;
  groups.forEach((r) => { r.group.position.copy(r.base); r.group.quaternion.copy(r.baseQ); r.move = null; });
  const base = baselineOf(S) ?? baselineOf(REC);
  if (base) frameModel(base); else if (cloudC) { sceneC.copy(cloudC); sceneR = cloudR; }
  clipCeiling.constant = ceiling - 0.2;
  if (s === 'capture') { video.loop = true; video.playbackRate = 1; video.play().catch(() => {}); caption('', ''); renderCaptureFacts(); }
  if (s === 'cloud') {
    cloudRate = Math.max(1, vdur() / 14);
    video.loop = false; video.currentTime = 0; video.playbackRate = Math.min(16, cloudRate); video.play().catch(() => {});
    caption('LingBot-Map', meta ? 'Reconstructing the floor from the video' : 'No point cloud yet: run reconstruct_scan');
    if (meta) $('#vp-note').textContent = `Precomputed: ${meta.inference_seconds ?? '?'} s of GPU inference on an ${GPU} · replayed in capture order`;
    if (cloud) revealTo(0);
    if (base) setCloudPoses(base, false);
    renderCloudFacts(0, 0);
  }
  if (s === 'bim') {
    video.pause(); if (cloud) revealTo(1e9);
    clearWorld();
    if (base) {
      const ax = Math.cos(cam.az + Math.PI / 2), ay = Math.sin(cam.az + Math.PI / 2), along = (e) => e.pose.pos[0] * ax + e.pose.pos[1] * ay;
      const RANK = { slab: 0, wall: 1, column: 2, door: 3, window: 3, opening: 3, equipment: 4 };
      applyModel(base, { animate: true, stagger: 2600, cloudAnim: false, order: (a, b) => (RANK[a.kind] ?? 5) - (RANK[b.kind] ?? 5) || along(a) - along(b) });
      groups.forEach((r) => (r.born += 700));
      caption('fit_bim', 'Point cloud → as-built model');
    } else caption('fit_bim', 'No as-built model yet: run fit_bim');
    renderBimFacts(base);
  }
  if (s === 'explore') {
    video.pause(); if (cloud) revealTo(1e9);
    const m = D.model && D.mode !== 'idle' ? D.model : base;
    if (m) applyModel(m, { flag: {}, cloudAnim: false });
    caption('explore', OBJ.labels ? 'Hover any point · click a machine to open it' : 'Click any machine to see what it is');
    renderExplorePanel();
  }
  if (s === 'design') { video.pause(); if (cloud) revealTo(1e9); enterDesign(); loadScenarioList(); }
  if (s === 'deliver') { video.pause(); if (cloud) revealTo(1e9); enterDeliver(); }
  renderInfo(); renderMentions();
  camStage();
}
const go = (d) => { const i = STAGES.indexOf(stage) + d; setStage(STAGES[(i + STAGES.length) % STAGES.length]); };

function renderCaptureFacts() {
  const v = video, bits = [];
  bits.push(fact(isFinite(v.duration) ? mmss(v.duration) : '–', 'video length'));
  if (v.videoWidth) bits.push(fact(`${v.videoWidth}<small>×${v.videoHeight}</small>`, 'phone video, RGB only'));
  if (meta) bits.push(fact(fmt(meta.frames), `frames sampled at ${meta.fps} fps<br>for reconstruction`));
  $('#cap-facts').innerHTML = bits.join('');
}
function renderCloudFacts(n, frame) {
  if (!meta) { $('#cloud-facts').innerHTML = fact('–', 'no reconstruction found'); return; }
  const done = n >= meta.n;
  $('#cloud-facts').innerHTML = fact(fmt(n), done ? 'coloured points' : 'points so far', done ? 'hl' : '')
    + fact(`${frame}<small>/ ${meta.frames}</small>`, 'frames fused')
    + fact(`${meta.inference_seconds ?? '–'}<small>s</small>`, `inference on an ${GPU}<br>(precomputed)`)
    + fact(`${meta.extent_m?.[0] ?? '–'}<small>× ${meta.extent_m?.[1] ?? '–'} m</small>`, 'floor extent');
}
function renderBimFacts(m) {
  if (!m) { $('#bim-facts').innerHTML = fact('–', 'no model yet'); $('#bim-fine').textContent = ''; return; }
  const c = count(m), sc = m.scan ?? {};
  $('#bim-facts').innerHTML = fact(c.wall ?? 0, 'walls') + fact((c.door ?? 0) + (c.window ?? 0), 'doors + windows')
    + fact(c.equipment ?? 0, 'equipment, coloured<br>from the scan')
    + (sc.fit_p50_mm != null ? fact(`${Math.round(sc.fit_p50_mm)}<small>mm</small>`, `median fit to the scan<br>P95 ${Math.round(sc.fit_p95_mm)} mm`, 'hl') : '');
  $('#bim-fine').textContent = [sc.perimeter && `Perimeter ${sc.perimeter}.`, sc.scale_source && `Scale: ${sc.scale_source}.`].filter(Boolean).join(' ');
}
function renderExplorePanel() {
  if (!$('#ex-facts')) return;
  const heroes = OBJ.list.filter((o) => o.mesh && OBJ.meshes.has(o.mesh));
  $('#ex-facts').innerHTML = OBJ.list.length
    ? fact(OBJ.list.length, 'objects segmented<br>in the point cloud') + (OBJ.labels ? fact(fmt(OBJ.labels.reduce((a, x) => a + (x > 0), 0)), 'points assigned<br>to an object') : '') + fact(heroes.length, 'with a generated<br>3D model', heroes.length ? 'hl' : '')
    : fact('–', 'segmentation not available yet:<br>showing the fitted boxes');
  $('#ex-hint').textContent = OBJ.labels ? 'Hover any point to light up the object it belongs to. Click to select, shift-click to add more.' : 'Hover or click any machine to see what it is. Shift-click to select more.';
  $('#ex-heroes').innerHTML = heroes.map((o) => { const id = elementForObject(o); return id ? `<button class="hero-btn" data-id="${esc(id)}"><span class="chip hero">3D</span>${esc(o.label)}<b>→</b></button>` : ''; }).join('');
  $$('#ex-heroes .hero-btn').forEach((b) => (b.onclick = () => openHero(b.dataset.id)));
  const sel = SEL.map(infoOf).filter(Boolean);
  $('#ex-sel').innerHTML = sel.length ? `<div class="mono-label">Selected</div><div class="mentions">${sel.map((x) => `<span class="mention">@${esc(shortName(x.label))}</span>`).join('')}</div>
    <button class="button primary" id="ex-ask">Ask the agent about ${sel.length > 1 ? 'these' : 'this'} →</button>` : '';
  $('#ex-ask')?.addEventListener('click', () => setStage('design'));
}

// ---------------------------------------------------------------- mentions + suggestions (selection -> request)
function renderMentions() {
  const box = $('#mentions'), ids = SEL.filter((id) => !id.startsWith('obj:'));
  box.innerHTML = ids.map((id) => `<span class="mention" data-id="${esc(id)}">@${esc(shortName(infoOf(id)?.label ?? id))}<button aria-label="Remove">×</button></span>`).join('');
  $$('#mentions .mention button').forEach((b) => (b.onclick = () => setSelection(SEL.filter((x) => x !== b.parentElement.dataset.id))));
  const n = ids.length, one = n === 1 ? infoOf(ids[0]) : null;
  const sugg = n === 0 ? [] : n === 1
    ? [`Move this ${one?.cat === 'machine' ? 'machine' : 'item'} to the east wall and keep a 1.2 m aisle`, 'What clearance does this need?', 'Replace this with a robot cell']
    : [`Move these ${n === 2 ? 'two' : n} machines to the east wall and keep a 1.2 m aisle`, 'Group these into one work cell with a guard fence'];
  $('#suggest').innerHTML = sugg.map((s) => `<button type="button" class="sugg">${esc(s)}</button>`).join('');
  $$('#suggest .sugg').forEach((b) => (b.onclick = () => { $('#brief-text').value = b.textContent; sendMessage(b.textContent, ids, 'brief'); }));
  if (n && $('#brief-text').value === DEFAULT_BRIEF) $('#brief-text').value = '';
  if (!n && !$('#brief-text').value) $('#brief-text').value = DEFAULT_BRIEF;
}

// ---------------------------------------------------------------- 05 design: live agent + replay + scenarios
// mode: idle (as-built, waiting for a brief) | live (following the project as the agent works) | replay (playing a session)
const D = { mode: 'idle', src: null, srcState: null, queue: [], liveShown: null, histSig: null, settled: true, busyUntil: 0, anim: null, evMs: 1000,
  model: null, val: null, seenMsg: new Set(), seenQ: new Set(), feedT: 0, answered: {}, loading: false };

function feedAdd(cls, text, tag = '') {
  const li = document.createElement('li'); li.className = cls;
  li.innerHTML = `<i></i><span>${esc(text)}</span><small>${esc(tag)}</small>`;
  $$('#feed li.now').forEach((x) => { x.classList.remove('now'); x.classList.add('done'); });
  if (!/you|ask|note/.test(cls)) li.classList.add('now');
  $('#feed').appendChild(li); $('#feed').scrollTop = $('#feed').scrollHeight;
}
function feedClear(src = '') { $('#feed').innerHTML = ''; $('#feed-src').textContent = src; D.seenMsg = new Set(); D.seenQ = new Set(); }
const mentionText = (m, st) => {
  const ids = Array.isArray(m.selected) ? m.selected : [];
  const els = st?.model?.elements ?? {};
  return ids.length ? ' · ' + ids.map((id) => '@' + shortName(nameOf(els[id]) ?? id)).join(' ') : '';
};
function timelineUpTo(st, t) {   // human requests and agent questions, interleaved with tool calls by timestamp
  if (!st) return;
  const items = [...st.messages.map((m) => ({ t: m.t, m })), ...st.questions.map((q) => ({ t: q.t, q }))].sort((a, b) => a.t - b.t);
  for (const it of items) {
    if (it.t > t || it.t < D.feedT) continue;
    if (it.m && !D.seenMsg.has(it.m.id)) { D.seenMsg.add(it.m.id); feedAdd('you', it.m.text + mentionText(it.m, st)); }
    if (it.q && !D.seenQ.has(it.q.id)) { D.seenQ.add(it.q.id); feedAdd('ask', it.q.question); }
  }
}
function eventRow(ev) {
  const v = ev.validation;
  let cls = '';
  if (ev.action === 'validate' && v) cls = v.code_ok && v.safety_ok && v.physics_ok !== false ? 'ok' : (v.counts?.fail ? 'bad' : 'hold');
  if (ev.actor === 'human') cls = 'ok';
  feedAdd(cls, ev.caption, ev.action);
}
const failFlags = (v) => {
  const f = {};
  for (const i of v?.issues ?? []) if (i.element && i.severity === 'hold') f[i.element] = HOLD;
  for (const i of v?.issues ?? []) if (i.element && i.severity === 'fail') f[i.element] = BAD;
  return f;
};

function enterDesign() {
  if (D.mode === 'idle' || !D.model) {
    const base = baselineOf(S) ?? baselineOf(REC);
    if (base) applyModel(base, { animate: true, stagger: 600 });
    D.model = base; D.val = null; pills(null); egressKey = ''; overlay.clear();
    if (D.mode === 'idle') {
      feedClear(''); $('#feed').dataset.empty = 'Send the brief, pick a pre-generated scenario, or replay the recorded session.';
      caption('as-built', 'Ready for a brief');
    }
  } else {
    applyModel(D.model, { flag: D.settled ? failFlags(D.val) : {}, cloudAnim: false }); pills(D.val); egressKey = ''; drawEgress(D.val);
    if (D.settled) settle(true);
  }
  D.busyUntil = Math.max(D.busyUntil, now() + 900);
}

async function startReplay({ liveOnly = false, label = null } = {}) {
  if (D.mode === 'replay' && D.queue.length && !liveOnly) return;
  const live = hasDesign(S);
  if (liveOnly && !live) { goLive(); D.settled = false; if (stage !== 'design') setStage('design'); return; }
  const st = live ? S : await getRecorded();
  const bi = baselineIdx(st.history);
  D.mode = 'replay'; D.src = live ? 'live' : 'recorded'; D.srcState = st; if (HERO) closeHero(); setSelection([]);
  D.queue = st.history.slice(bi + 1); D.evMs = clamp(26000 / Math.max(1, D.queue.length), 700, 1300);
  if (live) { D.liveShown = S.history.length; D.histSig = histSig(S); }
  D.feedT = st.history[bi]?.t ?? 0; D.settled = false; D.anim = null;
  feedClear(label ?? (live ? `replay · project history · ${D.queue.length} steps` : `replay · recorded session · ${D.queue.length} steps`));
  if (stage !== 'design') setStage('design');
  clearWorld(); applyModel(st.history[bi].model, { cloudAnim: false }); D.model = st.history[bi].model; D.val = null;
  pills(null); egressKey = ''; overlay.clear(); $('#signoff').hidden = true;
  caption('replay', label ? 'Replaying how the agent got there' : live ? 'Replaying the agent session on this project' : 'Replaying the recorded session');
  D.busyUntil = now() + 1100; $('#btn-skip').hidden = false;
}
function skipReplay() {
  if (!D.queue.length) return;
  D.anim = null; groups.forEach((r) => { r.group.position.copy(r.base); r.group.quaternion.copy(r.baseQ); });
  for (const ev of D.queue) { timelineUpTo(D.srcState, ev.t); eventRow(ev); D.model = ev.model; D.val = ev.validation; }
  D.queue = []; applyModel(D.model, { animate: true }); pills(D.val); drawEgress(D.val); D.busyUntil = 0;
}
function goLive() {   // follow the live project from here on
  D.mode = 'live'; D.src = 'live'; D.srcState = S; D.queue = []; D.liveShown = S.history.length; D.histSig = histSig(S); D.settled = false; D.anim = null;
  D.feedT = 0; feedClear('live · this project'); D.seenMsg = new Set(S.messages.map((m) => m.id)); D.seenQ = new Set(S.questions.map((q) => q.id));
  $('#btn-skip').hidden = true;
  D.model = S.model; D.val = S.validation; applyModel(S.model, { animate: true }); D.evMs = 900;
}

function stepDesign(t) {
  if (t < D.busyUntil) return;
  if (D.anim) { D.anim = null; groups.forEach((r) => { r.group.position.copy(r.base); r.group.quaternion.copy(r.baseQ); }); }
  if (D.queue.length) {
    const ev = D.queue.shift();
    timelineUpTo(D.srcState, ev.t);
    const v = ev.validation, failing = ev.action === 'validate' && v && !(v.code_ok && v.safety_ok && v.physics_ok !== false);
    const flag = {}; for (const id of ev.focus ?? []) flag[id] = failing ? BAD : LIME;
    applyModel(ev.model, { animate: true, flag }); D.model = ev.model; D.val = v;
    pills(v); drawEgress(v); if (!HERO) focusOn(ev.focus);
    caption(ev.actor === 'human' ? 'you' : ev.action, ev.caption); eventRow(ev);
    D.busyUntil = t + D.evMs + (ev.action === 'validate' ? 500 : 0) + (ev.action === 'move_element' ? 500 : 0);
    if (ev.animation?.frames?.length) {
      const names = movedNames(ev.animation);
      if (names.length) caption('MuJoCo', `Stability test · ${names.slice(0, 2).join(', ')} ${names.length > 1 ? 'fall' : 'falls'} over`);
      if (names.length) feedAdd('bad note', `MuJoCo tilt test: ${names.slice(0, 2).join(', ')} ${names.length > 1 ? 'fall' : 'falls'} over`, 'physics');
      D.anim = { frames: ev.animation.frames, fps: (ev.animation.fps || 25) * 0.6, t0: t + 450 };
      D.busyUntil = D.anim.t0 + ev.animation.frames.length * 1000 / D.anim.fps + 1800;
    }
    D.settled = false;
    return;
  }
  if (!D.settled && D.mode !== 'idle') settle();
}
function settle(quiet = false) {
  D.settled = true; $('#btn-skip').hidden = true;
  const recorded = D.src === 'recorded', st = recorded ? D.srcState : S;
  if (!st) return;
  timelineUpTo(st, Infinity);
  if (!recorded) {
    D.model = S.model; D.val = S.validation;
    applyModel(S.model, { animate: !quiet, flag: failFlags(S.validation) });
    if (D.mode === 'replay') D.mode = 'live';
  } else applyModel(D.model, { flag: failFlags(st.validation) });
  const v = recorded ? st.validation : S.validation;
  pills(v); drawEgress(v); if (!HERO) focus.goal = 0;
  caption(recorded ? 'recorded' : `R${st.model.revision ?? 1}`, st.agent_note || liveSummary(v) || st.history.at(-1)?.caption || '');
  showSignoff(st);
}
function showSignoff(st) {
  const q = st?.questions?.find((x) => x.answer == null && !D.answered[x.id]);
  const box = $('#signoff');
  if (!q || stage !== 'design') { box.hidden = true; box.dataset.id = ''; return; }
  const liveQ = S?.questions?.find((x) => x.id === q.id && x.question === q.question && x.answer == null);
  if (box.dataset.id === q.id && !box.hidden) return;
  box.dataset.id = q.id; box.hidden = false; $('#info').hidden = true;
  $('#so-q').textContent = q.question;
  $('#so-foot').textContent = liveQ ? '' : 'Recorded question: the live agent is not waiting on it, so your click is not sent.';
  $('#so-opts').innerHTML = '';
  q.options.forEach((o, i) => {
    const b = document.createElement('button'); b.className = 'button ' + (i === 0 ? 'primary' : 'secondary'); b.textContent = o;
    b.onclick = async () => {
      b.classList.add('hit'); $$('#so-opts button').forEach((x) => (x.disabled = true));
      D.answered[q.id] = o;
      if (liveQ) {
        try { await post('/api/answer', { id: q.id, answer: o }); await poll(true); toast(`Answer sent: ${o}`); }
        catch (ex) { toast('Could not send the answer: ' + ex.message, true); delete D.answered[q.id]; }
      } else toast(`Answered (replay only): ${o}`);
      feedAdd('you', o);
      setTimeout(() => { box.hidden = true; box.dataset.id = ''; renderInfo(); }, 500);
      caption('you', liveQ ? `Signed: “${o}” · the agent continues` : `Signed: “${o}”`);
    };
    $('#so-opts').appendChild(b);
  });
}

async function sendMessage(text, ids = [], where = 'brief') {
  text = (text ?? '').trim(); if (!text) return;
  ids = ids.filter((id) => id && !id.startsWith('obj:'));
  if (D.mode !== 'live') { if (D.mode === 'replay' && D.queue.length) skipReplay(); goLive(); }
  const btn = where === 'hero' ? $('#hero-send') : $('#btn-send'); btn.disabled = true;
  try {
    agent.sentText = text; agent.sentId = null; agent.pickedUp = false;
    await selfWrite(() => post('/api/message', ids.length ? { text, selected: ids } : { text }));
    const m = [...S.messages].reverse().find((x) => x.text === text); if (m) { agent.sentId = m.id; D.seenMsg.add(m.id); }
    agent.briefAt = now();
    feedAdd('you', text + (ids.length ? ' · ' + ids.map((id) => '@' + shortName(infoOf(id)?.label ?? id)).join(' ') : ''));
    caption('request', 'Sent · waiting for the agent');
    toast(ids.length ? `Sent with ${ids.length} referenced object${ids.length > 1 ? 's' : ''}` : 'Brief sent to the agent');
    if (where === 'brief') { setSelection([]); $('#brief-text').value = text; }
  } catch (ex) { toast('Could not send: ' + ex.message, true); }
  finally { setTimeout(() => (btn.disabled = false), 1500); }
}

const histSig = (s) => s.history.slice(0, 3).map((h) => `${h.seq}:${h.t}`);
const sameProject = (sig, s) => { const now = histSig(s), n = Math.min(sig.length, now.length); for (let i = 0; i < n; i++) if (sig[i] !== now[i]) return false; return true; };
function onState(prev) {
  if (!S) return;
  if (D.liveShown == null) { D.liveShown = S.history.length; D.histSig = histSig(S); }
  const replaced = D.histSig && !sameProject(D.histSig, S);
  if (S.history.length < D.liveShown || (replaced && !D.loading)) {   // reset (fit_bim) or another scenario loaded under us
    D.liveShown = S.history.length; D.histSig = histSig(S);
    if (!(D.mode === 'replay' && D.src === 'recorded')) {
      D.mode = 'idle'; D.queue = []; D.model = null; D.settled = true;
      if (stage === 'design') enterDesign();
    }
  } else if (S.history.length > D.liveShown && !D.loading) {
    const fresh = S.history.slice(D.liveShown);
    if (D.mode === 'idle') { goLive(); D.model = baselineOf(S); applyModel(D.model); D.liveShown = S.history.length - fresh.length; }
    if (D.mode === 'live' || (D.mode === 'replay' && D.src === 'live')) { D.queue.push(...S.history.slice(D.liveShown)); D.liveShown = S.history.length; D.srcState = S; D.settled = false; }
  } else if (prev && prev.seq !== S.seq && D.mode === 'live' && !D.queue.length) D.settled = false;   // note / question / answer changed
  if (D.mode === 'live') D.srcState = S;
  if (stage === 'deliver' && prev && prev.seq !== S.seq) renderDeliver(false);
}

function agentIndicator() {
  const t = now(), el = $('#agent-state'), openQ = S?.questions?.some((q) => q.answer == null);
  let cls = '', text = 'Agent idle';
  const silentSinceBrief = agent.briefAt && agent.at < agent.briefAt && t - agent.briefAt > 10000;
  if (D.mode === 'replay' && D.queue.length) { cls = 'replay'; text = 'Replaying a session'; }
  else if (agent.at && t - agent.at < 12000) { cls = 'working'; text = 'Agent working'; }
  else if (silentSinceBrief) { cls = 'missing'; text = 'No agent detected'; }
  else if (openQ && D.mode === 'live') { cls = 'waiting'; text = 'Waiting for your sign-off'; }
  if (el.dataset.c !== cls + text) { el.dataset.c = cls + text; el.className = 'agent-state ' + cls; el.querySelector('span').textContent = text; }
  $('#hint').hidden = !(silentSinceBrief && stage === 'design');
}

// pre-generated scenarios (list_scenarios / load_scenario): jump to a saved result, then replay how the agent got there
let scenarios = [];
async function loadScenarioList() {
  try { scenarios = (await mcp('list_scenarios')).scenarios.filter((s) => s.name !== 'autosave_before_load'); } catch { scenarios = []; }
  $('#scen-wrap').hidden = !scenarios.length;
  $('#scen').innerHTML = scenarios.map((s) => `<button class="scen" data-name="${esc(s.name)}"><b>${esc(s.title)}</b><small>${esc(s.revision)} · ${s.events} step${s.events === 1 ? '' : 's'}</small></button>`).join('');
  $$('#scen .scen').forEach((b) => (b.onclick = () => loadScenario(b.dataset.name, b)));
}
async function loadScenario(name, btn) {
  if (D.loading) return;
  D.loading = true; $$('#scen .scen').forEach((x) => (x.disabled = true)); btn.classList.add('busy');
  const sc = scenarios.find((s) => s.name === name);
  try {
    await selfWrite(() => mcp('load_scenario', { name }));
    D.liveShown = S.history.length; D.histSig = histSig(S); D.answered = {}; agent.briefAt = 0;
    closeHero(); setSelection([]);
    $$('#scen .scen').forEach((x) => x.classList.toggle('on', x.dataset.name === name));
    D.loading = false;
    await startReplay({ liveOnly: true, label: `pre-generated · ${sc?.title ?? name}` });
    toast(`Loaded pre-generated scenario: ${sc?.title ?? name}`);
  } catch (ex) { toast('Could not load: ' + ex.message, true); }
  finally { D.loading = false; btn.classList.remove('busy'); $$('#scen .scen').forEach((x) => (x.disabled = false)); }
}

// ---------------------------------------------------------------- 06 deliver
let qtyCache = null;
function deliverSource() { return D.src === 'recorded' && !hasDesign(S) && REC ? REC : S; }
function enterDeliver() {
  clearWorld(); renderDeliver(true);
  const st = deliverSource();
  $('#files').innerHTML = ''; $('#export-status').textContent = st === REC ? 'Showing the recorded session. Export writes the live project.' : '';
  if (st === S) mcp('get_quantities').then((q) => { qtyCache = q; renderQty(); }).catch(() => { $('#qty').innerHTML = '<p class="fine">Quantities unavailable.</p>'; });
  else { qtyCache = null; $('#qty').innerHTML = '<p class="fine">Quantities come from the live project (get_quantities).</p>'; }
}
function renderDeliver(animate) {
  const st = deliverSource(); if (!st) return;
  applyModel(st.model, { animate, stagger: animate ? 900 : 0, flag: failFlags(st.validation), cloudAnim: false });
  pills(st.validation); egressKey = ''; drawEgress(st.validation);
  const v = st.validation;
  caption(`R${st.model.revision ?? 1}`, liveSummary(v) || 'Current model');
  $('#dl-checks').innerHTML = ['code', 'safety', 'physics', 'wishes'].map((k) => `<span class="pill p-${k}"><i></i>${k[0].toUpperCase() + k.slice(1)}</span>`).join('');
  pills(v, $('#dl-checks'));
  $('#next').innerHTML = (st.next_steps ?? []).map((s) => `<li>${esc(s.title)}<small>${esc(s.who)}${s.why ? ' · ' + esc(s.why) : ''}</small></li>`).join('');
}
function renderQty() {
  const q = qtyCache; if (!q) return;
  const rows = (q.items ?? []).slice().sort((a, b) => b.cost_usd - a.cost_usd).slice(0, 5);
  $('#qty').innerHTML = `<div class="total">$${fmt(q.material_cost_usd ?? 0)}<small>material, rough order of magnitude</small></div>`
    + rows.map((r) => `<div class="row">${esc(r.name)}<span>${r.volume_m3} m³ · $${fmt(r.cost_usd)}</span></div>`).join('')
    + (q.items?.length ? '' : '<p class="fine">No proposed elements yet.</p>');
}
const FILE_TAG = (f) => f.endsWith('.ifc') ? ['IFC', 'IFC4 BIM model'] : f.endsWith('.glb') ? ['GLB', '3D for any viewer'] : f.endsWith('.mjcf.xml') ? ['MJCF', 'MuJoCo physics scene']
  : f === 'validation.json' ? ['JSON', 'every check + scan fit'] : f === 'model.json' ? ['JSON', 'parametric model'] : f.endsWith('.md') ? ['MD', 'package readme'] : ['SHA', 'checksums'];
async function doExport() {
  const b = $('#btn-export'); b.disabled = true; $('#export-status').textContent = 'Exporting… re-runs every check and the MuJoCo drop test';
  try {
    const r = await selfWrite(() => mcp('export_package'));
    const order = (f) => ['.ifc', '.glb', '.mjcf.xml', 'validation.json', 'model.json', 'README.md', 'SHA256SUMS'].findIndex((x) => f.endsWith(x));
    $('#files').innerHTML = r.files.slice().sort((a, b2) => order(a) - order(b2)).map((f) => { const [tag, what] = FILE_TAG(f);
      return `<li><a href="/data/export/${encodeURIComponent(f)}" download="${esc(f)}"><b>${tag}</b>${esc(f)}<small>${what}</small></a></li>`; }).join('');
    const el = r.elements ?? {}, d = r.drop_test ?? {};
    $('#export-status').textContent = [el.ifc_products != null && `${el.ifc_products} IFC elements read back`, d.dropped != null && `drop test: ${d.at_rest ?? '?'}/${d.dropped} at rest`].filter(Boolean).join(' · ');
    toast('Package exported');
  } catch (ex) { $('#export-status').textContent = 'Export failed: ' + ex.message; toast('Export failed', true); }
  finally { b.disabled = false; }
}

// ---------------------------------------------------------------- per-frame animation
function animateStage(t) {
  const st = t - stageAt;
  stepCloudPoses(t);
  if (stage === 'cloud' && meta) {
    const vt = Math.min(vdur(), st / 1000 * cloudRate), n = revealTo(vt / vdur() * meta.frames * 1.0001);
    if (!video.paused && Math.abs(video.currentTime - vt) > 0.5) video.currentTime = vt;
    if (n >= meta.n && !revealDoneAt) { revealDoneAt = t; caption('LingBot-Map', 'Coloured point cloud · gravity-aligned, metric'); }
    const fr = revealDoneAt ? 1 - span(t - revealDoneAt, 300, 1300) : 1;
    frustum.scale.setScalar(Math.max(0.001, fr)); frustum.visible = fr > 0.01;
    if (((t / 120) | 0) !== animateStage.last) { animateStage.last = (t / 120) | 0; renderCloudFacts(n, Math.min(meta.frames, Math.floor(vt * meta.fps) + 1)); }
    cloudU.uOpModel.value = cloudU.uOpScan.value = 1; cloudU.uSeg.value = 0;
  }
  if (stage === 'bim') {
    if (cloudU) { cloudU.uOpModel.value = 1 - 0.8 * ease(span(st, 900, 3800)); cloudU.uOpScan.value = 1; cloudU.uReveal.value = meta.n; cloudU.uFresh.value = 1; cloudU.uSeg.value = 0; }
    if (!wipeUser) {
      let w = null;
      if (st > 4600) w = st < 6200 ? -1.05 + 1.4 * ease(span(st, 4600, 6200)) : st < 7800 ? 0.35 - 0.7 * ease(span(st, 6200, 7800)) : -0.35 + 0.35 * ease(span(st, 7800, 8800));
      if (st > 6000 && st < 6100) caption('compare', 'Scan vs. rebuilt model · drag to compare');
      setWipe(w);
    } else {
      setWipe(wipeX);
      if (VIEW !== 'compare' && cloudU) applyViewLayers(st);
    }
  } else if (wipeX != null) setWipe(null);
  if (['explore', 'design', 'deliver'].includes(stage)) applyViewLayers(st);
  for (const r of groups.values()) updateElement(r, t);
  if (D.anim && t >= D.anim.t0 && stage === 'design') {
    const f = D.anim.frames[Math.min(D.anim.frames.length - 1, Math.floor((t - D.anim.t0) / 1000 * D.anim.fps))];
    for (const [id, p] of Object.entries(f)) { const r = groups.get(id); if (r) { r.group.position.set(p[0], p[1], p[2]); r.group.quaternion.set(p[4], p[5], p[6], p[3]); } }
  }
}
function applyViewLayers(st) {   // cloud / blend / model
  const cloudView = VIEW === 'cloud', blend = VIEW === 'blend';
  const k = ease(span(st, 200, 1600));
  cutNow = cloudView ? Infinity : k >= 1 ? CUT : (ceiling + 0.5) + (CUT - ceiling - 0.5) * k;
  clipCeiling.constant = cloudView ? ceiling - 0.2 : blend ? Math.max(CUT_TALL, 0.3) + 0.3 : Math.max(cutNow, 0.3) + 0.02;
  if (cloudU) {
    const op = cloudView ? 1 : blend ? 0.6 : 1 - 0.88 * ease(span(st, 0, 1200));
    cloudU.uOpModel.value = cloudU.uOpScan.value = op; cloudU.uFresh.value = 1; cloudU.uSeg.value = SEGMENTS && (cloudView || blend) ? 1 : 0;
    cloudU.uDim.value = HOVER && cloudView && labelIndexOf(HOVER) ? 1 : 0;
  }
  if (cloud) cloud.visible = true;
  for (const r of groups.values()) r.group.visible = !cloudView || r.el.id === HERO || (!OBJ.labels && SEL.includes(r.el.id));
  overlay.visible = !cloudView && (stage === 'design' || stage === 'deliver');
  $('#labels').style.visibility = cloudView ? 'hidden' : '';
  const dt = Math.min(200, now() - (applyViewLayers.t ?? now())); applyViewLayers.t = now();
  focus.w += (focus.goal - focus.w) * (1 - Math.exp(-dt / 450));
  updateEgress(now());
}

const v3 = new THREE.Vector3();
function tick(t) {
  requestAnimationFrame(tick);
  animateStage(t);
  if (stage === 'design') stepDesign(t);
  agentIndicator();
  if (stage !== 'capture') { updateCamera(t); renderer.render(scene, camera); doHover(t); }
  layoutObjectLabels(t);
  if (stage === 'design' || stage === 'deliver') for (const l of labels) {
    v3.copy(l.v).project(camera); l.d.style.left = ((v3.x + 1) / 2 * W) + 'px'; l.d.style.top = ((1 - v3.y) / 2 * H) + 'px';
  }
  heroTick();
}

// ---------------------------------------------------------------- capture-stage thumbnails (real frames from the video)
async function grabThumbs() {
  const v = document.createElement('video'); v.src = '/data/site.mp4'; v.muted = true; v.preload = 'auto';
  await new Promise((r) => { v.addEventListener('loadeddata', r, { once: true }); v.addEventListener('error', r, { once: true }); setTimeout(r, 6000); });
  if (!v.videoWidth) return;
  const c = document.createElement('canvas'); c.width = 424; c.height = 200; const g = c.getContext('2d');
  for (const f of [0.18, 0.5, 0.82]) {
    const at = f * v.duration;
    await new Promise((r) => { v.addEventListener('seeked', r, { once: true }); v.currentTime = at; setTimeout(r, 3000); });
    const s = Math.max(c.width / v.videoWidth, c.height / v.videoHeight), w = v.videoWidth * s, h = v.videoHeight * s;
    g.drawImage(v, (c.width - w) / 2, (c.height - h) / 2, w, h);
    const b = document.createElement('button'); b.title = `Jump to ${mmss(at)}`;
    b.innerHTML = `<img alt="Frame at ${mmss(at)}" src="${c.toDataURL('image/jpeg', 0.8)}">`;
    b.onclick = () => { video.currentTime = at; video.play().catch(() => {}); };
    $('#thumbs').appendChild(b);
  }
}

// ---------------------------------------------------------------- wiring
$('#btn-next').onclick = () => go(1);
$('#btn-back').onclick = () => go(-1);
$$('#stage-nav a').forEach((a) => (a.onclick = (e) => { e.preventDefault(); setStage(a.dataset.s); }));
$('.wordmark').onclick = (e) => { e.preventDefault(); setStage('capture'); };
for (const id of ['#btn-replay', '#btn-replay2', '#btn-replay3']) $(id).onclick = () => startReplay();
$('#btn-skip').onclick = skipReplay;
$('#brief').onsubmit = (ev) => { ev.preventDefault(); startReplay(); };   // presenter build: Send plays the recorded run, never a live agent
$('#btn-recloud').onclick = () => setStage('cloud');
$('#btn-rebim').onclick = () => setStage('bim');
$('#btn-export').onclick = doExport;
$('#hint-cmd').textContent = AGENT_CMD;
$('#btn-copy').onclick = () => navigator.clipboard?.writeText(AGENT_CMD).then(() => toast('Command copied'), () => toast('Copy failed: select the command', true));
$('#btn-reset').onclick = () => { $('#modal').hidden = false; $('#modal-yes').focus(); };
$('#modal-no').onclick = () => ($('#modal').hidden = true);
$('#modal-yes').onclick = async () => {
  const b = $('#modal-yes'); b.disabled = true; b.textContent = 'Refitting the as-built model…';
  try {
    D.loading = true;
    await selfWrite(() => mcp('fit_bim', { reset_design: true }));
    D.mode = 'idle'; D.queue = []; D.model = null; D.src = null; D.liveShown = S.history.length; D.histSig = histSig(S); D.answered = {};
    agent.briefAt = 0; agent.sentText = null; closeHero(); setSelection([]);
    $$('#scen .scen').forEach((x) => x.classList.remove('on'));
    $('#modal').hidden = true; toast('Reset to the as-built model'); setStage('design');
  } catch (ex) { toast('Reset failed: ' + ex.message, true); }
  finally { D.loading = false; b.disabled = false; b.textContent = 'Reset to as-built'; }
};
addEventListener('keydown', (e) => {
  if (/^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName) || !$('#modal').hidden) { if (e.key === 'Escape') $('#modal').hidden = true; return; }
  if (e.key === 'Escape') { if (HERO) closeHero(); else setSelection([]); return; }
  if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') { e.preventDefault(); go(1); }
  else if (e.key === 'ArrowLeft' || e.key === 'PageUp') { e.preventDefault(); if (STAGES.indexOf(stage) > 0) go(-1); }
});
addEventListener('hashchange', () => { const h = location.hash.slice(1); if (STAGES.includes(h) && h !== stage) setStage(h, { push: false }); });
video.addEventListener('loadedmetadata', () => stage === 'capture' && renderCaptureFacts());

// ---------------------------------------------------------------- boot
setStage(STAGES.includes(location.hash.slice(1)) ? location.hash.slice(1) : 'capture', { push: false });
await Promise.all([loadCloud().then(loadObjects), poll(true)]);
if (!hasDesign(S)) getRecorded().catch(() => {});
$('#loading').classList.add('off');
resize();
setStage(stage, { push: false });
setInterval(poll, 700);
requestAnimationFrame(tick);
grabThumbs();
window.__live = { project: (id, dz = 1) => { const r = groups.get(id); if (!r) return null; const b = elementBox(r.el); const c = b.getCenter(new THREE.Vector3()); c.z = b.max.z * 0.6; c.project(camera); const rr = vp.getBoundingClientRect(); return [rr.left + (c.x + 1) / 2 * W, rr.top + (1 - c.y) / 2 * H]; },
  heroBox: () => { const r = groups.get(HERO); return r && new THREE.Box3().setFromObject(r.group); }, groups, OBJ, S: () => S, setSelection, openHero, setView, pick: (x, y) => pick(x, y), stage: () => stage, D, MOVE };   // test hook
