// Capture Studio demo viewer: video -> point cloud -> BIM -> agentic design. Reads everything from the MCP server's state.
// Every animation is a function of clock(). With ?capture=1 the clock is virtual and advanced by window.__demo.frame(ms)
// (scripts/record_demo.py steps it at 30 fps and screenshots each frame), otherwise it is performance.now().
import * as THREE from 'three';
import { OrbitControls } from '/web/vendor/OrbitControls.js';
import { GLTFLoader } from '/web/vendor/GLTFLoader.js';
import { mergeGeometries } from '/web/vendor/BufferGeometryUtils.js';

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
const SCAN_MS = 6000, CLOUD_TARGET_MS = 14000, CLOUD_HOLD_MS = 2800, BIM_MS = 11000, BIM_TAG_SHIFT = 4400, UNDERSTAND_MS = 16000, HERO_MS = 12500;
let modelEditId = null;
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
const meshCut = new THREE.Plane(new THREE.Vector3(0, 0, -1), 1e6);   // keeps z <= constant: design cutaway / hero assemble for meshes
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
  if (stage === 'understand') return { target: sceneC, az: camShot.az0 + s * 0.035, el: 0.88, dist: fitDist(sceneR) * 0.64 };
  if (stage === 'replan') return { target: sceneC, az: camShot.az0 + s * 0.008, el: 1.02, dist: fitDist(sceneR) * 0.9 };
  if ((stage === 'hero' && !heroCam) || ((stage === 'hero' || stage === 'modeledit') && !focusC)) return { target: camShot.t0, az: camShot.az0 + s * 0.02, el: camShot.el0, dist: camShot.d0 };
  if (stage === 'hero' || stage === 'modeledit') return { target: focusC, az: camShot.az0 + s * (stage === 'hero' ? 0.2 : 0.12), el: stage === 'hero' ? 0.45 : 0.6, dist: focusD };
  // design: higher, slow drift, gentle push towards what the agent just touched
  const w = focus.w, tgt = sceneC.clone();
  if (focus.c && w > 0) tgt.lerp(focus.c, 0.4 * w);
  return { target: tgt, az: camShot.az0 + s * 0.012, el: 0.95, dist: fitDist(sceneR) * (0.98 - 0.12 * w) };
}

function camStage() { camFrom = { target: cam.target.clone(), az: cam.az, el: cam.el, dist: cam.dist }; camAt = clock(); camShot = { az0: cam.az, t0: cam.target.clone(), el0: cam.el, d0: cam.dist, at: clock() }; manual = false; }
let heroCam = false, focusC = null, focusD = 10, focusKey = null, held = false, autoKey = null;

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
uniform float uReveal, uLo, uFresh, uSize, uScale, uMinPx, uMaxPx, uTint, uBgOp, uBgClip, uDim, uLift, uObjOp; uniform vec3 uHi; uniform vec4 uWipe;
uniform sampler2D uObj;
attribute float label;
varying vec3 vColor; varying float vSide, vA;
vec3 hue(float h) { return clamp(abs(fract(h + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0, 0.0, 1.0); }
void main() {
  float id = float(gl_VertexID);
  if (id >= uReveal || id < uLo) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; return; }
  vec3 p = position; float hl = 0.0, hide = 0.0;
  int li = int(label + 0.5);
  if (li > 0) {           // per-object rigid move (origin -> current), highlight, dissolve
    vec4 A = texelFetch(uObj, ivec2(0, li), 0), B = texelFetch(uObj, ivec2(1, li), 0);
    float c = cos(B.x), s = sin(B.x); vec2 d = p.xy - A.xy;
    p.xy = vec2(c * d.x - s * d.y, s * d.x + c * d.y) + A.zw; hl = B.y; hide = B.z;
    p.z += uLift * hide * hide * (0.5 + fract(sin(id * 12.9898) * 43758.5453) * 1.5);
  } else if (p.z > uBgClip) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; return; }
  vA = li > 0 ? (1.0 - hide) * mix(uObjOp, 1.0, hl) : uBgOp;
  vec4 wp = modelMatrix * vec4(p, 1.0);
  vec4 mvPosition = viewMatrix * wp;
  float fresh = 1.0 - smoothstep(0.0, 1.0, (uReveal - id) / max(uFresh, 1.0));
  vColor = mix(color, uHi, fresh * 0.8);
  if (uTint > 0.0) {      // semantic tint: one hue per object, neutral grey for unlabelled points
    vec3 seg = label > 0.5 ? mix(vec3(0.92), hue(fract(label * 0.618034 + 0.07)), 0.78) : vec3(0.22, 0.24, 0.27) + color * 0.12;
    vColor = mix(vColor, seg, uTint);
  }
  vColor = mix(vColor, vColor * 0.2 + vec3(0.015), uDim * (1.0 - hl));
  vColor = mix(vColor, min(vec3(1.0), mix(vColor, uHi, 0.35) * 1.35), hl);
  vSide = dot(uWipe.xyz, wp.xyz) + uWipe.w;
  gl_PointSize = clamp(uSize * uScale / -mvPosition.z * (1.0 + fresh * 1.6 + hl * 0.35), uMinPx, uMaxPx);
  gl_Position = projectionMatrix * mvPosition;
  #include <clipping_planes_vertex>
}`;
const CLOUD_FS = `
#include <clipping_planes_pars_fragment>
uniform float uOpScan, uOpModel, uRound; varying vec3 vColor; varying float vSide, vA;
void main() {
  #include <clipping_planes_fragment>
  vec2 c = gl_PointCoord - 0.5; if (uRound > 0.5 && dot(c, c) > 0.25) discard;
  float a = vSide < 0.0 ? uOpScan : uOpModel * vA; if (a < 0.01) discard;
  gl_FragColor = vec4(vColor, a);
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
    uOpScan: { value: 1 }, uOpModel: { value: 1 }, uMinPx: { value: 1.5 }, uMaxPx: { value: 64 }, uRound: { value: 1 }, uTint: { value: 0 },
    uBgOp: { value: 1 }, uBgClip: { value: 1e6 }, uDim: { value: 0 }, uLift: { value: 0 }, uObjOp: { value: 1 }, uObj: { value: new THREE.DataTexture(new Float32Array(8), 2, 1, THREE.RGBAFormat, THREE.FloatType) } };
  cloudU.uObj.value.needsUpdate = true;
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
  const geo = meshFor(e);
  if (!geo && segObject(e)) { addParts(e, g, parts); return { g, parts, fades: [], seg: true }; }   // its points are the model
  if (geo) {                // segmented scan mesh (object-local, Z up, origin on the floor) instead of the fitted box
    const bb = geo.boundingBox, h = Math.max(0.05, bb.max.z - bb.min.z), bg = new THREE.Group();
    const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.78, metalness: 0.08, emissive: 0x000000,
      side: THREE.DoubleSide, clippingPlanes: [wipePlane, meshCut] }));
    mesh.position.z = -(bb.min.z + h / 2); mesh.userData.id = e.id; mesh.castShadow = mesh.receiveShadow = true; bg.add(mesh);
    if ((geo.index ? geo.index.count : geo.attributes.position.count) < 60000) {
      const ed = new THREE.LineSegments(geo.userData.edges ??= new THREE.EdgesGeometry(geo, 35), edgeMat(0x2b3038, 0.6));
      ed.material.clippingPlanes = [wipePlane, meshCut]; ed.position.z = mesh.position.z; bg.add(ed);
    }
    g.add(bg);
    parts.push({ bg, bottom: bb.min.z, h, z0: e.pose.pos[2], cut: null, hero: true });
    addParts(e, g, parts);
    return { g, parts, fades: [], hero: true };
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

function addParts(e, g, parts) {     // add_part add-ons: snapshot boxes carrying a part id
  for (const b of e.boxes ?? []) {
    if (!b.part) continue;
    const geo = b.shape === 'cylinder' ? new THREE.CylinderGeometry(Math.min(b.s[0], b.s[1]) / 2, Math.min(b.s[0], b.s[1]) / 2, b.s[2], 28).rotateX(Math.PI / 2) : new THREE.BoxGeometry(...b.s);
    const bg = new THREE.Group(); bg.position.set(b.c[0], b.c[1], b.c[2]);
    const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: hex(b.color, ACCENT), roughness: 0.55, metalness: 0.1, emissive: hex(b.color, ACCENT), emissiveIntensity: 0.12, clippingPlanes: [wipePlane] }));
    mesh.userData.id = e.id; mesh.userData.part = true; mesh.castShadow = mesh.receiveShadow = true; bg.add(mesh);
    const ed = new THREE.LineSegments(new THREE.EdgesGeometry(geo, 30), edgeMat(0xffffff, 0.5)); ed.userData.partEdge = true; bg.add(ed);
    g.add(bg);
    parts.push({ bg, bottom: b.c[2] - b.s[2] / 2, h: b.s[2], z0: e.pose.pos[2], cut: null, part: b.part, name: b.name, c: b.c, s: b.s });
  }
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
    const built = { g, parts, fades };
    const rec = { group: g, parts, fades, key, el: e, born: animate ? now + delay : 0, pop: animate && e.provenance === 'proposed',
      base: g.position.clone(), baseQ: g.quaternion.clone(), hero: !!meshFor(e), seg: !meshFor(e) && segObject(e) };
    if (old && animate) {           // same element rebuilt: only what is new animates (added parts pop in, the body stays)
      const had = new Set(old.parts.filter((p) => p.part).map((p) => p.part));
      rec.born = 0; for (const p of parts) if (p.part && !had.has(p.part)) { p.born = now; p.isNew = true; }
      if (old.group.position.distanceTo(g.position) > 0.01 || old.group.quaternion.angleTo(g.quaternion) > 0.01) rec.slide = { from: old.group.position.clone(), fromQ: old.group.quaternion.clone(), t0: now };
    }
    world.add(g); groups.set(e.id, rec);
    setFlag(rec, flag.includes(e.id) ? flagColor : null);
    updateElement(rec, now);
  }
  for (const [id, rec] of groups) if (!seen.has(id)) { world.remove(rec.group); groups.delete(id); sel.delete(id); }
  sel.forEach((id) => groups.get(id) && setFlag(groups.get(id), 0xffffff));
  syncObjects(model, !animate);
  $('#rev').textContent = model.revision ? 'R' + model.revision : '';
  buildLabels(model);
}

let cutNow = Infinity;
function updateElement(r, now) {
  let k = 1;
  if (r.born) { k = clamp((now - r.born) / (r.pop ? 650 : 750)); if (now >= r.born + 2000) r.born = 0; }
  const rise = r.born ? (r.pop ? easeBack(k) : easeOut(k)) : 1;
  for (const [m, o] of r.fades) m.opacity = o * clamp(k);
  if (r.slide) {        // moved element glides like its points
    const k2 = ease((now - r.slide.t0) / MOVE_MS);
    r.group.position.lerpVectors(r.slide.from, r.base, k2); r.group.quaternion.slerpQuaternions(r.slide.fromQ, r.baseQ, k2);
    if (k2 >= 1) r.slide = null;
  }
  if (r.hero) r.group.visible = heroMeshOn || r.focus === true;
  for (const p of r.parts) {
    let h = p.h;
    if (p.born) { const kp = clamp((now - p.born) / 600); h *= Math.max(0.001, easeBack(kp)); if (kp >= 1) p.born = 0; }
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
      o.material.emissive.setHex(color ?? (proposed ? ACCENT : 0)); o.material.emissiveIntensity = color === BAD ? 0.75 : color === 0xffffff ? 0.16 : color ? 0.45 : (proposed ? 0.2 : 0);
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
const TAGS = { scan: 'Video', cloud: 'Point cloud', bim: 'BIM', understand: 'Objects', hero: '3D model', replan: 'Re-plan', modeledit: 'Model edit', design: 'Agentic design' };
function caption(t) {
  $('#tag').textContent = TAGS[stage];
  const s = $('#caption span'); if (s.textContent === (t || '')) return;
  s.textContent = t || ''; s.classList.remove('fresh'); void s.offsetWidth; s.classList.add('fresh');
}
let toastUntil = 0;
function toast(t, ok = false) { $('#sent').textContent = t; $('#sent').className = ok ? 'ok' : ''; toastUntil = clock() + 2600; }

// ---------------------------------------------------------------- stages + replay
let S = null, stage = 'scan', queue = [], shown = 0, busyUntil = 0, anim = null, autoplay = false, stageAt = 0;
const sel = new Set();
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

function setStage(s, opts = {}) {
  if (follow) exitFollow();
  stage = s; stageAt = clock(); document.body.dataset.stage = s;
  const order = ['scan', 'cloud', 'bim', 'understand', 'design'];
  document.querySelectorAll('#steps button[data-s]').forEach((b) => { b.classList.toggle('on', b.dataset.s === s); b.classList.toggle('done', order.indexOf(b.dataset.s) < order.indexOf(s)); });
  if (cloud) { cloud.visible = s !== 'scan'; frustum.visible = pathLine.visible = s === 'cloud'; }
  world.visible = ['bim', 'design', 'understand', 'hero', 'replan', 'modeledit'].includes(s); overlay.visible = s === 'design'; $('#labels').style.display = s === 'design' ? '' : 'none';
  anim = null; revealDoneAt = 0; liveAt = 0; cutNow = Infinity; wipePlane.copy(NOCLIP()); meshCut.constant = 1e6;
  resetTags(); setModelOpacity(1); held = false; heroCam = false; focusC = null; autoKey = null; hoverKey = null; highlight([]);
  heroMeshOn = !['understand', 'hero', 'cloud', 'scan'].includes(s);
  groups.forEach((r) => { r.group.visible = true; r.focus = false; });
  objXf.forEach((X) => X && (X.hide = 0));
  if (cloudU) { cloudU.uTint.value = cloudU.uDim.value = cloudU.uLift.value = 0; cloudU.uBgOp.value = cloudU.uObjOp.value = 1; cloudU.uBgClip.value = 1e6; }
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
      if (hasLabels()) tagMode = { kind: 'seq', t0: stageAt + 3900, step: 380, max: 7, heroes: 2, t1: stageAt + 8500 };
      $('#stat').innerHTML = bits.join(' · ') + (base.scan?.fit_p50_mm != null ? ` · fit <b>${base.scan.fit_p50_mm} mm</b> P50` : '');
    } else caption('No BIM yet — run fit_bim');
  }
  if (['understand', 'hero', 'replan', 'modeledit'].includes(s)) { video.pause(); if (cloud) revealTo(1e9); }
  if (s === 'understand') {
    if (base) applyModel(base);
    world.visible = false;
    caption('The model understands the scan'); $('#stat').innerHTML = `<b>${OBJ?.objects?.length ?? 0}</b> objects segmented and identified`;
  }
  if (s === 'hero') {
    if (base) applyModel(base);
    const o = OBJ?.byKey.get(opts.key) ?? heroes()[0]; focusKey = o?.key ?? null;
    groups.forEach((r) => { const mine = r.el.object_key === focusKey; r.group.visible = mine; r.focus = mine; });
    if (o) { const w = objWorld(o); focusC = new THREE.Vector3(w.x, w.y, o.size[2] * 0.45); focusD = Math.max(3, fitDist(Math.hypot(...o.size) / 2) * 1.2); }
    meshCut.constant = -1;
    if (opts.clicked) stageAt -= 1000;
    caption(o ? 'Click an object to see its model' : 'No hero object in this scan');
  }
  if (s === 'replan' || s === 'modeledit') {
    const H = S?.history ?? [], act = s === 'replan' ? 'move_element' : 'add_part';
    const i0 = H.findIndex((h) => h.action === act), start = i0 > 0 ? i0 - 1 : Math.max(0, baseIdx);
    if (H[start]) applyModel(H[start].model); else if (base) applyModel(base);
    queue = i0 > 0 ? H.slice(start + 1) : []; shown = H.length; held = !!opts.hold;
    busyUntil = stageAt + 2000; pills(H[start]?.validation ?? null); S && (S._liveShown = null);
    evMs = 1500;
    if (s === 'modeledit') {
      const fid = H[i0]?.focus?.[0] ?? Object.values(S?.model?.elements ?? {}).find((e) => (e.boxes ?? []).some((b) => b.part))?.id;
      const r = groups.get(fid); focusKey = r?.el.object_key ?? null;
      if (r) { const b = elementBox(r.el), c = b.getCenter(new THREE.Vector3()), z = b.getSize(new THREE.Vector3()); focusC = new THREE.Vector3(c.x, c.y, z.z * 0.45); focusD = Math.max(3.5, fitDist(Math.hypot(z.x, z.y, z.z) / 2) * 1.5); }
      tagMode = { kind: 'parts', list: [] }; modelEditId = fid; cutNow = CUT;
      caption(r ? `Edit the model · ${r.el.label || r.el.name}` : 'No model edits in this scenario');
    } else caption(queue.length ? 'Re-plan the floor' : 'No moves in this scenario');
  }
  if (heroMeshOn) for (const o of heroes()) { const e = elementFor(o.key); if (e && meshFor(e) && objXf[o.index]) objXf[o.index].hide = 1; }   // the mesh stands in for its points
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
  if (!['design', 'replan', 'modeledit'].includes(stage) || !S || now < busyUntil || held) return;
  if (anim) { anim = null; groups.forEach((r) => { r.group.position.copy(r.base); r.group.quaternion.copy(r.baseQ); }); }
  if (queue.length) {
    const ev = queue.shift();
    const failing = ev.action === 'validate' && ev.validation && !(ev.validation.code_ok && ev.validation.safety_ok && ev.validation.physics_ok !== false);
    applyModel(ev.model, { animate: true, flag: ev.focus ?? [], flagColor: failing ? BAD : ACCENT });
    pills(ev.validation); focusOn(ev.focus ?? []);
    let cap = ev.caption || '';
    for (const id of ev.focus ?? []) { const e = ev.model.elements[id]; if (e?.label && e.name && cap.includes(e.name)) cap = cap.replace(e.name, e.label); }
    caption(cap);
    busyUntil = now + evMs + (ev.action === 'validate' ? 500 : 0) + (ev.action === 'move_element' ? MOVE_MS : 0);
    if (stage === 'modeledit' && tagMode?.kind === 'parts') {       // name each new part as it appears
      const r = groups.get(modelEditId);
      for (const p of r?.parts ?? []) if (p.isNew && !tagMode.list.some((q) => q.key === p.part)) {
        p.isNew = false; r.group.updateMatrixWorld(true);
        tagMode.list.push({ key: p.part, label: p.name, v: new THREE.Vector3(p.c[0], p.c[1], p.c[2] + p.s[2] / 2).applyMatrix4(r.group.matrixWorld), t: now + 250, prio: 1 });
      }
    }
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
    pills(S.validation); if (stage === 'design') { side(S, true); drawEgress(S.validation); }
    if (first) { caption(S.agent_note || liveSummary(S.validation) || S.history.at(-1)?.caption || ''); focus.goal = 0; liveAt = now; }
  }
}

async function poll() {
  try {
    const r = await fetch('/api/state?since=' + (S?.seq ?? -1));
    if (r.status !== 200) return;
    const prev = S; S = await r.json();
    if (prev) S._liveShown = prev._liveShown;
    if (['design', 'replan', 'modeledit'].includes(stage)) {
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
    : stage === 'bim' ? t >= BIM_MS + (hasLabels() ? BIM_TAG_SHIFT : 0)
    : stage === 'understand' ? t >= UNDERSTAND_MS
    : stage === 'hero' ? t >= HERO_MS
    : stage === 'replan' || stage === 'modeledit' ? !held && !queue.length && liveAt > 0 && clock() - liveAt >= 2500
    : (liveAt > 0 && clock() - liveAt >= 2500) || (!S?.history?.length && t > 3000);
  return { stage, t, done, queue: queue.length, question: !!S?.questions?.find((q) => q.answer == null), hoverKey, focusKey, held, seg: SEG,
    heroes: heroes().map((o) => o.key), objects: OBJ?.objects?.length ?? 0 };
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
    if (cloudU) {          // segmented: the objects' points stay, the shell's points give way to walls and slab
      const f = ease(span(t, 900, 3800));
      if (SEG) { cloudU.uOpModel.value = 1; cloudU.uBgOp.value = 1 - 0.85 * f; } else cloudU.uOpModel.value = 1 - 0.8 * f;
      cloudU.uOpScan.value = 1; cloudU.uReveal.value = meta.n; cloudU.uFresh.value = 1;
    }
    // compare wipe: sweep in from the left, rock, then hand the frame back to the model
    let w = null;
    const t0 = t; { const t = t0 - (hasLabels() ? BIM_TAG_SHIFT : 0);
    if (t > 4600 && t < 10200) {
      w = t < 6200 ? -1.05 + 1.4 * ease(span(t, 4600, 6200)) : t < 7800 ? 0.35 - 0.7 * ease(span(t, 6200, 7800)) : t < 8800 ? -0.35 + 0.35 * ease(span(t, 7800, 8800)) : -1.05 * ease(span(t, 9200, 10200));
      if (t > 6000 && t < 6100) caption('Scan vs. rebuilt model');
    } }
    setWipe(w);
  } else setWipe(null);
  if (stage === 'understand' && cloudU) {   // hover beats (autopilot unless a person is pointing), then every segment tinted
    cloudU.uReveal.value = meta.n; cloudU.uLo.value = 0; cloudU.uFresh.value = 1; cloudU.uOpModel.value = cloudU.uOpScan.value = 1;
    if (now - pointerAt > 2500) {
      const slot = Math.floor((t - 1200) / 2700), inSlot = (t - 1200) % 2700 < 2300 && slot >= 0 && slot < 4;
      if (inSlot && (autoKey == null || autoKey.slot !== slot)) {
        const pick = rankObjects().filter((o) => !autoKey?.used?.includes(o.key) && objectScreen(o.key)?.ok)[0];
        autoKey = { slot, key: pick?.key ?? null, used: [...(autoKey?.used ?? []), pick?.key] };
      }
      setHover(inSlot ? autoKey?.key ?? null : null);
    }
    const tint = ease(span(t, 12000, 13000));
    cloudU.uTint.value = tint;
    cloudU.uDim.value += ((hoverKey && tint === 0 ? 1 : 0) - cloudU.uDim.value) * 0.2;
    if (tint > 0) { setHover(null); caption(`Segmented into ${OBJ?.objects?.length ?? 0} objects`); }
    else if (hoverKey) caption('Hover an object: all of its points, and what it is');
  }
  if (stage === 'hero' && cloudU) {
    const o = OBJ?.byKey.get(focusKey);
    cloudU.uReveal.value = meta.n; cloudU.uLo.value = 0; cloudU.uFresh.value = 1; cloudU.uOpModel.value = cloudU.uOpScan.value = 1; cloudU.uLift.value = 1;
    if (o) {
      objXf[o.index].hide = ease(span(t, 1200, 2800));
      highlight(t < 1200 && t > 300 ? [o.key] : []);
      const f = ease(span(t, 1100, 2400));
      cloudU.uDim.value = 0.85 * f; cloudU.uBgOp.value = 1 - 0.7 * f; cloudU.uObjOp.value = 1 - 0.7 * f;
      meshCut.constant = o.size[2] * 1.08 * ease(span(t, 1700, 3600)) - 0.01;
      if (!heroCam && t > 1100) { heroCam = true; camStage(); }
      if (t > 1200) caption(o.label);
      if (t > 3600 && tagMode?.kind !== 'hero') {
        const m = (x) => (Math.round(x * 10) / 10).toFixed(1);
        tagMode = { kind: 'hero', key: o.key, t0: 0, props: `<span>${m(o.size[0])} × ${m(o.size[1])} × ${m(o.size[2])} m</span><span>${esc(o.category ?? 'object')}</span>` +
          (o.confidence != null ? `<span>${Math.round(o.confidence * 100)}% confidence</span>` : '') + `<span>${fmt(o.points ?? 0)} points</span>` };
      }
    }
  }
  if ((stage === 'replan' || stage === 'modeledit') && cloudU) {
    cloudU.uReveal.value = meta.n; cloudU.uLo.value = 0; cloudU.uFresh.value = 1; cloudU.uOpModel.value = cloudU.uOpScan.value = 1;
    cloudU.uBgOp.value = stage === 'replan' ? 0.35 : 0.2; cloudU.uBgClip.value = stage === 'replan' ? CUT + 0.6 : CUT; cloudU.uObjOp.value = stage === 'replan' ? 1 : 0.3;
    clipCeiling.constant = 1e6;
  }
  if (stage === 'design') {
    const k = ease(span(t, 200, 1600));
    cutNow = k >= 1 ? CUT : (ceiling + 0.5) + (CUT - ceiling - 0.5) * k;
    meshCut.constant = Math.max(cutNow, CUT_TALL);
    if (cloudU && SEG) {   // segmented: machines stay as points, the shell's points drop below the cut
      clipCeiling.constant = 1e6; cloudU.uBgClip.value = Math.max(cutNow, 0.3); cloudU.uOpModel.value = 1; cloudU.uBgOp.value = 1 - 0.75 * ease(span(t, 0, 1200)); cloudU.uFresh.value = 1;
    } else {
      clipCeiling.constant = Math.max(cutNow, 0.3) + 0.02;
      if (cloudU) { cloudU.uOpModel.value = 1 - 0.85 * ease(span(t, 0, 1200)); cloudU.uFresh.value = 1; }
    }
    focus.w += (focus.goal - focus.w) * 0.03;
    updateEgress(now);
  } else if (stage !== 'scan' && stage !== 'replan' && stage !== 'modeledit') clipCeiling.constant = ceiling - 0.2;
  updateObjects(now);
  for (const r of groups.values()) updateElement(r, now);
  if (anim && now >= anim.t0) {
    const f = anim.frames[Math.min(anim.frames.length - 1, Math.floor((now - anim.t0) / 1000 * anim.fps))];
    for (const [id, p] of Object.entries(f)) { const r = groups.get(id); if (r) { r.group.position.set(p[0], p[1], p[2]); r.group.quaternion.set(p[4], p[5], p[6], p[3]); } }
  }
  if (toastUntil && now > toastUntil) { toastUntil = 0; $('#sent').textContent = ''; $('#sent').className = ''; }
  updateTags(now);
}

// ---------------------------------------------------------------- interaction
document.querySelectorAll('#steps button[data-s]').forEach((b) => (b.onclick = () => { autoplay = false; setStage(b.dataset.s); }));
$('#play').onclick = () => { autoplay = true; setStage('scan'); };
$('#say form').onsubmit = (ev) => {
  ev.preventDefault(); const i = $('#say input'); if (!i.value.trim()) return;
  if (!window.__demo?.dry) post('/api/message', { text: i.value.trim(), selected: [...sel] }); i.value = ''; toast('Sent to the agent ✓', true); i.blur();
};
const ray = new THREE.Raycaster(); let down = null;
renderer.domElement.addEventListener('pointerdown', (e) => (down = [e.clientX, e.clientY]));
renderer.domElement.addEventListener('pointerup', (e) => {
  if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4 || follow || !['bim', 'understand', 'design', 'replan', 'modeledit'].includes(stage)) return;
  const nx = (e.clientX / innerWidth) * 2 - 1, ny = -(e.clientY / innerHeight) * 2 + 1;
  const key = SEG ? pickObject(nx, ny) : null;
  if (key && OBJ.byKey.get(key)?.hero && MESHES.has(OBJ.byKey.get(key).mesh) && !e.shiftKey && (stage === 'bim' || stage === 'understand')) { setStage('hero', { key, clicked: true }); return; }
  let id = key ? elementFor(key)?.id ?? null : null;
  if (!id) {
    ray.setFromCamera(new THREE.Vector2(nx, ny), camera);
    const hit = ray.intersectObjects(world.children, true).find((h) => h.object.userData.id && !h.object.userData.ghost && groups.get(h.object.userData.id)?.el.kind !== 'slab' && h.object.visible);
    id = hit?.object.userData.id ?? null;
  }
  if (!e.shiftKey) for (const k of [...sel]) if (k !== id) { sel.delete(k); groups.get(k) && setFlag(groups.get(k), null); }
  if (id) { if (e.shiftKey && sel.has(id)) { sel.delete(id); setFlag(groups.get(id), null); } else { sel.add(id); setFlag(groups.get(id), 0xffffff); } }
  highlight([...sel].map((i) => groups.get(i)?.el.object_key).filter(Boolean));
  renderChips();
});
function renderChips() {      // selected elements travel with the next message
  $('#chips').innerHTML = [...sel].map((id) => `<span class="chip" data-id="${id}">@${esc(groups.get(id)?.el.label || groups.get(id)?.el.name || id)}<b>×</b></span>`).join('');
  $('#chips').querySelectorAll('.chip b').forEach((b) => (b.onclick = () => { const id = b.parentNode.dataset.id; sel.delete(id); groups.get(id) && setFlag(groups.get(id), null); renderChips(); }));
}
$('#steps button[data-s="understand"]').hidden = true; $('#steps .sep-u').hidden = true;


// ---------------------------------------------------------------- segmented objects: labels per point, hero meshes, rigid moves
// objects.json (all objects: label/function/pos/size/yaw), cloud_labels.bin (uint16 per point, k = objects[k-1]),
// objects.glb (hero meshes only, object-local, stored glTF Y-up). Every piece is optional.
let OBJ = null, SEG = false;
const MESHES = new Map();                       // mesh name -> BufferGeometry (Z-up, object-local, origin on the floor)
const objXf = [];                               // per label index: { o, from, to, t0, hl, hlFrom, hlTo, hlT0, hide }
let objTex = null;
const boxesMode = PARAMS.get('boxes') === '1';
let heroMeshOn = true;
const toLinear = (c) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));

async function loadObjects() {
  try { const r = await fetch('/data/scan/objects.json'); if (r.ok) OBJ = await r.json(); } catch { OBJ = null; }
  if (!OBJ?.objects?.length) { OBJ = null; return; }
  OBJ.byKey = new Map(OBJ.objects.map((o, i) => [o.key, { ...o, index: o.index ?? i + 1 }]));
  if (OBJ.objects.some((o) => o.mesh)) {
    try {
      const gltf = await new GLTFLoader().loadAsync('/data/scan/objects.glb');
      gltf.scene.updateMatrixWorld(true);
      const yup = String(OBJ.glb_up ?? 'Y').toUpperCase() === 'Y', lists = new Map();
      gltf.scene.traverse((o) => {
        if (!o.isMesh) return;
        const g = o.geometry.clone(); g.applyMatrix4(o.matrixWorld); if (yup) g.rotateX(Math.PI / 2);
        const c = g.attributes.color;
        if (c && OBJ.vertex_colors !== 'linear') {   // most exporters write sRGB bytes; three expects linear vertex colours
          const a = new Float32Array(c.count * 3);
          for (let i = 0; i < c.count; i++) { a[i * 3] = toLinear(c.getX(i)); a[i * 3 + 1] = toLinear(c.getY(i)); a[i * 3 + 2] = toLinear(c.getZ(i)); }
          g.setAttribute('color', new THREE.BufferAttribute(a, 3));
        }
        g.userData.map = o.material?.map ?? null;
        const name = o.name || o.parent?.name;
        for (const n of new Set([name, o.parent?.name, o.geometry.name].filter(Boolean))) { if (!lists.has(n)) lists.set(n, []); lists.get(n).push(g); }
      });
      for (const [n, gs] of lists) {
        let g = gs[0];
        if (gs.length > 1) { try { g = mergeGeometries(gs.map((x) => x.index ? x.toNonIndexed() : x)) ?? gs[0]; g.userData.map = gs[0].userData.map; } catch { /* keep first */ } }
        if (!g.attributes.normal) g.computeVertexNormals();
        g.computeBoundingBox(); MESHES.set(n, g);
      }
    } catch (e) { console.warn('objects.glb not loaded', e); }
  }
  const N = Math.max(0, ...[...OBJ.byKey.values()].map((o) => o.index));
  objTex = new THREE.DataTexture(new Float32Array(2 * (N + 1) * 4), 2, N + 1, THREE.RGBAFormat, THREE.FloatType);
  objTex.needsUpdate = true;
  for (const o of OBJ.byKey.values()) {
    const id = { x: o.pos[0], y: o.pos[1], yaw: 0 };
    objXf[o.index] = { o, from: id, to: id, cur: { ...id }, t0: 0, hl: 0, hlFrom: 0, hlTo: 0, hlT0: 0, hide: 0 };
  }
  if (cloud) {
    cloudU.uObj.value = objTex;
    try {
      const r = await fetch('/data/scan/cloud_labels.bin?v=' + meta.n);
      if (r.ok) { const b = await r.arrayBuffer(); if (b.byteLength >= meta.n * 2) { cloud.geometry.setAttribute('label', new THREE.BufferAttribute(new Uint16Array(b, 0, meta.n), 1)); SEG = true; } }
    } catch { /* no labels */ }
  }
  $('#steps button[data-s="understand"]').hidden = !SEG;
  $('#steps .sep-u').hidden = !SEG;
}

function meshFor(e) {
  if (!OBJ || e.provenance === 'proposed' || boxesMode) return null;
  const name = e.mesh ?? OBJ.byKey.get(e.object_key)?.mesh;
  return name ? MESHES.get(name) ?? null : null;
}
const segObject = (e) => SEG && !boxesMode && e.provenance !== 'proposed' && e.object_key && OBJ?.byKey.has(e.object_key);
const hasLabels = () => !!OBJ || [...groups.values()].some((r) => r.el.label);

function syncObjects(model, instant = false) {     // rigid transform origin -> current pose for every object's points
  if (!OBJ) return;
  const now = clock(), seen = new Set();
  for (const e of Object.values(model?.elements ?? {})) {
    const o = e.object_key && OBJ.byKey.get(e.object_key); if (!o) continue;
    const X = objXf[o.index]; seen.add(o.index);
    const og = e.origin ?? { pos: o.pos, yaw_deg: o.yaw_deg };
    const to = { x: e.pose.pos[0], y: e.pose.pos[1], yaw: e.pose.yaw - THREE.MathUtils.degToRad(og.yaw_deg ?? 0), ox: og.pos[0], oy: og.pos[1] };
    setXf(X, to, instant, now);
  }
  for (const o of OBJ.byKey.values()) if (!seen.has(o.index)) setXf(objXf[o.index], { x: o.pos[0], y: o.pos[1], yaw: 0, ox: o.pos[0], oy: o.pos[1] }, instant || !model, now);
}
function setXf(X, to, instant, now) {
  to.ox ??= X.o.pos[0]; to.oy ??= X.o.pos[1];
  const same = Math.abs(to.x - X.to.x) + Math.abs(to.y - X.to.y) + Math.abs(to.yaw - X.to.yaw) < 1e-4;
  if (same && !instant) return;
  X.from = instant ? to : { ...X.cur, ox: X.cur.ox ?? to.ox, oy: X.cur.oy ?? to.oy }; X.to = to; X.t0 = now;
  if (instant) X.cur = { ...to };
}
const MOVE_MS = 1700;
function updateObjects(now) {
  if (!objTex) return;
  const d = objTex.image.data;
  for (let i = 1; i < objXf.length; i++) {
    const X = objXf[i]; if (!X) continue;
    const k = ease((now - X.t0) / MOVE_MS), f = X.from, t = X.to;
    X.cur = { x: f.x + (t.x - f.x) * k, y: f.y + (t.y - f.y) * k, yaw: f.yaw + (t.yaw - f.yaw) * k, ox: t.ox, oy: t.oy };
    X.hl = X.hlFrom + (X.hlTo - X.hlFrom) * ease((now - X.hlT0) / 280);
    d.set([X.cur.ox, X.cur.oy, X.cur.x, X.cur.y], i * 8);
    d.set([X.cur.yaw, X.hl, X.hide, 0], i * 8 + 4);
  }
  objTex.needsUpdate = true;
}
function highlight(keys, now = clock()) {        // light up these objects' points (eased); others go back
  const want = new Set(keys);
  for (let i = 1; i < objXf.length; i++) {
    const X = objXf[i]; if (!X) continue;
    const to = want.has(X.o.key) ? 1 : 0;
    if (X.hlTo !== to) { X.hlFrom = X.hl; X.hlTo = to; X.hlT0 = now; }
  }
}
function objWorld(o) {      // current world placement of an object (follows moves)
  const X = objXf[o.index], c = X?.cur ?? { x: o.pos[0], y: o.pos[1], yaw: 0 };
  return { x: c.x, y: c.y, yaw: THREE.MathUtils.degToRad(o.yaw_deg ?? 0) + (c.yaw ?? 0) };
}

const _inv = new THREE.Matrix4(), _m = new THREE.Matrix4(), _r = new THREE.Ray(), _bx = new THREE.Box3(), _hit = new THREE.Vector3();
function pickObject(ndcX, ndcY) {     // ray vs each object's oriented box -> nearest key
  if (!OBJ) return null;
  ray.setFromCamera(new THREE.Vector2(ndcX, ndcY), camera);
  let best = null, bd = Infinity;
  for (const o of OBJ.byKey.values()) {
    const w = objWorld(o);
    _m.makeRotationZ(w.yaw).setPosition(w.x, w.y, 0); _inv.copy(_m).invert();
    _r.copy(ray.ray).applyMatrix4(_inv);
    _bx.min.set(-o.size[0] / 2, -o.size[1] / 2, 0); _bx.max.set(o.size[0] / 2, o.size[1] / 2, o.size[2]);
    if (_r.intersectBox(_bx, _hit)) { const d = _hit.applyMatrix4(_m).distanceTo(camera.position); if (d < bd) { bd = d; best = o.key; } }
  }
  return best;
}
function objectScreen(key) {
  const o = OBJ?.byKey.get(key); if (!o) return null;
  const w = objWorld(o); v3.set(w.x, w.y, o.size[2] * 0.55).project(camera);
  return { x: (v3.x + 1) / 2 * innerWidth, y: (1 - v3.y) / 2 * innerHeight, ok: v3.z < 1 && Math.abs(v3.x) < 0.95 && Math.abs(v3.y) < 0.95 };
}
const elementFor = (key) => [...groups.values()].find((r) => r.el.object_key === key)?.el;
const heroes = () => (OBJ ? [...OBJ.byKey.values()].filter((o) => o.hero && o.mesh && MESHES.has(o.mesh)) : []);
const rankObjects = () => (OBJ ? [...OBJ.byKey.values()].sort((a, b) => (b.hero ? 1 : 0) - (a.hero ? 1 : 0) || b.size[0] * b.size[1] * b.size[2] * (0.5 + (b.confidence ?? 0.5)) - a.size[0] * a.size[1] * a.size[2] * (0.5 + (a.confidence ?? 0.5))) : []);

// ---------------------------------------------------------------- labels: leader line + pill, choreographed per stage
const tags = new Map();         // key -> { d, on, tOn, tOff, stem, sx, sy, side }
let tagMode = null;             // { kind: 'seq'|'hover'|'follow'|'hero'|'parts', ... }
function resetTags() { tags.forEach((t) => t.d.remove()); tags.clear(); tagMode = null; }

function anchorsMap() {
  const out = new Map();
  if (OBJ) for (const o of OBJ.byKey.values()) {
    const w = objWorld(o);
    out.set(o.key, { key: o.key, label: o.label, fn: o.function, conf: o.confidence, cat: o.category, size: o.size, hero: !!o.hero,
      v: new THREE.Vector3(w.x, w.y, o.size[2]), prio: o.size[0] * o.size[1] * o.size[2] * (0.5 + (o.confidence ?? 0.5)) + (o.hero ? 1e3 : 0) });
  }
  if (world.visible) for (const r of groups.values()) {
    const e = r.el; if (!e.label || e.provenance === 'proposed') continue;
    const k = e.object_key ?? e.id, b = elementBox(e), c = b.getCenter(new THREE.Vector3()), s = b.getSize(new THREE.Vector3());
    const prev = out.get(k);
    out.set(k, { ...(prev ?? {}), key: k, label: e.label, fn: e.function ?? prev?.fn, conf: e.confidence ?? prev?.conf, size: prev?.size ?? [s.x, s.y, s.z],
      v: prev ? prev.v : new THREE.Vector3(c.x, c.y, b.max.z), prio: prev?.prio ?? s.x * s.y * s.z });
  }
  return out;
}
function tagEl(a) {
  let t = tags.get(a.key);
  if (!t) {
    const d = document.createElement('div'); d.className = 'tag';
    d.innerHTML = `<i></i><b></b><div class="tp"><span></span><small></small><em></em></div>`;
    $('#tags').appendChild(d); t = { d, on: false, tOn: -1e9, tOff: -1e9, stem: 56, sx: 0, sy: 0 }; tags.set(a.key, t);
  }
  return t;
}
function tagContent(t, a, hero, props) {
  const sig = a.label + '|' + (hero ? a.fn : '') + '|' + (props ?? '');
  if (t.sig === sig) return; t.sig = sig;
  t.d.querySelector('span').textContent = a.label;
  t.d.querySelector('small').textContent = hero && a.fn ? a.fn : '';
  t.d.querySelector('em').innerHTML = props ?? '';
  t.d.classList.toggle('hero', !!(hero && a.fn)); t.d.classList.toggle('props', !!props);
  t.w = null;
}
let followHero = null;
function updateTags(T) {
  const layer = $('#tags');
  if (!tagMode) { layer.style.display = 'none'; return; }
  layer.style.display = '';
  const A = anchorsMap(), W = innerWidth, H = innerHeight;
  const proj = (v) => { v3.copy(v).project(camera); return { x: (v3.x + 1) / 2 * W, y: (1 - v3.y) / 2 * H, ok: v3.z < 1 && Math.abs(v3.x) < 0.9 && Math.abs(v3.y) < 0.9 }; };
  let want = [];
  const m = tagMode;
  if (m.kind === 'seq') {
    if (!m.set && T >= m.t0) m.set = [...A.values()].filter((a) => proj(a.v).ok).sort((a, b) => b.prio - a.prio).slice(0, m.max).map((a) => a.key);
    const n = Math.floor((T - m.t0) / m.step) + 1;
    if (m.set && T < m.t1) want = m.set.slice(0, Math.max(0, n)).map((k, i) => ({ a: A.get(k), hero: i < m.heroes })).filter((w) => w.a);
  } else if (m.kind === 'hover' || m.kind === 'hero') {
    if (m.key && A.get(m.key) && T >= (m.t0 ?? -1e12)) want = [{ a: A.get(m.key), hero: true, props: m.kind === 'hero' ? m.props : null }];
  } else if (m.kind === 'parts') {
    want = (m.list ?? []).filter((p) => T >= p.t).map((p) => ({ a: p, hero: false }));
  } else if (m.kind === 'follow') {
    const eye = camera.position;
    const cand = [...A.values()].map((a) => ({ a, d: a.v.distanceTo(eye), p: proj(a.v) })).filter((c) => c.d < 8 && c.p.ok).sort((x, y) => x.d - y.d);
    const keep = cand.filter((c) => tags.get(c.a.key)?.on);
    const pick = [...keep];
    for (const c of cand) { if (pick.length >= 3) break; if (!pick.includes(c)) pick.push(c); }
    if (!pick.some((c) => c.a.key === followHero)) followHero = pick[0]?.a.key ?? null;
    want = pick.slice(0, 3).map((c) => ({ a: c.a, hero: c.a.key === followHero }));
  }
  const wantKeys = new Set(want.map((w) => w.a.key));
  for (const [k, t] of tags) if (t.on && !wantKeys.has(k)) { t.on = false; t.tOff = T; }
  const placed = [];
  const order = [...want.map((w) => w.a.key), ...[...tags.keys()].filter((k) => !wantKeys.has(k))];
  for (const k of order) {
    const w = want.find((x) => x.a.key === k), t = w ? tagEl(w.a) : tags.get(k);
    if (w) { if (!t.on) { t.on = true; t.tOn = T; } tagContent(t, w.a, w.hero, w.props); }
    const alpha = t.on ? clamp((T - t.tOn) / 380) : 1 - clamp((T - t.tOff) / 320);
    if (alpha <= 0.001) { t.d.style.opacity = 0; continue; }
    const a = w?.a ?? A.get(k);
    if (a) { const p = proj(a.v); if (p.ok || !t.placed) { t.sx = p.x; t.sy = p.y; } }
    const pill = t.d.querySelector('.tp');
    if (t.w == null) { t.w = pill.offsetWidth; t.h = pill.offsetHeight; }
    let ok = false;
    for (const stem of [t.stem, 56, 96, 136, 176]) {
      for (const side of [t.side ?? 1, -(t.side ?? 1)]) {
        const x0 = side > 0 ? t.sx - 14 : t.sx - t.w + 14, r = { x0: x0 - 6, x1: x0 + t.w + 6, y0: t.sy - stem - t.h - 6, y1: t.sy - stem + 6 };
        if (r.x0 < 12 || r.x1 > W - 12 || r.y0 < 70 || r.y1 > H - 90) continue;
        if (placed.some((q) => r.x0 < q.x1 && r.x1 > q.x0 && r.y0 < q.y1 && r.y1 > q.y0)) continue;
        placed.push(r); t.stem = stem; t.side = side; ok = true; break;
      }
      if (ok) break;
    }
    t.placed = true;
    if (!ok) { t.d.style.opacity = 0; continue; }
    t.d.style.opacity = alpha;
    t.d.style.transform = `translate(${t.sx}px, ${t.sy}px)`;
    t.d.querySelector('b').style.height = t.stem + 'px'; t.d.querySelector('b').style.top = -t.stem + 'px';
    pill.style.bottom = t.stem + 'px';
    pill.style.left = t.side > 0 ? '-14px' : (14 - t.w) + 'px';
    pill.style.transform = `translateY(${(1 - alpha) * 8}px) scale(${0.92 + 0.08 * alpha})`;
  }
}

// hover: points of the object under the cursor light up, the rest dims
let hoverKey = null, hoverAt = -1e9, pointerAt = -1e9;
renderer.domElement.addEventListener('pointermove', (e) => {
  if (!SEG || follow || !['bim', 'understand', 'design'].includes(stage)) return;
  pointerAt = clock();
  const k = pickObject((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
  renderer.domElement.style.cursor = k ? 'pointer' : '';
  setHover(k);
});
function setHover(k) {
  if (k === hoverKey) return;
  hoverKey = k; hoverAt = clock(); highlight(k ? [k] : []);
  if (!tagMode || tagMode.kind === 'hover') tagMode = k ? { kind: 'hover', key: k } : null;
}

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
  if (!follow) { follow = { t: CAPTURE ? 0 : video.currentTime, layer, labels: PARAMS.get('labels') === '1', tint: PARAMS.get('tint') === '1' }; if (!CAPTURE) { video.loop = true; video.playbackRate = 1; video.play().catch(() => {}); } }
  follow.layer = layer; document.body.dataset.follow = layer;
  camera.fov = followFov(); camera.near = 0.03; camera.updateProjectionMatrix(); updateScale();
  const hasCloud = layer === 'cloud' || layer.startsWith('cloud+'), which = layer.includes('bim') ? 'bim' : layer.includes('design') ? 'design' : null;
  anim = null; cutNow = Infinity; setWipe(null); clipCeiling.constant = 1e6; overlay.visible = false; $('#labels').style.display = 'none';
  heroMeshOn = true; highlight([]);
  if (cloud) {
    cloud.visible = hasCloud || (!!which && SEG); frustum.visible = pathLine.visible = false; cloud.geometry.setDrawRange(0, meta.n);
    cloudU.uBgOp.value = hasCloud || !which ? 1 : 0; cloudU.uBgClip.value = 1e6; cloudU.uDim.value = cloudU.uLift.value = 0; cloudU.uObjOp.value = 1; meshCut.constant = 1e6;   // model layers keep the objects' own points
    const u = cloudU; u.uReveal.value = meta.n; u.uFresh.value = 1; u.uOpModel.value = u.uOpScan.value = 1;
    u.uSize.value = (meta.voxel_m ?? 0.02) * 1.3; u.uMinPx.value = 1.5; u.uMaxPx.value = 13; u.uRound.value = 0;
  }
  world.visible = !!which;
  objXf.forEach((X) => X && (X.hide = 0));
  if (which) {
    base = baseline();
    const m = which === 'bim' ? base : S?.model;
    if (m) applyModel(m);
    groups.forEach((r) => { r.born = 0; r.group.position.copy(r.base); r.group.quaternion.copy(r.baseQ); updateElement(r, clock()); });
    groups.forEach((r) => { setFlag(r, null); r.near = null; });
    setModelOpacity(hasCloud ? 0.45 : 1);
    if (which === 'design') groups.forEach(addGhost); else dropGhosts();     // proposed work stays visible through scanned equipment
    for (const o of heroes()) if (objXf[o.index]) objXf[o.index].hide = elementFor(o.key) && meshFor(elementFor(o.key)) ? 1 : 0;   // the mesh stands in for its points

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
  if (cloudU) cloudU.uTint.value = follow.tint && cloud.visible ? 1 : 0;
  updateObjects(clock());
  if (follow.labels) { if (tagMode?.kind !== 'follow') tagMode = { kind: 'follow' }; } else tagMode = null;
  for (const r of groups.values()) {      // never render from inside a box: hide elements that contain the eye
    if (r.el.kind === 'slab' || r.el.kind === 'space') continue;
    r.box ??= elementBox(r.el);
    const k = clamp((r.box.distanceToPoint(camera.position) - 0.35) / 1.0);   // fade boxes the eye is in or right against
    r.group.visible = k > 0.02;
    if (r.near !== k) { r.near = k; nearFade(r, k); }
    if (r.hero && !heroMeshOn) r.group.visible = false;
  }
  renderer.render(scene, camera);
  updateTags(follow.t * 1000);
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
    if (status().done) { holdAt ||= now; if (now - holdAt > 900) { holdAt = 0; const nx = { scan: 'cloud', cloud: 'bim', bim: SEG ? 'understand' : 'design', understand: 'design' }[stage]; if (nx) setStage(nx); else autoplay = false; } }
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
  go: (s, opts) => { autoplay = false; setStage(s, opts ?? {}); return status(); },
  release: () => { held = false; busyUntil = clock() + 300; setHover(null); return status(); },
  objectScreen: (k) => objectScreen(k),
  hoverKey: () => hoverKey,
  replanTargets: () => {        // elements whose objects move in the queued history (for @chips)
    const now = new Map([...groups.values()].map((r) => [r.el.id, r.el]));
    const out = [];
    for (const ev of queue) for (const e of Object.values(ev.model.elements)) {
      const a = now.get(e.id); if (a?.object_key && !out.some((x) => x.id === e.id) && Math.hypot(e.pose.pos[0] - a.pose.pos[0], e.pose.pos[1] - a.pose.pos[1]) > 0.05) out.push({ id: e.id, key: a.object_key, label: a.label });
    }
    return out;
  },
  dry: false,
  partNames: () => queue.filter((h) => h.action === 'add_part').map((h) => (h.caption || '').split(' added to ')[0]).filter(Boolean),
  _orient() {   // test aid: world centroid of red / yellow vertices of each hero mesh vs its pose
    const out = [];
    for (const r of groups.values()) {
      const m = r.parts.find((p) => p.hero)?.bg.children.find((o) => o.isMesh); if (!m) continue;
      m.updateMatrixWorld(true); const c = m.geometry.attributes.color, P = m.geometry.attributes.position, acc = { red: [0, 0, 0, 0], yellow: [0, 0, 0, 0] };
      for (let i = 0; i < P.count; i++) {
        const R = c.getX(i), G = c.getY(i), B = c.getZ(i), k = R > 0.4 && G < 0.15 ? 'red' : R > 0.5 && G > 0.4 && B < 0.15 ? 'yellow' : null; if (!k) continue;
        const v = new THREE.Vector3(P.getX(i), P.getY(i), P.getZ(i)).applyMatrix4(m.matrixWorld); acc[k][0] += v.x; acc[k][1] += v.y; acc[k][2] += v.z; acc[k][3]++;
      }
      const f = (a) => a[3] ? a.slice(0, 3).map((x) => +(x / a[3]).toFixed(2)) : null;
      out.push({ key: r.el.object_key, pos: r.el.pose.pos, yaw_deg: +(r.el.pose.yaw * 180 / Math.PI).toFixed(1), size: r.el.size, red_column: f(acc.red), yellow_knob: f(acc.yellow) });
    }
    return out;
  },
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
  async follow(spec, t) {    // first-person render of one layer at video time t (s): 'cloud', 'bim:labels', 'cloud:tint', ...
    const [layer, ...o] = spec.split(':');
    if (!follow || follow.layer !== layer) enterFollow(layer);
    follow.t = t; follow.labels = o.includes('labels'); follow.tint = o.includes('tint');
    if (layer === 'video') await seekVideo(Math.min(vdur() - 0.02, t));
    followTick();
    return { layer, t, frame: t / frameDt(), fov: camera.fov };
  },
  followInfo: () => ({ frameDt: frameDt(), fov: camera.fov, frames: meta?.frames, duration: vdur() }),
};

if (!CAPTURE) addEventListener('error', () => {});
await loadCloud();
await loadObjects();
await new Promise((r) => (video.readyState >= 1 ? r() : (video.addEventListener('loadedmetadata', r, { once: true }), video.addEventListener('error', r, { once: true }), setTimeout(r, 4000))));
await poll();
const start = location.hash.slice(1);
setStage(['scan', 'cloud', 'bim', 'understand', 'hero', 'replan', 'modeledit', 'design'].includes(start) ? start : 'scan');
if (PARAMS.get('follow') && meta) enterFollow(PARAMS.get('follow'));
if (CAPTURE) { video.pause(); window.__demo.ready = true; }
else {
  setInterval(poll, 800);
  const loop = (now) => { requestAnimationFrame(loop); tick(now); };
  requestAnimationFrame(loop);
}
