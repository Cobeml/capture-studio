"""Hero 3D model of one scanned machine: generated shape + real texture from the walkthrough video.

    python -m capture_studio.hero obj_05 --frame 181          # (after capture_studio.objects)

1. shape    Hunyuan3D-2 (tencent/Hunyuan3D-2, hunyuan3d-dit-v2-0) generates a complete, clean mesh from
            the best SAM-2.1-masked crop of the object (image -> 3D). Licence: Tencent Hunyuan 3D 2.0
            Community License (not licensed in the EU, UK or South Korea) -- recorded in objects.json.
2. register the mesh is scaled / rotated (about Z only) / translated onto the object's scanned points
            (yaw search + similarity ICP), so it sits exactly where the points are, in the frozen frame.
3. texture  xatlas UV atlas; every texel is coloured from the real video frames that see it (refined
            LingBot-Map poses, ray-cast self-occlusion, inside the object's SAM mask, frontal + close views
            preferred); texels no frame saw take the colour of the nearest seen surface.
4. export   objects.glb (one node per hero, object-local frame, glTF Y-up) + preview renders.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from . import fit, objects, store

ROOT = store.ROOT
HY3D_REPO = ROOT / "third_party" / "hunyuan3d-2"
HY3D_MODELS = ROOT / "models" / "hy3dgen"
MESH_SOURCE = "image-to-3d:hunyuan3d-2 + video texture"
LICENSE_NOTE = ("Shape generated with Tencent Hunyuan3D-2 (Tencent Hunyuan 3D 2.0 Community License; "
                "territory excludes EU, UK, South Korea). Texture: projected from the walkthrough video.")


# ---------------------------------------------------------------- 1. shape

def masked_crop(frame_bgr, mask, pad=1.15) -> np.ndarray:
    """RGBA square crop of the object (background transparent)."""
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    m = lab == 1 + int(np.argmax(st[1:, 4]))
    ys, xs = np.nonzero(m)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    rgba = np.dstack([frame_bgr[:, :, ::-1], (m * 255).astype(np.uint8)])[y0:y1 + 1, x0:x1 + 1]
    S = int(max(rgba.shape[:2]) * pad)
    out = np.zeros((S, S, 4), np.uint8)
    oy, ox = (S - rgba.shape[0]) // 2, (S - rgba.shape[1]) // 2
    out[oy:oy + rgba.shape[0], ox:ox + rgba.shape[1]] = rgba
    return out


def sam_mask_hr(frame_bgr, box) -> np.ndarray:
    import torch
    sys.path.insert(0, str(ROOT / "third_party" / "sam2"))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    pred = SAM2ImagePredictor(build_sam2(objects.SAM2_CFG, str(objects.SAM2_CKPT), device="cuda"))
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred.set_image(np.ascontiguousarray(frame_bgr[:, :, ::-1]))
        m, _, _ = pred.predict(box=np.array(box, np.float32), multimask_output=False)
    del pred
    torch.cuda.empty_cache()
    return np.asarray(m)[0] > 0.5


def generate_shape(rgba: np.ndarray, seed=1234, steps=50, octree=384, max_faces=120_000):
    import os
    import torch
    from PIL import Image
    os.environ.setdefault("HY3DGEN_MODELS", str(HY3D_MODELS))
    sys.path.insert(0, str(HY3D_REPO))
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FloaterRemover, DegenerateFaceRemover, FaceReducer
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2", subfolder="hunyuan3d-dit-v2-0", use_safetensors=True, variant="fp16")
    mesh = pipe(image=Image.fromarray(rgba, "RGBA"), num_inference_steps=steps, octree_resolution=octree,
                generator=torch.manual_seed(seed), output_type="trimesh")[0]
    del pipe
    torch.cuda.empty_cache()
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh, max_facenum=max_faces)
    return mesh


# ---------------------------------------------------------------- 2. register onto the scan

def _rz(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def register(V_yup: np.ndarray, F: np.ndarray, Q: np.ndarray, yaw0: float | None = None):
    """Similarity (s, yaw, t) placing the generated mesh (Y-up, arbitrary scale) onto scan points Q (Z-up).
    Returns world-space vertices and the fit residual (scan -> mesh distances)."""
    V = np.stack([V_yup[:, 0], -V_yup[:, 2], V_yup[:, 1]], 1)        # Y-up -> Z-up
    V = V - [0, 0, V[:, 2].min()]
    hq = float(np.percentile(Q[:, 2], 99.5))
    s0 = hq / float(V[:, 2].max())
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V * s0), o3d.utility.Vector3iVector(F))
    qc = np.r_[np.median(Q[:, :2], 0), 0.0]
    qpc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Q[:: max(1, len(Q) // 30000)].astype(np.float64)))

    def score(Vw):
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(
            o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(Vw), o3d.utility.Vector3iVector(F))))
        d = scene.compute_distance(o3d.core.Tensor(np.asarray(qpc.points), dtype=o3d.core.float32)).numpy()
        return float(np.mean(np.minimum(d, 0.3))), d

    best = None
    Vs = V * s0
    vc = np.r_[(Vs[:, :2].max(0) + Vs[:, :2].min(0)) / 2, 0.0]
    yaws = range(0, 360, 5) if yaw0 is None else [yaw0 + d for d in range(-40, 41, 4)]
    for yaw in yaws:
        for k in (0.85, 0.95, 1.05):
            Vw = ((Vs - vc) * k) @ _rz(yaw).T + qc
            sc, _ = score(Vw)
            if best is None or sc < best[0]:
                best = (sc, yaw, k)
    _, yaw, k = best
    # refine with similarity ICP: scan points (source) onto the mesh surface (target) = the one-sided
    # scan->mesh objective (the scan only saw some faces); rotation is then projected to yaw only
    Vw = ((Vs - vc) * k) @ _rz(yaw).T + qc
    reg = o3d.pipelines.registration
    tgt = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(Vw), o3d.utility.Vector3iVector(F)) \
        .sample_points_uniformly(200_000)
    T = np.eye(4)
    for dmax in (0.25, 0.10, 0.05):
        r = reg.registration_icp(qpc, tgt, dmax, T, reg.TransformationEstimationPointToPoint(with_scaling=False),
                                 reg.ICPConvergenceCriteria(max_iteration=60))
        T = r.transformation
        sc3 = np.cbrt(np.linalg.det(T[:3, :3]))
        ang = math.degrees(math.atan2(T[1, 0], T[0, 0]))
        T = np.eye(4)
        T[:3, :3] = sc3 * _rz(ang)
        T[:3, 3] = r.transformation[:3, 3]
        T[2, 3] = 0.0                                   # both stand on the floor
    Ti = np.linalg.inv(T)                               # mesh follows the inverse of the scan->mesh motion
    Vw = Vw @ Ti[:3, :3].T + Ti[:3, 3]
    Vw[:, 2] -= Vw[:, 2].min()
    yaw = yaw - math.degrees(math.atan2(T[1, 0], T[0, 0]))
    k = k / np.cbrt(np.linalg.det(T[:3, :3]))
    sc, d = score(Vw)
    return Vw, {"yaw_deg": float(yaw), "scale": float(s0 * k), "mean_scan_to_mesh_m": sc,
                "p50_mm": float(np.median(d) * 1000), "p90_mm": float(np.percentile(d, 90) * 1000)}


# ---------------------------------------------------------------- 3. texture from the video

def frame_cameras(z, frame, Ts, Wd, Hd):
    """Per frame: (K_hr 3x3, R 3x3, t 3) mapping MODEL coordinates to the hi-res video image."""
    E, K = z["extrinsic"], z["intrinsic"]
    S, H, W = z["points"].shape[:3]
    Rf, sf, tf = np.asarray(frame["R"]), float(frame["s"]), np.asarray(frame["t"])
    out = []
    for i in range(S):
        R, t = E[i, :3, :3], E[i, :3, 3]
        if Ts is not None:
            Ri = Ts[i, :3, :3].T
            R, t = R @ Ri, R @ (-Ri @ Ts[i, :3, 3]) + t
        # model -> raw: x_raw = Rf^T (X - tf) / sf ; raw -> cam: R x_raw + t  (scale-free projection)
        Rm = R @ Rf.T / sf
        tm = t - Rm @ tf
        Kh = K[i].copy()
        Kh[0] *= Wd / W
        Kh[1] *= Hd / H
        out.append((Kh, Rm, tm))
    return out


def uv_atlas(V, F, size=2048):
    import xatlas
    vmap, Fi, UV = xatlas.parametrize(V.astype(np.float32), F.astype(np.uint32))
    return vmap, Fi.astype(np.int64), UV.astype(np.float32)


def texel_positions(V, N, F, UV, size):
    """Rasterise the atlas: per covered texel its 3D position, normal and a validity mask."""
    pos = np.zeros((size, size, 3), np.float32)
    nor = np.zeros((size, size, 3), np.float32)
    cov = np.zeros((size, size), bool)
    P = UV * (size - 1)
    for f in range(len(F)):
        a, b, c = F[f]
        tri = P[[a, b, c]]
        x0, y0 = np.floor(tri.min(0)).astype(int)
        x1, y1 = np.ceil(tri.max(0)).astype(int)
        if x1 < x0 or y1 < y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        p = np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], 1)
        v0, v1 = tri[1] - tri[0], tri[2] - tri[0]
        den = v0[0] * v1[1] - v1[0] * v0[1]
        if abs(den) < 1e-12:
            continue
        w = p - tri[0]
        l1 = (w[:, 0] * v1[1] - v1[0] * w[:, 1]) / den
        l2 = (v0[0] * w[:, 1] - w[:, 0] * v0[1]) / den
        l0 = 1 - l1 - l2
        eps = 0.6 / max(np.abs(den) ** 0.5, 1e-6)       # include border texels (dilated by ~half a texel)
        ok = (l0 >= -eps) & (l1 >= -eps) & (l2 >= -eps)
        if not ok.any():
            continue
        X = l0[ok, None] * V[a] + l1[ok, None] * V[b] + l2[ok, None] * V[c]
        Nn = l0[ok, None] * N[a] + l1[ok, None] * N[b] + l2[ok, None] * N[c]
        yy, xx = np.clip(ys.ravel()[ok], 0, size - 1), np.clip(xs.ravel()[ok], 0, size - 1)
        pos[yy, xx], nor[yy, xx], cov[yy, xx] = X, Nn, True
    nor /= np.maximum(np.linalg.norm(nor, axis=2, keepdims=True), 1e-9)
    return pos, nor, cov


def _bilinear(img, u, v):
    x0, y0 = np.floor(u).astype(int), np.floor(v).astype(int)
    fx, fy = (u - x0)[:, None], (v - y0)[:, None]
    x0, y0 = np.clip(x0, 0, img.shape[1] - 2), np.clip(y0, 0, img.shape[0] - 2)
    a, b = img[y0, x0].astype(np.float64), img[y0, x0 + 1].astype(np.float64)
    c, d = img[y0 + 1, x0].astype(np.float64), img[y0 + 1, x0 + 1].astype(np.float64)
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def bake_texture(V, F, UV, frames, cams, view_ids, masks_hr, size=2048):
    """Colour every texel from the frames that see it; unseen texels from the nearest seen surface."""
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))
    mesh.compute_vertex_normals()
    N = np.asarray(mesh.vertex_normals)
    pos, nor, cov = texel_positions(V, N, F, UV, size)
    idx = np.flatnonzero(cov.ravel())
    X, Nn = pos.reshape(-1, 3)[idx], nor.reshape(-1, 3)[idx]
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    Hd, Wd = frames.shape[1:3]
    acc = np.zeros((len(idx), 3))
    wsum = np.zeros(len(idx))
    best_w = np.zeros(len(idx))
    best_c = np.zeros((len(idx), 3))
    for f in view_ids:
        K, R, t = cams[f]
        C = -np.linalg.solve(R, t)
        xc = X @ R.T + t
        zc = xc[:, 2]
        u = K[0, 0] * xc[:, 0] / zc + K[0, 2]
        v = K[1, 1] * xc[:, 1] / zc + K[1, 2]
        ok = (zc > 0.2) & (u >= 1) & (u < Wd - 2) & (v >= 1) & (v < Hd - 2)
        d = C - X
        dist = np.linalg.norm(d, axis=1)
        d /= dist[:, None]
        cosv = np.einsum("ij,ij->i", Nn, d)
        ok &= cosv > 0.15
        if f in masks_hr:
            m = masks_hr[f]
            ok &= m[np.clip(v.astype(int), 0, Hd - 1), np.clip(u.astype(int), 0, Wd - 1)]
        sel = np.flatnonzero(ok)
        if not len(sel):
            continue
        rays = np.concatenate([X[sel] + d[sel] * 0.01, d[sel]], 1).astype(np.float32)
        hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        vis = ~np.isfinite(hit) | (hit > dist[sel] - 0.02)
        sel = sel[vis]
        col = _bilinear(np.asarray(frames[f]), u[sel], v[sel])
        w = cosv[sel] ** 3 / np.maximum(dist[sel], 0.5) ** 2
        acc[sel] += col * w[:, None]
        wsum[sel] += w
        better = w > best_w[sel]
        best_w[sel[better]] = w[better]
        best_c[sel[better]] = col[better]
    seen = wsum > 0
    # crisp but seam-free: lean on the best view, blend in the weighted mean
    colour = np.where(seen[:, None], 0.65 * best_c + 0.35 * acc / np.maximum(wsum, 1e-9)[:, None], 0)
    if (~seen).any() and seen.any():
        # faces no frame saw get the machine's own dominant paint (the body above the plinth, the plinth
        # below), feathered into the nearest seen colour over a few cm: no streaks, no borrowed neighbours
        from scipy.spatial import cKDTree
        zmin = float(V[:, 2].min())
        low = X[:, 2] < zmin + 0.30
        lum = colour.mean(1)

        def paint(sel_):
            c = colour[seen & sel_]
            if len(c) < 50:
                c = colour[seen]
            l = c.mean(1)
            return np.median(c[l >= np.percentile(l, 40)], axis=0)
        body, plinth = paint(~low), paint(low)
        base = np.where(low[:, None], plinth, body)
        tree = cKDTree(X[seen])
        dd, j = tree.query(X[~seen], k=8)
        near = np.median(colour[seen][j], axis=1)
        w = np.clip(1 - dd.mean(1) / 0.06, 0, 1)[:, None]
        colour[~seen] = w * near + (1 - w) * base[~seen]
    tex = np.zeros((size * size, 3), np.uint8)
    tex[idx] = np.clip(colour, 0, 255).astype(np.uint8)
    tex = tex.reshape(size, size, 3)
    # pad islands so mip-mapping does not pull in black
    holes = (~cov).astype(np.uint8)
    tex = cv2.inpaint(tex, holes, 3, cv2.INPAINT_TELEA) if holes.any() else tex
    return tex, float(seen.mean())


# ---------------------------------------------------------------- 4. export + preview

def to_local(Vw, pos, yaw_deg):
    return (Vw - [pos[0], pos[1], 0.0]) @ _rz(yaw_deg)       # row-vector form of Rz(-yaw) @ (X - p)


def subdivide_uv(V, F, UV):
    """One midpoint subdivision (x4 faces), interpolating positions and UVs (seams are already split)."""
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    uniq, inv = np.unique(E, axis=0, return_inverse=True)
    inv = inv.ravel()
    n = len(V)
    Vn = np.concatenate([V, (V[uniq[:, 0]] + V[uniq[:, 1]]) / 2])
    UVn = np.concatenate([UV, (UV[uniq[:, 0]] + UV[uniq[:, 1]]) / 2])
    m = len(F)
    a, b, c = n + inv[:m], n + inv[m:2 * m], n + inv[2 * m:]
    Fn = np.concatenate([np.stack([F[:, 0], a, c], 1), np.stack([a, F[:, 1], b], 1),
                         np.stack([c, b, F[:, 2]], 1), np.stack([a, b, c], 1)])
    return Vn, Fn, UVn


def export_glb(heroes: list[dict], path: Path, vertex_colors=True):
    """heroes: [{key, V_local (Z-up), F, UV, tex (BGR)}] -> one glTF binary, one node+mesh per hero, stored
    glTF Y-up (model (x, y, z) -> glTF (x, z, -y)). Each primitive carries the embedded baseColor texture
    (JPEG) AND, for viewers that only use vertex colours, the texture baked into COLOR_0 (sRGB bytes) on a
    once-subdivided mesh so the colours keep the detail."""
    import pygltflib as g
    blob = bytearray()
    views, accs, meshes, nodes, mats, texs, imgs = [], [], [], [], [], [], []

    def add_view(data: bytes, target=None):
        while len(blob) % 4:
            blob.append(0)
        views.append(g.BufferView(buffer=0, byteOffset=len(blob), byteLength=len(data), target=target))
        blob.extend(data)
        return len(views) - 1

    def add_acc(arr, ctype, atype, target, normalized=False, minmax=False):
        v = add_view(arr.tobytes(), target)
        kw = {"min": arr.min(0).tolist(), "max": arr.max(0).tolist()} if minmax else {}
        accs.append(g.Accessor(bufferView=v, componentType=ctype, count=len(arr), type=atype, normalized=normalized, **kw))
        return len(accs) - 1

    for h in heroes:
        V, F, UV = h["V_local"], h["F"], h["UV"]
        if vertex_colors:
            V, F, UV = subdivide_uv(V, F, UV)
        Vg = np.stack([V[:, 0], V[:, 2], -V[:, 1]], 1).astype(np.float32)
        mo = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(Vg.astype(np.float64)), o3d.utility.Vector3iVector(F))
        mo.compute_vertex_normals()
        N = np.asarray(mo.vertex_normals).astype(np.float32)
        S = h["tex"].shape[0]
        attrs = {"POSITION": add_acc(Vg, g.FLOAT, g.VEC3, g.ARRAY_BUFFER, minmax=True),
                 "NORMAL": add_acc(N, g.FLOAT, g.VEC3, g.ARRAY_BUFFER),
                 "TEXCOORD_0": add_acc(UV.astype(np.float32), g.FLOAT, g.VEC2, g.ARRAY_BUFFER)}
        if vertex_colors:
            rgb = _bilinear(h["tex"][:, :, ::-1], UV[:, 0] * (S - 1), UV[:, 1] * (S - 1))
            col = np.concatenate([np.clip(np.round(rgb), 0, 255), np.full((len(rgb), 1), 255)], 1).astype(np.uint8)
            attrs["COLOR_0"] = add_acc(col, g.UNSIGNED_BYTE, g.VEC4, g.ARRAY_BUFFER, normalized=True)
        ind = add_acc(F.astype(np.uint32).ravel(), g.UNSIGNED_INT, g.SCALAR, g.ELEMENT_ARRAY_BUFFER)
        jpg = cv2.imencode(".jpg", h["tex"], [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
        imgs.append(g.Image(bufferView=add_view(jpg), mimeType="image/jpeg", name=h["key"]))
        texs.append(g.Texture(source=len(imgs) - 1, sampler=0))
        mats.append(g.Material(name=h["key"], pbrMetallicRoughness=g.PbrMetallicRoughness(
            baseColorTexture=g.TextureInfo(index=len(texs) - 1), metallicFactor=0.1, roughnessFactor=0.7)))
        meshes.append(g.Mesh(name=h["key"], primitives=[g.Primitive(attributes=g.Attributes(**attrs), indices=ind,
                                                                     material=len(mats) - 1)]))
        nodes.append(g.Node(name=h["key"], mesh=len(meshes) - 1))
    gl = g.GLTF2(asset=g.Asset(version="2.0", generator="capture_studio.hero"), scene=0,
                 scenes=[g.Scene(nodes=list(range(len(nodes))))], nodes=nodes, meshes=meshes, materials=mats,
                 textures=texs, images=imgs, samplers=[g.Sampler(magFilter=9729, minFilter=9987, wrapS=33071, wrapT=33071)],
                 accessors=accs, bufferViews=views, buffers=[g.Buffer(byteLength=len(blob))])
    gl.set_binary_blob(bytes(blob))
    gl.save_binary(str(path))


def render_mesh(V, F, UV, tex, eye, target, W=900, H=700, fov=40, bg=(32, 32, 36)):
    """Textured preview render from a look-at camera (Z up)."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    R = np.stack([r, -u, f])
    fy = H / 2 / math.tan(math.radians(fov) / 2)
    K = np.array([[fy, 0, W / 2], [0, fy, H / 2], [0, 0, 1.0]])
    bgimg = np.empty((H, W, 3))
    bgimg[:] = bg
    img, _ = render_from_camera(V, F, UV, tex, K, R, -R @ eye, W, H, bg=bgimg)
    return img


# ---------------------------------------------------------------- driver

def build(key: str, frame_idx: int, seed=1234, shape_file: str | None = None) -> dict:
    t0 = time.time()
    sd = objects.scan_dir()
    doc = json.loads((sd / "objects.json").read_text())
    obj = next(o for o in doc["objects"] if o["key"] == key)
    meta, P, col = objects.read_cloud()
    lab = np.fromfile(sd / "cloud_labels.bin", dtype=np.uint16)
    Q = P[lab == obj["index"]].astype(np.float64)
    z = np.load(sd / "raw_predictions.npz")
    frames = objects.video_frames(n=len(z["extrinsic"]))
    dets = json.loads((sd / objects.WORK_NAME / "detections.json").read_text())
    work = sd / objects.WORK_NAME / f"hero_{key}"
    work.mkdir(parents=True, exist_ok=True)

    # which detections belong to this object: those whose lifted pixels land on its points
    Pm, keep, nc = fit.model_pixels(z)
    from scipy.spatial import cKDTree
    tree = cKDTree(Q)
    S, H, W = z["points"].shape[:3]
    mk = np.load(sd / objects.WORK_NAME / "masks.npz")
    views = []
    for f in range(S):
        if f"f{f}" not in mk.files:
            continue
        ms = np.unpackbits(mk[f"f{f}"], axis=-1)[..., :W].astype(bool)
        for j, m in enumerate(ms):
            q = Pm[f][m & keep[f]]
            if len(q) < 100:
                continue
            d, _ = tree.query(q[:: max(1, len(q) // 400)], distance_upper_bound=0.05)
            frac = float(np.isfinite(d).mean())
            if frac > 0.5:
                views.append((f, j, frac, int(m.sum())))
    del Pm, keep, nc
    view_ids = sorted({v[0] for v in views})
    print(f"  {key}: {len(Q)} scan points, {len(view_ids)} frames see it")

    # 1. shape from the chosen frame's mask
    cand = sorted([v for v in views if v[0] == frame_idx], key=lambda v: -v[3])
    box = dets[str(frame_idx)][cand[0][1]]["box"] if cand else None
    if box is None:
        raise SystemExit(f"frame {frame_idx} has no detection of {key}; frames: {view_ids}")
    shape_path = work / "shape.ply"
    if shape_file:
        import shutil
        shutil.copy(shape_file, shape_path)
    if not shape_path.exists():
        mhr = sam_mask_hr(frames[frame_idx], box)
        rgba = masked_crop(frames[frame_idx], mhr)
        cv2.imwrite(str(work / "input.png"), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
        mesh = generate_shape(rgba, seed=seed)
        mesh.export(shape_path)
    import trimesh
    tm = trimesh.load(shape_path, process=False)
    V0, F = np.asarray(tm.vertices, np.float64), np.asarray(tm.faces, np.int64)
    t_shape = time.time() - t0

    # 2. register: yaw seeded from the input camera (the generator's front faces it), then yaw / scale / x / y
    #    optimised so the mesh silhouette matches the object's SAM masks in up to 10 real frames
    frame = fit.load_frame()
    Ts = fit.refine_frames(z)
    camsL = frame_cameras(z, frame, Ts, W, H)
    camsH = frame_cameras(z, frame, Ts, frames.shape[2], frames.shape[1])
    Kc, Rc, tc = camsH[frame_idx]
    cam = -np.linalg.solve(Rc, tc)
    d = cam[:2] - np.median(Q[:, :2], 0)
    yaw0 = math.degrees(math.atan2(d[1], d[0])) + 90.0
    by_area, seen_f = [], set()
    for f, j, frac, area in sorted(views, key=lambda v: -v[3]):
        if f not in seen_f:
            by_area.append((f, j))
            seen_f.add(f)
    lowmask = {f: np.unpackbits(mk[f"f{f}"], axis=-1)[..., :W].astype(bool)[j] for f, j in by_area}
    sil = sorted(by_area[:10])
    Vw, reg = fit_silhouettes(V0, F, [(*camsL[f], lowmask[f]) for f, _ in sil], yaw0, Q)
    print("  registration:", {k: round(v, 3) for k, v in reg.items()})

    # 3. texture from the real frames (masks: the object's SAM masks, upsampled and dilated)
    amax = max(a for (_, _, _, a) in views)
    area_of = {(f, j): a for (f, j, _, a) in views}
    sel = sorted([v for v in by_area if area_of[v] >= 0.3 * amax][:24])
    masks_hr = {f: cv2.dilate(cv2.resize(lowmask[f].astype(np.uint8), (frames.shape[2], frames.shape[1]),
                                         interpolation=cv2.INTER_NEAREST), np.ones((15, 15), np.uint8)).astype(bool)
                for f, _ in sel}
    cams = camsH
    if len(F) > 80_000:                              # lighter atlas / GLB
        mo = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(Vw), o3d.utility.Vector3iVector(F))
        mo = mo.simplify_quadric_decimation(80_000)
        Vw, F = np.asarray(mo.vertices), np.asarray(mo.triangles).astype(np.int64)
    vmap, Fi, UV = uv_atlas(Vw, F)
    Vt = Vw[vmap]
    tex, seen_frac = bake_texture(Vt, Fi, UV, frames, cams, [f for f, _ in sel], masks_hr)
    cv2.imwrite(str(work / "texture.png"), tex)
    t_tex = time.time() - t0 - t_shape

    # object pose for the GLB / objects.json: keep the scan OBB pose, refit size to the model
    pos, yaw = obj["pos"], obj["yaw_deg"]
    Vl = to_local(Vt, pos, yaw)
    ext = Vl.max(0) - Vl.min(0)
    hero = {"key": key, "V_local": Vl, "F": Fi, "UV": UV, "tex": tex, "Vw": Vt,
            "info": {"mesh_source": MESH_SOURCE, "mesh_license": LICENSE_NOTE, "hero_input_frame": frame_idx,
                     "silhouette_frames": [int(f) for f, _ in sil],
                     "texture_frames": [int(f) for f, _ in sel], "texture_seen_fraction": round(seen_frac, 3),
                     "registration": {k: round(v, 4) for k, v in reg.items()},
                     "mesh_faces": int(len(Fi)), "model_size_m": [round(float(v), 3) for v in ext]},
            "seconds": {"shape": round(t_shape, 1), "texture": round(t_tex, 1)}}
    np.savez_compressed(work / "hero.npz", V_local=Vl, F=Fi, UV=UV, Vw=Vt)
    (work / "hero.json").write_text(json.dumps(hero["info"], indent=1))
    return hero


def write_outputs(keys: list[str]):
    """Gather built heroes -> objects.glb, objects.json (mesh/hero/mesh_source), preview renders."""
    sd = objects.scan_dir()
    doc = json.loads((sd / "objects.json").read_text())
    heroes = []
    for k in keys:
        w = sd / objects.WORK_NAME / f"hero_{k}"
        z = np.load(w / "hero.npz")
        heroes.append({"key": k, "V_local": z["V_local"], "F": z["F"], "UV": z["UV"], "Vw": z["Vw"],
                       "tex": cv2.imread(str(w / "texture.png")), "info": json.loads((w / "hero.json").read_text())})
    export_glb(heroes, sd / "objects.glb")
    for o in doc["objects"]:
        h = next((h for h in heroes if h["key"] == o["key"]), None)
        o["mesh"], o["hero"] = (o["key"], True) if h else (None, False)
        for f in ("mesh_source", "mesh_license", "hero_input_frame", "texture_frames", "registration", "model_size_m",
                  "vertex_colors", "silhouette_frames"):
            o.pop(f, None)
        if h:
            o.update({f: h["info"][f] for f in ("mesh_source", "mesh_license", "hero_input_frame", "texture_frames",
                                                 "registration", "model_size_m") if f in h["info"]})
            o["vertex_colors"] = "srgb"
            o["mesh_source"] = MESH_SOURCE
    doc["glb"] = "objects.glb: hero objects only; one node+mesh per hero named by its key; object-local frame " \
                 "(origin at pos on the floor, +x along length at yaw_deg), glTF Y-up (model = (x, -z, y))"
    (sd / "objects.json").write_text(json.dumps(doc, indent=1))
    pv = sd / "objects_preview"
    pv.mkdir(exist_ok=True)
    for h in heroes:
        V = h["V_local"]
        c = (V.max(0) + V.min(0)) / 2
        r = float(np.linalg.norm(V.max(0) - V.min(0)))
        tiles = []
        for az in (-35, 35, 145, 215):
            e = c + r * 1.2 * np.array([math.cos(math.radians(az)), math.sin(math.radians(az)), 0.45])
            tiles.append(render_mesh(V, h["F"], h["UV"], h["tex"], e, c, W=640, H=480))
        cv2.imwrite(str(pv / f"{h['key']}_mesh.jpg"), np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("keys", nargs="+", help="object keys, e.g. obj_05")
    ap.add_argument("--frame", type=int, action="append", help="input frame per key (same order)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--shape", action="append", help="use this generated mesh (.ply) instead of generating")
    ap.add_argument("--export-only", action="store_true", help="only rebuild objects.glb / objects.json")
    a = ap.parse_args()
    if not a.export_only:
        for n, (k, f) in enumerate(zip(a.keys, a.frame or [])):
            h = build(k, f, a.seed, (a.shape or [None] * len(a.keys))[n])
            print(k, h["seconds"], h["info"]["registration"], "texture seen", h["info"]["texture_seen_fraction"])
    write_outputs(a.keys)



def render_from_camera(V, F, UV, tex, K, R, t, W, H, bg=None):
    """Render the mesh (model frame) through a real video camera (K, R, t: model -> image)."""
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))
    mesh.compute_triangle_normals()
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    vv, uu = np.mgrid[0:H, 0:W]
    d_cam = np.stack([(uu + 0.5 - K[0, 2]) / K[0, 0], (vv + 0.5 - K[1, 2]) / K[1, 1], np.ones_like(uu, float)], -1)
    Rn = R / np.linalg.norm(R[0])
    d = d_cam.reshape(-1, 3) @ Rn                                # cam -> model directions
    C = -np.linalg.solve(R, t)
    rays = np.concatenate([np.broadcast_to(C, d.shape), d / np.linalg.norm(d, axis=1, keepdims=True)], 1)
    ans = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))
    hit = np.isfinite(ans["t_hit"].numpy())
    img = np.zeros((H * W, 3)) if bg is None else bg.reshape(-1, 3).astype(np.float64).copy()
    pid, bc = ans["primitive_ids"].numpy()[hit], ans["primitive_uvs"].numpy()[hit]
    f = F[pid]
    uv = (1 - bc[:, 0] - bc[:, 1])[:, None] * UV[f[:, 0]] + bc[:, :1] * UV[f[:, 1]] + bc[:, 1:] * UV[f[:, 2]]
    S = tex.shape[0]
    c = tex[np.clip((uv[:, 1] * (S - 1)).astype(int), 0, S - 1), np.clip((uv[:, 0] * (S - 1)).astype(int), 0, S - 1)]
    n = np.asarray(mesh.triangle_normals)[pid]
    shade = 0.6 + 0.4 * np.abs(np.einsum("ij,ij->i", n, rays[hit, 3:]))
    img[hit] = c * shade[:, None]
    return np.clip(img, 0, 255).astype(np.uint8).reshape(H, W, 3), hit.reshape(H, W)


def _silhouette(scene, K, R, t, W, H, step=2):
    vv, uu = np.mgrid[0:H:step, 0:W:step]
    d_cam = np.stack([(uu + 0.5 - K[0, 2]) / K[0, 0], (vv + 0.5 - K[1, 2]) / K[1, 1], np.ones_like(uu, float)], -1)
    Rn = R / np.linalg.norm(R[0])
    d = d_cam.reshape(-1, 3) @ Rn
    C = -np.linalg.solve(R, t)
    rays = np.concatenate([np.broadcast_to(C, d.shape), d / np.linalg.norm(d, axis=1, keepdims=True)], 1)
    hit = np.isfinite(scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))["t_hit"].numpy())
    return hit.reshape(vv.shape)


def fit_silhouettes(V_yup, F, views, yaw0, Q, step=2):
    """Pose the generated mesh so its silhouette matches the object's SAM masks in several real frames.
    views: [(K, R, t, mask (H, W) bool)] at the point-map resolution. Optimises yaw, scale, x, y (z: floor).
    Returns world vertices and the mean silhouette IoU."""
    from scipy.optimize import minimize
    V = np.stack([V_yup[:, 0], -V_yup[:, 2], V_yup[:, 1]], 1)
    V = V - [0, 0, V[:, 2].min()]
    V[:, :2] -= (V[:, :2].max(0) + V[:, :2].min(0)) / 2
    s0 = float(np.percentile(Q[:, 2], 99.5)) / float(V[:, 2].max())
    c0 = np.median(Q[:, :2], 0)
    tm = o3d.t.geometry.TriangleMesh(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.int32)))
    masks = [m[::step, ::step] for (_, _, _, m) in views]

    def place(p):
        yaw, ls, tx, ty = p
        return (V * s0 * math.exp(ls)) @ _rz(yaw).T + [c0[0] + tx, c0[1] + ty, 0.0]

    def iou(p):
        Vw = place(p)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.core.Tensor(Vw.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
        out = []
        for (K, R, t, _), m in zip(views, masks):
            h = _silhouette(scene, K, R, t, m.shape[1] * step, m.shape[0] * step, step)
            out.append((h & m).sum() / max((h | m).sum(), 1))
        return float(np.mean(out))

    best = None
    for dy in range(-60, 61, 10):
        for ls in (-0.15, 0.0, 0.15):
            for tx, ty in ((0, 0), (0.4, 0), (-0.4, 0), (0, 0.4), (0, -0.4)):
                p = (yaw0 + dy, ls, tx, ty)
                v = iou(p)
                if best is None or v > best[0]:
                    best = (v, p)
    r = minimize(lambda p: -iou(p), np.array(best[1], float), method="Powell",
                 options={"xtol": 1e-3, "ftol": 1e-4, "maxfev": 600})
    p = r.x if -r.fun > best[0] else np.array(best[1])
    return place(p), {"yaw_deg": float(p[0]), "scale": float(s0 * math.exp(p[1])), "silhouette_iou": float(max(-r.fun, best[0]))}


if __name__ == "__main__":
    main()
