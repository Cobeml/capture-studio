"""Point cloud -> metric, gravity-aligned cloud -> as-built BIM shell.

prepare_cloud()  LingBot-Map predictions -> filtered cloud in the model frame
                 (Z up, floor at z=0, walls axis-aligned, metres) + viewer binary.
fit_shell()      wall faces by plane-sweep on oriented normals, faces paired into walls with
                 real thickness, openings found by *ray crossing* (a camera ray that passes
                 through a wall plane proves a hole there), slab from the wall extents.
                 Every fitted element records its residual to the scan (P50/P95 mm).
"""
from __future__ import annotations

import json
import math

import numpy as np
import open3d as o3d
from scipy import ndimage
from scipy.signal import find_peaks

from . import bim, store

SCAN = store.DATA / "scan"
CAMERA_HEIGHT_M = 1.5       # scale prior for video-only capture: phone held at about chest/eye height


def cloud_meta() -> dict | None:
    p = SCAN / "meta.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    return {k: m[k] for k in ("n", "frames", "fps", "ceiling_height", "scale", "scale_source", "extent_m", "inference_seconds")}


def set_scale_from_ceiling(h: float):
    SCAN.mkdir(parents=True, exist_ok=True)
    (SCAN / "calibration.json").write_text(json.dumps({"ceiling_height_m": h}))
    prepare_cloud()


def _rot_to_z(n):
    n = n / np.linalg.norm(n)
    v = np.cross(n, [0, 0, 1.0])
    s, c = np.linalg.norm(v), n[2]
    if s < 1e-8:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s ** 2)


def _normals(P, cams_of_points, radius):
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P.astype(np.float64)))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=40))
    n = np.asarray(pc.normals)
    flip = np.einsum("ij,ij->i", n, cams_of_points - P) < 0     # make normals face the camera that saw them
    n[flip] *= -1
    return n.astype(np.float32)


def prepare_cloud(max_points=1_600_000) -> dict:
    z = np.load(SCAN / "raw_predictions.npz")
    pts, rgb, conf, extr = z["points"], z["rgb"], z["conf"], z["extrinsic"]
    S, H, W, _ = pts.shape
    R, t = extr[:, :3, :3], extr[:, :3, 3]
    C = -np.einsum("sji,sj->si", R, t)             # camera centres
    fwd = R[:, 2, :]
    up_cam = -R[:, 1, :]

    # --- filter: confidence + depth discontinuities (flying pixels), stride 2
    depth = np.einsum("sij,shwj->shwi", R, pts)[..., 2] + t[:, None, None, 2]
    gy, gx = np.gradient(depth, axis=(1, 2))
    edge = np.hypot(gx, gy) / np.maximum(depth, 1e-6) > 0.03
    keep = (conf > max(1.5, float(np.percentile(conf, 20)))) & ~edge & (depth > 0)
    sub = np.zeros_like(keep)
    sub[:, ::2, ::2] = True
    keep &= sub
    f_idx = np.broadcast_to(np.arange(S)[:, None, None], keep.shape)[keep]
    P = pts[keep].astype(np.float64)
    col = rgb[keep]

    # --- gravity: floor plane = big plane, normal ~ mean camera 'up', below the cameras
    up0 = up_cam.mean(0)
    up0 /= np.linalg.norm(up0)
    h = P @ up0
    low = P[h < np.percentile(h, 35)]
    low = low[np.random.default_rng(0).choice(len(low), min(len(low), 300_000), replace=False)]
    span = float(np.linalg.norm(C.max(0) - C.min(0))) + 1e-6
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(low))
    n_floor, d_floor = up0, None
    for _ in range(4):
        (a, b, c, d), inl = pc.segment_plane(0.01 * span, 3, 400)
        n = np.array([a, b, c])
        if abs(n @ up0) > 0.9:
            sgn = 1.0 if n @ up0 > 0 else -1.0
            n_floor, d_floor = n * sgn, d * sgn
            break
        pc = pc.select_by_index(inl, invert=True)
    Rg = _rot_to_z(n_floor)
    P, C, fwd = P @ Rg.T, C @ Rg.T, fwd @ Rg.T
    z0 = -d_floor if d_floor is not None else float(np.percentile(P[:, 2], 2))
    P[:, 2] -= z0
    C[:, 2] -= z0

    # --- scale
    scale = CAMERA_HEIGHT_M / float(np.median(C[:, 2]))
    source = f"assumed phone height {CAMERA_HEIGHT_M} m above floor"
    P *= scale
    C *= scale

    def ceiling_of(P):
        zz = P[(P[:, 2] > 1.9) & (P[:, 2] < 5.0), 2]
        hist, edges = np.histogram(zz, bins=np.arange(1.9, 5.0, 0.02))
        hist = ndimage.gaussian_filter1d(hist.astype(float), 1.5)
        return float(edges[int(np.argmax(hist))] + 0.01)

    ceil = ceiling_of(P)
    cal = SCAN / "calibration.json"
    if cal.exists():
        known = json.loads(cal.read_text())["ceiling_height_m"]
        k = known / ceil
        P *= k
        C *= k
        scale *= k
        ceil, source = known, f"measured ceiling height {known} m"

    # --- yaw: make the dominant wall direction axis-aligned
    cam_pts = C[f_idx]
    band = (P[:, 2] > 0.4) & (P[:, 2] < ceil - 0.4)
    idx = np.flatnonzero(band)
    idx = idx[np.random.default_rng(1).choice(len(idx), min(len(idx), 400_000), replace=False)]
    n = _normals(P[idx], cam_pts[idx], 0.12)
    horiz = np.abs(n[:, 2]) < 0.2
    ang = np.mod(np.arctan2(n[horiz, 1], n[horiz, 0]), np.pi / 2)
    hist, edges = np.histogram(ang, bins=180, range=(0, np.pi / 2))
    hist = ndimage.gaussian_filter1d(hist.astype(float), 2, mode="wrap")
    yaw = float(edges[int(np.argmax(hist))] + (edges[1] - edges[0]) / 2)
    c_, s_ = math.cos(-yaw), math.sin(-yaw)
    Rz = np.array([[c_, -s_, 0], [s_, c_, 0], [0, 0, 1]])
    P, C, fwd = P @ Rz.T, C @ Rz.T, fwd @ Rz.T

    # --- origin at the min corner of the walls; crop far outliers (things seen through windows)
    wb = P[band]
    lo = np.percentile(wb[:, :2], 0.5, axis=0)
    hi = np.percentile(wb[:, :2], 99.5, axis=0)
    P[:, :2] -= lo
    C[:, :2] -= lo
    ext = hi - lo
    inside = (P[:, 0] > -2.5) & (P[:, 0] < ext[0] + 2.5) & (P[:, 1] > -2.5) & (P[:, 1] < ext[1] + 2.5) & \
             (P[:, 2] > -0.4) & (P[:, 2] < ceil + 0.4)
    P, col, f_idx = P[inside], col[inside], f_idx[inside]

    # --- voxel dedupe keeping the earliest frame (so the viewer can reveal the scan in capture order)
    vox = 0.02
    while True:
        keys = np.floor(P / vox).astype(np.int64)
        keys = (keys[:, 0] + 4096) | ((keys[:, 1] + 4096) << 21) | ((keys[:, 2] + 4096) << 42)
        _, first = np.unique(keys, return_index=True)
        if len(first) <= max_points:
            break
        vox *= 1.25
    first.sort()
    P, col, f_idx = P[first], col[first], f_idx[first]

    SCAN.mkdir(parents=True, exist_ok=True)
    np.savez(SCAN / "aligned.npz", P=P.astype(np.float32), frame=f_idx.astype(np.int32), C=C.astype(np.float32),
             ceiling=ceil)
    with open(SCAN / "cloud.bin", "wb") as f:
        f.write(P.astype(np.float32).tobytes())
        f.write(col.astype(np.uint8).tobytes())
    offsets = np.searchsorted(f_idx, np.arange(S + 1)).tolist()
    meta = {
        "n": int(len(P)), "frames": int(S), "fps": float(z["fps"]), "frame_offsets": offsets,
        "cameras": [[round(float(v), 3) for v in (*C[i], *fwd[i])] for i in range(S)],
        "center": [round(float(ext[0] / 2), 2), round(float(ext[1] / 2), 2), 1.0],
        "radius": round(float(np.linalg.norm(ext) / 2), 2), "extent_m": [round(float(v), 2) for v in ext],
        "ceiling_height": round(ceil, 3), "scale": round(scale, 5), "scale_source": source, "voxel_m": round(vox, 4),
        "inference_seconds": round(float(z["inference_seconds"]), 1),
    }
    (SCAN / "meta.json").write_text(json.dumps(meta))
    return cloud_meta()


# ---------------------------------------------------------------- shell fitting

def _runs(mask, min_len, close):
    """Index runs [a, b) of True in a 1-D mask, closing gaps <= close and dropping runs < min_len."""
    m = ndimage.binary_closing(np.pad(mask, close), structure=np.ones(close + 1))[close:-close] if close else mask
    lab, n = ndimage.label(m)
    out = []
    for sl, in ndimage.find_objects(lab):
        if sl.stop - sl.start >= min_len:
            out.append((sl.start, sl.stop))
    return out


def _faces(P, N, H, cell=0.05):
    """Vertical planar face segments: {axis, sign, c, b0, b1, members(mask over P)}."""
    faces = []
    band = (P[:, 2] > 0.1) & (P[:, 2] < H - 0.1)
    for a in (0, 1):
        b = 1 - a
        for sign in (1, -1):
            sel = band & (sign * N[:, a] > 0.88)
            if sel.sum() < 500:
                continue
            c = P[sel, a]
            edges = np.arange(c.min() - 0.1, c.max() + 0.1, 0.02)
            hist, _ = np.histogram(c, edges)
            sm = ndimage.gaussian_filter1d(hist.astype(float), 1.2)
            peaks, _ = find_peaks(sm, height=max(0.03 * sm.max(), 150), distance=5, prominence=0.3 * max(0.03 * sm.max(), 150))
            for pk in peaks:
                c0 = edges[pk] + 0.01
                near = sel & (np.abs(P[:, a] - c0) < 0.05)
                c0 = float(np.median(P[near, a]))
                near = sel & (np.abs(P[:, a] - c0) < 0.05)
                u, zz = P[near, b], P[near, 2]
                u0 = math.floor(u.min() / cell) * cell
                grid = np.zeros((int((u.max() - u0) / cell) + 1, int(H / 0.1) + 1), bool)
                grid[((u - u0) / cell).astype(int), np.clip((zz / 0.1).astype(int), 0, grid.shape[1] - 1)] = True
                covered = grid.sum(1) >= 4                       # at least 0.4 m of height has surface here
                for i0, i1 in _runs(covered, int(0.5 / cell), int(0.25 / cell)):
                    b0, b1 = u0 + i0 * cell, u0 + i1 * cell
                    mem = near & (P[:, b] >= b0) & (P[:, b] <= b1)
                    if mem.sum() < 300:
                        continue
                    faces.append({"axis": a, "sign": sign, "c": c0, "b0": float(b0), "b1": float(b1), "mem": mem})
    return faces


def _seen_through(face, faces, P, Cp, H, n=3000) -> float:
    """Fraction of a face's points whose camera ray first passes through another wall face."""
    idx = np.flatnonzero(face["mem"])
    idx = idx[:: max(1, len(idx) // n)]
    p, c = P[idx], Cp[idx]
    through = np.zeros(len(idx), bool)
    for g in faces:
        if g is face:
            continue
        a, b = g["axis"], 1 - g["axis"]
        den = p[:, a] - c[:, a]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (g["c"] - c[:, a]) / den
        x = c + t[:, None] * (p - c)
        through |= (t > 0.03) & (t < 0.97) & (x[:, b] > g["b0"]) & (x[:, b] < g["b1"]) & (x[:, 2] > 0) & (x[:, 2] < H)
    return float(through.mean())


def _openings_on_face(face, P, Cp, H, cell=0.05):
    """Holes in a face: cells crossed by camera rays and free of surface points."""
    a, b, sign, c0 = face["axis"], 1 - face["axis"], face["sign"], face["c"]
    L = face["b1"] - face["b0"]
    nu, nz = int(round(L / cell)), int(round(H / cell))
    occ = np.zeros((nu, nz), bool)
    m = face["mem"]
    ui = ((P[m, b] - face["b0"]) / cell).astype(int)
    zi = (P[m, 2] / cell).astype(int)
    ok = (ui >= 0) & (ui < nu) & (zi >= 0) & (zi < nz)
    occ[ui[ok], zi[ok]] = True
    # rays: camera on the room side of the plane, point beyond it
    sc = sign * (Cp[:, a] - c0)
    sp = sign * (P[:, a] - c0)
    thru = (sc > 0.05) & (sp < -0.12)
    tt = sc[thru] / (sc[thru] - sp[thru])
    X = Cp[thru] + tt[:, None] * (P[thru] - Cp[thru])
    ui = ((X[:, b] - face["b0"]) / cell).astype(int)
    zi = (X[:, 2] / cell).astype(int)
    ok = (ui >= 0) & (ui < nu) & (zi >= 0) & (zi < nz)
    cross = np.zeros((nu, nz), np.int32)
    np.add.at(cross, (ui[ok], zi[ok]), 1)
    hole = (cross >= 2) & ~ndimage.binary_dilation(occ, iterations=1)
    hole = ndimage.binary_closing(hole, structure=np.ones((5, 5)))
    hole = ndimage.binary_opening(hole, structure=np.ones((5, 5)))
    lab, n = ndimage.label(hole)
    out = []
    for k, (su, sz) in enumerate(ndimage.find_objects(lab), 1):
        fill = float((lab[su, sz] == k).mean())
        w, hgt = (su.stop - su.start + 2) * cell, (sz.stop - sz.start + 2) * cell
        if w < 0.55 or hgt < 0.55 or fill < 0.6:
            continue
        # the occupancy dilation above shaved one cell off every side of the hole: give it back
        su = slice(max(su.start - 1, 0), min(su.stop + 1, nu))
        sz = slice(max(sz.start - 1, 0), min(sz.stop + 1, nz))
        sill, top = sz.start * cell, sz.stop * cell
        if sill < 0.3:
            sill = 0.0
        if top > H - 0.2:
            top = H
        out.append({"u0": su.start * cell, "u1": su.stop * cell, "sill": sill, "top": top, "rays": int(cross[su, sz].sum())})
    return out


def fit_shell() -> tuple[dict, dict]:
    z = np.load(SCAN / "aligned.npz")
    P, frame, C, H = z["P"].astype(np.float64), z["frame"], z["C"].astype(np.float64), float(z["ceiling"])
    meta = json.loads((SCAN / "meta.json").read_text())
    Cp = C[frame]
    N = _normals(P, Cp, 0.10)

    all_faces = _faces(P, N, H)
    faces = [f for f in all_faces if _seen_through(f, all_faces, P, Cp, H) < 0.5]
    # pair opposite faces of the same wall body
    used, walls = set(), []
    for i, f in enumerate(faces):
        if f["sign"] != -1 or i in used:
            continue
        best = None
        for j, g in enumerate(faces):
            if j in used or g["axis"] != f["axis"] or g["sign"] != 1:
                continue
            t = g["c"] - f["c"]
            ov = min(f["b1"], g["b1"]) - max(f["b0"], g["b0"])
            if 0.06 <= t <= 0.45 and ov > 0.5 * min(f["b1"] - f["b0"], g["b1"] - g["b0"]):
                if best is None or ov > best[1]:
                    best = (j, ov)
        if best:
            used |= {i, best[0]}
            walls.append([f, faces[best[0]]])
    walls += [[f] for i, f in enumerate(faces) if i not in used]

    model = bim.empty_model("Gandhi Bazaar flat — main hall")
    model["level"]["ceiling_height"] = round(H, 2)
    recs = []
    for fs in walls:
        a = fs[0]["axis"]
        if len(fs) == 2:
            c = (fs[0]["c"] + fs[1]["c"]) / 2
            th = round(fs[1]["c"] - fs[0]["c"], 2)
        else:
            th = 0.2
            c = fs[0]["c"] - fs[0]["sign"] * th / 2
        b0, b1 = min(f["b0"] for f in fs), max(f["b1"] for f in fs)
        res = np.concatenate([np.abs(P[f["mem"], a] - f["c"]) for f in fs]) * 1000
        recs.append({"axis": a, "c": c, "th": th, "b0": b0, "b1": b1, "faces": fs,
                     "fit": {"p50": round(float(np.percentile(res, 50)), 1), "p95": round(float(np.percentile(res, 95)), 1), "n": int(len(res))}})

    # corners: run wall ends into perpendicular walls they nearly meet
    for r in recs:
        for o in recs:
            if o["axis"] == r["axis"]:
                continue
            if o["b0"] - 0.4 <= r["c"] <= o["b1"] + 0.4:
                if abs(r["b0"] - o["c"]) < 0.4:
                    r["b0"] = o["c"]
                if abs(r["b1"] - o["c"]) < 0.4:
                    r["b1"] = o["c"]

    first_cam = C[0, :2]
    doorish = []
    for k, r in enumerate(sorted(recs, key=lambda r: (r["axis"], r["c"]))):
        a = r["axis"]
        start = [r["c"], r["b0"]] if a == 0 else [r["b0"], r["c"]]
        end = [r["c"], r["b1"]] if a == 0 else [r["b1"], r["c"]]
        w = bim.add(model, "wall", "scanned", name=f"Wall {chr(65 + k % 26)}{k // 26 or ''}",
                    start=[round(v, 2) for v in start], end=[round(v, 2) for v in end], thickness=r["th"],
                    height=round(H, 2), material="masonry", fit=r["fit"], faces_seen=len(r["faces"]))
        ops = []
        for f in r["faces"]:
            for o in _openings_on_face(f, P, Cp, H):
                u0, u1 = o["u0"] + f["b0"] - r["b0"], o["u1"] + f["b0"] - r["b0"]
                if any(u0 < q["u1"] and q["u0"] < u1 for q in ops):
                    continue
                ops.append({**o, "u0": u0, "u1": u1})
        L = bim.wall_length(w)
        for o in sorted(ops, key=lambda o: o["u0"]):
            u0, u1 = max(0.0, o["u0"]), min(L, o["u1"])
            if u1 - u0 < 0.5:
                continue
            kind = "window" if o["sill"] > 0.3 else ("door" if (u1 - u0) <= 1.4 and o["top"] < H - 0.2 else "void")
            try:
                e = bim.add(model, kind, "scanned", host=w["id"], offset=round(u0, 2), width=round(u1 - u0, 2),
                            sill=round(o["sill"], 2), height=round(min(o["top"], w["height"]) - o["sill"], 2), rays=o["rays"])
            except ValueError:
                continue
            if kind == "window":
                w["exterior"] = True
            else:
                doorish.append(e)
    if doorish:   # the walkthrough starts at the entrance: nearest doorway to the first camera is the exit
        ex = min(doorish, key=lambda e: math.dist(bim.opening_center(model, e), first_cam))
        ex["exit"] = True
        ex["name"] = "Entrance"

    ws = [e for e in model["elements"].values() if e["kind"] == "wall"]
    xs = [p[0] for w in ws for p in (w["start"], w["end"])] + list(C[:, 0])
    ys = [p[1] for w in ws for p in (w["start"], w["end"])] + list(C[:, 1])
    bim.add(model, "slab", "scanned", name="Floor slab", min=[round(min(xs) - 0.25, 2), round(min(ys) - 0.25, 2)],
            max=[round(max(xs) + 0.25, 2), round(max(ys) + 0.25, 2)], top=0.0, thickness=0.15, material="concrete")

    fits = [w["fit"] for w in ws]
    model["scan"] = {
        "source": "data/site.mp4 (phone walkthrough)", "reconstruction": "LingBot-Map (streaming)",
        "frames": meta["frames"], "points": meta["n"], "scale_source": meta["scale_source"],
        "fit_p50_mm": round(float(np.median([f["p50"] for f in fits])), 1) if fits else None,
        "fit_p95_mm": round(float(max(f["p95"] for f in fits)), 1) if fits else None,
    }
    counts = bim.summary(model)["counts"]
    return model, {"counts": counts, "ceiling_height_m": round(H, 2), "scan": model["scan"],
                   "note": "Scale is an assumption until one dimension is tape-checked (set_scale)."}
