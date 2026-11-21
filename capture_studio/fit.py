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

import cv2
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
    return {k: m.get(k) for k in ("n", "frames", "fps", "ceiling_height", "ceiling_source", "scale", "scale_source", "extent_m", "inference_seconds")}


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


# ---------------------------------------------------------------- dense cloud
#
# LingBot-Map predicts one point map per frame. Neighbouring frames disagree by ~3 % of depth (pose error
# of ~1 deg / a few cm per frame at 2 fps), which is what made every surface look thick and "rough".
# 1) refine_frames(): pairwise point-to-plane ICP between frames i, i+1..i+3 (+ revisits), solved as one
#    pose graph whose nodes are tied to LingBot's own poses by a weak prior (an anchor node), so local
#    disagreement is removed without letting the trajectory drift: the frozen model frame stays put.
# 2) dense_pixels(): every pixel is re-projected into its +-3 neighbours; it is kept only if >= 2 of them
#    see the same surface at that depth (and few see *through* it: flying pixels / ghosts), and its position
#    is the average of the agreeing observations (multi-view fusion -> thinner surfaces).
# 3) prepare_cloud(): full-resolution fused pixels -> frozen model frame -> 1 cm voxel average.

REFINE_SIGMA_T, REFINE_SIGMA_R, REFINE_SIGMA_ICP = 0.03, 0.02, 0.004    # raw units (~x3.8 = metres), radians


def load_frame() -> dict | None:
    p = SCAN / "frame.json"
    return json.loads(p.read_text()) if p.exists() else None


def _icp_clouds(pts, conf, edge, depth, C, vox):
    clouds = []
    for i in range(len(pts)):
        m = (conf[i] > 2.0) & ~edge[i] & (depth[i] > 0)
        p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[i][m].astype(np.float64))).voxel_down_sample(vox)
        p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vox * 5, max_nn=30))
        p.orient_normals_towards_camera_location(C[i])
        clouds.append(p)
    return clouds


def refine_frames(z=None, force=False) -> np.ndarray:
    """Per-frame rigid corrections (S,4,4) in raw coordinates, cached in frame_refine.npz."""
    cache = SCAN / "frame_refine.npz"
    z = z if z is not None else np.load(SCAN / "raw_predictions.npz")
    E = z["extrinsic"]
    S = len(E)
    if cache.exists() and not force:
        c = np.load(cache)
        if c["T"].shape[0] == S:
            return c["T"]
    reg = o3d.pipelines.registration
    pts, conf = z["points"], z["conf"]
    R, t = E[:, :3, :3], E[:, :3, 3]
    C = -np.einsum("sji,sj->si", R, t)
    fwd = R[:, 2, :]
    depth = np.einsum("sij,shwj->shwi", R, pts)[..., 2] + t[:, None, None, 2]
    gy, gx = np.gradient(depth, axis=(1, 2))
    edge = np.hypot(gx, gy) / np.maximum(depth, 1e-6) > 0.03
    clouds = _icp_clouds(pts, conf, edge, depth, C, 0.003)
    pairs = [(i, j) for i in range(S) for j in range(i + 1, min(S, i + 4))]
    for i in range(0, S, 2):                      # revisits: same place, same heading, later in the walk
        for j in range(i + 8, S, 2):
            if np.linalg.norm(C[i] - C[j]) < 0.35 and fwd[i] @ fwd[j] > 0.75:
                pairs.append((i, j))
    pg = reg.PoseGraph()
    for _ in range(S + 1):                        # node S = anchor (LingBot's own poses), fixed
        pg.nodes.append(reg.PoseGraphNode(np.eye(4)))
    prior = np.diag([1 / REFINE_SIGMA_R ** 2] * 3 + [1 / REFINE_SIGMA_T ** 2] * 3)
    for i in range(S):
        pg.edges.append(reg.PoseGraphEdge(S, i, np.eye(4), prior, uncertain=False))
    have = set()
    for i, j in pairs:
        T = np.eye(4)
        for d in (0.02, 0.008):
            r = reg.registration_icp(clouds[i], clouds[j], d, T,
                                     reg.TransformationEstimationPointToPlane(reg.TukeyLoss(k=d)),
                                     reg.ICPConvergenceCriteria(max_iteration=40))
            T = r.transformation
        if r.fitness < 0.3:
            continue
        info = reg.get_information_matrix_from_point_clouds(clouds[i], clouds[j], 0.008, T)
        info = info / max(info[5, 5], 1.0) / REFINE_SIGMA_ICP ** 2      # per-correspondence information
        pg.edges.append(reg.PoseGraphEdge(i, j, T, info, uncertain=(j - i > 1)))
        have.add((i, j))
    for i in range(S - 1):
        if (i, i + 1) not in have:
            pg.edges.append(reg.PoseGraphEdge(i, i + 1, np.eye(4), np.eye(6), uncertain=False))
    opt = reg.GlobalOptimizationOption(max_correspondence_distance=0.008, edge_prune_threshold=0.25,
                                       reference_node=S, preference_loop_closure=1.0)
    reg.global_optimization(pg, reg.GlobalOptimizationLevenbergMarquardt(),
                            reg.GlobalOptimizationConvergenceCriteria(), opt)
    Ts = np.stack([n.pose for n in pg.nodes])[:S]
    # remove any common rigid motion: corrected camera centres best-fit back onto LingBot's
    Cc = np.einsum("sij,sj->si", Ts[:, :3, :3], C) + Ts[:, :3, 3]
    mu_a, mu_b = Cc.mean(0), C.mean(0)
    U, _, Vt = np.linalg.svd((Cc - mu_a).T @ (C - mu_b))
    Ra = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    A = np.eye(4)
    A[:3, :3], A[:3, 3] = Ra, mu_b - Ra @ mu_a
    Ts = np.einsum("ij,sjk->sik", A, Ts)
    np.savez(cache, T=Ts, pairs=np.array(sorted(have)))
    return Ts


def dense_pixels(z, Ts=None, tau=0.015, nbrs=(1, 2, 3), min_cons=2, min_conf=1.2):
    """Multi-view consistency filter + fusion of LingBot's per-pixel points (GPU if available).

    Returns (P (S,H,W,3) float32 fused points in *raw* coordinates, keep (S,H,W) bool, n_cons (S,H,W) uint8)."""
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pts = torch.from_numpy(np.ascontiguousarray(z["points"])).to(dev)
    conf = torch.from_numpy(z["conf"]).to(dev)
    E = torch.from_numpy(z["extrinsic"]).float().to(dev)
    K = torch.from_numpy(z["intrinsic"]).float().to(dev)
    S, H, W, _ = pts.shape
    R, t = E[:, :, :3], E[:, :, 3]
    if Ts is not None:
        Tc = torch.from_numpy(Ts).float().to(dev)
        pts = torch.einsum("sij,shwj->shwi", Tc[:, :3, :3], pts) + Tc[:, None, None, :3, 3]
        Ri = Tc[:, :3, :3].transpose(1, 2)
        ti = -torch.einsum("sij,sj->si", Ri, Tc[:, :3, 3])
        R, t = R @ Ri, torch.einsum("sij,sj->si", R, ti) + t
    D = torch.einsum("sij,shwj->shwi", R, pts)[..., 2] + t[:, None, None, 2]
    gy, gx = torch.gradient(D, dim=(1, 2))
    edge = torch.hypot(gx, gy) / D.clamp_min(1e-6) > 0.03
    out = torch.empty_like(pts)
    keep = torch.zeros((S, H, W), dtype=torch.bool, device=dev)
    ncons = torch.zeros((S, H, W), dtype=torch.uint8, device=dev)
    for i in range(S):
        X = pts[i].reshape(-1, 3)
        acc, cnt = X.clone(), torch.ones(len(X), device=dev)
        nviol = torch.zeros(len(X), device=dev)
        for dj in nbrs:
            for j in (i - dj, i + dj):
                if not 0 <= j < S:
                    continue
                Xc = X @ R[j].T + t[j]
                zz = Xc[:, 2]
                u = (K[j, 0, 0] * Xc[:, 0] / zz + K[j, 0, 2]).round().long()
                v = (K[j, 1, 1] * Xc[:, 1] / zz + K[j, 1, 2]).round().long()
                ok = (zz > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
                uu, vv = u.clamp(0, W - 1), v.clamp(0, H - 1)
                dj_ = D[j, vv, uu]
                ok &= (conf[j, vv, uu] > min_conf) & ~edge[j, vv, uu]
                rel = (zz - dj_) / zz.clamp_min(1e-6)
                cons = ok & (rel.abs() < tau)
                nviol += (ok & (rel < -3 * tau)).float()
                acc[cons] += pts[j, vv[cons], uu[cons]]
                cnt += cons.float()
        nc = cnt - 1
        k = (conf[i].reshape(-1) > min_conf) & ~edge[i].reshape(-1) & (D[i].reshape(-1) > 0) & \
            (nc >= min_cons) & (nviol <= nc / 2)
        out[i] = (acc / cnt[:, None]).reshape(H, W, 3)
        keep[i] = k.reshape(H, W)
        ncons[i] = nc.clamp(max=255).to(torch.uint8).reshape(H, W)
    return out.cpu().numpy(), keep.cpu().numpy(), ncons.cpu().numpy()


def to_model(P, frame):
    """x_model = s * R @ x_raw + t (frame.json)."""
    return (frame["s"] * (np.asarray(P, np.float64) @ np.asarray(frame["R"]).T) + np.asarray(frame["t"]))


def model_pixels(z=None, refine=True):
    """Per-pixel fused points in the model frame + validity, for segmentation (objects.py)."""
    z = z if z is not None else np.load(SCAN / "raw_predictions.npz")
    frame = load_frame()
    Ts = refine_frames(z) if refine else None
    P, keep, ncons = dense_pixels(z, Ts)
    S, H, W, _ = P.shape
    Pm = to_model(P.reshape(-1, 3), frame).astype(np.float32).reshape(S, H, W, 3)
    return Pm, keep, ncons


def _umeyama(A, B):
    """Similarity (s, R, t) with B ~ s R A + t."""
    mu_a, mu_b = A.mean(0), B.mean(0)
    A0, B0 = A - mu_a, B - mu_b
    U, D, Vt = np.linalg.svd(B0.T @ A0 / len(A))
    Sd = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ Sd @ Vt
    s = float(np.trace(np.diag(D) @ Sd) / (A0 ** 2).sum(1).mean())
    return s, R, mu_b - s * R @ mu_a


def prepare_cloud(max_points=4_500_000, voxel=0.008, refine=True) -> dict:
    z = np.load(SCAN / "raw_predictions.npz")
    pts, rgb, conf, extr = z["points"], z["rgb"], z["conf"], z["extrinsic"]
    S, H, W, _ = pts.shape
    R, t = extr[:, :3, :3], extr[:, :3, 3]
    C = -np.einsum("sji,sj->si", R, t)             # camera centres (LingBot's poses: the published path)
    C_raw = C.copy()
    fwd = R[:, 2, :]
    up_cam = -R[:, 1, :]
    upv = up_cam.copy()
    o3d.utility.random.seed(0)                     # deterministic RANSAC: refits land in the same frame

    Ts = refine_frames(z) if refine else None
    Pd, keep, _ = dense_pixels(z, Ts)
    f_idx = np.broadcast_to(np.arange(S)[:, None, None], keep.shape)[keep]
    P = Pd[keep].astype(np.float64)
    col = rgb[keep]
    del Pd
    frame = load_frame()
    if frame is not None:
        return _finish_cloud(z, frame, P, col, f_idx, C_raw, fwd, upv, max_points, voxel)

    P_raw = P.copy()
    # --- no frame.json yet: estimate one (gravity, scale, yaw, origin), then freeze it
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
    P, C, fwd, upv = P @ Rg.T, C @ Rg.T, fwd @ Rg.T, upv @ Rg.T
    z0 = -d_floor if d_floor is not None else float(np.percentile(P[:, 2], 2))
    P[:, 2] -= z0
    C[:, 2] -= z0

    # --- scale
    scale = CAMERA_HEIGHT_M / float(np.median(C[:, 2]))
    source = f"assumed phone height {CAMERA_HEIGHT_M} m above floor"
    P *= scale
    C *= scale

    def ceiling_of(P):
        """Soffit height: the highest distinct horizontal layer above head height. A forward-looking
        walkthrough of a hall often never sees the roof, so fall back to the top of what was seen."""
        zz = P[(P[:, 2] > 1.9) & (P[:, 2] < 14.0), 2]
        hist, edges = np.histogram(zz, bins=np.arange(1.9, 14.0, 0.04))
        hist = ndimage.gaussian_filter1d(hist.astype(float), 2.0)
        peaks, _ = find_peaks(hist, prominence=0.25 * hist.max(), height=0.35 * hist.max())
        if len(peaks):
            return float(edges[int(peaks[-1])] + 0.02), "detected"
        return float(np.clip(np.percentile(zz, 98), 2.6, 12.0)), "not observed — top of scanned geometry"

    # floor at exactly z=0: refine the RANSAC offset with the mode of the lowest layer
    hz, ez = np.histogram(P[(P[:, 2] > -0.5) & (P[:, 2] < 0.5), 2], bins=100)
    dz = float(ez[int(np.argmax(ndimage.gaussian_filter1d(hz.astype(float), 2)))] + 0.005)
    P[:, 2] -= dz
    C[:, 2] -= dz
    k0 = CAMERA_HEIGHT_M / float(np.median(C[:, 2]))
    P *= k0
    C *= k0
    scale *= k0
    ceil, ceil_source = ceiling_of(P)
    cal = SCAN / "calibration.json"
    if cal.exists():
        known = json.loads(cal.read_text())["ceiling_height_m"]
        k = known / ceil
        P *= k
        C *= k
        scale *= k
        ceil, source, ceil_source = known, f"measured ceiling height {known} m", "measured"

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
    P, C, fwd, upv = P @ Rz.T, C @ Rz.T, fwd @ Rz.T, upv @ Rz.T

    # --- origin at the min corner of the walls; crop far outliers (things seen through windows)
    wb = P[band]
    lo = np.percentile(wb[:, :2], 0.5, axis=0)
    hi = np.percentile(wb[:, :2], 99.5, axis=0)
    P[:, :2] -= lo
    C[:, :2] -= lo
    ext = hi - lo
    s_, R_, t_ = _umeyama(C_raw, C)
    frame = {"R": R_.tolist(), "s": s_, "t": t_.tolist(), "ceiling_height": round(float(ceil), 3),
             "ceiling_source": ceil_source, "scale_source": source, "extent_m": [round(float(v), 2) for v in ext],
             "note": "x_model = s * R @ x_raw + t"}
    (SCAN / "frame.json").write_text(json.dumps(frame))
    return _finish_cloud(z, frame, P_raw, col, f_idx, C_raw, R[:, 2, :], up_cam, max_points, voxel)


def _finish_cloud(z, frame, P_raw, col, f_idx, C_raw, fwd_raw, up_raw, max_points, voxel) -> dict:
    """Raw fused points -> frozen model frame -> voxel average -> cloud.bin / aligned.npz / meta.json."""
    S, H, W = z["points"].shape[:3]
    Rf, s_, tf = np.asarray(frame["R"]), float(frame["s"]), np.asarray(frame["t"])
    P = to_model(P_raw, frame)
    C = to_model(C_raw, frame)
    fwd, upv = fwd_raw @ Rf.T, up_raw @ Rf.T
    ceil = float(frame["ceiling_height"])
    if "extent_m" in frame:
        ext = np.asarray(frame["extent_m"], float)
    else:
        band = (P[:, 2] > 0.4) & (P[:, 2] < ceil - 0.4)
        ext = np.percentile(P[band][:, :2], 99.5, axis=0)
    inside = (P[:, 0] > -2.5) & (P[:, 0] < ext[0] + 2.5) & (P[:, 1] > -2.5) & (P[:, 1] < ext[1] + 2.5) & \
             (P[:, 2] > -0.4) & (P[:, 2] < ceil + 0.4)
    P, col, f_idx = P[inside], col[inside], f_idx[inside]

    # --- voxel average (fused position, mean colour), stamped with the earliest frame that saw the voxel
    #     so the viewer can still reveal the scan in capture order
    vox = voxel
    while True:
        keys = np.floor(P / vox).astype(np.int64)
        keys = (keys[:, 0] + 8192) | ((keys[:, 1] + 8192) << 21) | ((keys[:, 2] + 8192) << 42)
        uk, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
        n_ok = int((cnt >= 2).sum())             # a voxel needs two fused observations (single hits are noise)
        if n_ok <= max_points:
            break
        vox *= 1.1
    acc = np.zeros((len(uk), 3))
    for k in range(3):
        acc[:, k] = np.bincount(inv, weights=P[:, k], minlength=len(uk))
    cacc = np.stack([np.bincount(inv, weights=col[:, k].astype(np.float64), minlength=len(uk)) for k in range(3)], 1)
    fmin = np.full(len(uk), S, np.int64)
    np.minimum.at(fmin, inv, f_idx)
    ok = cnt >= 2
    P = acc[ok] / cnt[ok, None]
    col = np.clip(np.round(cacc[ok] / cnt[ok, None]), 0, 255).astype(np.uint8)
    f_idx = fmin[ok]
    # --- statistical outlier removal (isolated specks left after fusion)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    _, ind = pc.remove_statistical_outlier(nb_neighbors=12, std_ratio=2.5)
    ind = np.asarray(ind)
    P, col, f_idx = P[ind], col[ind], f_idx[ind]
    order = np.argsort(f_idx, kind="stable")
    P, col, f_idx = P[order], col[order], f_idx[order]

    SCAN.mkdir(parents=True, exist_ok=True)
    np.savez(SCAN / "aligned.npz", P=P.astype(np.float32), frame=f_idx.astype(np.int32), C=C.astype(np.float32),
             ceiling=ceil)
    with open(SCAN / "cloud.bin", "wb") as f:
        f.write(P.astype(np.float32).tobytes())
        f.write(col.astype(np.uint8).tobytes())
    for stale in ("cloud_labels.bin",):          # point order changed: per-point labels no longer apply
        (SCAN / stale).unlink(missing_ok=True)
    offsets = np.searchsorted(f_idx, np.arange(S + 1)).tolist()
    meta = {
        "n": int(len(P)), "frames": int(S), "fps": float(z["fps"]), "frame_offsets": offsets,
        "frame_dt": float(z["frame_dt"]) if "frame_dt" in z else _frame_dt(float(z["fps"])),   # seconds of video per frame
        # per frame: position, forward, up (model frame) — enough to re-fly the capture path exactly
        "cameras": [[round(float(v), 4) for v in (*C[i], *fwd[i], *upv[i])] for i in range(S)],
        "fov_y_deg": round(float(np.degrees(2 * np.arctan(H / 2 / np.median(z["intrinsic"][:, 1, 1])))), 2),
        "aspect": round(W / H, 4),
        "center": [round(float(ext[0] / 2), 2), round(float(ext[1] / 2), 2), 1.0],
        "radius": round(float(np.linalg.norm(ext) / 2), 2), "extent_m": [round(float(v), 2) for v in ext],
        "ceiling_height": round(ceil, 3), "ceiling_source": frame.get("ceiling_source", ""),
        "view_clip": round(min(ceil - 0.2, 3.4), 2), "scale": round(s_, 5),
        "scale_source": frame.get("scale_source", ""), "voxel_m": round(vox, 4),
        "inference_seconds": round(float(z["inference_seconds"]), 1),
        "cloud": {"method": "LingBot-Map full-res pixels, ICP pose-graph frame refinement, multi-view consistency "
                            "filter + fusion, voxel average", "frame": "frame.json (frozen)"},
    }
    if (SCAN / "objects.json").exists():
        meta["objects"] = len(json.loads((SCAN / "objects.json").read_text()).get("objects", []))
    (SCAN / "meta.json").write_text(json.dumps(meta))
    return cloud_meta()


def _frame_dt(fps: float) -> float:
    """Seconds of video per LingBot frame: the loader keeps every round(src_fps / fps)-th source frame."""
    cap = cv2.VideoCapture(str(store.ROOT / "data" / "site.mp4"))
    src = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()
    return max(1, round(src / fps)) / src if src > 0 else 1.0 / fps


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
                top = np.where(grid.any(1), grid.shape[1] - 1 - np.argmax(grid[:, ::-1], axis=1), 0) * 0.1
                for i0, i1 in _runs(covered, int(0.5 / cell), int(0.25 / cell)):
                    if np.median(top[i0:i1][covered[i0:i1]]) < min(H - 0.5, 2.6):
                        continue                                 # stops well short of the ceiling: a machine or rack side
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


def _equipment(P, colors, walls, H, cell=0.05, max_items=40):
    """Free-standing objects on the floor -> oriented boxes coloured from the scan."""
    zmax = min(H - 0.3, 2.8)
    m = (P[:, 2] > 0.5) & (P[:, 2] < zmax)            # above floor clutter, below services
    for w in walls:                                   # drop points belonging to walls
        a = np.array(w["start"]); b = np.array(w["end"]); d = b - a; L2 = float(d @ d)
        t = np.clip(((P[:, :2] - a) @ d) / L2, 0, 1)
        m &= np.linalg.norm(P[:, :2] - (a + t[:, None] * d), axis=1) > w["thickness"] / 2 + 0.15
    Q, C = P[m], colors[m]
    if len(Q) < 500:
        return []
    lo = Q[:, :2].min(0)
    ij = ((Q[:, :2] - lo) / cell).astype(int)
    shape = ij.max(0) + 1
    cnt = np.zeros(shape, np.int32)
    np.add.at(cnt, (ij[:, 0], ij[:, 1]), 1)
    occ = ndimage.binary_opening(ndimage.binary_closing(cnt >= 3, structure=np.ones((3, 3))), structure=np.ones((3, 3)))
    # cores = cells at least 20 cm inside a blob; every occupied cell joins its nearest core.
    # Thin bridges (cables, noise streaks, touching neighbours) no longer weld machines together.
    dist = ndimage.distance_transform_edt(occ)
    cores, n = ndimage.label(dist >= 4)
    _, (ni, nj) = ndimage.distance_transform_edt(cores == 0, return_indices=True)
    lab = np.where(occ & (ndimage.distance_transform_edt(cores == 0) <= 7), cores[ni, nj], 0)
    for thr in (8, 12):                               # blobs still larger than ~9 m2: split again with deeper cores
        for k, sl in enumerate(ndimage.find_objects(lab), 1):
            if sl is None:
                continue
            mk = lab[sl] == k
            if mk.sum() * cell * cell < 9.0:
                continue
            sub, m_sub = ndimage.label(ndimage.distance_transform_edt(mk) >= thr)
            if m_sub < 2:
                continue
            _, (si, sj) = ndimage.distance_transform_edt(sub == 0, return_indices=True)
            new = sub[si, sj]
            n_max = lab.max()
            lab[sl] = np.where(mk, np.where(new > 1, n_max + new - 1, k), lab[sl])
    n = int(lab.max())
    pl = lab[ij[:, 0], ij[:, 1]]
    out = []
    for k in range(1, n + 1):
        sel = pl == k
        if sel.sum() < 400:
            continue
        q = Q[sel]
        h = float(np.percentile(q[:, 2], 97))
        (cx, cy), (sx, sy), ang = cv2.minAreaRect(q[::max(1, len(q) // 5000), :2].astype(np.float32))
        if sx * sy < 0.3 or h < 0.8 or min(sx, sy) < 0.35:
            continue
        if sx < sy:
            sx, sy, ang = sy, sx, ang + 90
        ang = ((ang + 90) % 180) - 90
        snap = round(ang / 90) * 90
        if abs(ang - snap) < 12:                      # most plant is set out square to the building
            ang = snap
            c, s_ = math.cos(math.radians(ang)), math.sin(math.radians(ang))
            u = (q[:, 0] - cx) * c + (q[:, 1] - cy) * s_
            v = -(q[:, 0] - cx) * s_ + (q[:, 1] - cy) * c
            (u0, u1), (v0, v1) = np.percentile(u, [1, 99]), np.percentile(v, [1, 99])
            cx, cy = cx + (u0 + u1) / 2 * c - (v0 + v1) / 2 * s_, cy + (u0 + u1) / 2 * s_ + (v0 + v1) / 2 * c
            sx, sy = u1 - u0, v1 - v0
        col = np.median(C[sel], axis=0).astype(int)
        cat = "rack" if h > 2.4 and sy < 1.4 else "bench" if h < 1.15 else "machine"
        out.append({"pos": [round(float(cx), 2), round(float(cy), 2)], "size": [round(float(sx), 2), round(float(sy), 2), round(h, 2)],
                    "yaw_deg": round(float(ang), 1), "category": cat, "points": int(sel.sum()),
                    "color": "#%02x%02x%02x" % tuple(int(x) for x in col)})
    out.sort(key=lambda e: -e["size"][0] * e["size"][1])
    return out[:max_items]


def _close_perimeter(recs, P, C, lo, hi, H):
    """A cluttered hall hides most of its walls behind plant. Close the shell as a rectangle through the
    wall faces that *were* seen (else the extent of tall geometry); each side reports how much scan supports it."""
    side_c = {}
    for a, bound, sign in ((0, lo[0], 1), (0, hi[0], -1), (1, lo[1], 1), (1, hi[1], -1)):
        cands = [r for r in recs if r["axis"] == a and abs(r["c"] - bound) < 1.0 and r["b1"] - r["b0"] >= 1.5]
        L = sum(r["b1"] - r["b0"] for r in cands)
        side_c[(a, sign)] = (sum(r["c"] * (r["b1"] - r["b0"]) for r in cands) / L if cands else bound - sign * 0.1, L)
    out = []
    for (a, sign), (c, seen) in side_c.items():
        b0, b1 = side_c[(1 - a, 1)][0], side_c[(1 - a, -1)][0]
        near = (np.abs(P[:, a] - (c + sign * 0.1)) < 0.12) & (P[:, 2] > 0.3) & (P[:, 1 - a] > b0) & (P[:, 1 - a] < b1)
        res = np.abs(P[near, a] - np.median(P[near, a])) * 1000 if near.sum() > 50 else np.array([0.0])
        covered = len(np.unique((P[near, 1 - a] / 0.25).astype(int))) * 0.25 / max(b1 - b0, 1e-6)
        out.append({"axis": a, "c": float(c), "th": 0.2, "b0": float(b0), "b1": float(b1), "faces": [],
                    "fit": {"p50": round(float(np.percentile(res, 50)), 1), "p95": round(float(np.percentile(res, 95)), 1), "n": int(near.sum())},
                    "perimeter": True, "sign": sign, "scan_coverage": round(float(min(covered, 1.0)), 2)})
    x0, x1 = side_c[(0, 1)][0], side_c[(0, -1)][0]
    y0, y1 = side_c[(1, 1)][0], side_c[(1, -1)][0]
    for r in recs:                       # keep real interior partitions only
        (c0, c1), (d0, d1) = ((x0, x1), (y0, y1)) if r["axis"] == 0 else ((y0, y1), (x0, x1))
        if c0 + 0.7 < r["c"] < c1 - 0.7 and r["b1"] - r["b0"] >= 2.0 and r["b0"] > d0 - 0.3 and r["b1"] < d1 + 0.3:
            out.append(r)
    # doorways: wherever the person filming walked through (or started at) a perimeter line
    for r in out:
        if not r.get("perimeter"):
            continue
        a, c = r["axis"], r["c"]
        side = np.sign(C[:, a] - c)
        hits = [float((C[i, 1 - a] + C[i + 1, 1 - a]) / 2) for i in range(len(C) - 1) if side[i] != side[i + 1]]
        if abs(C[0, a] - c) < 1.0:
            hits.append(float(C[0, 1 - a]))
        r["walked_through"] = sorted({round(h, 1) for h in hits})
    return out


# ---------------------------------------------------------------- scanned objects -> equipment

BIM_CATEGORY = {"machine": "machine", "robot": "robot", "rack": "rack", "bench": "bench", "tank": "tank",
                "conveyor": "conveyor", "cart": "bench", "cabinet": "rack", "other": "machine"}


def load_objects() -> list[dict] | None:
    p = SCAN / "objects.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("objects", [])


def _object_equipment(objs: list[dict], walls: list[dict]) -> list[dict]:
    """One scanned equipment element per segmented object. Boxes are kept clear of the perimeter walls'
    faces (physics / clash / egress all use the box)."""
    if walls:
        xs = [p[0] for w in walls for p in (w["start"], w["end"])]
        ys = [p[1] for w in walls for p in (w["start"], w["end"])]
        th = max(w["thickness"] for w in walls) / 2 + 0.02
        x0, x1, y0, y1 = min(xs) + th, max(xs) - th, min(ys) + th, max(ys) - th
    out = []
    for o in objs:
        (cx, cy), (l, d, h), yaw = o["pos"], o["size"], float(o.get("yaw_deg", 0.0))
        if walls:
            if not (x0 - 0.5 < cx < x1 + 0.5 and y0 - 0.5 < cy < y1 + 0.5):
                continue                          # seen through a doorway / outside the shell
            c, s_ = abs(math.cos(math.radians(yaw))), abs(math.sin(math.radians(yaw)))
            hx, hy = (l * c + d * s_) / 2, (l * s_ + d * c) / 2      # half extents of the footprint AABB
            for lo_, hi_, ax in ((x0, x1, 0), (y0, y1, 1)):
                cc, hh = (cx, hx) if ax == 0 else (cy, hy)
                if hi_ - lo_ < 2 * hh:              # cannot fit at all: shrink to the room
                    k = (hi_ - lo_) / (2 * hh)
                    l, d, hx, hy = l * k, d * k, hx * k, hy * k
                    hh = hh * k
                cc = min(max(cc, lo_ + hh), hi_ - hh)
                if ax == 0:
                    cx = cc
                else:
                    cy = cc
        cat = o.get("category", "other")
        bcat = BIM_CATEGORY.get(cat, "machine")
        pos = [round(float(cx), 2), round(float(cy), 2)]
        out.append({
            "name": o["label"], "pos": pos, "size": [round(float(l), 2), round(float(d), 2), round(float(h), 2)],
            "yaw_deg": round(yaw, 1), "category": bcat, "object_category": cat,
            "material": "racking" if bcat == "rack" else "machine", "anchored": True, "fixed": True,
            "color": o.get("color", "#888888"), "points": int(o.get("points", 0)),
            "object_key": o["key"], "label": o["label"], "function": o.get("function", ""),
            "confidence": o.get("confidence"), "mesh": o.get("mesh"), "hero": bool(o.get("hero", False)),
            "origin": {"pos": pos, "yaw_deg": round(yaw, 1)},
        })
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

    model = bim.empty_model("Scanned site")
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

    tall = P[P[:, 2] > min(1.8, 0.6 * H)]
    lo, hi = np.percentile(tall[:, :2], 1, axis=0), np.percentile(tall[:, :2], 99, axis=0)
    perimeter_inferred = sum(r["b1"] - r["b0"] for r in recs) < 0.6 * 2 * float((hi - lo).sum())
    if perimeter_inferred:
        recs = _close_perimeter(recs, P, C, lo, hi, H)

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
                    height=round(H, 2), material="masonry", fit=r["fit"], faces_seen=len(r["faces"]),
                    **({"exterior": True, "basis": "perimeter closed from partial scan", "scan_coverage": r["scan_coverage"]}
                       if r.get("perimeter") else {"basis": "fitted"}))
        ops = []
        for b in r.get("walked_through", []):
            u = min(max(b - r["b0"] - 0.6, 0.1), bim.wall_length(w) - 1.3)
            if all(abs(u - q["offset"]) > 1.5 for q in bim.openings_of(model, w["id"])):
                doorish.append(bim.add(model, "door", "scanned", host=w["id"], offset=round(u, 2), width=1.2, sill=0.0,
                                       height=2.1, basis="camera path passes through"))
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
    cb = SCAN / "cloud.bin"
    colors = (np.fromfile(cb, dtype=np.uint8)[len(P) * 12:].reshape(-1, 3) if cb.exists()
              else np.full((len(P), 3), 150, np.uint8))
    objs = load_objects()
    if objs is not None:                          # segmented + named objects (capture_studio.objects)
        eq = _object_equipment(objs, ws)
        for e in eq:
            bim.add(model, "equipment", "scanned", **e)
    else:                                         # fallback: geometric occupancy clustering -> boxes
        eq = _equipment(P, colors, ws, H)
        if perimeter_inferred:
            x0, x1 = min(w["start"][0] for w in ws), max(w["end"][0] for w in ws)
            y0, y1 = min(w["start"][1] for w in ws), max(w["end"][1] for w in ws)
            eq = [e for e in eq if x0 < e["pos"][0] < x1 and y0 < e["pos"][1] < y1]
        names = {"machine": "Machine", "rack": "Rack", "bench": "Bench"}
        for i, e in enumerate(eq):
            n = sum(x["category"] == e["category"] for x in eq[:i]) + 1
            bim.add(model, "equipment", "scanned", name=f"{names[e['category']]} {n}", material="racking" if e["category"] == "rack" else "machine",
                    anchored=True, fixed=True, **e)
    xs = [p[0] for w in ws for p in (w["start"], w["end"])] + list(C[:, 0]) + [q[0] for e in eq for q in bim.footprint(model, {"kind": "equipment", **e})]
    ys = [p[1] for w in ws for p in (w["start"], w["end"])] + list(C[:, 1]) + [q[1] for e in eq for q in bim.footprint(model, {"kind": "equipment", **e})]
    bim.add(model, "slab", "scanned", name="Floor slab", min=[round(min(xs) - 0.25, 2), round(min(ys) - 0.25, 2)],
            max=[round(max(xs) + 0.25, 2), round(max(ys) + 0.25, 2)], top=0.0, thickness=0.15, material="concrete")

    fits = [w["fit"] for w in ws]
    model["scan"] = {
        "source": "data/site.mp4 (phone walkthrough)", "reconstruction": "LingBot-Map (streaming)",
        "frames": meta["frames"], "points": meta["n"], "scale_source": meta["scale_source"],
        "objects": "segmented by Gemini + SAM 2.1 (capture_studio.objects)" if objs is not None else "geometric clustering",
        "fit_p50_mm": round(float(np.median([f["p50"] for f in fits])), 1) if fits else None,
        "fit_p95_mm": round(float(max(f["p95"] for f in fits)), 1) if fits else None,
    }
    counts = bim.summary(model)["counts"]
    model["scan"]["perimeter"] = "closed from partial scan (walls hidden behind equipment)" if perimeter_inferred else "fitted"
    return model, {"counts": counts, "ceiling_height_m": round(H, 2), "scan": model["scan"],
                   "note": "Scale is an assumption until one dimension is tape-checked (set_scale)."}
