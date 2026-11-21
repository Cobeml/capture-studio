"""Scene understanding: the dense cloud -> physical objects, each named and explained by Gemini.

    python -m capture_studio.objects                       # detect -> segment -> label -> hero meshes -> files
    python -m capture_studio.objects --labels-from FILE    # offline: reuse labels (objects.json / labels json)

Stages (each cached under data/scan/objects_work/ so a re-run only redoes what changed):
  1. detect   Gemini (vision) boxes every whole piece of equipment / furniture on keyframes.
  2. masks    SAM 2.1 turns each box into a pixel mask on the full-resolution video frame.
  3. lift     masks -> 3D through LingBot-Map's per-pixel points (refined frame poses, frozen model frame).
  4. fuse     masks of the same physical object in different frames are merged by 3D voxel overlap;
              every voxel is then voted to at most one object -> cloud_labels.bin.
  5. label    per object the 2-3 best views (crops + outlined full frame) -> Gemini -> label / category /
              function JSON (then human-reviewed overrides from objects_work/review.json are applied).
  6. hero     one or two hero objects get a high-fidelity textured 3D model (capture_studio/hero.py ->
              objects.glb); a re-run here keeps heroes that were already built.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from . import fit, store

ROOT = store.ROOT
WORK_NAME = "objects_work"
GEMINI_DET_MODEL = "gemini-4-carbon"      # strongest tier: boxes and names are what the demo shows
GEMINI_LABEL_MODEL = "gemini-4-carbon"
CATEGORIES = ("machine", "robot", "rack", "bench", "tank", "conveyor", "cart", "cabinet", "other")


def scan_dir() -> Path:
    return fit.SCAN


def work_dir() -> Path:
    d = scan_dir() / WORK_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- Gemini

def _load_dotenv(path: Path | None = None) -> None:
    """Same minimal .env reader as scripts/agent_gemini.py (never overrides the environment)."""
    path = path or ROOT / ".env"
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        elif " #" in val:
            val = val.split(" #", 1)[0].rstrip()
        if key and key not in os.environ:
            os.environ[key] = val


def gemini_client():
    _load_dotenv()
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("no GEMINI_API_KEY / GOOGLE_API_KEY (environment or .env)")
    from google import genai
    from google.genai import types
    return genai.Client(api_key=key, http_options=types.HttpOptions(timeout=180_000))


def _jpeg(img_bgr, q=90) -> bytes:
    return cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()


def gemini_json(client, model: str, parts: list, retries=4):
    from google.genai import types
    contents = [types.Part.from_bytes(data=p, mime_type="image/jpeg") if isinstance(p, bytes) else p for p in parts]
    last = None
    for k in range(retries):
        try:
            r = client.models.generate_content(model=model, contents=contents, config=types.GenerateContentConfig(
                response_mime_type="application/json", temperature=0.2))
            return json.loads(r.text)
        except Exception as ex:  # rate limits / transient 5xx / malformed JSON: back off and retry
            last = ex
            time.sleep(3 * 2 ** k)
    raise RuntimeError(f"Gemini call failed: {last}")


# ---------------------------------------------------------------- frames

def video_frames(video=None, n=None, interval=None) -> np.ndarray:
    """Full-resolution video frames matching LingBot-Map's frame indices (every `interval`-th frame)."""
    video = str(video or ROOT / "data" / "site.mp4")
    meta = json.loads((scan_dir() / "meta.json").read_text())
    cap = cv2.VideoCapture(video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = interval or max(1, round(meta["frame_dt"] * src_fps))
    n = n or meta["frames"]
    out, i = [], 0
    while len(out) < n:
        ok, f = cap.read()
        if not ok:
            break
        if i % interval == 0:
            out.append(f)
        i += 1
    cap.release()
    return np.stack(out)


# ---------------------------------------------------------------- 1. detection

DETECT_PROMPT = """This is a frame from a walkthrough video of a machine shop. Detect every distinct physical object that a facility layout would treat as one piece of equipment or furniture: machine tools (CNC lathes, vertical machining centres, manual lathes, milling machines, saws, shears, grinders, drill presses), robots, shelving / storage racks, bar-stock racks, workbenches, tables, tool carts, tool cabinets, parts washers, drums/tanks, compressors, conveyors. Each box must cover the WHOLE object (not a part of it such as a door, control panel or chuck). Do not box walls, floor, ceiling, pipes, cables, lights, people or small loose items (smaller than ~40 cm). Include objects partly cut off by the frame edge if their visible part is substantial.
Return JSON: a list of {"box_2d": [ymin, xmin, ymax, xmax] normalised to 0-1000, "label": short specific name}."""


def detect(frames: np.ndarray, keyframes: list[int], workers=8) -> dict:
    """{frame index: [{"box": [x0, y0, x1, y1] px in the video frame, "label": str}]}, cached per frame."""
    path = work_dir() / "detections.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    todo = [k for k in keyframes if str(k) not in cache]
    if todo:
        client = gemini_client()
        Hh, Ww = frames.shape[1:3]

        def one(k):
            dets = gemini_json(client, GEMINI_DET_MODEL, [_jpeg(frames[k]), DETECT_PROMPT])
            out = []
            for d in dets if isinstance(dets, list) else []:
                try:
                    y0, x0, y1, x1 = [float(v) for v in d["box_2d"]]
                except (KeyError, TypeError, ValueError):
                    continue
                box = [x0 / 1000 * Ww, y0 / 1000 * Hh, x1 / 1000 * Ww, y1 / 1000 * Hh]
                if box[2] - box[0] > 8 and box[3] - box[1] > 8:
                    out.append({"box": [round(v, 1) for v in box], "label": str(d.get("label", ""))[:60]})
            return k, out

        t0 = time.time()
        with cf.ThreadPoolExecutor(workers) as ex:
            for n, fut in enumerate(cf.as_completed([ex.submit(one, k) for k in todo])):
                try:
                    k, out = fut.result()
                except Exception as e:  # one failed frame must not kill the run
                    print("  detect failed:", e)
                    continue
                cache[str(k)] = out
                if n % 10 == 0:
                    path.write_text(json.dumps(cache))
                    print(f"  detect {n + 1}/{len(todo)}  {time.time() - t0:.0f}s")
        path.write_text(json.dumps(cache))
    return {int(k): v for k, v in cache.items() if int(k) in set(keyframes)}


# ---------------------------------------------------------------- 2. masks (SAM 2.1, box prompts)

SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CKPT = ROOT / "models" / "sam2" / "sam2.1_hiera_large.pt"


def sam_masks(frames: np.ndarray, dets: dict, out_hw: tuple[int, int]) -> dict:
    """{frame: bool array (n_det, H, W)} at the point-map resolution, cached in masks.npz."""
    path = work_dir() / "masks.npz"
    H, W = out_hw
    sig = json.dumps({str(k): [d["box"] for d in v] for k, v in sorted(dets.items())})
    if path.exists():
        z = np.load(path, allow_pickle=False)
        if str(z["sig"]) == sig:
            return {int(k[1:]): np.unpackbits(z[k], axis=-1)[..., :W].astype(bool) for k in z.files if k.startswith("f")}
    import torch
    sys.path.insert(0, str(ROOT / "third_party" / "sam2"))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    pred = SAM2ImagePredictor(build_sam2(SAM2_CFG, str(SAM2_CKPT), device="cuda"))
    out = {}
    t0 = time.time()
    for k, ds in sorted(dets.items()):
        if not ds:
            continue
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred.set_image(np.ascontiguousarray(frames[k][:, :, ::-1]))
            m, score, _ = pred.predict(box=np.array([d["box"] for d in ds], np.float32), multimask_output=False)
        m = np.asarray(m).reshape(len(ds), *frames.shape[1:3]).astype(np.float32)
        small = np.stack([cv2.resize(x, (W, H), interpolation=cv2.INTER_AREA) > 0.5 for x in m])
        out[k] = small
    print(f"  SAM 2.1: {sum(len(v) for v in out.values())} masks on {len(out)} frames in {time.time() - t0:.0f}s")
    del pred
    torch.cuda.empty_cache()
    np.savez_compressed(path, sig=np.array(sig), **{f"f{k}": np.packbits(v, axis=-1) for k, v in out.items()})
    return out


# ---------------------------------------------------------------- 3-4. lift + fuse

VOX = 0.04          # voting voxel (m)
FLOOR_Z = 0.05      # points this close to the floor belong to the floor, never to an object


def _vkey(P, vox=VOX):
    q = np.floor(np.asarray(P) / vox).astype(np.int64) + 4096
    return q[:, 0] | (q[:, 1] << 13) | (q[:, 2] << 26)


def _main_depth_cluster(d, gap=0.25):
    """Indices of the largest run of depths without a gap > `gap` (drops background seen past an edge)."""
    o = np.argsort(d)
    breaks = np.flatnonzero(np.diff(d[o]) > gap)
    runs = np.split(o, breaks + 1)
    return max(runs, key=len)


_NB = np.array([dx + (dy << 13) + (dz << 26) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)], np.int64)


def lift_masks(Pm, valid, masks, dets, cams, ceiling):
    """Per detection: the compact voxel ids it covers (and their 1-voxel dilation); per frame the voxel ids
    it observes (for the visibility-normalised vote)."""
    S, H, W, _ = Pm.shape
    valid = valid & (Pm[..., 2] > FLOOR_Z) & (Pm[..., 2] < ceiling - 0.1)
    seen_keys = [np.unique(_vkey(Pm[f][valid[f]])) for f in range(S)]
    allk = np.unique(np.concatenate(seen_keys))
    seen = [np.searchsorted(allk, s) for s in seen_keys]
    recs = []
    er = np.ones((3, 3), np.uint8)
    for f, ms in sorted(masks.items()):
        for j, m in enumerate(ms):
            mm = cv2.erode(m.astype(np.uint8), er).astype(bool) & valid[f]
            if mm.sum() < 40:
                continue
            P = Pm[f][mm]
            d = np.linalg.norm(P - cams[f], axis=1)
            P = P[_main_depth_cluster(d)]
            if len(P) < 40:
                continue
            keys, cnt = np.unique(_vkey(P), return_counts=True)
            if (cnt >= 2).sum() > 10:
                keys = keys[cnt >= 2]
            ids = np.searchsorted(allk, keys)
            nb = np.unique((keys[:, None] + _NB[None]).ravel())
            pos = np.clip(np.searchsorted(allk, nb), 0, len(allk) - 1)
            dil = pos[allk[pos] == nb]
            recs.append({"frame": f, "det": j, "label": dets[f][j]["label"], "vox": ids, "dil": dil,
                         "area": int(m.sum()), "box": dets[f][j]["box"], "depth": float(np.median(d))})
    return recs, allk, seen


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def fuse(recs, n_vox, seen, S, ov_thr=0.3, min_hits=1):
    """Masks -> instances (union-find on dilated 3D voxel overlap), then each voxel voted to <= 1 instance."""
    from scipy import sparse
    n = len(recs)
    rows = np.concatenate([np.full(len(r["vox"]), i) for i, r in enumerate(recs)])
    cols = np.concatenate([r["vox"] for r in recs])
    M = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, n_vox))
    drows = np.concatenate([np.full(len(r["dil"]), i) for i, r in enumerate(recs)])
    dcols = np.concatenate([r["dil"] for r in recs])
    MD = sparse.csr_matrix((np.ones(len(drows), np.float32), (drows, dcols)), shape=(n, n_vox))
    near = (M @ MD.T).tocsr()                       # near[i, j] = |V_i within one voxel of V_j|
    size = np.asarray(M.sum(1)).ravel()
    nt = near.T.tocsr()
    c = near.tocoo()
    cand = []
    for a, b, v in zip(c.row, c.col, c.data):
        if a < b and recs[a]["frame"] != recs[b]["frame"]:
            ov = (v + nt[a, b]) / (size[a] + size[b])
            if ov > ov_thr:
                cand.append((ov, a, b))
    cand.sort(reverse=True)
    # constrained agglomeration: two masks that Gemini drew as *different* boxes in the same frame are
    # different objects, so their components may never merge (this is what keeps side-by-side machines apart)
    uf = _UF(n)
    members = {i: {recs[i]["frame"]: [recs[i]["box"]]} for i in range(n)}

    def box_iou(p, q):
        ix = max(0.0, min(p[2], q[2]) - max(p[0], q[0]))
        iy = max(0.0, min(p[3], q[3]) - max(p[1], q[1]))
        inter = ix * iy
        return inter / ((p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - inter + 1e-9)

    for ov, a, b in cand:
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb:
            continue
        ma, mb = members[ra], members[rb]
        if len(ma) > len(mb):
            ra, rb, ma, mb = rb, ra, mb, ma
        if any(box_iou(p, q) < 0.5 for f, bs in ma.items() if f in mb for p in bs for q in mb[f]):
            continue
        uf.p[ra] = rb
        for f, bs in ma.items():
            mb.setdefault(f, []).extend(bs)
        del members[ra]
    roots = sorted({uf.find(i) for i in range(n)})
    rid = {r: k for k, r in enumerate(roots)}
    inst = np.array([rid[uf.find(i)] for i in range(n)])
    K = len(roots)
    A = sparse.csr_matrix((np.ones(n), (inst, np.arange(n))), shape=(K, n))
    Hh = (A @ M).tocsr()                                           # hits per instance x voxel
    fr = np.array([r["frame"] for r in recs])
    B = sparse.csr_matrix((np.ones(n), (inst, fr)), shape=(K, S))
    B.data[:] = 1.0                                                # a frame counts once per instance
    B.sum_duplicates()
    B.data[:] = 1.0
    srows = np.concatenate([np.full(len(s), f) for f, s in enumerate(seen)])
    scols = np.concatenate(seen)
    Seen = sparse.csr_matrix((np.ones(len(srows), np.float32), (srows, scols)), shape=(S, n_vox))
    Vis = (B @ Seen).tocsr()
    Hc = Hh.tocoo()
    vis = np.asarray(Vis[Hc.row, Hc.col]).ravel()
    score = Hc.data / np.maximum(vis, 1)
    ok = (Hc.data >= min_hits) & (score >= 0.5)
    k, v = Hc.row[ok], Hc.col[ok]
    sc = score[ok] + 0.01 * np.minimum(Hc.data[ok], 20)
    o = np.lexsort((-sc, v))                                        # per voxel, best score first
    v, k = v[o], k[o]
    first = np.r_[True, v[1:] != v[:-1]]
    lab = np.zeros(n_vox, np.int32)
    lab[v[first]] = k[first] + 1
    return inst, lab


# ---------------------------------------------------------------- 4b. instances -> clean objects

KEYWORDS = [  # Gemini's free-text detector label -> coarse category (used for merging + as a labelling hint)
    ("robot", "robot"), ("conveyor", "conveyor"), ("drum", "tank"), ("barrel", "tank"), ("tank", "tank"),
    ("cart", "cart"), ("dolly", "cart"), ("trolley", "cart"), ("cabinet", "cabinet"), ("locker", "cabinet"),
    ("rack", "rack"), ("shelv", "rack"), ("shelf", "rack"), ("bench", "bench"), ("table", "bench"),
    ("stand", "bench"), ("desk", "bench"),
]


def coarse_category(label: str) -> str:
    s = label.lower()
    for k, c in KEYWORDS:
        if k in s:
            return c
    return "machine"


def point_voxel_labels(P, allk, lab):
    k = _vkey(P)
    i = np.clip(np.searchsorted(allk, k), 0, len(allk) - 1)
    pl = np.where(allk[i] == k, lab[i], 0)
    pl[P[:, 2] < FLOOR_Z] = 0
    return pl


def wall_mask(P, walls, margin=0.12):
    """Points on (or just in front of) the fitted wall faces: never part of an object."""
    m = np.zeros(len(P), bool)
    for w in walls:
        a, b = np.asarray(w["start"], float), np.asarray(w["end"], float)
        d = b - a
        L2 = float(d @ d)
        if L2 < 1e-6:
            continue
        t = np.clip(((P[:, :2] - a) @ d) / L2, 0, 1)
        m |= np.linalg.norm(P[:, :2] - (a + t[:, None] * d), axis=1) < w["thickness"] / 2 + margin
    return m


def _grid_cc(P, cell=0.04, dilate=1):
    """Connected components of points on a voxel grid (26-connected, after `dilate` voxel dilation)."""
    q = np.floor(P / cell).astype(np.int64)
    q -= q.min(0)
    shape = q.max(0) + 1
    occ = np.zeros(shape, bool)
    occ[q[:, 0], q[:, 1], q[:, 2]] = True
    big = ndimage.binary_dilation(occ, iterations=dilate) if dilate else occ
    lab, n = ndimage.label(big, structure=np.ones((3, 3, 3)))
    return lab[q[:, 0], q[:, 1], q[:, 2]], n


def obb(P, snap_deg=12.0, lo_pct=1.0, hi_pct=99.0):
    """Oriented floor box of points: centre (x, y), size (length >= depth, height), yaw (deg, (-90, 90])."""
    xy = P[:, :2].astype(np.float32)
    sub = np.ascontiguousarray(xy[:: max(1, len(xy) // 20000)])
    (cx, cy), (sx, sy), ang = cv2.minAreaRect(sub)
    if sx < sy:
        ang += 90
    ang = ((ang + 90) % 180) - 90
    snap = round(ang / 90) * 90
    if abs(ang - snap) < snap_deg:                   # plant is set out square to the building
        ang = snap
    c, s = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    u = (P[:, 0] - cx) * c + (P[:, 1] - cy) * s
    v = -(P[:, 0] - cx) * s + (P[:, 1] - cy) * c
    (u0, u1), (v0, v1) = np.percentile(u, [lo_pct, hi_pct]), np.percentile(v, [lo_pct, hi_pct])
    if u1 - u0 < v1 - v0:                            # keep +x along the longer side
        ang = ang + 90 if ang <= 0 else ang - 90
        c, s = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        u, v = (P[:, 0] - cx) * c + (P[:, 1] - cy) * s, -(P[:, 0] - cx) * s + (P[:, 1] - cy) * c
        (u0, u1), (v0, v1) = np.percentile(u, [lo_pct, hi_pct]), np.percentile(v, [lo_pct, hi_pct])
    cx, cy = cx + (u0 + u1) / 2 * c - (v0 + v1) / 2 * s, cy + (u0 + u1) / 2 * s + (v0 + v1) / 2 * c
    h = float(np.percentile(P[:, 2], 99.5))
    return [float(cx), float(cy)], [float(u1 - u0), float(v1 - v0), h], float(ang)


def obb_corners(pos, size, yaw):
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    hx, hy = size[0] / 2, size[1] / 2
    return np.array([[pos[0] + x * c - y * s, pos[1] + x * s + y * c]
                     for x, y in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))], np.float32)


def _overlap_area(a, b):
    area, _ = cv2.intersectConvexConvex(a, b)
    return float(max(area, 0.0))


def clean_instances(P, pl, recs, inst, walls, min_points=1500, min_masks=2):
    """Point labels -> cleaned, merged, grown, renumbered labels + per-object stats."""
    pl = pl.copy()
    pl[wall_mask(P, walls)] = 0
    labels_by_inst = {}
    for r, k in zip(recs, inst):
        labels_by_inst.setdefault(k + 1, []).append(r)
    # 1. keep the main connected body of each instance (drops stray voxels voted from far masks)
    for k in np.unique(pl[pl > 0]):
        idx = np.flatnonzero(pl == k)
        cc, n = _grid_cc(P[idx], 0.04, 1)
        if n > 1:
            sizes = np.bincount(cc)
            sizes[0] = 0
            keep = sizes >= max(0.15 * sizes.max(), 200)
            pl[idx[~keep[cc]]] = 0
    # 2. merge fragments of one physical object: touching + same coarse category, or a tiny/sparsely
    #    observed fragment touching a bigger one
    def stats():
        ks = np.unique(pl[pl > 0])
        return ks, {k: int((pl == k).sum()) for k in ks}
    changed = True
    while changed:
        changed = False
        ks, cnt = stats()
        if len(ks) < 2:
            break
        q = np.floor(P[pl > 0] / 0.04).astype(np.int64)
        lk = pl[pl > 0]
        key = (q[:, 0] + 4096) | ((q[:, 1] + 4096) << 13) | ((q[:, 2] + 4096) << 26)
        vk, vi = np.unique(key * 4096 + lk, return_index=True)   # (voxel, label) pairs
        vox, vl = vk // 4096, vk % 4096
        nbv = (vox[:, None] + _NB[None]).ravel()
        nbl = np.repeat(vl, len(_NB))
        o = np.argsort(vox)
        vs, ls = vox[o], vl[o]
        pos = np.searchsorted(vs, nbv)
        contact = {}
        for off in range(4):                                       # a voxel can carry a few labels
            p2 = np.clip(pos + off, 0, len(vs) - 1)
            hit = (vs[p2] == nbv) & (ls[p2] != nbl)
            for a, b in zip(nbl[hit], ls[p2][hit]):
                contact[(int(a), int(b))] = contact.get((int(a), int(b)), 0) + 1
        nvox = {k: int((vl == k).sum()) for k in ks}
        best = None
        for (a, b), c in contact.items():
            small, big = (a, b) if nvox[a] <= nvox[b] else (b, a)
            ratio = c / 27 / max(nvox[small], 1)
            ra, rb = labels_by_inst.get(small, []), labels_by_inst.get(big, [])
            ca = max({coarse_category(r["label"]) for r in ra} or {"machine"}, key=lambda x: sum(coarse_category(r["label"]) == x for r in ra))
            cb = max({coarse_category(r["label"]) for r in rb} or {"machine"}, key=lambda x: sum(coarse_category(r["label"]) == x for r in rb))
            weak = len(ra) < min_masks or cnt[small] < min_points
            if (ratio > 0.25 and ca == cb) or (weak and ratio > 0.1):
                if best is None or ratio > best[0]:
                    best = (ratio, small, big)
        if best:
            _, small, big = best
            pl[pl == small] = big
            labels_by_inst.setdefault(big, []).extend(labels_by_inst.pop(small, []))
            changed = True
    # 3. drop what is still too small / too sparsely observed to be a real object
    ks, cnt = stats()
    for k in ks:
        q = P[pl == k]
        ext = q.max(0) - q.min(0)
        if cnt[k] < min_points or len(labels_by_inst.get(k, [])) < min_masks or np.percentile(q[:, 2], 99) < 0.25 \
                or max(ext[0], ext[1]) < 0.25:
            pl[pl == k] = 0
    # 4. grow into unlabelled, non-floor, non-wall points touching the object inside its box (completes
    #    surfaces the masks only half covered: tops, legs, edges)
    free = (pl == 0) & (P[:, 2] > FLOOR_Z) & ~wall_mask(P, walls)
    from scipy.spatial import cKDTree
    for _ in range(4):
        src = np.flatnonzero(pl > 0)
        cand = np.flatnonzero(free & (pl == 0))
        if not len(cand):
            break
        tree = cKDTree(P[src])
        d, j = tree.query(P[cand], k=1, distance_upper_bound=0.03)
        ok = np.isfinite(d)
        if not ok.any():
            break
        newl = pl[src[j[ok]]]
        boxes = {}
        for k in np.unique(newl):
            q = P[pl == k]
            boxes[k] = (q.min(0) - 0.05, q.max(0) + 0.05)
        sel = cand[ok]
        inb = np.array([np.all((P[i] >= boxes[k][0]) & (P[i] <= boxes[k][1])) for i, k in zip(sel, newl)]) \
            if len(sel) < 200000 else np.ones(len(sel), bool)
        pl[sel[inb]] = newl[inb]
    # renumber by floor area (descending)
    ks = [k for k in np.unique(pl[pl > 0])]
    info = []
    for k in ks:
        q = P[pl == k]
        pos, size, yaw = obb(q)
        info.append((size[0] * size[1], k, pos, size, yaw))
    info.sort(key=lambda x: -x[0])
    remap = np.zeros(pl.max() + 1, np.int32)
    objs = []
    for n, (_, k, pos, size, yaw) in enumerate(info, 1):
        remap[k] = n
        objs.append({"index": n, "src": int(k), "pos": pos, "size": size, "yaw_deg": yaw,
                     "dets": labels_by_inst.get(k, [])})
    return remap[pl], objs


# ---------------------------------------------------------------- 5. labelling (Gemini vision)

LABEL_PROMPT = """You are looking at one physical object on the floor of a machine shop, segmented from a walkthrough video.
Images 1-{n} are crops of the object from different views; the last image is a full frame with the object outlined in green.
A detector called it: {hint}.
Identify THIS outlined object (not its neighbours). Return strict JSON:
{{"label": specific name; include make/model ONLY if the text is clearly legible in the images (e.g. "Haas SL-30 CNC lathe", "Fadal VMC 4020A vertical machining centre"), otherwise a generic specific name such as "Manual engine lathe" or "Mobile tool-holder cart",
 "category": one of machine|robot|rack|bench|tank|conveyor|cart|cabinet|other,
 "function": one plain sentence (max 25 words) on what it is for / what it does on a shop floor,
 "confidence": 0..1}}"""


def best_views(obj, frames_shape, k=3, gap=4):
    Hh, Ww = frames_shape[1:3]
    scored = []
    for r in obj["dets"]:
        x0, y0, x1, y1 = r["box"]
        border = (x0 < 4) + (y0 < 4) + (x1 > Ww - 4) + (y1 > Hh - 4)
        scored.append((r["area"] * (0.55 ** border), r))
    scored.sort(key=lambda x: -x[0])
    out = []
    for _, r in scored:
        if all(abs(r["frame"] - q["frame"]) >= gap for q in out):
            out.append(r)
        if len(out) == k:
            break
    return out


def crop(frame, box, pad=0.15, size=512):
    Hh, Ww = frame.shape[:2]
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    x0, x1 = int(max(0, x0 - pad * w)), int(min(Ww, x1 + pad * w))
    y0, y1 = int(max(0, y0 - pad * h)), int(min(Hh, y1 + pad * h))
    c = frame[y0:y1, x0:x1]
    s = size / max(c.shape[:2])
    return cv2.resize(c, (max(1, int(c.shape[1] * s)), max(1, int(c.shape[0] * s))), interpolation=cv2.INTER_AREA)


def outlined(frame, mask_small, box):
    img = frame.copy()
    if mask_small is not None:
        m = cv2.resize(mask_small.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cs, -1, (0, 255, 0), 3)
    else:
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 3)
    return img


def label_objects(objs, frames, masks, workers=6) -> dict:
    """{src instance id (str): {label, category, function, confidence}} via Gemini; cached in labels_gemini.json."""
    from collections import Counter
    path = work_dir() / "labels_gemini.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    client = gemini_client()

    def one(o):
        views = o["views_recs"]
        hint = ", ".join(f"{l} ({n}x)" for l, n in Counter(r["label"] for r in o["dets"]).most_common(3))
        parts = [_jpeg(crop(frames[r["frame"]], r["box"])) for r in views]
        r0 = views[0]
        ms = masks.get(r0["frame"])
        parts.append(_jpeg(cv2.resize(outlined(frames[r0["frame"]], ms[r0["det"]] if ms is not None else None, r0["box"]),
                                      (960, 540)), 85))
        parts.append(LABEL_PROMPT.format(n=len(views), hint=hint))
        res = gemini_json(client, GEMINI_LABEL_MODEL, parts)
        if isinstance(res, list):
            res = res[0]
        return o["source_key"], res

    todo = [o for o in objs if o["source_key"] not in cache]
    with cf.ThreadPoolExecutor(workers) as ex:
        for fut in cf.as_completed([ex.submit(one, o) for o in todo]):
            try:
                k, res = fut.result()
                cache[k] = res
            except Exception as e:
                print("  label failed:", e)
    path.write_text(json.dumps(cache, indent=1))
    return cache


def preview_montage(o, frames, masks, out_path):
    tiles = []
    for r in o["views_recs"]:
        ms = masks.get(r["frame"])
        img = outlined(frames[r["frame"]], ms[r["det"]] if ms is not None else None, r["box"])
        tiles.append(cv2.resize(crop(img, r["box"], 0.25, 400), (400, 300)))
    while len(tiles) < 3:
        tiles.append(np.zeros((300, 400, 3), np.uint8))
    img = np.hstack(tiles)
    bar = np.full((60, img.shape[1], 3), 30, np.uint8)
    cv2.putText(bar, f"{o['key']}  {o['label']}"[:80], (8, 24), 0, 0.7, (255, 255, 255), 2)
    cv2.putText(bar, o.get("function", "")[:120], (8, 50), 0, 0.45, (200, 200, 200), 1)
    cv2.imwrite(str(out_path), np.vstack([bar, img]), [cv2.IMWRITE_JPEG_QUALITY, 85])


# ---------------------------------------------------------------- driver

def read_cloud():
    meta = json.loads((scan_dir() / "meta.json").read_text())
    n = meta["n"]
    raw = np.fromfile(scan_dir() / "cloud.bin", dtype=np.uint8)
    return meta, raw[: n * 12].view(np.float32).reshape(n, 3), raw[n * 12:].reshape(n, 3)


def _shrink_overlaps(objs, P, pl, thr=0.3):
    """Boxes of distinct objects must not grossly overlap (clash/egress/physics use them): tighten the
    extents of an overlapping pair to robust 5-95 % percentiles."""
    for a in objs:
        for b in objs:
            if a["index"] >= b["index"]:
                continue
            ca, cb = obb_corners(a["pos"], a["size"], a["yaw_deg"]), obb_corners(b["pos"], b["size"], b["yaw_deg"])
            inter = _overlap_area(ca, cb)
            small = min(a["size"][0] * a["size"][1], b["size"][0] * b["size"][1])
            if small > 0 and inter / small > thr:
                o = min((a, b), key=lambda x: x["size"][0] * x["size"][1])      # tighten the smaller one
                if not o.get("tight"):
                    o["pos"], o["size"], o["yaw_deg"] = obb(P[pl == o["index"]], lo_pct=5, hi_pct=95)
                    o["tight"] = True


def run(labels_from: str | None = None, write_previews=True) -> dict:
    T = {}
    t0 = time.time()
    meta, P, col = read_cloud()
    z = np.load(scan_dir() / "raw_predictions.npz")
    S, H, W = z["points"].shape[:3]
    frames = video_frames(n=S)
    T["frames"] = time.time() - t0
    dets = detect(frames, list(range(S)))
    T["detect"] = time.time() - t0 - sum(T.values())
    masks = sam_masks(frames, dets, (H, W))
    T["sam"] = time.time() - t0 - sum(T.values())
    Pm, keep, nc = fit.model_pixels(z)
    valid = (nc >= 1) & (z["conf"] > 1.2)
    cams = np.array(meta["cameras"])[:, :3]
    recs, allk, seen = lift_masks(Pm, valid, masks, dets, cams, meta["ceiling_height"])
    del Pm, keep, nc, valid
    inst, lab = fuse(recs, len(allk), seen, S)
    T["lift+fuse"] = time.time() - t0 - sum(T.values())
    model, _ = fit.fit_shell()
    walls = [e for e in model["elements"].values() if e["kind"] == "wall"]
    pl = point_voxel_labels(P, allk, lab)
    pl, objs = clean_instances(P, pl, recs, inst, walls)
    for o in objs:
        o["source_key"] = f"seg_{o['index']:02d}"
    # human review (objects_work/review.json, keyed by the pre-merge seg_NN keys):
    #   {"merge": {"seg_44": "seg_05", ...}, "labels": {"seg_11": {"label": ..., "label_source": ...}}}
    rp = work_dir() / "review.json"
    review = json.loads(rp.read_text()) if rp.exists() else {}
    by_src = {o["source_key"]: o for o in objs}
    for a_, b_ in review.get("merge", {}).items():
        while b_ in review.get("merge", {}):
            b_ = review["merge"][b_]
        if a_ in by_src and b_ in by_src:
            pl[pl == by_src[a_]["index"]] = by_src[b_]["index"]
            by_src[b_]["dets"] = by_src[b_]["dets"] + by_src[a_]["dets"]
            by_src[b_].setdefault("merged", []).append(a_)
            objs.remove(by_src.pop(a_))
    for o in objs:                                   # refit boxes, renumber by floor area
        o["pos"], o["size"], o["yaw_deg"] = obb(P[pl == o["index"]])
    objs.sort(key=lambda o: -o["size"][0] * o["size"][1])
    remap = np.zeros(int(pl.max()) + 1, np.int32)
    for n, o in enumerate(objs, 1):
        remap[o["index"]] = n
        o["index"] = n
    pl = remap[pl]
    _shrink_overlaps(objs, P, pl)
    T["clean"] = time.time() - t0 - sum(T.values())
    for o in objs:
        o["key"] = f"obj_{o['index']:02d}"
        o["views_recs"] = best_views(o, frames.shape)

    # labels: Gemini (or a file for offline re-runs), then the human review overrides
    if labels_from:
        src = json.loads(Path(labels_from).read_text())
        if isinstance(src, dict) and "objects" in src:            # an objects.json: match on the seg key
            src = {o.get("source_key", o["key"]): o for o in src["objects"]}
        labels = {k: {f: v.get(f) for f in ("label", "category", "function", "confidence", "label_source")}
                  for k, v in src.items()}
    else:
        labels = label_objects(objs, frames, masks)
    T["label"] = time.time() - t0 - sum(T.values())
    hero = set()
    out = []
    for o in objs:
        lb = dict(labels.get(o["source_key"], {}))
        source = lb.pop("label_source", None) or "gemini"
        if not lb.get("label"):
            from collections import Counter
            lb = {"label": Counter(r["label"] for r in o["dets"]).most_common(1)[0][0], "category": "other",
                  "function": "", "confidence": 0.3}
            source = "detector"
        rv = review.get("labels", {}).get(o["source_key"])
        if rv:
            lb.update({k: v for k, v in rv.items() if k in ("label", "category", "function", "confidence")})
            source = rv.get("label_source", "gemini+reviewed")
        cat = lb.get("category") if lb.get("category") in CATEGORIES else "other"
        m = pl == o["index"]
        q = P[m]
        rgb = np.median(col[m], axis=0).astype(int)
        rec = {
            "key": o["key"], "index": o["index"], "label": str(lb["label"]), "category": cat,
            "function": str(lb.get("function", "")), "confidence": round(float(lb.get("confidence") or 0.5), 2),
            "label_source": source,
            "pos": [round(o["pos"][0], 3), round(o["pos"][1], 3)],
            "size": [round(v, 3) for v in o["size"]], "yaw_deg": round(o["yaw_deg"], 1),
            "color": "#%02x%02x%02x" % tuple(int(x) for x in rgb), "points": int(m.sum()),
            "point_count": int(m.sum()),
            "bbox_min": [round(float(v), 3) for v in q.min(0)], "bbox_max": [round(float(v), 3) for v in q.max(0)],
            "mesh": o["key"] if o["key"] in hero else None, "hero": o["key"] in hero,
            "views": [int(r["frame"]) for r in o["views_recs"]], "source_key": o["source_key"],
            **({"merged_from": o["merged"]} if o.get("merged") else {}),
            "detector_votes": len(o["dets"]),
        }
        out.append(rec)
        o.update(rec)
    doc = {"version": 1, "glb_up": "Y", "frame": "model (frame.json), metres, Z up",
           "segmentation": "Gemini (gemini-4-carbon) boxes on all keyframes -> SAM 2.1 masks -> lifted through "
                           "LingBot-Map points -> fused across frames by 3D voxel overlap (cannot-link: different "
                           "boxes in one frame) -> visibility-normalised voxel vote",
           "labels": "Gemini vision on the best 3 views per object, reviewed by hand where marked",
           "objects": out}
    sd = scan_dir()
    pl.astype(np.uint16).tofile(sd / "cloud_labels.bin")
    (sd / "objects.json").write_text(json.dumps(doc, indent=1))
    meta["objects"] = len(out)
    (sd / "meta.json").write_text(json.dumps(meta))
    if write_previews:
        pv = sd / "objects_preview"
        pv.mkdir(exist_ok=True)
        for o in objs:
            preview_montage(o, frames, masks, pv / f"{o['key']}.jpg")
    built = sorted(p.name[5:] for p in work_dir().glob("hero_*") if (p / "hero.json").exists())
    built = [k for k in built if any(o["key"] == k for o in out)]
    if built:                                       # keep hero models already built (capture_studio.hero)
        from . import hero as hero_mod
        hero_mod.write_outputs(built)
        doc = json.loads((sd / "objects.json").read_text())
    T["write"] = time.time() - t0 - sum(T.values())
    print("stage seconds:", {k: round(v, 1) for k, v in T.items()})
    return doc


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--labels-from", help="objects.json or {key: {label, category, function, confidence}} "
                                          "to reuse instead of calling Gemini")
    ap.add_argument("--no-previews", action="store_true")
    a = ap.parse_args()
    doc = run(a.labels_from, not a.no_previews)
    for o in doc["objects"]:
        print(f"{o['key']}  {o['label']:<45} {o['category']:<8} {o['points']:>7} pts  {o['label_source']}")


if __name__ == "__main__":
    main()


def transfer_labels(old_P, old_lab, new_P, radius=0.03):
    """Carry per-point object labels over to a re-built cloud without re-segmenting (object poses stay
    exactly as they were): each new point takes the label of the nearest old point within `radius`,
    floor points stay 0."""
    from scipy.spatial import cKDTree
    tree = cKDTree(old_P)
    d, j = tree.query(new_P, k=1, distance_upper_bound=radius)
    lab = np.zeros(len(new_P), np.uint16)
    ok = np.isfinite(d)
    lab[ok] = old_lab[j[ok]]
    lab[new_P[:, 2] < FLOOR_Z] = 0
    return lab


def refresh_after_cloud_change(old_P, old_lab):
    """After prepare_cloud() rebuilt cloud.bin: transfer labels, update per-object point stats
    (counts / AABB) in objects.json; positions, sizes, yaw, keys and labels are untouched."""
    meta, P, col = read_cloud()
    lab = transfer_labels(old_P, old_lab, P)
    sd = scan_dir()
    lab.tofile(sd / "cloud_labels.bin")
    doc = json.loads((sd / "objects.json").read_text())
    for o in doc["objects"]:
        m = lab == o["index"]
        if m.any():
            q = P[m]
            o["points"] = o["point_count"] = int(m.sum())
            o["bbox_min"] = [round(float(v), 3) for v in q.min(0)]
            o["bbox_max"] = [round(float(v), 3) for v in q.max(0)]
    (sd / "objects.json").write_text(json.dumps(doc, indent=1))
    meta["objects"] = len(doc["objects"])
    (sd / "meta.json").write_text(json.dumps(meta))
    return int((lab > 0).sum())
