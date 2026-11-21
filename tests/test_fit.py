"""Shell fitting against ground truth: a synthetic scan of a known two-room shell.

Points are sampled on every surface a walking camera could see (visibility is ray-tested
against the true walls, so doorways and windows let rays through exactly as in a real scan),
with 8 mm noise. The fitter must recover walls, thicknesses and openings within tolerance.
"""
import json
import math

import numpy as np
import pytest

from capture_studio import bim

H = 2.9
MACHINES = [(2.0, 3.6, 1.6, 0.9, 1.4), (5.6, 3.8, 1.2, 1.0, 2.0)]   # cx, cy, length, depth, height


def truth():
    m = bim.empty_model("truth")
    w = {}
    for name, a, b, t in (("S", [0, 0], [7, 0], 0.2), ("E", [7, 0], [7, 5], 0.2), ("N", [7, 5], [0, 5], 0.2),
                          ("W", [0, 5], [0, 0], 0.2), ("P", [4, 0], [4, 5], 0.15)):
        w[name] = bim.add(m, "wall", "scanned", name=name, start=a, end=b, thickness=t, height=H, material="masonry")
    bim.add(m, "door", "scanned", host=w["W"]["id"], offset=2.0, width=1.0, height=2.1, sill=0.0)      # entrance
    bim.add(m, "door", "scanned", host=w["P"]["id"], offset=1.5, width=0.9, height=2.1, sill=0.0)      # between rooms
    bim.add(m, "window", "scanned", host=w["S"]["id"], offset=1.2, width=1.6, height=1.3, sill=0.9)
    return m


def _blocked(m, C, P):
    """True where segment C->P passes through any wall solid (shrunk a little so surface points survive)."""
    hit = np.zeros(len(P), bool)
    for e in m["elements"].values():
        if e["kind"] != "wall":
            continue
        ps = bim.pose(m, e)
        c, s = math.cos(ps["yaw"]), math.sin(ps["yaw"])
        R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        o, d = (C - ps["pos"]) @ R.T, (P - C) @ R.T
        for b in bim.solids(m, e):
            lo = np.array(b["c"]) - np.array(b["s"]) / 2 + 0.02
            hi = np.array(b["c"]) + np.array(b["s"]) / 2 - 0.02
            with np.errstate(divide="ignore", invalid="ignore"):
                t1, t2 = (lo - o) / d, (hi - o) / d
            tn = np.nanmax(np.minimum(t1, t2), axis=1)
            tf = np.nanmin(np.maximum(t1, t2), axis=1)
            hit |= (tn < tf) & (tf > 0) & (tn < 0.995)
    return hit


def synth(tmp):
    rng = np.random.default_rng(0)
    m = truth()
    cams = np.array([[x, y, 1.5] for x, y in
                     [(0.5, 2.5), (0.8, 2.5), (1.5, 1.5), (2.5, 2.5), (3.2, 3.8), (3.3, 2.0), (3.4, 1.0),
                      (4.6, 2.0), (5.5, 2.5), (6.2, 4.0), (5.5, 1.0), (2.0, 4.0), (1.0, 1.0)]])
    pts = []
    g = 0.025
    for e in m["elements"].values():           # both faces + reveals of every wall solid
        if e["kind"] != "wall":
            continue
        ps = bim.pose(m, e)
        c, s = math.cos(ps["yaw"]), math.sin(ps["yaw"])
        for b in bim.solids(m, e):
            u = np.arange(b["c"][0] - b["s"][0] / 2, b["c"][0] + b["s"][0] / 2, g)
            z = np.arange(b["c"][2] - b["s"][2] / 2, b["c"][2] + b["s"][2] / 2, g)
            U, Z = np.meshgrid(u, z)
            for side in (-1, 1):
                loc = np.stack([U.ravel(), np.full(U.size, side * b["s"][1] / 2), Z.ravel()], 1)
                pts.append(np.stack([ps["pos"][0] + loc[:, 0] * c - loc[:, 1] * s,
                                     ps["pos"][1] + loc[:, 0] * s + loc[:, 1] * c, loc[:, 2]], 1))
    for (cx, cy, sx, sy, h) in MACHINES:                                         # box "machines": 4 sides + top
        for u in np.arange(-sx / 2, sx / 2, g):
            for side in (-sy / 2, sy / 2):
                z = np.arange(0, h, g)
                pts.append(np.stack([np.full(z.size, cx + u), np.full(z.size, cy + side), z], 1))
        for v in np.arange(-sy / 2, sy / 2, g):
            for side in (-sx / 2, sx / 2):
                z = np.arange(0, h, g)
                pts.append(np.stack([np.full(z.size, cx + side), np.full(z.size, cy + v), z], 1))
        U, V = np.meshgrid(np.arange(-sx / 2, sx / 2, g), np.arange(-sy / 2, sy / 2, g))
        pts.append(np.stack([cx + U.ravel(), cy + V.ravel(), np.full(U.size, h)], 1))
    X, Y = np.meshgrid(np.arange(0.1, 6.9, g * 1.5), np.arange(0.1, 4.9, g * 1.5))
    pts.append(np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1))            # floor
    pts.append(np.stack([X.ravel(), Y.ravel(), np.full(X.size, H)], 1))          # ceiling
    X, Z = np.meshgrid(np.arange(-1, 8, g * 2), np.arange(0, 4, g * 2))
    pts.append(np.stack([X.ravel(), np.full(X.size, -3.0), Z.ravel()], 1))       # building across the street
    Y, Z = np.meshgrid(np.arange(0, 5, g * 2), np.arange(0, 2.5, g * 2))
    pts.append(np.stack([np.full(Y.size, -1.6), Y.ravel(), Z.ravel()], 1))       # lobby wall outside the entrance
    X, Y = np.meshgrid(np.arange(-1.6, 0, g * 1.5), np.arange(0, 5, g * 1.5))
    pts.append(np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1))            # lobby floor
    pts.append(np.stack([X.ravel(), Y.ravel(), np.full(X.size, 2.5)], 1))        # lobby ceiling
    P = np.concatenate(pts)
    keep = np.zeros(len(P), bool)
    cam = np.zeros(len(P), int)
    for _ in range(6):                                                           # try a few cameras per point
        todo = np.flatnonzero(~keep)
        k = rng.integers(0, len(cams), len(todo))
        ok = ~_blocked(m, cams[k], P[todo])
        keep[todo[ok]] = True
        cam[todo[ok]] = k[ok]
    P, cam = P[keep], cam[keep]
    P = P + rng.normal(0, 0.008, P.shape)
    order = np.argsort(cam, kind="stable")
    scan = tmp / "scan"
    scan.mkdir(parents=True)
    np.savez(scan / "aligned.npz", P=P[order].astype(np.float32), frame=cam[order].astype(np.int32),
             C=cams.astype(np.float32), ceiling=H)
    (scan / "meta.json").write_text(json.dumps({"frames": len(cams), "n": len(P), "scale_source": "synthetic"}))
    return m


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    from capture_studio import fit
    tmp = tmp_path_factory.mktemp("data")
    synth(tmp)
    old = fit.SCAN
    fit.SCAN = tmp / "scan"
    try:
        model, report = fit.fit_shell()
    finally:
        fit.SCAN = old
    return model, report


def _wall_at(model, axis, c, tol=0.06):
    for e in model["elements"].values():
        if e["kind"] == "wall" and abs(e["start"][axis] - c) < tol and abs(e["end"][axis] - c) < tol:
            yield e


def test_walls_recovered(fitted):
    model, report = fitted
    for axis, c in ((0, 0.0), (0, 7.0), (0, 4.0), (1, 0.0), (1, 5.0)):
        assert list(_wall_at(model, axis, c)), f"no wall on axis {axis} at {c}: {report}"
    assert report["scan"]["fit_p50_mm"] < 12


def test_partition_thickness_from_two_faces(fitted):
    model, _ = fitted
    p = max(_wall_at(model, 0, 4.0), key=bim.wall_length)
    assert p["faces_seen"] == 2 and abs(p["thickness"] - 0.15) <= 0.03


def test_openings_recovered(fitted):
    model, _ = fitted
    ops = [e for e in model["elements"].values() if e["kind"] in bim.OPENING_KINDS]
    def near(kind, xy, w, tol=0.2):
        return [o for o in ops if o["kind"] == kind and math.dist(bim.opening_center(model, o), xy) < tol + 0.1
                and abs(o["width"] - w) < tol]
    assert near("door", [4.0, 1.95], 0.9), [bim.brief(model, o) for o in ops]
    assert near("door", [0.0, 2.5], 1.0), [bim.brief(model, o) for o in ops]
    win = near("window", [2.0, 0.0], 1.6)
    assert win and abs(win[0]["sill"] - 0.9) < 0.15 and abs(win[0]["height"] - 1.3) < 0.2
    assert any(o.get("exit") for o in ops)


def test_surfaces_seen_through_openings_are_not_walls(fitted):
    model, _ = fitted
    assert not list(_wall_at(model, 0, -1.6, 0.2)) and not list(_wall_at(model, 1, -3.0, 0.2))
    assert sum(e["kind"] == "wall" for e in model["elements"].values()) == 5


def test_equipment_recovered_as_boxes_not_walls(fitted):
    model, _ = fitted
    eq = [e for e in model["elements"].values() if e["kind"] == "equipment"]
    assert len(eq) == len(MACHINES), [bim.brief(model, e) for e in eq]
    for cx, cy, sx, sy, h in MACHINES:
        e = min(eq, key=lambda e: math.dist(e["pos"], (cx, cy)))
        assert math.dist(e["pos"], (cx, cy)) < 0.12
        assert abs(e["size"][0] - sx) < 0.15 and abs(e["size"][1] - sy) < 0.15 and abs(e["size"][2] - h) < 0.15
        assert e["provenance"] == "scanned" and e["color"].startswith("#")


OBJECTS = [  # what capture_studio.objects writes: segmented + named objects (one hangs past the east wall)
    {"key": "obj_01", "index": 1, "label": "Haas SL-30 CNC lathe", "category": "machine",
     "function": "Turns round metal parts.", "confidence": 0.9, "label_source": "gemini+reviewed",
     "pos": [2.0, 3.6], "size": [1.6, 0.9, 1.4], "yaw_deg": 0.0, "color": "#c0c0c0", "points": 5000,
     "mesh": "obj_01", "hero": True, "views": [1, 2]},
    {"key": "obj_02", "index": 2, "label": "Tool cart", "category": "cart", "function": "Holds tool holders.",
     "confidence": 0.8, "label_source": "gemini", "pos": [6.6, 3.8], "size": [1.2, 0.6, 1.0], "yaw_deg": 90.0,
     "color": "#cc2222", "points": 900, "mesh": None, "views": [3]},
    {"key": "obj_03", "index": 3, "label": "Car outside", "category": "other", "function": "", "confidence": 0.5,
     "label_source": "gemini", "pos": [5.0, -3.0], "size": [4.0, 1.8, 1.4], "yaw_deg": 0.0, "color": "#333333",
     "points": 100, "mesh": None, "views": [4]},
]


def test_objects_json_replaces_box_clustering(tmp_path):
    from capture_studio import fit
    synth(tmp_path)
    (tmp_path / "scan" / "objects.json").write_text(json.dumps({"version": 1, "objects": OBJECTS}))
    old = fit.SCAN
    fit.SCAN = tmp_path / "scan"
    try:
        model, report = fit.fit_shell()
    finally:
        fit.SCAN = old
    eq = {e["object_key"]: e for e in model["elements"].values() if e["kind"] == "equipment"}
    assert set(eq) == {"obj_01", "obj_02"}                         # outside the shell -> dropped
    lathe, cart = eq["obj_01"], eq["obj_02"]
    assert lathe["name"] == lathe["label"] == "Haas SL-30 CNC lathe" and lathe["category"] == "machine"
    assert lathe["function"] and lathe["mesh"] == "obj_01" and lathe["hero"] and lathe["confidence"] == 0.9
    assert lathe["material"] == "machine" and lathe["anchored"] and lathe["fixed"] and lathe["provenance"] == "scanned"
    assert lathe["origin"] == {"pos": lathe["pos"], "yaw_deg": lathe["yaw_deg"]}
    assert cart["category"] == "bench" and cart["object_category"] == "cart" and cart["mesh"] is None
    # the cart's box poked through the east wall face (x = 6.9): it is pushed back inside the room
    xs = [p[0] for p in bim.footprint(model, cart)]
    assert max(xs) <= 6.9 + 1e-6, xs
    assert "label" in bim.brief(model, lathe) and report["scan"]["objects"].startswith("segmented")
