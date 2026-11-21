"""BIM model: elements, geometry, spaces, quantities.

Conventions (borrowed from the Moving Atoms as-built packages): metres, Z-up, local scan
frame; every element has a stable id `<kind>_<hash>`, an IFC class, a provenance status
("scanned" = fitted to the point cloud, "proposed" = designed by agent/human) and, for
scanned elements, a fit residual against the scan (P50/P95 in mm).

Element geometry is parametric (wall = centreline + thickness + height, hosted openings…).
`solids(model, el)` turns any element into boxes in the element's local frame; the viewer,
MuJoCo and the exporters all consume those same boxes, so they can never disagree.
"""
from __future__ import annotations

import copy
import hashlib
import math
import time

IFC_CLASS = {
    "slab": "IfcSlab", "wall": "IfcWall", "door": "IfcDoor", "window": "IfcWindow",
    "void": "IfcOpeningElement", "column": "IfcColumn", "beam": "IfcBeam",
    "fixture": "IfcFurnishingElement", "space": "IfcSpace",
}

# density kg/m3, unit cost (USD per m3, rough order of magnitude for takeoff only)
MATERIALS = {
    "concrete": {"density": 2400, "cost_m3": 180, "color": "#9aa0a6"},
    "masonry": {"density": 1900, "cost_m3": 220, "color": "#b9b2a6"},
    "aac_block": {"density": 650, "cost_m3": 140, "color": "#d9d6cf"},
    "drywall": {"density": 250, "cost_m3": 310, "color": "#e8e6e1"},
    "timber": {"density": 500, "cost_m3": 650, "color": "#c49a6c"},
    "steel": {"density": 7850, "cost_m3": 9500, "color": "#5f6b7a"},
    "glass": {"density": 2500, "cost_m3": 3000, "color": "#9fd3e6"},
    "casework": {"density": 350, "cost_m3": 900, "color": "#c9b79c"},
}

OPENING_KINDS = ("door", "window", "void")


def new_id(kind: str, model: dict) -> str:
    seed = f"{kind}{time.time_ns()}{len(model['elements'])}".encode()
    return f"{kind}_{hashlib.sha1(seed).hexdigest()[:8]}"


def empty_model(name="Untitled site") -> dict:
    return {
        "project": {"name": name, "units": "m", "up": "Z", "code_profile": "NBC_INDIA_2016"},
        "revision": 0,
        "level": {"name": "L1", "elevation": 0.0, "ceiling_height": 2.9},
        "scan": {},
        "elements": {},
    }


def add(model: dict, kind: str, provenance="proposed", **props) -> dict:
    if kind not in IFC_CLASS:
        raise ValueError(f"unknown kind {kind!r}; one of {sorted(IFC_CLASS)}")
    el = {"id": new_id(kind, model), "kind": kind, "ifc_class": IFC_CLASS[kind],
          "provenance": provenance, **props}
    el.setdefault("name", f"{kind} {sum(e['kind'] == kind for e in model['elements'].values()) + 1}")
    validate_element(model, el)
    model["elements"][el["id"]] = el
    return el


def validate_element(model: dict, el: dict):
    k = el["kind"]
    if k == "wall":
        L = wall_length(el)
        if L < 0.1:
            raise ValueError("wall shorter than 0.1 m")
        if not 0.05 <= el["thickness"] <= 1.0:
            raise ValueError("wall thickness must be 0.05–1.0 m")
        if el["height"] <= 0:
            raise ValueError("wall height must be positive")
    if k in OPENING_KINDS:
        host = model["elements"].get(el["host"])
        if not host or host["kind"] != "wall":
            raise ValueError(f"host {el['host']!r} is not a wall")
        L = wall_length(host)
        if el["offset"] < 0 or el["offset"] + el["width"] > L + 1e-6:
            raise ValueError(f"opening [{el['offset']:.2f}, {el['offset'] + el['width']:.2f}] m "
                             f"does not fit in host wall of length {L:.2f} m")
        if el["sill"] < 0 or el["sill"] + el["height"] > host["height"] + 1e-6:
            raise ValueError("opening exceeds host wall height")
        for o in openings_of(model, host["id"]):
            if o["id"] != el.get("id") and el["offset"] < o["offset"] + o["width"] and \
                    o["offset"] < el["offset"] + el["width"] and \
                    el["sill"] < o["sill"] + o["height"] and o["sill"] < el["sill"] + el["height"]:
                raise ValueError(f"opening overlaps {o['id']}")
    if "material" in el and el["material"] not in MATERIALS:
        raise ValueError(f"unknown material {el['material']!r}; one of {sorted(MATERIALS)}")


def wall_length(w: dict) -> float:
    return math.dist(w["start"], w["end"])


def wall_yaw(w: dict) -> float:
    return math.atan2(w["end"][1] - w["start"][1], w["end"][0] - w["start"][0])


def openings_of(model: dict, wall_id: str) -> list[dict]:
    return sorted((e for e in model["elements"].values()
                   if e["kind"] in OPENING_KINDS and e.get("host") == wall_id),
                  key=lambda e: e["offset"])


def opening_center(model: dict, o: dict) -> list[float]:
    w = model["elements"][o["host"]]
    yaw = wall_yaw(w)
    u = o["offset"] + o["width"] / 2
    return [w["start"][0] + u * math.cos(yaw), w["start"][1] + u * math.sin(yaw)]


def pose(model: dict, el: dict) -> dict:
    """Element local frame: origin (world xyz) + yaw about Z."""
    k = el["kind"]
    if k == "wall":
        return {"pos": [el["start"][0], el["start"][1], el.get("base", 0.0)], "yaw": wall_yaw(el)}
    if k in OPENING_KINDS:
        w = model["elements"][el["host"]]
        p = pose(model, w)
        c, s = math.cos(p["yaw"]), math.sin(p["yaw"])
        return {"pos": [p["pos"][0] + el["offset"] * c, p["pos"][1] + el["offset"] * s,
                        p["pos"][2] + el["sill"]], "yaw": p["yaw"]}
    if k == "beam":
        return {"pos": [el["start"][0], el["start"][1], el["z_top"] - el["depth"]],
                "yaw": math.atan2(el["end"][1] - el["start"][1], el["end"][0] - el["start"][0])}
    if k == "slab":
        return {"pos": [el["min"][0], el["min"][1], el["top"] - el["thickness"]], "yaw": 0.0}
    if k == "space":
        return {"pos": [0, 0, 0], "yaw": 0.0}
    return {"pos": [el["pos"][0], el["pos"][1], el.get("base", 0.0)], "yaw": math.radians(el.get("yaw_deg", 0.0))}


def solids(model: dict, el: dict) -> list[dict]:
    """Boxes in the element's local frame: {"c": centre xyz, "s": full size xyz}."""
    k = el["kind"]
    if k == "wall":
        L, t, h = wall_length(el), el["thickness"], el["height"]
        out, u = [], 0.0
        for o in openings_of(model, el["id"]):
            a, b = o["offset"], o["offset"] + o["width"]
            if a > u + 1e-4:
                out.append({"c": [(u + a) / 2, 0, h / 2], "s": [a - u, t, h]})
            if o["sill"] > 1e-3:
                out.append({"c": [(a + b) / 2, 0, o["sill"] / 2], "s": [b - a, t, o["sill"]]})
            top = o["sill"] + o["height"]
            if top < h - 1e-3:
                out.append({"c": [(a + b) / 2, 0, (top + h) / 2], "s": [b - a, t, h - top]})
            u = max(u, b)
        if L > u + 1e-4:
            out.append({"c": [(u + L) / 2, 0, h / 2], "s": [L - u, t, h]})
        return out
    if k in ("door", "window"):
        th = 0.04
        return [{"c": [el["width"] / 2, 0, el["height"] / 2], "s": [el["width"], th, el["height"]]}]
    if k == "void":
        return []
    if k == "column":
        a, b = el["size"]
        return [{"c": [0, 0, el["height"] / 2], "s": [a, b, el["height"]]}]
    if k == "beam":
        L = math.dist(el["start"], el["end"])
        return [{"c": [L / 2, 0, el["depth"] / 2], "s": [L, el["width"], el["depth"]]}]
    if k == "slab":
        sx, sy = el["max"][0] - el["min"][0], el["max"][1] - el["min"][1]
        return [{"c": [sx / 2, sy / 2, el["thickness"] / 2], "s": [sx, sy, el["thickness"]]}]
    if k == "fixture":
        l, d, h = el["size"]
        return [{"c": [0, 0, h / 2], "s": [l, d, h]}]
    return []


def volume(model: dict, el: dict) -> float:
    return sum(b["s"][0] * b["s"][1] * b["s"][2] for b in solids(model, el))


def mass(model: dict, el: dict) -> float:
    return volume(model, el) * MATERIALS[el.get("material", "concrete")]["density"]


# ---------------------------------------------------------------- spaces / 2D helpers

def poly_area(poly) -> float:
    return abs(sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1]
                   for i in range(len(poly)))) / 2


def poly_centroid(poly) -> list[float]:
    return [sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly)]


def point_in_poly(p, poly) -> bool:
    x, y, inside = p[0], p[1], False
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def dist_point_seg(p, a, b) -> float:
    ax, ay, bx, by = a[0], a[1], b[0], b[1]
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0 if L2 == 0 else max(0, min(1, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2))
    return math.hypot(p[0] - ax - t * dx, p[1] - ay - t * dy)


def dist_point_poly_edge(p, poly) -> float:
    return min(dist_point_seg(p, poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly)))


def space_min_width(poly) -> float:
    """Narrowest extent of the polygon over edge-aligned directions (exact for rectangles)."""
    best = float("inf")
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        L = math.hypot(x2 - x1, y2 - y1)
        if L < 1e-6:
            continue
        nx, ny = -(y2 - y1) / L, (x2 - x1) / L
        proj = [p[0] * nx + p[1] * ny for p in poly]
        best = min(best, max(proj) - min(proj))
    return best


def space_openings(model: dict, space: dict, tol=0.35) -> list[dict]:
    """Doors/windows/voids sitting on the boundary of a space."""
    return [o for o in model["elements"].values() if o["kind"] in OPENING_KINDS
            and dist_point_poly_edge(opening_center(model, o), space["polygon"]) < tol]


# ---------------------------------------------------------------- summaries

def summary(model: dict) -> dict:
    els = list(model["elements"].values())
    by_kind: dict[str, int] = {}
    for e in els:
        by_kind[e["kind"]] = by_kind.get(e["kind"], 0) + 1
    return {
        "project": model["project"], "revision": f"R{model['revision']}", "level": model["level"],
        "counts": by_kind,
        "scanned": sum(e["provenance"] == "scanned" for e in els),
        "proposed": sum(e["provenance"] == "proposed" for e in els),
        "elements": [brief(model, e) for e in els],
    }


def brief(model: dict, e: dict) -> dict:
    b = {"id": e["id"], "kind": e["kind"], "name": e["name"], "provenance": e["provenance"]}
    k = e["kind"]
    if k == "wall":
        b.update(start=_r(e["start"]), end=_r(e["end"]), length=round(wall_length(e), 2),
                 height=e["height"], thickness=e["thickness"], material=e.get("material"),
                 openings=[o["id"] for o in openings_of(model, e["id"])])
    elif k in OPENING_KINDS:
        b.update(host=e["host"], offset=round(e["offset"], 2), width=round(e["width"], 2),
                 sill=round(e["sill"], 2), height=round(e["height"], 2),
                 center=_r(opening_center(model, e)), exit=e.get("exit", False))
    elif k == "space":
        b.update(use=e["use"], area=round(poly_area(e["polygon"]), 2),
                 min_width=round(space_min_width(e["polygon"]), 2), polygon=[_r(p) for p in e["polygon"]])
    elif k == "slab":
        b.update(min=_r(e["min"]), max=_r(e["max"]), top=e["top"])
    else:
        b.update({key: e[key] for key in ("pos", "size", "yaw_deg", "start", "end", "height", "material", "anchored")
                  if key in e})
    if "fit" in e:
        b["fit_mm"] = e["fit"]
    return b


def _r(p):
    return [round(float(v), 3) for v in p]


def quantities(model: dict) -> dict:
    """Takeoff for proposed work only (what has to be bought / built)."""
    rows, total = [], 0.0
    for e in model["elements"].values():
        if e["provenance"] != "proposed" or e["kind"] in ("space", "void"):
            continue
        mat = e.get("material", "timber" if e["kind"] == "door" else "glass" if e["kind"] == "window" else "concrete")
        vol = volume(model, e)
        cost = vol * MATERIALS[mat]["cost_m3"]
        row = {"id": e["id"], "name": e["name"], "kind": e["kind"], "material": mat,
               "volume_m3": round(vol, 3), "mass_kg": round(vol * MATERIALS[mat]["density"]),
               "cost_usd": round(cost)}
        if e["kind"] == "wall":
            row["face_area_m2"] = round(vol / e["thickness"], 2)
        rows.append(row)
        total += cost
    return {"items": rows, "material_cost_usd": round(total),
            "note": "Rough order-of-magnitude material cost for proposed elements only."}


def snapshot(model: dict) -> dict:
    """Render-ready copy: every element gets pose + boxes + colour."""
    out = copy.deepcopy(model)
    for e in out["elements"].values():
        e["pose"] = pose(model, e)
        e["boxes"] = solids(model, e)
        mat = e.get("material") or ("timber" if e["kind"] == "door" else "glass" if e["kind"] == "window" else "concrete")
        e["color"] = MATERIALS[mat]["color"]
    return out
