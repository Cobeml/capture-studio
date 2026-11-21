"""Regulatory + safety checks over the BIM model.

Rule values are a *demo subset* transcribed from memory of the cited codes and must be
verified against the adopted edition before any real submission. Profiles are plain data
so a jurisdiction can be swapped or tuned without touching the checks.

Every issue: {key, category: code|safety, severity: fail|warn|hold, rule, ref, element, message, fix}
  fail = must change the design, warn = action on site, hold = needs a human sign-off.
"""
from __future__ import annotations

import heapq
import math

import numpy as np
from scipy import ndimage

from . import bim

HABITABLE = ("living", "bedroom", "office", "dining", "study")

PROFILES = {
    "NBC_INDIA_2016": {
        "title": "National Building Code of India 2016 (Part 3 & 4) — demo subset",
        "room": {  # use: (min area m2, min width m, min height m)
            "living": (9.5, 2.4, 2.75), "bedroom": (9.5, 2.4, 2.75), "office": (9.5, 2.4, 2.75),
            "dining": (9.5, 2.4, 2.75), "study": (9.5, 2.4, 2.75),
            "kitchen": (5.0, 1.8, 2.75), "bathroom": (1.8, 1.2, 2.1), "store": (3.0, 1.2, 2.2),
            "corridor": (0.0, 1.0, 2.1),
        },
        "light_vent_ratio": 0.10,           # openable area / floor area, habitable + kitchen
        "door_clear_width": {"default": 0.90, "bathroom": 0.75, "store": 0.75},
        "door_min_height": 2.0,
        "egress_clear_width": 0.90,
        "egress_travel_max": 22.5,
        "guard_height": 1.0,
        "lintel_span_max": 1.2,
        "refs": {"room": "NBC 2016 Pt 3 cl. 12.2", "light": "NBC 2016 Pt 8 §1 cl. 4", "door": "NBC 2016 Pt 3 Annex B",
                 "egress": "NBC 2016 Pt 4 cl. 4", "guard": "NBC 2016 Pt 3 cl. 12.19", "demo": "NBC 2016 Pt 7 / IS 4130"},
    },
    "OSHA_1910": {
        "title": "OSHA 29 CFR 1910 general industry + NFPA 101 industrial egress — demo subset",
        "industrial": True, "rooms_need_doors": False,
        "room": {  # zones on a production floor: (min area m2, min clear width m, min height m)
            "work_cell": (0.0, 0.0, 2.3), "assembly": (0.0, 0.0, 2.3), "storage": (0.0, 0.0, 2.3),
            "aisle": (0.0, 0.9, 2.3), "forklift_aisle": (0.0, 3.0, 2.3), "office": (0.0, 0.0, 2.3),
            "exit_access": (0.0, 0.9, 2.3),
        },
        "light_vent_ratio": 0.0,
        "door_clear_width": {"default": 0.813},
        "door_min_height": 2.03,
        "egress_clear_width": 0.711,        # 28 in exit access
        "egress_travel_max": 61.0,          # 200 ft, general industrial, unsprinklered
        "guard_height": 1.07,               # 42 in guardrail
        "lintel_span_max": 1.2,
        "machine_clearance": 0.6,           # working space around machines (demo value)
        "lateral_g": 0.25,                  # equipment / rack anchorage screen
        "refs": {"room": "29 CFR 1910.22 / 1910.176(a)", "light": "-", "door": "29 CFR 1910.36(g)",
                 "egress": "29 CFR 1910.36–37 / NFPA 101 ch. 40", "guard": "29 CFR 1910.29(b)",
                 "demo": "29 CFR 1926.850", "machine": "29 CFR 1910.212", "aisle": "29 CFR 1910.176(a)",
                 "clearance": "29 CFR 1910.22(a) / maker's manual"},
    },
    "IRC_2021": {
        "title": "International Residential Code 2021 — demo subset",
        "room": {u: (6.5, 2.134, 2.134) for u in HABITABLE} | {
            "kitchen": (0.0, 0.0, 2.134), "bathroom": (0.0, 0.0, 2.032), "store": (0.0, 0.0, 2.0),
            "corridor": (0.0, 0.914, 2.134)},
        "light_vent_ratio": 0.08,
        "door_clear_width": {"default": 0.813, "bathroom": 0.61, "store": 0.61},
        "door_min_height": 1.981,
        "egress_clear_width": 0.914,
        "egress_travel_max": 60.0,
        "guard_height": 0.914,
        "lintel_span_max": 1.2,
        "refs": {"room": "IRC R304/R305", "light": "IRC R303.1", "door": "IRC R311.2", "egress": "IRC R311",
                 "guard": "IRC R312.1", "demo": "IRC R301.1 / OSHA 1926.850"},
    },
}


def check(model: dict, approvals: dict | None = None) -> dict:
    approvals = approvals or {}
    prof = PROFILES[model["project"]["code_profile"]]
    els = model["elements"]
    issues: list[dict] = []

    def issue(key, category, severity, rule, ref, element, message, fix):
        if severity == "hold" and key in approvals:
            return
        issues.append({"key": key, "category": category, "severity": severity, "rule": rule,
                       "ref": prof["refs"].get(ref, ref), "element": element, "message": message, "fix": fix})

    spaces = [e for e in els.values() if e["kind"] == "space"]
    ceiling = model["level"]["ceiling_height"]

    # ---- rooms: area, width, height, daylight/ventilation, doors
    for s in spaces:
        need = prof["room"].get(s["use"])
        area = bim.poly_area(s["polygon"])
        width = bim.space_min_width(s["polygon"])
        if need:
            a, w, h = need
            if area < a - 1e-6:
                issue(f"area:{s['id']}", "code", "fail", "Minimum room area", "room", s["id"],
                      f"{s['name']} is {area:.1f} m², minimum for {s['use']} is {a} m².",
                      f"Enlarge by {a - area:.1f} m² (move a partition) or change its use.")
            if width < w - 1e-6:
                issue(f"width:{s['id']}", "code", "fail", "Minimum room width", "room", s["id"],
                      f"{s['name']} is {width:.2f} m wide, minimum is {w} m.", "Widen the room.")
            if ceiling < h - 1e-6:
                issue(f"height:{s['id']}", "code", "fail", "Minimum ceiling height", "room", s["id"],
                      f"Clear height {ceiling:.2f} m is below the {h} m minimum for {s['use']}.",
                      "Confirm the scanned ceiling height on site (set_scale) or change use.")
        ops = bim.space_openings(model, s)
        doors = [o for o in ops if o["kind"] in ("door", "void") and o["sill"] < 0.05]
        if s["use"] in HABITABLE + ("kitchen",):
            vent = sum(o["width"] * o["height"] for o in ops
                       if o["kind"] in ("window", "void") and els[o["host"]].get("exterior"))
            need_v = prof["light_vent_ratio"] * area
            if vent < need_v - 1e-6:
                issue(f"light:{s['id']}", "code", "fail", "Natural light & ventilation", "light", s["id"],
                      f"{s['name']} has {vent:.2f} m² of exterior openings; needs {need_v:.2f} m² "
                      f"({prof['light_vent_ratio']:.0%} of floor area).",
                      "Include an exterior window in this room, or add/enlarge one.")
        if not doors and prof.get("rooms_need_doors", True):
            issue(f"access:{s['id']}", "code", "fail", "Room has no door", "door", s["id"],
                  f"{s['name']} has no door or opening on its boundary.", "Add a door.")
        for d in doors:
            if d["kind"] != "door":
                continue
            minw = prof["door_clear_width"].get(s["use"], prof["door_clear_width"]["default"])
            if d["width"] < minw - 1e-6:
                issue(f"doorw:{d['id']}:{s['id']}", "code", "fail", "Door clear width", "door", d["id"],
                      f"{d['name']} is {d['width']:.2f} m wide; {s['name']} needs ≥ {minw} m.",
                      f"Widen to {minw} m.")
            if d["height"] < prof["door_min_height"] - 1e-6:
                issue(f"doorh:{d['id']}", "code", "fail", "Door height", "door", d["id"],
                      f"{d['name']} is {d['height']:.2f} m high; minimum {prof['door_min_height']} m.",
                      "Raise the door head.")

    # ---- egress: every space must reach the exit at the required clear width
    egress = egress_paths(model, prof["egress_clear_width"])
    if spaces and egress["exit"] is None:
        issue("egress:noexit", "code", "fail", "Means of egress", "egress", None,
              "No exit is defined.", "Mark the exit door with exit=true, or add a space with use 'exit_access' at the way out.")
    for sid, r in egress["paths"].items():
        name = els[sid]["name"]
        if r is None:
            issue(f"egress:{sid}", "code", "fail", "Means of egress", "egress", sid,
                  f"No {prof['egress_clear_width']} m-wide clear route from {name} to the exit.",
                  "Widen the pinch point (door or gap between walls/fixtures) on the way out.")
        elif r["length"] > prof["egress_travel_max"]:
            issue(f"travel:{sid}", "code", "fail", "Egress travel distance", "egress", sid,
                  f"Travel from {name} is {r['length']:.1f} m; limit {prof['egress_travel_max']} m.",
                  "Shorten the route or add an exit.")

    # ---- industrial: guarding, anchorage, working clearance, aisles kept clear
    if prof.get("industrial"):
        solid = [e for e in els.values() if e["kind"] in ("equipment", "fixture", "column")]
        aisles = [e for e in spaces if e["use"] in ("aisle", "forklift_aisle", "exit_access")]
        for e in solid:
            if e["provenance"] == "proposed" and e["kind"] == "equipment":
                if e.get("hazard") and not e.get("guarded"):
                    issue(f"guarding:{e['id']}", "safety", "fail", "Machine guarding", "machine", e["id"],
                          f"{e['name']} has exposed hazards (point of operation / moving parts) and no guarding.",
                          "Specify guarding: set guarded=true once a fixed guard, interlocked enclosure or light curtain is defined.")
                if e.get("fixed", True) and not e.get("anchored"):
                    issue(f"anchor:{e['id']}", "safety", "fail", "Machine anchorage", "machine", e["id"],
                          f"{e['name']} is a fixed machine and is not anchored to the slab.",
                          "Set anchored=true (anchor bolts per maker's foundation drawing).")
                for o in solid:
                    if o["id"] == e["id"] or (o["provenance"] == "proposed" and o["id"] < e["id"]):
                        continue
                    gap = bim.plan_gap(model, e, o)
                    need = prof["machine_clearance"]
                    if 0.0 < gap < need - 1e-6:
                        issue(f"clearance:{e['id']}:{o['id']}", "code", "fail", "Working clearance", "clearance", e["id"],
                              f"{e['name']} is {gap:.2f} m from {o['name']}; needs ≥ {need} m to operate and maintain it.",
                              f"Move it {need - gap:.2f} m further away.")
            for a in aisles:
                fp = bim.footprint(model, e)
                pts = fp + [bim.poly_centroid(fp)] if fp else []
                if any(bim.point_in_poly(q, a["polygon"]) for q in pts):
                    sev = "fail" if e["provenance"] == "proposed" else "warn"
                    issue(f"aisle:{e['id']}:{a['id']}", "safety", sev, "Aisle obstructed", "aisle", e["id"],
                          f"{e['name']} stands in {a['name']}; aisles must be kept clear.",
                          "Move it out of the aisle or reroute the aisle around it.")

    # ---- safety
    for o in els.values():
        if o["kind"] not in bim.OPENING_KINDS:
            continue
        host = els[o["host"]]
        if host.get("exterior") and o["kind"] in ("window", "void") and not o.get("exit") \
                and o["sill"] < prof["guard_height"] - 0.1 and not o.get("guard"):
            issue(f"guard:{o['id']}", "safety", "warn", "Fall protection at low-sill opening", "guard", o["id"],
                  f"{o['name']} has a {o['sill']:.2f} m sill on an exterior wall — fall hazard during works.",
                  f"Install a {prof['guard_height']} m guard / temporary edge protection (set guard=true once specified).")
        if o["provenance"] == "proposed" and host["provenance"] == "scanned":
            issue(f"cut:{o['id']}", "safety", "hold", "New opening in existing wall", "demo", o["id"],
                  f"{o['name']} cuts into existing wall {host['name']}; it may be load-bearing.",
                  "Structural engineer to confirm and specify lintel/needling before cutting. Ask the user to sign off.")
        if o["provenance"] == "proposed" and o["width"] > prof["lintel_span_max"] and not o.get("lintel") \
                and o["sill"] + o["height"] < host["height"] - 0.05:
            issue(f"lintel:{o['id']}", "safety", "fail", "Lintel over opening", "demo", o["id"],
                  f"{o['name']} spans {o['width']:.2f} m with masonry above and no lintel specified.",
                  "Set lintel (e.g. 'RC 150x200') on the opening.")
    for r in model.get("removed", []):
        issue(f"demo:{r['id']}", "safety", "hold", "Demolition of existing wall", "demo", r["id"],
              f"Existing {r['name']} is removed; structural role is unknown from a scan.",
              "Structural engineer sign-off and a demolition method statement. Ask the user to sign off.")

    counts = {s: sum(i["severity"] == s for i in issues) for s in ("fail", "warn", "hold")}
    return {"profile": prof["title"], "ok": counts["fail"] == 0 and counts["hold"] == 0,
            "counts": counts, "issues": issues,
            "egress": {"exit": egress["exit"],
                       "paths": {k: (v and {"length": round(v["length"], 2), "points": v["points"]})
                                 for k, v in egress["paths"].items()}}}


# ---------------------------------------------------------------- egress routing

def _grid(model: dict, cell=0.05, pad=0.3):
    slab = next((e for e in model["elements"].values() if e["kind"] == "slab"), None)
    if slab is None:
        return None
    x0, y0 = slab["min"][0] - pad, slab["min"][1] - pad
    nx = int(math.ceil((slab["max"][0] + pad - x0) / cell))
    ny = int(math.ceil((slab["max"][1] + pad - y0) / cell))
    xs = x0 + (np.arange(nx) + 0.5) * cell
    ys = y0 + (np.arange(ny) + 0.5) * cell
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    blocked = np.zeros((nx, ny), bool)
    for e in model["elements"].values():
        if e["kind"] in ("space", "slab", "door", "window", "void", "beam"):
            continue
        p = bim.pose(model, e)
        c, s = math.cos(p["yaw"]), math.sin(p["yaw"])
        lx = (X - p["pos"][0]) * c + (Y - p["pos"][1]) * s
        ly = -(X - p["pos"][0]) * s + (Y - p["pos"][1]) * c
        for b in bim.solids(model, e):
            zlo = p["pos"][2] + b["c"][2] - b["s"][2] / 2
            zhi = zlo + b["s"][2]
            if zhi < 0.1 or zlo > 2.0:      # below ankle or above head height: does not block walking
                continue
            blocked |= (np.abs(lx - b["c"][0]) <= b["s"][0] / 2) & (np.abs(ly - b["c"][1]) <= b["s"][1] / 2)
    return {"x0": x0, "y0": y0, "cell": cell, "blocked": blocked}


def egress_paths(model: dict, clear_width: float) -> dict:
    els = model["elements"]
    spaces = [e for e in els.values() if e["kind"] == "space"]
    exit_door = next((e for e in els.values() if e["kind"] in ("door", "void") and e.get("exit")), None)
    if exit_door is None:       # open-plan scans rarely capture the door itself: a declared exit-access zone will do
        exit_door = next((e for e in spaces if e["use"] == "exit_access"), None)
    spaces = [e for e in spaces if e is not exit_door]
    g = _grid(model)
    if not spaces or exit_door is None or g is None:
        return {"exit": exit_door and exit_door["id"], "paths": {}}
    cell = g["cell"]
    # a person needs clear_width: inflate obstacles by half of it (less one cell of tolerance)
    r = max(1, int(round((clear_width / 2) / cell)) - 1)
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    free = ~ndimage.binary_dilation(g["blocked"], structure=(xx * xx + yy * yy <= r * r))

    def to_cell(p):
        return int((p[0] - g["x0"]) / cell), int((p[1] - g["y0"]) / cell)

    def to_xy(c):
        return [round(g["x0"] + (c[0] + 0.5) * cell, 3), round(g["y0"] + (c[1] + 0.5) * cell, 3)]

    goal = to_cell(bim.poly_centroid(exit_door["polygon"]) if exit_door["kind"] == "space"
                   else bim.opening_center(model, exit_door))
    nx, ny = free.shape
    if not (0 <= goal[0] < nx and 0 <= goal[1] < ny) or not free[goal]:
        return {"exit": exit_door["id"], "paths": {s["id"]: None for s in spaces}}
    # Dijkstra from the exit over free cells (8-connected)
    dist = np.full(free.shape, np.inf)
    prev = {}
    dist[goal] = 0
    pq = [(0.0, goal)]
    steps = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142)]
    while pq:
        d, (i, j) = heapq.heappop(pq)
        if d > dist[i, j]:
            continue
        for di, dj, w in steps:
            a, b = i + di, j + dj
            if 0 <= a < nx and 0 <= b < ny and free[a, b] and d + w < dist[a, b]:
                dist[a, b] = d + w
                prev[(a, b)] = (i, j)
                heapq.heappush(pq, (d + w, (a, b)))
    out = {}
    for s in spaces:
        # start from the reachable free cell inside the room that is farthest from the exit
        xs = [p[0] for p in s["polygon"]]
        ys = [p[1] for p in s["polygon"]]
        i0, j0 = to_cell((min(xs), min(ys)))
        i1, j1 = to_cell((max(xs), max(ys)))
        best, best_d = None, -1.0
        for i in range(max(i0, 0), min(i1 + 1, nx), 2):
            for j in range(max(j0, 0), min(j1 + 1, ny), 2):
                if free[i, j] and np.isfinite(dist[i, j]) and dist[i, j] > best_d \
                        and bim.point_in_poly(to_xy((i, j)), s["polygon"]):
                    best, best_d = (i, j), dist[i, j]
        if best is None:
            out[s["id"]] = None
            continue
        pts, c = [], best
        while c != goal:
            pts.append(to_xy(c))
            c = prev[c]
        pts.append(to_xy(goal))
        out[s["id"]] = {"length": float(best_d * cell), "points": pts[::4] + [pts[-1]]}
    return {"exit": exit_door["id"], "paths": out}
