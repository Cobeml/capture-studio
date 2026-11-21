"""MuJoCo physical-robustness checks.

The scanned shell (as-built slab + walls) is static collision geometry — an as-built model
is fixed. Every *proposed* element becomes a free rigid body with real mass, resting on the
slab by contact. Structural connections are explicit weld constraints:

  * a proposed wall whose end abuts another wall is tied to it (toothed / wall ties),
  * `head_restraint: true` ties a wall to the soffit, `anchored: true` fixes an element down.

Tests
  clash      initial interpenetration between elements that are not connected
  service    gravity settle, then a tilt test: gravity gains a horizontal component of
             `lateral_g` in 4 directions. Anything that slides > 20 mm or tips > 2° fails.
  erection   same with every tie removed and a small lateral load: what needs temporary
             propping while it is being built.
  drop       boxes dropped across the slab must come to rest on it (collider sanity check).
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from . import bim

G = 9.81
DYNAMIC_KINDS = ("wall", "fixture", "equipment", "column", "beam")
SLIDE_LIMIT, TILT_LIMIT_DEG = 0.02, 2.0


def _quat(yaw):
    return f"{math.cos(yaw / 2):.6f} 0 0 {math.sin(yaw / 2):.6f}"


def find_ties(model: dict) -> dict[str, list[str]]:
    """Connections for each proposed element: other element ids, or 'world'."""
    els = model["elements"]
    walls = [e for e in els.values() if e["kind"] == "wall"]
    ties: dict[str, list[str]] = {}
    for e in els.values():
        if e["provenance"] != "proposed" or e["kind"] not in DYNAMIC_KINDS:
            continue
        t: list[str] = []
        if e.get("anchored") or e.get("head_restraint"):
            t.append("world")
        if e["kind"] == "wall":
            for w in walls:
                if w["id"] == e["id"]:
                    continue
                reach = w["thickness"] / 2 + e["thickness"] / 2 + 0.03
                end_on = any(bim.dist_point_seg(p, w["start"], w["end"]) < reach for p in (e["start"], e["end"]))
                butted = any(bim.dist_point_seg(p, e["start"], e["end"]) < reach for p in (w["start"], w["end"]))
                if end_on or butted:
                    t.append(w["id"])
        if e["kind"] == "beam":      # a beam bears on whatever is under its ends
            for o in els.values():
                if o["kind"] in ("wall", "column") and o["id"] != e["id"]:
                    for p in (e["start"], e["end"]):
                        d = bim.dist_point_seg(p, o["start"], o["end"]) if o["kind"] == "wall" \
                            else math.dist(p, o["pos"])
                        if d < 0.25 and o["id"] not in t:
                            t.append(o["id"])
        ties[e["id"]] = t
    return ties


def build_mjcf(model: dict, with_ties=True, drop_boxes: list | None = None, isolate=False) -> tuple[str, dict]:
    """isolate=True: proposed elements do not collide with each other (each is judged on its own)."""
    els = model["elements"]
    ties = find_ties(model)
    bodies, welds, excludes = [], [], []
    dynamic = []
    for e in els.values():
        if e["kind"] in ("space", "void", "door", "window"):
            continue
        boxes = bim.solids(model, e)
        if not boxes:
            continue
        p = bim.pose(model, e)
        is_dyn = e["provenance"] == "proposed" and e["kind"] in DYNAMIC_KINDS
        dens = bim.density(model, e)
        mask = ('contype="2" conaffinity="1" ' if is_dyn else 'contype="1" conaffinity="2" ') if isolate else ""
        geoms = "".join(
            f'<geom type="box" {mask}pos="{b["c"][0]:.4f} {b["c"][1]:.4f} {b["c"][2]:.4f}" '
            f'size="{b["s"][0] / 2:.4f} {b["s"][1] / 2:.4f} {b["s"][2] / 2:.4f}" density="{dens}"/>'
            for b in boxes)
        joint = "<freejoint/>" if is_dyn else ""
        # dynamic bodies start 1 mm above their bearing so the first contact is clean
        z = p["pos"][2] + (0.001 if is_dyn else 0.0)
        bodies.append(f'<body name="{e["id"]}" pos="{p["pos"][0]:.4f} {p["pos"][1]:.4f} {z:.4f}" '
                      f'quat="{_quat(p["yaw"])}">{joint}{geoms}</body>')
        if is_dyn:
            dynamic.append(e["id"])
    for eid, tl in ties.items():
        for other in tl:
            if other == "world":
                if with_ties:
                    welds.append(f'<weld name="{eid}|world" body1="{eid}"/>')
            else:
                excludes.append(f'<exclude body1="{eid}" body2="{other}"/>')
                if with_ties and not (other in ties and eid in ties[other] and other < eid):
                    welds.append(f'<weld name="{eid}|{other}" body1="{eid}" body2="{other}"/>')
    for i, (x, y, z) in enumerate(drop_boxes or []):
        bodies.append(f'<body name="drop{i}" pos="{x:.3f} {y:.3f} {z:.3f}"><freejoint/>'
                      f'<geom type="box" size="0.15 0.15 0.15" density="400"/></body>')
    xml = f"""<mujoco model="capture_studio">
  <option timestep="0.002" gravity="0 0 -{G}" integrator="implicitfast" cone="elliptic" impratio="10" noslip_iterations="4"/>
  <default><geom friction="0.6 0.005 0.0001" solref="0.004 1"/></default>
  <worldbody>{''.join(bodies)}</worldbody>
  <contact>{''.join(dict.fromkeys(excludes))}</contact>
  <equality>{''.join(welds)}</equality>
</mujoco>"""
    return xml, {"dynamic": dynamic, "ties": ties}


def _clashes(m, d, model) -> list[dict]:
    mujoco.mj_forward(m, d)
    seen, out = set(), []
    for c in d.contact[: d.ncon]:
        if c.dist > -0.02:
            continue
        a = m.body(m.geom_bodyid[c.geom1]).name
        b = m.body(m.geom_bodyid[c.geom2]).name
        pa = model["elements"].get(a, {}).get("provenance")
        pb = model["elements"].get(b, {}).get("provenance")
        if "proposed" not in (pa, pb) or (a, b) in seen:
            continue
        seen.add((a, b))
        out.append({"a": a, "b": b, "depth_mm": round(-c.dist * 1000)})
    return out


def _tilt_deg(q0, q1) -> float:
    r0, r1 = np.zeros(9), np.zeros(9)
    mujoco.mju_quat2Mat(r0, q0)
    mujoco.mju_quat2Mat(r1, q1)
    z0, z1 = r0.reshape(3, 3)[:, 2], r1.reshape(3, 3)[:, 2]
    return math.degrees(math.acos(max(-1.0, min(1.0, float(z0 @ z1)))))


def _tilt_test(m, d, ids, lateral_g, seconds, record=False):
    """Returns per-body worst (slide m, tilt deg), peak weld forces, optional trajectory."""
    start_q, start_v = d.qpos.copy(), d.qvel.copy()
    ref = {i: (d.body(i).xpos.copy(), d.body(i).xquat.copy()) for i in ids}
    worst = {i: [0.0, 0.0, 0] for i in ids}
    tie_peak = {m.equality(k).name: 0.0 for k in range(m.neq)}
    frames_by_dir = []
    for k, ang in enumerate((0, math.pi / 2, math.pi, 3 * math.pi / 2)):
        d.qpos[:], d.qvel[:] = start_q, start_v
        m.opt.gravity[:] = (lateral_g * G * math.cos(ang), lateral_g * G * math.sin(ang), -G)
        frames = []
        for step in range(int(seconds / m.opt.timestep)):
            mujoco.mj_step(m, d)
            if step % 20 == 0:
                if record:
                    frames.append({i: [round(float(v), 4) for v in (*d.body(i).xpos, *d.body(i).xquat)] for i in ids})
                for r in range(d.nefc):
                    if d.efc_type[r] == mujoco.mjtConstraint.mjCNSTR_EQUALITY:
                        nm = m.equality(d.efc_id[r]).name
                        tie_peak[nm] = max(tie_peak[nm], abs(float(d.efc_force[r])))
        for i in ids:
            slide = float(np.linalg.norm(d.body(i).xpos - ref[i][0]))
            tilt = _tilt_deg(ref[i][1], d.body(i).xquat)
            if slide > worst[i][0] or tilt > worst[i][1]:
                worst[i] = [max(slide, worst[i][0]), max(tilt, worst[i][1]), k]
        frames_by_dir.append(frames)
    m.opt.gravity[:] = (0, 0, -G)
    d.qpos[:], d.qvel[:] = start_q, start_v
    return worst, tie_peak, frames_by_dir


def run(model: dict, lateral_g: float | None = None, erection_g=0.05) -> dict:
    """Full robustness report for the current design."""
    els = model["elements"]
    if lateral_g is None:
        from . import codes
        lateral_g = codes.PROFILES[model["project"]["code_profile"]].get("lateral_g", 0.15)
    xml, info = build_mjcf(model, with_ties=True)
    ids = info["dynamic"]
    report = {"lateral_g": lateral_g, "erection_g": erection_g, "bodies": {}, "issues": [],
              "ties": [], "clashes": [], "animation": None, "ok": True}
    if not ids:
        report["note"] = "No proposed physical elements yet — nothing to test."
        return report
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    report["clashes"] = _clashes(m, d, model)
    clashing = {c["a"] for c in report["clashes"]} | {c["b"] for c in report["clashes"]}
    for c in report["clashes"]:
        report["issues"].append({
            "key": f"clash:{c['a']}:{c['b']}", "category": "physics", "severity": "fail", "rule": "Clash",
            "ref": "MuJoCo contact", "element": c["a"] if els[c["a"]]["provenance"] == "proposed" else c["b"],
            "message": f"{els[c['a']]['name']} overlaps {els[c['b']]['name']} by {c['depth_mm']} mm.",
            "fix": "Move or resize one of them."})
    if clashing:
        report["ok"] = False
        report["note"] = "Dynamics skipped until clashes are resolved (overlapping solids would explode apart)."
        return report

    # service state: settle under gravity, then tilt test with ties active
    p0 = {i: d.body(i).xpos.copy() for i in ids}
    for _ in range(500):
        mujoco.mj_step(m, d)
    settle = {i: float(np.linalg.norm(d.body(i).xpos - p0[i])) for i in ids}
    worst, tie_peak, frames = _tilt_test(m, d, ids, lateral_g, 1.6, record=True)

    # erection state: no ties
    xml2, _ = build_mjcf(model, with_ties=False, isolate=True)
    m2 = mujoco.MjModel.from_xml_string(xml2)
    d2 = mujoco.MjData(m2)
    for _ in range(500):
        mujoco.mj_step(m2, d2)
    worst2, _, _ = _tilt_test(m2, d2, ids, erection_g, 1.6)

    anim_dir, anim_score = 0, -1.0
    for i in ids:
        e = els[i]
        slide, tilt, k = worst[i]
        stable = slide < SLIDE_LIMIT and tilt < TILT_LIMIT_DEG and settle[i] < 0.01
        free_ok = worst2[i][0] < SLIDE_LIMIT and worst2[i][1] < TILT_LIMIT_DEG
        report["bodies"][i] = {
            "name": e["name"], "mass_kg": round(float(m.body(i).subtreemass[0])),
            "ties": info["ties"].get(i, []), "settle_mm": round(settle[i] * 1000, 1),
            "slide_mm": round(slide * 1000, 1), "tilt_deg": round(tilt, 2),
            "stable": stable, "stable_without_ties": free_ok}
        if not stable:
            report["ok"] = False
            if max(slide / SLIDE_LIMIT, tilt / TILT_LIMIT_DEG) > anim_score:
                anim_dir, anim_score = k, max(slide / SLIDE_LIMIT, tilt / TILT_LIMIT_DEG)
            how = "topples" if tilt >= TILT_LIMIT_DEG else "slides"
            fix = ("Run it into an existing wall so it can be tied in, or set head_restraint=true "
                   "(fixed to the soffit)." if e["kind"] == "wall" else "Set anchored=true (anchor bolts to the slab) or make it lower / deeper.")
            report["issues"].append({
                "key": f"stability:{i}", "category": "physics", "severity": "fail",
                "rule": "Lateral stability", "ref": f"MuJoCo tilt test {lateral_g} g", "element": i,
                "message": f"{e['name']} {how} under {lateral_g} g lateral load "
                           f"(tilt {tilt:.0f}°, moves {slide * 1000:.0f} mm) — it is free-standing.",
                "fix": fix})
        elif not free_ok:
            report["issues"].append({
                "key": f"propping:{i}", "category": "safety", "severity": "warn",
                "rule": "Temporary stability during construction", "ref": f"MuJoCo erection test {erection_g} g",
                "element": i,
                "message": f"{e['name']} is only stable once tied in; it falls at {erection_g} g while free-standing.",
                "fix": "Prop/brace during construction until ties and head restraint are complete."})
    for name, f in tie_peak.items():
        a, b = name.split("|")
        report["ties"].append({"element": a, "to": b, "peak_force_kN": round(f / 1000, 2)})
    report["animation"] = {"direction_deg": anim_dir * 90, "fps": 25, "frames": frames[anim_dir]}
    return report


def drop_test(model: dict, n=24, seed=0) -> dict:
    """Moving-Atoms-style collider check: drop boxes over the slab, they must rest on it."""
    slab = next((e for e in model["elements"].values() if e["kind"] == "slab"), None)
    if slab is None:
        return {"ok": False, "note": "no slab"}
    rng = np.random.default_rng(seed)
    pts = [(rng.uniform(slab["min"][0] + 0.6, slab["max"][0] - 0.6),
            rng.uniform(slab["min"][1] + 0.6, slab["max"][1] - 0.6),
            model["level"]["ceiling_height"] + 0.6 + 0.4 * i) for i in range(n)]
    xml, _ = build_mjcf(model, with_ties=True, drop_boxes=pts)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    for _ in range(int(6.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    zs = [float(d.body(f"drop{i}").xpos[2]) for i in range(n)]
    sp = [float(np.linalg.norm(d.body(f"drop{i}").cvel[3:])) for i in range(n)]
    fell = sum(z < slab["top"] - 0.05 for z in zs)
    return {"dropped": n, "fell_through": fell, "at_rest": sum(s < 0.05 for s in sp),
            "on_floor": sum(abs(z - slab["top"] - 0.15) < 0.03 for z in zs),
            "on_elements": sum(z > slab["top"] + 0.2 for z in zs), "ok": fell == 0}
