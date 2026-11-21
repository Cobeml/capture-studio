"""Deliverable package: IFC4 + GLB + MuJoCo MJCF + model.json + validation.json + README + SHA256SUMS.

Layout and self-checks follow the Moving Atoms as-built packages: every exported element is
counted back out of the written files, scan-fit residuals are reported, and a physics drop
test proves the colliders are sound.
"""
from __future__ import annotations

import hashlib
import json
import math
import time

import ifcopenshell
import ifcopenshell.api
import ifcopenshell.guid
import numpy as np
import trimesh

from . import bim, physics, store


def _ifc(model: dict, path):
    run = ifcopenshell.api.run
    f = run("project.create_file", version="IFC4")
    project = run("root.create_entity", f, ifc_class="IfcProject", name=model["project"]["name"])
    run("unit.assign_unit", f, length={"is_metric": True, "raw": "METERS"})
    ctx = run("context.add_context", f, context_type="Model")
    body = run("context.add_context", f, context_type="Model", context_identifier="Body",
               target_view="MODEL_VIEW", parent=ctx)
    site = run("root.create_entity", f, ifc_class="IfcSite", name="Site")
    building = run("root.create_entity", f, ifc_class="IfcBuilding", name=model["project"]["name"])
    storey = run("root.create_entity", f, ifc_class="IfcBuildingStorey", name=model["level"]["name"])
    run("aggregate.assign_object", f, products=[site], relating_object=project)
    run("aggregate.assign_object", f, products=[building], relating_object=site)
    run("aggregate.assign_object", f, products=[storey], relating_object=building)

    def pt(*v):
        return f.create_entity("IfcCartesianPoint", Coordinates=[float(x) for x in v])

    def dr(*v):
        return f.create_entity("IfcDirection", DirectionRatios=[float(x) for x in v])

    def placement(pos, yaw):
        ax = f.create_entity("IfcAxis2Placement3D", Location=pt(*pos), Axis=dr(0, 0, 1),
                             RefDirection=dr(math.cos(yaw), math.sin(yaw), 0))
        return f.create_entity("IfcLocalPlacement", RelativePlacement=ax)

    def extrude(profile, base, depth):
        pos = f.create_entity("IfcAxis2Placement3D", Location=pt(*base))
        return f.create_entity("IfcExtrudedAreaSolid", SweptArea=profile, Position=pos,
                               ExtrudedDirection=dr(0, 0, 1), Depth=float(depth))

    n = 0
    for e in model["elements"].values():
        if e["kind"] == "void":
            continue
        ent = run("root.create_entity", f, ifc_class=e["ifc_class"], name=e["name"])
        p = bim.pose(model, e)
        ent.ObjectPlacement = placement(p["pos"], p["yaw"])
        items = []
        if e["kind"] == "space":
            poly = f.create_entity("IfcPolyline", Points=[pt(*q) for q in e["polygon"] + [e["polygon"][0]]])
            prof = f.create_entity("IfcArbitraryClosedProfileDef", ProfileType="AREA", OuterCurve=poly)
            items.append(extrude(prof, (0, 0, 0), model["level"]["ceiling_height"]))
        for b in bim.solids(model, e):
            prof = f.create_entity("IfcRectangleProfileDef", ProfileType="AREA", XDim=float(b["s"][0]), YDim=float(b["s"][1]))
            items.append(extrude(prof, (b["c"][0], b["c"][1], b["c"][2] - b["s"][2] / 2), b["s"][2]))
        if items:
            rep = f.create_entity("IfcShapeRepresentation", ContextOfItems=body, RepresentationIdentifier="Body",
                                  RepresentationType="SweptSolid", Items=items)
            ent.Representation = f.create_entity("IfcProductDefinitionShape", Representations=[rep])
        if e["kind"] == "space":        # spaces decompose the storey; physical elements are contained in it
            run("aggregate.assign_object", f, products=[ent], relating_object=storey)
        else:
            run("spatial.assign_container", f, products=[ent], relating_structure=storey)
        props = {"Id": e["id"], "Provenance": e["provenance"], "Revision": f"R{model['revision']}"}
        if e.get("material"):
            props["Material"] = e["material"]
        if "fit" in e:
            props |= {"FitP50_mm": float(e["fit"]["p50"]), "FitP95_mm": float(e["fit"]["p95"]), "FitPoints": int(e["fit"]["n"])}
        for k in ("exit", "guard", "anchored", "head_restraint", "exterior"):
            if e.get(k):
                props[k.title().replace("_", "")] = True
        if e.get("lintel"):
            props["Lintel"] = e["lintel"]
        pset = run("pset.add_pset", f, product=ent, name="CaptureStudio_Element")
        run("pset.edit_pset", f, pset=pset, properties=props)
        n += 1
    f.write(str(path))
    return n


def _glb(model: dict, path):
    scene = trimesh.Scene()
    for e in bim.snapshot(model)["elements"].values():
        if e["kind"] == "space":
            continue
        T = trimesh.transformations.rotation_matrix(e["pose"]["yaw"], [0, 0, 1])
        T[:3, 3] = e["pose"]["pos"]
        rgba = [int(e["color"][i:i + 2], 16) for i in (1, 3, 5)] + [255]
        if e["provenance"] == "proposed":
            rgba = [255, 138, 61, 255]
        for i, b in enumerate(e["boxes"]):
            box = trimesh.creation.box(extents=b["s"])
            box.apply_translation(b["c"])
            box.apply_transform(T)
            box.visual.face_colors = rgba
            scene.add_geometry(box, node_name=f"{e['id']}_{i}", geom_name=f"{e['id']}_{i}")
    scene.apply_transform(trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0]))  # Z-up -> glTF Y-up
    scene.export(str(path))


def write_package(state: dict, v: dict) -> dict:
    model = state["model"]
    out = store.DATA / "export"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.iterdir():
        old.unlink()
    rev = f"R{model['revision']}"
    stem = f"capture_studio_{rev}"

    (out / "model.json").write_text(json.dumps(model, indent=1))
    n_ifc = _ifc(model, out / f"{stem}.ifc")
    _glb(model, out / f"{stem}.glb")
    xml, info = physics.build_mjcf(model)
    (out / f"{stem}.mjcf.xml").write_text(xml)

    # read everything back
    ifc = ifcopenshell.open(str(out / f"{stem}.ifc"))
    expected = sum(e["kind"] != "void" for e in model["elements"].values())
    import mujoco
    mj = mujoco.MjModel.from_xml_path(str(out / f"{stem}.mjcf.xml"))
    glb = trimesh.load(str(out / f"{stem}.glb"))
    fits = [e["fit"] for e in model["elements"].values() if "fit" in e]
    validation = {
        "revision": rev, "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "units": "m", "up_axis": "Z",
        "elements": {"expected": expected, "ifc_products": len(ifc.by_type("IfcProduct")) - 3,  # minus site/building/storey
                     "ifc_schema": ifc.schema, "glb_meshes": len(glb.geometry), "mjcf_bodies": mj.nbody - 1,
                     "mjcf_dynamic": len(info["dynamic"]), "mjcf_ties": mj.neq},
        "scan_fit_mm": {"elements": len(fits),
                        "p50_median": round(float(np.median([x["p50"] for x in fits])), 1) if fits else None,
                        "p95_max": round(float(max(x["p95"] for x in fits)), 1) if fits else None,
                        "per_element": {e["id"]: e["fit"] for e in model["elements"].values() if "fit" in e}},
        "scan": model.get("scan", {}),
        "checks": {"profile": v["profile"], "code_ok": v["code_ok"], "safety_ok": v["safety_ok"],
                   "physics_ok": v["physics_ok"], "requirements_ok": v["desires_ok"], "counts": v["counts"],
                   "open_issues": [{k: i[k] for k in ("key", "severity", "category", "rule", "ref", "message", "fix")} for i in v["issues"]]},
        "physics": {"lateral_g": v["physics"]["lateral_g"], "bodies": v["physics"]["bodies"], "ties": v["physics"]["ties"],
                    "drop_test": physics.drop_test(model)},
        "approvals": state["approvals"], "requirements": state["requirements"], "next_steps": state["next_steps"],
        "quantities": bim.quantities(model),
    }
    (out / "validation.json").write_text(json.dumps(validation, indent=1))

    c = validation["checks"]
    d = validation["physics"]["drop_test"]
    tick = lambda ok: "pass" if ok else "OPEN"  # noqa: E731
    readme = f"""# {model['project']['name']} — {rev}

Capture Studio package. Metres, Z-up, local scan frame (origin at the shell's min corner).
As-built elements were fitted to a LingBot-Map point cloud reconstructed from a phone video
(provenance `scanned`, with scan-fit residuals); design elements are `proposed`.

## Files
- `{stem}.ifc` — IFC4, one storey, property set `CaptureStudio_Element` on every element (id, provenance, fit P50/P95, material).
- `{stem}.glb` — the same geometry for viewers (proposed work in orange).
- `{stem}.mjcf.xml` — MuJoCo scene: scanned shell static, proposed elements dynamic, ties as weld constraints.
- `model.json` — parametric source model. `validation.json` — full check results. `SHA256SUMS`.

## Checks
- Elements: {validation['elements']['ifc_products']} in IFC of {expected} expected; {validation['elements']['mjcf_bodies']} physics bodies, {validation['elements']['mjcf_ties']} ties.
- Scan fit: median P50 {validation['scan_fit_mm']['p50_median']} mm, worst P95 {validation['scan_fit_mm']['p95_max']} mm over {len(fits)} fitted elements.
- Code ({c['profile']}): {tick(c['code_ok'])}. Safety: {tick(c['safety_ok'])}. Physics ({validation['physics']['lateral_g']} g tilt test): {tick(c['physics_ok'])}. Client requirements: {tick(c['requirements_ok'])}.
- Drop test: {d.get('dropped')} boxes dropped, {d.get('fell_through')} fell through the slab, {d.get('at_rest')} at rest.
- Open items: {c['counts']['fail']} fail, {c['counts']['hold']} hold, {c['counts']['warn']} site actions.

## Next steps
""" + "\n".join(f"{i + 1}. **{s['title']}** ({s['who']}) — {s['why']}" for i, s in enumerate(state["next_steps"])) + """

## Limitations
Scale comes from a video-only reconstruction and one calibration assumption (see `scan` in
validation.json) — tape-check one dimension before setting out. Code rules are a demo subset
and not a compliance certificate. The MuJoCo test is a rigid-body stability screen, not a
structural design.
"""
    (out / "README.md").write_text(readme)
    sums = "".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in sorted(out.iterdir()))
    (out / "SHA256SUMS").write_text(sums)
    return {"dir": str(out), "files": sorted(p.name for p in out.iterdir()),
            "elements": validation["elements"], "scan_fit_mm": {k: validation["scan_fit_mm"][k] for k in ("p50_median", "p95_max")},
            "checks": {k: c[k] for k in ("code_ok", "safety_ok", "physics_ok", "requirements_ok", "counts")},
            "drop_test": d}
