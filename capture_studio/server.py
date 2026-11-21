"""Capture Studio MCP server + demo viewer (one process).

    python -m capture_studio.server            # http://127.0.0.1:8765  (viewer)  +  /mcp  (MCP, streamable HTTP)
    python -m capture_studio.server --stdio    # MCP over stdio (state is shared through data/project/state.json)

Any MCP client (Claude, Gemini CLI / google-genai, …) drives the same workflow:
  reconstruct_scan -> fit_bim -> read user wishes -> edit -> validate -> fix -> commit_revision -> export_package
"""
from __future__ import annotations

import copy
import functools
import inspect
import subprocess
import sys
import time

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response

from . import bim, codes, physics, store

mcp = FastMCP(
    "capture-studio", host="127.0.0.1", port=8765, stateless_http=True, json_response=True,
    instructions=(
        "Capture Studio turns a phone video of a construction site into an editable BIM model.\n"
        "Workflow: 1) reconstruct_scan  2) fit_bim (as-built shell, provenance=scanned)  "
        "3) get_user_messages / list_requirements to learn what the client wants  "
        "4) design with add_space / add_wall / add_opening / add_fixture / update_element / remove_element  "
        "5) validate — fix every 'fail', resolve every 'hold' by asking the user (ask_user -> record_approval), "
        "re-validate until ok  6) set_next_steps, commit_revision, export_package.\n"
        "Units are metres, Z up, origin at the shell's min corner. Use say() for one short line "
        "telling the human what you are doing. Never invent user approval."
    ),
)


# ---------------------------------------------------------------- helpers

def _validate(state: dict, run_physics=True) -> dict:
    model = state["model"]
    code = codes.check(model, state["approvals"])
    prev = state.get("validation") or {}
    phys = physics.run(model) if run_physics else prev.get("physics")
    issues = list(code["issues"]) + (phys["issues"] if (phys and run_physics) else
                                     [i for i in prev.get("issues", []) if i["category"] == "physics" or i["key"].startswith("propping:")])
    for r in state["requirements"]:
        _eval_requirement(model, r)
    cat = lambda c, sev: [i for i in issues if i["category"] == c and i["severity"] in sev]  # noqa: E731
    v = {
        "code_ok": not cat("code", ("fail",)),
        "safety_ok": not cat("safety", ("fail", "hold")),
        "physics_ok": (not cat("physics", ("fail",))) if phys else None,
        "physics_stale": not run_physics,
        "desires_ok": bool(state["requirements"]) and all(r["status"] == "met" for r in state["requirements"]),
        "counts": {s: sum(i["severity"] == s for i in issues) for s in ("fail", "warn", "hold")},
        "profile": code["profile"], "issues": issues, "egress": code["egress"], "physics": phys,
    }
    v["ok"] = bool(v["code_ok"] and v["safety_ok"] and v["physics_ok"] and
                   (v["desires_ok"] or not state["requirements"]))
    state["validation"] = v
    return v


def _eval_requirement(model: dict, r: dict):
    c = r.get("check")
    if not c:
        return
    els = model["elements"].values()
    if c["type"] == "space":
        ok = any(e["kind"] == "space" and e["use"] == c["use"] and bim.poly_area(e["polygon"]) >= c.get("min_area", 0)
                 for e in els)
    elif c["type"] == "element":
        ok = any(e["kind"] == c["kind"] and c.get("name_contains", "").lower() in e["name"].lower()
                 and e["provenance"] == "proposed" for e in els)
    else:
        return
    r["status"] = "met" if ok else "open"
    r["evidence"] = "checked automatically against the model"


def _brief_validation(v: dict) -> dict:
    return {"ok": v["ok"], "code_ok": v["code_ok"], "safety_ok": v["safety_ok"], "physics_ok": v["physics_ok"],
            "physics_stale": v["physics_stale"], "desires_ok": v["desires_ok"], "counts": v["counts"],
            "issues": [{k: i[k] for k in ("key", "category", "severity", "rule", "element", "message", "fix")}
                       for i in v["issues"]]}


def edit(caption_fn):
    """Decorator for mutating tools: load -> mutate -> quick code check -> record -> save."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            state = store.load()
            before = copy.deepcopy(state["model"])
            try:
                result = fn(state, *a, **kw)
            except (ValueError, KeyError) as ex:
                return {"error": str(ex)}
            before_keys = {i["key"] for i in (state.get("validation") or {}).get("issues", [])}
            v = _validate(state, run_physics=False)
            focus = [result["id"]] if isinstance(result, dict) and "id" in result else []
            if state["model"] != before:
                state["stage"] = "design"
            store.record(state, "agent", fn.__name__, caption_fn(result, kw), focus)
            store.save(state)
            now = {i["key"]: i for i in v["issues"]}
            return {"result": result, "revision": f"R{state['model']['revision']}", "seq": state["seq"],
                    "new_issues": [f"[{i['severity']}] {i['message']} → {i['fix']}" for k, i in now.items() if k not in before_keys],
                    "cleared_issues": sorted(before_keys - set(now)),
                    "open": v["counts"], "note": "physics not re-run; call validate()"}
        # hide the injected `state` parameter from the MCP tool schema
        sig = inspect.signature(fn)
        wrapper.__signature__ = sig.replace(parameters=list(sig.parameters.values())[1:], return_annotation=dict)
        wrapper.__annotations__ = {k: v for k, v in fn.__annotations__.items() if k != "state"} | {"return": dict}
        del wrapper.__wrapped__
        return wrapper
    return deco


def _el(state, element_id):
    try:
        return state["model"]["elements"][element_id]
    except KeyError:
        raise ValueError(f"no element {element_id!r}; call get_model() for valid ids")


# ---------------------------------------------------------------- pipeline tools

@mcp.tool()
def pipeline_status() -> dict:
    """Where the project is: scan / cloud / as-built BIM / design, with counts and open items."""
    from . import fit
    s = store.load()
    return {"stage": s["stage"], "video": str(store.DATA / "site.mp4"), "video_exists": (store.DATA / "site.mp4").exists(),
            "point_cloud": fit.cloud_meta(), "model": {k: v for k, v in bim.summary(s["model"]).items() if k != "elements"},
            "requirements": len(s["requirements"]), "unread_user_messages": sum(not m["read"] for m in s["messages"]),
            "validation": s["validation"] and _brief_validation(s["validation"])["counts"], "seq": s["seq"]}


@mcp.tool()
def reconstruct_scan(fps: int = 6, force: bool = False) -> dict:
    """Stage 1→2. Run LingBot-Map on the phone video (data/site.mp4) to get a coloured point cloud
    and camera path, then gravity-align and scale it to metres. Cached after the first run
    (GPU, ~1–2 min); pass force=true to recompute."""
    from . import fit
    raw = store.DATA / "scan" / "raw_predictions.npz"
    if force or not raw.exists():
        p = subprocess.run([sys.executable, "-m", "capture_studio.reconstruct", "--fps", str(fps)],
                           cwd=store.ROOT, capture_output=True, text=True)
        if p.returncode != 0:
            return {"error": p.stderr[-1500:]}
    meta = fit.prepare_cloud()
    s = store.load()
    if s["stage"] == "scan":
        s["stage"] = "cloud"
        store.save(s)
    return meta


@mcp.tool()
def fit_bim(reset_design: bool = False) -> dict:
    """Stage 2→3. Fit the as-built shell (slab, walls, doors, windows) to the point cloud and
    create revision R1. Every fitted element is provenance='scanned' and carries its fit residual
    to the scan (P50/P95 mm). Refuses to overwrite an existing design unless reset_design=true."""
    from . import fit
    s = store.load()
    if any(e["provenance"] == "proposed" for e in s["model"]["elements"].values()) and not reset_design:
        return {"error": "a design already exists; pass reset_design=true to discard it and refit"}
    keep = {k: s[k] for k in ("messages", "requirements")}
    s = store.fresh() | keep
    s["model"], report = fit.fit_shell()
    s["model"]["revision"] = 1
    s["stage"] = "bim"
    _validate(s, run_physics=False)
    store.record(s, "agent", "fit_bim", "As-built model fitted to the scan", [])
    store.save(s)
    return {"report": report, "model": bim.summary(s["model"])}


@mcp.tool()
def set_scale(known_ceiling_height_m: float) -> dict:
    """Recalibrate metric scale from one tape-measured dimension: the true floor-to-ceiling height.
    Video-only reconstruction has no absolute scale, so the default assumes the phone was held
    1.5 m above the floor. Refits the as-built model (discarding the design)."""
    from . import fit
    fit.set_scale_from_ceiling(known_ceiling_height_m)
    return fit_bim(reset_design=True)


# ---------------------------------------------------------------- read tools

@mcp.tool()
def get_model() -> dict:
    """The whole BIM model as a compact list: ids, kinds, coordinates (m), provenance, fit residuals."""
    return bim.summary(store.load()["model"])


@mcp.tool()
def get_element(element_id: str) -> dict:
    """Full properties of one element."""
    s = store.load()
    try:
        return _el(s, element_id)
    except ValueError as ex:
        return {"error": str(ex)}


@mcp.tool()
def get_quantities() -> dict:
    """Material takeoff and rough material cost for proposed (new-build) elements."""
    return bim.quantities(store.load()["model"])


# ---------------------------------------------------------------- design tools

@mcp.tool()
@edit(lambda r, kw: f"Wall added · {r['length']} m")
def add_wall(state, start: list[float], end: list[float], thickness: float = 0.1, height: float | None = None,
             material: str = "aac_block", name: str | None = None, head_restraint: bool = False) -> dict:
    """Add a partition wall along a centreline from start [x,y] to end [x,y].
    height defaults to the ceiling height. material: aac_block | masonry | drywall | concrete | timber.
    A wall whose end touches another wall is tied into it; head_restraint=true fixes its top to the soffit.
    Free-standing walls will fail the physics check."""
    m = state["model"]
    props = dict(start=start, end=end, thickness=thickness, height=height or m["level"]["ceiling_height"],
                 material=material, head_restraint=head_restraint)
    if name:
        props["name"] = name
    return bim.brief(m, bim.add(m, "wall", **props))


@mcp.tool()
@edit(lambda r, kw: f"{r['kind'].title()} added · {r['width']} m")
def add_opening(state, host_wall_id: str, kind: str, offset: float, width: float, height: float = 2.1,
                sill: float = 0.0, name: str | None = None, is_exit: bool = False, lintel: str | None = None) -> dict:
    """Cut a door / window / void into a wall. offset = distance (m) from the wall's start point to
    the near edge of the opening. Doors have sill 0. Openings over 1.2 m wide with wall above need
    lintel (e.g. 'RC 150x200'). Cutting into a scanned wall raises a structural 'hold'."""
    m = state["model"]
    if kind not in bim.OPENING_KINDS:
        raise ValueError(f"kind must be one of {bim.OPENING_KINDS}")
    _el(state, host_wall_id)
    props = dict(host=host_wall_id, offset=offset, width=width, height=height, sill=sill, exit=is_exit)
    if name:
        props["name"] = name
    if lintel:
        props["lintel"] = lintel
    return bim.brief(m, bim.add(m, kind, **props))


@mcp.tool()
@edit(lambda r, kw: f"{r['name']} placed")
def add_fixture(state, name: str, pos: list[float], size: list[float], yaw_deg: float = 0.0,
                material: str = "casework", anchored: bool = False) -> dict:
    """Place a box-shaped fixture (kitchen island, counter, wardrobe, shelving, equipment).
    pos = [x,y] of its centre on the floor, size = [length, depth, height] in m.
    Tall slender items tip over in the physics check unless anchored=true."""
    m = state["model"]
    return bim.brief(m, bim.add(m, "fixture", name=name, pos=pos, size=size, yaw_deg=yaw_deg,
                                material=material, anchored=anchored))


@mcp.tool()
@edit(lambda r, kw: f"Column added")
def add_column(state, pos: list[float], size: list[float] = [0.23, 0.23], height: float | None = None,
               material: str = "concrete", anchored: bool = True, name: str | None = None) -> dict:
    """Add a column at [x,y] with section size [a,b]."""
    m = state["model"]
    props = dict(pos=pos, size=size, height=height or m["level"]["ceiling_height"], material=material, anchored=anchored)
    if name:
        props["name"] = name
    return bim.brief(m, bim.add(m, "column", **props))


@mcp.tool()
@edit(lambda r, kw: f"{r['name']} · {r['area']} m²")
def add_space(state, name: str, use: str, polygon: list[list[float]]) -> dict:
    """Declare a room: polygon = [[x,y],…] corners in order (use the wall centrelines).
    use: living | bedroom | office | dining | study | kitchen | bathroom | store | corridor.
    Code rules (area, width, daylight, doors, egress) are evaluated per space."""
    m = state["model"]
    prof = codes.PROFILES[m["project"]["code_profile"]]
    if use not in prof["room"]:
        raise ValueError(f"use must be one of {sorted(prof['room'])}")
    if len(polygon) < 3:
        raise ValueError("polygon needs at least 3 points")
    return bim.brief(m, bim.add(m, "space", name=name, use=use, polygon=polygon))


@mcp.tool()
@edit(lambda r, kw: f"{r['name']} updated")
def update_element(state, element_id: str, changes: dict) -> dict:
    """Change properties of an element, e.g. {"end":[4.2,3.0]}, {"width":0.9}, {"anchored":true},
    {"guard":true}, {"polygon":[…]}, {"use":"office"}. Geometry is re-validated."""
    m = state["model"]
    el = _el(state, element_id)
    bad = set(changes) & {"id", "kind", "ifc_class", "provenance", "fit"}
    if bad:
        raise ValueError(f"cannot change {sorted(bad)}")
    trial = {**el, **changes}
    bim.validate_element(m, trial)
    el.update(changes)
    for o in bim.openings_of(m, element_id) if el["kind"] == "wall" else []:
        bim.validate_element(m, o)
    return bim.brief(m, el)


@mcp.tool()
@edit(lambda r, kw: f"{r['name']} removed")
def remove_element(state, element_id: str) -> dict:
    """Delete an element (and openings hosted in it). Removing a *scanned* wall is demolition:
    it is allowed but raises a structural 'hold' that only the user can sign off."""
    m = state["model"]
    el = _el(state, element_id)
    for o in list(m["elements"].values()):
        if o.get("host") == element_id:
            del m["elements"][o["id"]]
    del m["elements"][element_id]
    if el["provenance"] == "scanned" and el["kind"] == "wall":
        m.setdefault("removed", []).append({"id": el["id"], "name": el["name"]})
    return {"id": element_id, "name": el["name"], "removed": True}


@mcp.tool()
def undo() -> dict:
    """Revert the model to the state before the last edit."""
    s = store.load()
    if len(s["history"]) < 2:
        return {"error": "nothing to undo"}
    s["history"].pop()
    s["model"] = copy.deepcopy(s["history"][-1]["model"])
    _validate(s, run_physics=False)
    s["seq"] += 1
    store.save(s)
    return {"restored_to_seq": s["history"][-1]["seq"], "model": bim.summary(s["model"])["counts"]}


# ---------------------------------------------------------------- checks

@mcp.tool()
def validate() -> dict:
    """Run everything: building-code rules, site-safety rules, MuJoCo physical robustness
    (clash, lateral tilt test, erection-stage stability) and the client's requirements.
    Returns every open issue with a concrete fix. 'fail' = change the design; 'hold' = needs
    the user's sign-off; 'warn' = site action to carry into next steps."""
    s = store.load()
    v = _validate(s, run_physics=True)
    ph = v["physics"]
    store.record(s, "agent", "validate",
                 "All checks pass" if v["ok"] else f"{v['counts']['fail']} to fix · {v['counts']['hold']} to sign off",
                 [i["element"] for i in v["issues"] if i["element"] and i["severity"] != "warn"])
    store.save(s)
    out = _brief_validation(v)
    out["physics"] = {"bodies": ph["bodies"], "ties": ph["ties"], "clashes": ph["clashes"], "note": ph.get("note")}
    out["egress_m"] = {k: (p and p["length"]) for k, p in v["egress"]["paths"].items()}
    out["requirements"] = s["requirements"]
    return out


@mcp.tool()
def set_code_profile(profile: str) -> dict:
    """Switch jurisdiction: NBC_INDIA_2016 or IRC_2021."""
    if profile not in codes.PROFILES:
        return {"error": f"one of {sorted(codes.PROFILES)}"}
    s = store.load()
    s["model"]["project"]["code_profile"] = profile
    v = _validate(s, run_physics=False)
    store.save(s)
    return _brief_validation(v)


@mcp.tool()
def get_code_rules() -> dict:
    """The numeric rules of the active code profile (room sizes, door widths, egress, guards…)."""
    s = store.load()
    return codes.PROFILES[s["model"]["project"]["code_profile"]]


# ---------------------------------------------------------------- human in the loop

@mcp.tool()
def get_user_messages(mark_read: bool = True) -> dict:
    """Unread messages the human typed in the viewer (wishes, corrections, answers)."""
    s = store.load()
    new = [m for m in s["messages"] if not m["read"]]
    if mark_read and new:
        for m in new:
            m["read"] = True
        store.save(s)
    return {"messages": new, "open_questions": [q for q in s["questions"] if q["answer"] is None]}


@mcp.tool()
def add_requirement(text: str, check: dict | None = None) -> dict:
    """Record something the client wants so it is tracked to completion. Optional auto-check:
    {"type":"space","use":"office","min_area":9.5}  or  {"type":"element","kind":"fixture","name_contains":"island"}.
    Without a check, close it yourself with resolve_requirement once the model satisfies it."""
    s = store.load()
    r = {"id": f"req_{len(s['requirements']) + 1}", "text": text, "check": check, "status": "open", "evidence": ""}
    s["requirements"].append(r)
    _validate(s, run_physics=False)
    s["seq"] += 1
    store.save(s)
    return r


@mcp.tool()
def resolve_requirement(requirement_id: str, status: str, evidence: str) -> dict:
    """Mark a requirement met / unmet with one line of evidence (element ids, measured values)."""
    s = store.load()
    r = next((r for r in s["requirements"] if r["id"] == requirement_id), None)
    if r is None or status not in ("met", "unmet", "open"):
        return {"error": "unknown requirement or status (met|unmet|open)"}
    r["status"], r["evidence"] = status, evidence
    _validate(s, run_physics=False)
    s["seq"] += 1
    store.save(s)
    return r


@mcp.tool()
def list_requirements() -> dict:
    """Client requirements and their status."""
    return {"requirements": store.load()["requirements"]}


@mcp.tool()
def ask_user(question: str, options: list[str]) -> dict:
    """Put a decision in front of the human (shown as buttons in the viewer). Use it for sign-offs
    ('hold' issues) and real design choices. Poll get_user_answer for the reply."""
    s = store.load()
    q = {"id": f"q_{len(s['questions']) + 1}", "question": question, "options": options, "answer": None, "t": time.time()}
    s["questions"].append(q)
    s["seq"] += 1
    store.save(s)
    return q


@mcp.tool()
def get_user_answer(question_id: str, wait_seconds: int = 0) -> dict:
    """The human's answer to ask_user (answer is null until they click). Optionally wait up to 120 s."""
    deadline = time.time() + min(wait_seconds, 120)
    while True:
        s = store.load()
        q = next((q for q in s["questions"] if q["id"] == question_id), None)
        if q is None:
            return {"error": "unknown question"}
        if q["answer"] is not None or time.time() >= deadline:
            return q
        time.sleep(1)


@mcp.tool()
def record_approval(issue_key: str, question_id: str) -> dict:
    """Clear a 'hold' issue using the human's answer to a question you asked about it.
    Only an affirmative human answer counts; the approval is stored with the answer text."""
    s = store.load()
    q = next((q for q in s["questions"] if q["id"] == question_id), None)
    if q is None or q["answer"] is None:
        return {"error": "no human answer yet for that question"}
    if not q["answer"].lower().startswith(("yes", "approve", "confirm", "ok")):
        return {"error": f"the human answered {q['answer']!r} — not an approval; change the design instead"}
    s["approvals"][issue_key] = {"question": q["question"], "answer": q["answer"], "t": time.time()}
    v = _validate(s, run_physics=False)
    store.record(s, "human", "approve", f"Signed off: {q['answer']}", [])
    store.save(s)
    return {"approved": issue_key, "open": v["counts"]}


@mcp.tool()
def say(text: str) -> dict:
    """One short line shown to the human in the viewer about what you are doing or decided."""
    s = store.load()
    s["agent_note"] = text[:160]
    s["seq"] += 1
    store.save(s)
    return {"ok": True}


# ---------------------------------------------------------------- deliver

@mcp.tool()
def set_next_steps(steps: list[dict]) -> dict:
    """Publish the practical plan that follows from this design. Each step:
    {"title": str, "who": "engineer|contractor|owner|authority", "why": str}. Order matters.
    Include sign-offs, permits, temporary works from 'warn' issues, procurement and sequencing."""
    s = store.load()
    s["next_steps"] = [{"title": x.get("title", ""), "who": x.get("who", ""), "why": x.get("why", "")} for x in steps]
    s["seq"] += 1
    store.save(s)
    return {"count": len(s["next_steps"])}


@mcp.tool()
def commit_revision(note: str) -> dict:
    """Freeze the current design as the next revision (R2, R3, …). Refuses while validation has
    'fail' or 'hold' items."""
    s = store.load()
    v = _validate(s, run_physics=True)
    if not (v["code_ok"] and v["safety_ok"] and v["physics_ok"]):
        store.save(s)
        return {"error": "not committed: open fail/hold issues", "validation": _brief_validation(v)}
    s["model"]["revision"] += 1
    s["model"]["revision_note"] = note
    s["stage"] = "design"
    store.record(s, "agent", "commit_revision", f"R{s['model']['revision']} · {note}"[:80], [])
    store.save(s)
    return {"revision": f"R{s['model']['revision']}", "note": note}


@mcp.tool()
def export_package() -> dict:
    """Write the deliverable to data/export/: IFC4, GLB, MuJoCo MJCF, model.json,
    validation.json (checks + scan-fit residuals + drop test), README.md and SHA256SUMS."""
    from . import export
    s = store.load()
    v = _validate(s, run_physics=True)
    store.save(s)
    return export.write_package(s, v)


# ---------------------------------------------------------------- viewer + API

WEB = store.ROOT / "web"


@mcp.custom_route("/", methods=["GET"])
async def index(_: Request):
    return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-store"})


@mcp.custom_route("/web/{path:path}", methods=["GET"])
async def web_file(r: Request):
    f = (WEB / r.path_params["path"]).resolve()
    return FileResponse(f, headers={"Cache-Control": "no-store"}) if f.is_file() and WEB in f.parents else Response(status_code=404)


@mcp.custom_route("/data/{path:path}", methods=["GET"])
async def data_file(r: Request):
    f = (store.DATA / r.path_params["path"]).resolve()
    return FileResponse(f) if f.is_file() and store.DATA in f.parents else Response(status_code=404)


@mcp.custom_route("/api/state", methods=["GET"])
async def api_state(r: Request):
    s = store.load()
    if r.query_params.get("since") == str(s["seq"]):
        return Response(status_code=204)
    return JSONResponse(store.view(s))


@mcp.custom_route("/api/message", methods=["POST"])
async def api_message(r: Request):
    body = await r.json()
    s = store.load()
    s["messages"].append({"id": f"msg_{len(s['messages']) + 1}", "text": str(body.get("text", ""))[:500],
                          "selected": body.get("selected"), "t": time.time(), "read": False})
    s["seq"] += 1
    store.save(s)
    return JSONResponse({"ok": True})


@mcp.custom_route("/api/answer", methods=["POST"])
async def api_answer(r: Request):
    body = await r.json()
    s = store.load()
    for q in s["questions"]:
        if q["id"] == body.get("id") and q["answer"] is None:
            q["answer"] = str(body.get("answer", ""))[:200]
    s["seq"] += 1
    store.save(s)
    return JSONResponse({"ok": True})


def main():
    mcp.run(transport="stdio" if "--stdio" in sys.argv else "streamable-http")


if __name__ == "__main__":
    main()
