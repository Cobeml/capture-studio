"""Independent design review with the Gemini Decisions API (model gemini-decision-1).

The rule engine (codes.py) and MuJoCo (physics.py) answer questions that can be computed.
This module asks the ones that cannot: closed yes/no and rubric questions about whether the
design *really* does what was asked — "is the client's requirement delivered, or merely
relabelled?", "is a machine flagged guarded actually enclosed by modelled guarding?" — over
one shared input (the brief, the design as JSON, the rule results and a plan drawing).
The API returns a calibrated probability and a confidence per question and generates no text,
so every answer is a bounded decision we can threshold:

    confident yes  -> nothing to do
    confident no   -> 'fail' issue (category 'review'), the agent must fix it
    unsure         -> 'warn' issue: a human should look

It is optional: with no API key or SDK support, review() returns {"available": False}.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import cv2
import numpy as np

from . import bim

MODEL = os.environ.get("GEMINI_DECISION_MODEL", "gemini-decision-1")
PASS, FAIL, MIN_CONF = 0.70, 0.30, 0.60


def _load_env():
    """Read KEY=VALUE pairs from the repo's .env into the process (never printed, never overriding)."""
    f = Path(__file__).resolve().parent.parent / ".env"
    if not f.exists():
        return
    for line in f.read_text().splitlines():
        line = line.strip().removeprefix("export ")
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _client():
    _load_env()
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return None
    try:
        from google import genai
        c = genai.Client(api_key=key, http_options={"api_version": "v1beta"})
        return c if hasattr(c, "decisions") else None
    except Exception:
        return None


def plan_png(model: dict, px_per_m=36) -> bytes:
    """Plan drawing the reviewer looks at: existing = grey, proposed = orange, zones = green outline, exit = red ring."""
    slab = next((e for e in model["elements"].values() if e["kind"] == "slab"), None)
    if slab is None:
        return b""
    S = px_per_m
    x0, y0 = slab["min"][0] - 0.5, slab["min"][1] - 0.5
    W, H = int((slab["max"][0] + 0.5 - x0) * S), int((slab["max"][1] + 0.5 - y0) * S)
    img = np.full((H, W, 3), 245, np.uint8)
    px = lambda p: (int((p[0] - x0) * S), H - int((p[1] - y0) * S))  # noqa: E731
    for e in model["elements"].values():
        new = bim.is_new(e)
        if e["kind"] in ("equipment", "fixture", "column"):
            fp = bim.footprint(model, e)
            for i in range(0, len(fp), 4):
                quad = np.array([px(p) for p in fp[i:i + 4]], np.int32)
                cv2.fillPoly(img, [quad], (40, 130, 255) if new else (150, 150, 150))
                cv2.polylines(img, [quad], True, (60, 60, 60), 1)
        elif e["kind"] == "wall":
            for b in bim.solids(model, e):
                if b["c"][2] - b["s"][2] / 2 > 0.3:
                    continue                                   # lintels above openings: leave the gap visible
                p, yaw = bim.pose(model, e), bim.wall_yaw(e)
                c, s = np.cos(yaw), np.sin(yaw)
                a = (p["pos"][0] + (b["c"][0] - b["s"][0] / 2) * c, p["pos"][1] + (b["c"][0] - b["s"][0] / 2) * s)
                d = (p["pos"][0] + (b["c"][0] + b["s"][0] / 2) * c, p["pos"][1] + (b["c"][0] + b["s"][0] / 2) * s)
                cv2.line(img, px(a), px(d), (0, 100, 230) if new else (30, 30, 30), max(2, int(e["thickness"] * S)))
        elif e["kind"] == "space":
            cv2.polylines(img, [np.array([px(p) for p in e["polygon"]], np.int32)], True, (60, 160, 60), 2)
            cv2.putText(img, e["name"][:22], px(bim.poly_centroid(e["polygon"])), 0, 0.42, (30, 110, 30), 1, cv2.LINE_AA)
    for e in model["elements"].values():
        if e["kind"] in bim.OPENING_KINDS and e.get("exit"):
            cv2.circle(img, px(bim.opening_center(model, e)), 9, (0, 0, 220), 3)
    return cv2.imencode(".png", img)[1].tobytes()


def build_request(state: dict, validation: dict | None) -> tuple[list, dict, dict]:
    """(input blocks, questions, meta) for one review of the current design."""
    model = state["model"]
    els = model["elements"]
    new = [bim.brief(model, e) | ({"parts": [q["name"] for q in e["parts"]]} if e.get("parts") else {})
           for e in els.values() if bim.is_new(e)]
    context = {
        "site": model["project"].get("site_type", "site"), "rules": model["project"]["code_profile"],
        "client_requirements": [{"id": r["id"], "text": r["text"]} for r in state["requirements"]],
        "client_messages": [m["text"] for m in state["messages"]][-6:],
        "proposed_or_relocated_elements": new,
        "existing_elements": len(els) - len(new),
        "rule_engine_open_issues": [{"severity": i["severity"], "rule": i["rule"], "message": i["message"]}
                                    for i in (validation or {}).get("issues", []) if i["category"] != "review"],
    }
    blocks = [{"type": "text", "text":
               "Design review of a proposed change to a scanned site. Units are metres. JSON follows, then a plan "
               "drawing (existing = grey, proposed or relocated = orange, zones = green outlines, exit = red ring).\n"
               + json.dumps(context)}]
    png = plan_png(model)
    if png:
        blocks.append({"type": "image", "data": base64.b64encode(png).decode(), "mime_type": "image/png", "resolution": "high"})
    questions, meta = {}, {}
    for r in state["requirements"]:
        k = f"req_{r['id']}"
        questions[k] = {"type": "noul",
                        "instructions": f"Does the proposed design actually deliver this client requirement: \"{r['text']}\"?",
                        "criteria": {"true_description": "Specific proposed elements or zones in the JSON clearly provide it.",
                                     "false_description": "Missing, only partly provided, or satisfied only by relabelling "
                                                          "something that already existed."}}
        meta[k] = {"element": None, "rule": "Requirement really delivered", "what": r["text"], "gate": True}
    for e in els.values():
        if bim.is_new(e) and e["kind"] == "equipment" and e.get("guarded"):
            k = f"guard_{e['id']}"
            questions[k] = {"type": "noul",
                            "instructions": f"The machine \"{e['name']}\" at {e['pos']} is flagged as guarded. Is physical guarding "
                                            "for it actually modelled (fence or wall runs around it with a gate, an enclosure or "
                                            "light-curtain part attached to it)?",
                            "criteria": {"true_description": "Guard elements or parts enclosing or protecting this machine are in the design.",
                                         "false_description": "Only the flag is set; nothing in the design physically guards it."}}
            meta[k] = {"element": e["id"], "rule": "Guarding actually modelled", "what": e["name"], "gate": True}
    if png and new:
        questions["no_overlap"] = {"type": "noul",
                                   "instructions": "Looking at the plan drawing: are all orange (proposed) shapes clear of the grey (existing) shapes?",
                                   "criteria": {"true_description": "No orange shape overlaps or touches a grey shape.",
                                                "false_description": "At least one orange shape overlaps a grey shape."}}
        meta["no_overlap"] = {"element": None, "rule": "Plan looks clash-free", "what": "plan drawing", "gate": False}
        questions["plan_quality"] = {"type": "score",
                                     "instructions": "Rate the proposed layout as a practical, buildable change to this floor.",
                                     "criteria": ["Unusable: blocks circulation or ignores the brief.",
                                                  "Weak: delivers part of the brief with obvious layout problems.",
                                                  "Sound: delivers the brief with a workable layout.",
                                                  "Strong: delivers the brief with clear circulation, guarding and access."]}
        meta["plan_quality"] = {"element": None, "rule": "Layout quality", "what": "overall", "gate": False}
    return blocks, questions, meta


def review(state: dict, validation: dict | None = None) -> dict:
    """Run the review. Returns {"available", "model", "answers": {...}, "issues": [...], "usage"}."""
    client = _client()
    if client is None:
        return {"available": False, "reason": "no GEMINI_API_KEY or google-genai without decisions support"}
    blocks, questions, meta = build_request(state, validation)
    if not questions:
        return {"available": True, "model": MODEL, "answers": {}, "issues": [], "note": "nothing to review yet"}
    try:
        res = client.decisions.create(model=MODEL, input=blocks, questions=questions)
    except Exception as ex:                                       # network, quota, schema drift in a v1beta API
        return {"available": False, "reason": f"{type(ex).__name__}: {str(ex)[:300]}"}
    answers, issues = {}, []
    for k, q in questions.items():
        a = res.answers[k]
        conf = float(getattr(a, "confidence", 0.0) or 0.0)
        m = meta[k]
        if q["type"] == "score":
            answers[k] = {"rule": m["rule"], "score": round(float(a.score), 2), "of": len(q["criteria"]) - 1, "confidence": round(conf, 2)}
            continue
        p = float(a.noul)
        answers[k] = {"rule": m["rule"], "about": m["what"], "p_yes": round(p, 2), "confidence": round(conf, 2)}
        if p >= PASS and conf >= MIN_CONF:
            continue
        sure_no = p <= FAIL and conf >= MIN_CONF and m["gate"]
        issues.append({
            "key": f"review:{k}", "category": "review", "severity": "fail" if sure_no else "warn", "rule": m["rule"],
            "ref": f"Gemini Decisions ({MODEL})", "element": m["element"],
            "message": (f"Independent review says NO (p={p:.2f}): {m['what']}." if sure_no else
                        f"Independent review is unsure (p={p:.2f}, confidence {conf:.2f}): {m['what']}."),
            "fix": "Model what is missing (real elements, not flags or labels), then validate again." if sure_no
                   else "Have a person check this point before relying on it."})
    usage = getattr(res, "usage", None)
    return {"available": True, "model": MODEL, "answers": answers, "issues": issues,
            "usage": str(getattr(usage, "input_tokens_by_modality", "")) or None}
