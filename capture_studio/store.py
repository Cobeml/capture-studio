"""Project state on disk — the single source of truth shared by the MCP tools and the viewer."""
from __future__ import annotations

import copy
import json
import os
import threading
import time
from pathlib import Path

from . import bim

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("CAPTURE_STUDIO_DATA", ROOT / "data"))
STATE = DATA / "project" / "state.json"
_lock = threading.RLock()


def fresh() -> dict:
    return {
        "seq": 0, "model": bim.empty_model(), "history": [],
        "requirements": [], "messages": [], "questions": [], "approvals": {},
        "validation": None, "next_steps": [], "agent_note": "", "stage": "scan",
    }


def load() -> dict:
    with _lock:
        if STATE.exists():
            return json.loads(STATE.read_text())
        return fresh()


def save(state: dict):
    with _lock:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATE.with_suffix(".tmp")
        tmp.write_text(json.dumps(state))
        tmp.replace(STATE)


def record(state: dict, actor: str, action: str, caption: str, focus: list[str] | None = None):
    """Append a history event (with a full model copy, so the viewer can replay and we can undo)."""
    state["seq"] += 1
    state["history"].append({
        "seq": state["seq"], "t": time.time(), "actor": actor, "action": action,
        "caption": caption, "focus": focus or [], "model": copy.deepcopy(state["model"]),
        "validation": summarize_validation(state.get("validation")),
    })


def summarize_validation(v: dict | None) -> dict | None:
    if not v:
        return None
    return {k: v.get(k) for k in ("code_ok", "safety_ok", "physics_ok", "desires_ok", "counts")}


def view(state: dict) -> dict:
    """What the viewer gets: render-ready snapshots for the live model and every history step."""
    out = {k: state[k] for k in ("seq", "requirements", "messages", "questions", "validation",
                                 "next_steps", "agent_note", "stage")}
    out["model"] = bim.snapshot(state["model"])
    out["history"] = [{**{k: h[k] for k in ("seq", "t", "actor", "action", "caption", "focus", "validation")},
                       "model": bim.snapshot(h["model"])} for h in state["history"]]
    return out
