"""The deliverable package is written and counts back correctly."""
import json

from capture_studio import bim, export, physics, codes, store
from test_core import factory


def test_package_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA", tmp_path)
    m = factory()
    m["revision"] = 2
    bim.add(m, "equipment", name="CNC mill", pos=[2.5, 1.0], size=[2.4, 1.6, 2.2], category="machine",
            material="machine", mass_kg=4500, hazard=True, guarded=True, anchored=True, fixed=True)
    state = store.fresh() | {"model": m, "next_steps": [{"title": "Pour machine pad", "who": "contractor", "why": "anchorage"}]}
    code, phys = codes.check(m), physics.run(m)
    v = {"profile": code["profile"], "code_ok": True, "safety_ok": True, "physics_ok": phys["ok"], "desires_ok": True,
         "counts": code["counts"], "issues": code["issues"] + phys["issues"], "physics": phys}
    out = export.write_package(state, v)
    assert {"README.md", "SHA256SUMS", "validation.json", "model.json"} <= set(out["files"])
    assert any(f.endswith(".ifc") for f in out["files"]) and any(f.endswith(".glb") for f in out["files"])
    val = json.loads((tmp_path / "export" / "validation.json").read_text())
    assert val["elements"]["ifc_products"] == val["elements"]["expected"]
    assert val["physics"]["drop_test"]["fell_through"] == 0
