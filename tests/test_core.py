"""Core behaviour on a synthetic 8 x 6 m shell (no scan needed):  python -m pytest tests -q"""
from capture_studio import bim, codes, physics


def shell():
    m = bim.empty_model("test")
    H = 2.9
    m["level"]["ceiling_height"] = H
    bim.add(m, "slab", "scanned", min=[-0.1, -0.1], max=[8.1, 6.1], top=0.0, thickness=0.15, material="concrete")
    w = {}
    for name, a, b in (("S", [0, 0], [8, 0]), ("E", [8, 0], [8, 6]), ("N", [8, 6], [0, 6]), ("W", [0, 6], [0, 0])):
        w[name] = bim.add(m, "wall", "scanned", name=name, start=a, end=b, thickness=0.2, height=H,
                          material="masonry", exterior=True)
    bim.add(m, "door", "scanned", host=w["W"]["id"], offset=2.5, width=1.0, height=2.1, sill=0.0, exit=True)
    bim.add(m, "window", "scanned", host=w["S"]["id"], offset=1.0, width=2.5, height=1.5, sill=1.0)
    bim.add(m, "window", "scanned", host=w["S"]["id"], offset=5.5, width=1.5, height=1.2, sill=1.0)
    return m, w


def keys(result):
    return {i["key"].split(":")[0] for i in result["issues"]}


def test_wall_boxes_cut_by_openings():
    m, w = shell()
    boxes = bim.solids(m, w["S"])
    assert len(boxes) == 7          # 3 piers + 2 sills + 2 heads
    full = 8 * 0.2 * 2.9
    assert abs(bim.volume(m, w["S"]) - (full - (2.5 * 1.5 + 1.5 * 1.2) * 0.2)) < 1e-6


def test_code_flags_small_dark_room_and_narrow_door():
    m, w = shell()
    p = bim.add(m, "wall", start=[5.5, 0], end=[5.5, 6], thickness=0.1, height=2.9, material="aac_block")
    bim.add(m, "door", host=p["id"], offset=2.0, width=0.7, height=2.1, sill=0.0)
    bim.add(m, "space", name="Living", use="living", polygon=[[0, 0], [5.5, 0], [5.5, 6], [0, 6]])
    bim.add(m, "space", name="Office", use="office", polygon=[[5.5, 0], [8, 0], [8, 6], [5.5, 6]])
    r = codes.check(m)
    k = keys(r)
    assert "doorw" in k               # 0.7 m door < 0.9 m
    assert "egress" in k              # and nobody fits through it at 0.9 m clear width
    assert not r["ok"]
    door = next(e for e in m["elements"].values() if e["kind"] == "door" and e["provenance"] == "proposed")
    door["width"] = 1.0
    r = codes.check(m)
    assert r["ok"], r["issues"]
    assert 3 < r["egress"]["paths"][next(e["id"] for e in m["elements"].values() if e.get("use") == "office")]["length"] < 15


def test_cutting_scanned_wall_needs_signoff():
    m, w = shell()
    o = bim.add(m, "door", host=w["N"]["id"], offset=3, width=1.0, height=2.1, sill=0.0)
    r = codes.check(m)
    assert f"cut:{o['id']}" in {i["key"] for i in r["issues"]}
    assert codes.check(m, {f"cut:{o['id']}": {}})["counts"]["hold"] == 0


def test_freestanding_wall_topples_tied_wall_holds():
    m, _ = shell()
    free = bim.add(m, "wall", name="free", start=[2, 2], end=[5, 2], thickness=0.1, height=2.9, material="aac_block")
    tied = bim.add(m, "wall", name="tied", start=[6, 0], end=[6, 6], thickness=0.1, height=2.9, material="aac_block")
    r = physics.run(m)
    assert not r["bodies"][free["id"]]["stable"]
    assert r["bodies"][tied["id"]]["stable"]
    assert not r["bodies"][tied["id"]]["stable_without_ties"]
    assert any(i["key"] == f"stability:{free['id']}" for i in r["issues"])
    assert any(i["key"] == f"propping:{tied['id']}" for i in r["issues"])
    free["head_restraint"] = True
    assert physics.run(m)["ok"]


def test_tall_cabinet_needs_anchor_island_does_not():
    m, _ = shell()
    cab = bim.add(m, "fixture", name="wardrobe", pos=[4, 3], size=[1.2, 0.3, 2.4], material="casework")
    isl = bim.add(m, "fixture", name="island", pos=[2, 4], size=[1.8, 0.9, 0.9], material="casework")
    r = physics.run(m)
    assert not r["bodies"][cab["id"]]["stable"] and r["bodies"][isl["id"]]["stable"]
    cab["anchored"] = True
    assert physics.run(m)["ok"]


def test_clash_detected():
    m, _ = shell()
    bim.add(m, "fixture", name="in the wall", pos=[8.0, 3], size=[1, 1, 1], material="casework")
    r = physics.run(m)
    assert r["clashes"] and not r["ok"]


def test_drop_test_nothing_falls_through():
    m, _ = shell()
    d = physics.drop_test(m, n=12)
    assert d["ok"] and d["fell_through"] == 0 and d["at_rest"] >= 10
