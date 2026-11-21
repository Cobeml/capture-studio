"""Scene-understanding helpers: oriented boxes of segmented points and the hero GLB convention."""
import math

import numpy as np

from capture_studio import objects


def _box_points(cx, cy, l, d, h, yaw, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    u, v, z = rng.uniform(-l / 2, l / 2, n), rng.uniform(-d / 2, d / 2, n), rng.uniform(0.05, h, n)
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    return np.stack([cx + u * c - v * s, cy + u * s + v * c, z], 1)


def test_obb_recovers_rotated_box():
    for yaw in (0.0, 30.0, -60.0, 90.0):
        pos, size, ang = objects.obb(_box_points(4.0, 2.0, 2.4, 0.9, 1.8, yaw))
        assert math.dist(pos, (4.0, 2.0)) < 0.05
        assert abs(size[0] - 2.4) < 0.08 and abs(size[1] - 0.9) < 0.08 and abs(size[2] - 1.8) < 0.05
        diff = (ang - yaw + 90) % 180 - 90                           # box symmetry: yaw mod 180
        assert abs(diff) < 2.0, (yaw, ang)


def test_coarse_category():
    assert objects.coarse_category("CNC tool holder cart") == "cart"
    assert objects.coarse_category("Wire shelving rack") == "rack"
    assert objects.coarse_category("Haas SL-30 CNC lathe") == "machine"


def test_hero_glb_is_object_local_and_y_up(tmp_path):
    import trimesh
    from capture_studio import hero
    # a 2 x 1 x 1.5 m block standing on the floor at the object's origin, long side along local +x
    V = np.array([[x, y, z] for x in (-1, 1) for y in (-0.5, 0.5) for z in (0, 1.5)], float)
    F = trimesh.convex.convex_hull(V).faces
    UV = np.zeros((len(V), 2), np.float32)
    tex = np.full((8, 8, 3), 200, np.uint8)
    path = tmp_path / "o.glb"
    hero.export_glb([{"key": "obj_01", "V_local": V, "F": F, "UV": UV, "tex": tex}], path)
    sc = trimesh.load(path)
    assert "obj_01" in sc.graph.nodes
    g = sc.dump(concatenate=True).vertices                           # glTF Y-up as stored
    ext = g.max(0) - g.min(0)
    assert abs(ext[0] - 2.0) < 1e-4 and abs(ext[1] - 1.5) < 1e-4 and abs(ext[2] - 1.0) < 1e-4
    model = np.stack([g[:, 0], -g[:, 2], g[:, 1]], 1)               # viewer: gltf (x,y,z) -> model (x,-z,y)
    assert abs(model[:, 2].min()) < 1e-6 and abs(model[:, 2].max() - 1.5) < 1e-6


def test_to_local_inverts_pose():
    from capture_studio import hero
    Vw = _box_points(5.0, 3.0, 2.0, 1.0, 1.0, 30.0, n=100)
    L = hero.to_local(Vw, [5.0, 3.0], 30.0)
    assert np.all(np.abs(L[:, 0]) <= 1.0 + 1e-6) and np.all(np.abs(L[:, 1]) <= 0.5 + 1e-6)
