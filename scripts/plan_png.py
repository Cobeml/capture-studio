"""Plan drawing of the current BIM model with a metre grid (dev aid): python scripts/plan_png.py OUT.png"""
import json, sys, math
import numpy as np, cv2
from capture_studio import bim, store
m = store.load()["model"]; S = 44
slab = next(e for e in m["elements"].values() if e["kind"] == "slab")
x0, y0 = slab["min"][0] - 1, slab["min"][1] - 1
W, H = int((slab["max"][0] + 1 - x0) * S), int((slab["max"][1] + 1 - y0) * S)
img = np.full((H, W, 3), 22, np.uint8)
px = lambda p: (int((p[0] - x0) * S), H - int((p[1] - y0) * S))
for x in range(int(x0), int(slab["max"][0]) + 2):
    cv2.line(img, px((x, y0)), px((x, y0 + 99)), (45, 45, 45), 1); cv2.putText(img, str(x), (px((x, 0))[0] + 2, H - 4), 0, 0.4, (0, 255, 255), 1)
for y in range(int(y0), int(slab["max"][1]) + 2):
    cv2.line(img, px((x0, y)), px((x0 + 99, y)), (45, 45, 45), 1); cv2.putText(img, str(y), (2, px((0, y))[1] - 2), 0, 0.4, (0, 255, 255), 1)
for e in m["elements"].values():
    if e["kind"] in ("equipment", "fixture", "column"):
        fp = np.array([px(p) for p in bim.footprint(m, e)], np.int32)
        col = (60, 140, 255) if e["provenance"] == "proposed" else tuple(int(e.get("color", "#888888")[i:i + 2], 16) for i in (5, 3, 1))
        cv2.fillPoly(img, [fp], col); cv2.polylines(img, [fp], True, (230, 230, 230), 1)
        cv2.putText(img, e["name"].replace("Machine ", "M").replace("Rack ", "R").replace("Bench ", "B"), tuple(fp.mean(0).astype(int) - (10, 0)), 0, 0.38, (255, 255, 255), 1)
    elif e["kind"] == "wall":
        cv2.line(img, px(e["start"]), px(e["end"]), (200, 200, 200) if e["provenance"] == "scanned" else (60, 140, 255), max(2, int(e["thickness"] * S)))
    elif e["kind"] == "space":
        cv2.polylines(img, [np.array([px(p) for p in e["polygon"]], np.int32)], True, (120, 220, 120), 2)
for e in m["elements"].values():
    if e["kind"] in bim.OPENING_KINDS:
        c = bim.opening_center(m, e); cv2.circle(img, px(c), 7, (0, 0, 255) if e.get("exit") else (255, 200, 0), 2)
cv2.imwrite(sys.argv[1], img)
