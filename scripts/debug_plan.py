"""Top-down + side renders of the aligned cloud (dev aid):  python scripts/debug_plan.py OUT.png [--labels]

--labels  colour points by object (cloud_labels.bin) and write each object's key at its centre."""
import json
import sys

import cv2
import numpy as np

from capture_studio import fit

z = np.load(fit.SCAN / "aligned.npz"); P, C, H = z["P"], z["C"], float(z["ceiling"])
raw = np.fromfile(fit.SCAN / "cloud.bin", dtype=np.uint8)[len(P) * 12:].reshape(-1, 3).copy()
labels = None
if "--labels" in sys.argv and (fit.SCAN / "cloud_labels.bin").exists():
    labels = np.fromfile(fit.SCAN / "cloud_labels.bin", dtype=np.uint16).astype(int)
    pal = np.random.default_rng(3).integers(60, 255, (labels.max() + 2, 3)).astype(np.uint8)
    raw = (raw * 0.35).astype(np.uint8)
    raw[labels > 0] = pal[labels[labels > 0]]
s = 40  # px per metre
lo = P[:, :2].min(0); ext = P[:, :2].max(0) - lo
W, Hh = int(ext[0] * s) + 1, int(ext[1] * s) + 1


def px(x, y):
    return int((x - lo[0]) * s), Hh - 1 - int((y - lo[1]) * s)


def top(mask):
    img = np.zeros((Hh, W, 3), np.uint8)
    q = P[mask]; c = raw[mask]; o = np.argsort(q[:, 2])
    u = ((q[o, 0] - lo[0]) * s).astype(int); v = Hh - 1 - ((q[o, 1] - lo[1]) * s).astype(int)
    img[v, u] = c[o][:, ::-1]
    return img


a = top(P[:, 2] < min(H - 0.3, 2.2))
for i in range(len(C) - 1):
    cv2.line(a, px(*C[i, :2]), px(*C[i + 1, :2]), (0, 140, 255), 1)
for x in range(int(np.ceil(lo[0])), int(lo[0] + ext[0]) + 1, 5):
    cv2.putText(a, str(x), (px(x, 0)[0], 12), 0, 0.4, (0, 255, 255), 1)
if labels is not None and (fit.SCAN / "objects.json").exists():
    for o in json.loads((fit.SCAN / "objects.json").read_text())["objects"]:
        cv2.putText(a, o["key"][4:], px(*o["pos"]), 0, 0.4, (255, 255, 255), 1)
side = np.zeros((int((H + 1.5) * s), W, 3), np.uint8)
u = ((P[:, 0] - lo[0]) * s).astype(int); v = np.clip(side.shape[0] - 1 - ((P[:, 2] + 0.5) * s).astype(int), 0, side.shape[0] - 1)
side[v, u] = raw[:, ::-1]
cv2.imwrite(sys.argv[1], np.vstack([a, side]))
print("extent", ext.round(1), "ceiling", H, "n", len(P), "img", a.shape)
