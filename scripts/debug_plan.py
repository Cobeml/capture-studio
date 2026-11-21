"""Top-down + side renders of the aligned cloud (dev aid):  python scripts/debug_plan.py OUT.png"""
import sys
import numpy as np, cv2
from capture_studio import fit
z = np.load(fit.SCAN / "aligned.npz"); P, C, H = z["P"], z["C"], float(z["ceiling"])
raw = np.fromfile(fit.SCAN / "cloud.bin", dtype=np.uint8)[len(P) * 12:].reshape(-1, 3)
s = 40  # px per metre
lo = P[:, :2].min(0); ext = P[:, :2].max(0) - lo
W, Hh = int(ext[0] * s) + 1, int(ext[1] * s) + 1
def top(mask):
    img = np.zeros((Hh, W, 3), np.uint8)
    q = P[mask]; c = raw[mask]; o = np.argsort(q[:, 2])
    u = ((q[o, 0] - lo[0]) * s).astype(int); v = Hh - 1 - ((q[o, 1] - lo[1]) * s).astype(int)
    img[v, u] = c[o][:, ::-1]
    return img
a = top(P[:, 2] < min(H - 0.3, 2.2))
for i in range(len(C) - 1):
    p = lambda k: (int((C[k, 0] - lo[0]) * s), Hh - 1 - int((C[k, 1] - lo[1]) * s))
    cv2.line(a, p(i), p(i + 1), (0, 140, 255), 1)
for x in range(0, int(ext[0]) + 1, 5): cv2.putText(a, str(x), (int(x * s), 12), 0, 0.4, (0, 255, 255), 1)
side = np.zeros((int((H + 1.5) * s), W, 3), np.uint8)
u = ((P[:, 0] - lo[0]) * s).astype(int); v = np.clip(side.shape[0] - 1 - ((P[:, 2] + 0.5) * s).astype(int), 0, side.shape[0] - 1)
side[v, u] = raw[:, ::-1]
cv2.imwrite(sys.argv[1], np.vstack([a, side]))
print("extent", ext.round(1), "ceiling", H, "n", len(P), "img", a.shape)
