"""Record demo footage of the viewer, frame by frame, at 1920x1080 / 30 fps.

    python scripts/record_demo.py OUT_DIR [BASE_URL] [options]

Writes OUT_DIR/01_video.mp4, 02_point_cloud.mp4, 03_bim.mp4, 04_agentic_design.mp4 and full_demo.mp4.

The page is opened with ?capture=1: its clock is virtual and only advances when this script calls
window.__demo.frame(ms), which also seeks the <video> to the matching time and pins CSS transitions to the
same clock. Each frame is then screenshotted and piped to ffmpeg, so the output is smooth however slowly
the GPU (or SwiftShader) renders. The stages run back to back in one page, so the clips cut together cleanly.

Human-in-the-loop footage at the end of the design stage (skip with --no-hitl):
  * types --message into "Tell the agent…" and presses Send (a real POST /api/message: the agent will see it);
  * shows the pending ask_user question, or, if there is none, calls the MCP tool ask_user itself with a
    question derived from the open issues (or --question / --options), then clicks the first answer
    (a real POST /api/answer).

Follow mode (--mode follow, or all): first-person clips locked to the phone's path, for match cuts.
  follow_video.mp4 (the source video, trimmed/retimed by ffmpeg), follow_cloud.mp4, follow_bim.mp4 (as-built),
  follow_design.mp4 (live model) all have the same frame count and timing: frame k is video time
  START + k * speed / fps. The viewer puts its camera on the interpolated LingBot-Map pose for that time with
  the video's horizontal FOV. Composites built from them: follow_matchcut.mp4 (video -> cloud -> BIM -> design
  wipes mid-motion) and follow_split.mp4 (2x2 synced panels). Read-only on the server.

    python scripts/record_demo.py OUT http://127.0.0.1:8642 --mode follow [--follow-range 87,112.5] [--follow-speed 1]
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
STAGES = [("scan", "01_video"), ("cloud", "02_point_cloud"), ("bim", "03_bim"), ("design", "04_agentic_design")]
MAX_S = {"scan": 10, "cloud": 40, "bim": 16, "design": 150}
GL_ARGS = {
    "gpu": ["--use-gl=angle", "--use-angle=gl-egl", "--enable-gpu", "--ignore-gpu-blocklist", "--enable-gpu-rasterization"],
    "swiftshader": ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"],
}


def parse():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out")
    ap.add_argument("base", nargs="?", default="http://127.0.0.1:8642")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--gl", choices=["auto", "gpu", "swiftshader"], default="auto")
    ap.add_argument("--stages", default="scan,cloud,bim,design", help="comma list (subset of scan,cloud,bim,design)")
    ap.add_argument("--no-hitl", action="store_true", help="no typed message / question card in the design stage")
    ap.add_argument("--message", default="Keep a clear aisle to the exit")
    ap.add_argument("--question", default=None, help="ask_user question if none is pending (default: derived from issues)")
    ap.add_argument("--options", default=None, help="comma-separated answers for --question")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--mode", choices=["stages", "follow", "all"], default="stages")
    ap.add_argument("--follow-range", default="87,112.5", help="START,END in source-video seconds ('all' = whole clip)")
    ap.add_argument("--follow-speed", type=float, default=1.0, help="playback speed of the follow clips (2 = twice as fast)")
    ap.add_argument("--follow-win", type=float, default=12, help="show cloud points from +-N reconstruction frames (0 = all)")
    ap.add_argument("--follow-smooth", type=float, default=0, help="gaussian smoothing of the pose track, sigma in frames")
    ap.add_argument("--follow-layers", default="cloud,bim,design", help="rendered layers (the video layer is always made)")
    return ap.parse_args()


def chrome():
    exe = sorted(glob.glob(os.path.expanduser("~/.cache/ms-playwright/chromium-*/chrome-linux*/chrome")))
    return exe[-1] if exe else None


def launch(p, mode, w, h):
    args = GL_ARGS[mode] + ["--autoplay-policy=no-user-gesture-required", "--hide-scrollbars", "--force-color-profile=srgb"]
    b = p.chromium.launch(executable_path=chrome(), args=args)
    pg = b.new_page(viewport={"width": w, "height": h}, device_scale_factor=1)
    gl = pg.evaluate("""() => { const g = document.createElement('canvas').getContext('webgl2'); if (!g) return null;
        const e = g.getExtension('WEBGL_debug_renderer_info'); return e ? g.getParameter(e.UNMASKED_RENDERER_WEBGL) : g.getParameter(g.RENDERER); }""")
    return b, pg, gl


def encoder(path, fps, crf):
    return subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(fps), "-c:v", "mjpeg", "-i", "-",
                             "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p", "-r", str(fps),
                             "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)


def state(base):
    with urllib.request.urlopen(base + "/api/state", timeout=10) as r:
        return json.loads(r.read())


def mcp(base, tool, args):
    env = {**os.environ, "CAPTURE_STUDIO_MCP": base.rstrip("/") + "/mcp"}
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "mcp_call.py"), tool, json.dumps(args)], env=env,
                       capture_output=True, text=True, timeout=60)
    if r.returncode:
        raise RuntimeError(f"{tool} failed: {r.stderr[-400:]}")
    return r.stdout


def default_question(S):
    iss = (S.get("validation") or {}).get("issues") or []
    for sev, opts in (("hold", ["Approve", "Reject"]), ("fail", ["Yes, fix it", "Leave it"])):
        for i in iss:
            if i.get("severity") == sev:
                msg = i.get("message", i.get("rule", "")).split(" — ")[0].rstrip(".")
                return (f"{msg}. Approve?" if sev == "hold" else f"{msg}. Apply the suggested fix?"), opts
    return "Happy with this layout?", ["Approve", "Change it"]


class Recorder:
    def __init__(self, pg, fps, crf):
        self.pg, self.fps, self.crf, self.t = pg, fps, crf, 0.0
        self.cdp = pg.context.new_cdp_session(pg)
        self.enc = None
        self.frames = 0

    def frame(self, n=1):
        st = None
        for _ in range(n):
            self.t += 1000.0 / self.fps
            st = self.pg.evaluate("t => window.__demo.frame(t)", self.t)
            img = self.cdp.send("Page.captureScreenshot", {"format": "jpeg", "quality": 94, "optimizeForSpeed": True})["data"]
            self.enc.stdin.write(base64.b64decode(img))
            self.frames += 1
        return st

    def seconds(self, s):
        return self.frame(max(1, round(s * self.fps)))

    def center(self, sel):
        b = self.pg.locator(sel).first.bounding_box()
        return b["x"] + b["width"] / 2, b["y"] + b["height"] / 2

    def move(self, sel, ms=800):
        x, y = self.center(sel)
        self.pg.evaluate("([x, y, ms]) => window.__demo.cursor(x, y, ms)", [x, y, ms])
        self.seconds(ms / 1000 + 0.15)
        return x, y

    def click(self, sel):
        x, y = self.move(sel)
        self.pg.evaluate("() => window.__demo.click()")
        self.pg.mouse.click(x, y)
        self.frame(4)


def hitl(rec, args, base):
    pg = rec.pg
    rec.seconds(0.8)
    if args.message:
        rec.click("#say input")
        for ch in args.message:
            pg.keyboard.type(ch)
            rec.frame(2)
        rec.seconds(0.4)
        rec.click("#say form button")
        pg.evaluate("() => window.__demo.poll()")
        rec.seconds(1.8)
    st = pg.evaluate("() => window.__demo.poll()")
    if not st["question"]:
        q, opts = default_question(state(base))
        if args.question:
            q, opts = args.question, [o.strip() for o in (args.options or "Approve,Not yet").split(",")]
        print("  ask_user:", q, opts)
        mcp(base, "ask_user", {"question": q, "options": opts})
        for _ in range(20):
            st = pg.evaluate("() => window.__demo.poll()")
            if st["question"]:
                break
            time.sleep(0.2)
    if st["question"]:
        rec.seconds(2.2)
        rec.click("#q button")
        for _ in range(20):
            st = pg.evaluate("() => window.__demo.poll()")
            if not st["question"]:
                break
            time.sleep(0.2)
        rec.seconds(2.2)
    pg.evaluate("() => window.__demo.cursor(innerWidth * 0.7, innerHeight * 0.6, 900, false)")
    rec.seconds(1.2)


def main():
    args = parse()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    base = args.base.rstrip("/")
    w, h = (int(x) for x in args.size.split("x"))
    want = [s.strip() for s in args.stages.split(",")]
    with sync_playwright() as p:
        modes = ["gpu", "swiftshader"] if args.gl == "auto" else [args.gl]
        for mode in modes:
            b, pg, gl = launch(p, mode, w, h)
            if gl and (mode == "swiftshader" or "SwiftShader" not in gl):
                break
            b.close()
        print(f"renderer: {gl} ({mode})")
        if args.mode in ("follow", "all"):
            follow(pg, args, base, out, w, h)
            if args.mode == "follow":
                b.close()
                return
        errors = []
        pg.on("console", lambda m: m.type == "error" and errors.append(m.text))
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto(base + "/?capture=1")
        pg.wait_for_function("window.__demo && window.__demo.ready", timeout=120000)
        rec = Recorder(pg, args.fps, args.crf)
        clips = []
        for stage, name in STAGES:
            if stage not in want:
                continue
            path = out / f"{name}.mp4"
            rec.enc = encoder(path, args.fps, args.crf)
            t0, f0 = time.time(), rec.frames
            pg.evaluate("s => window.__demo.go(s)", stage)
            n = 0
            while n < MAX_S[stage] * args.fps:
                st = rec.frame(); n += 1
                if st["done"]:
                    break
            if stage == "design" and not args.no_hitl:
                hitl(rec, args, base)
            rec.seconds(0.6 if stage != "design" else 1.0)
            rec.enc.stdin.close(); rec.enc.wait()
            nf = rec.frames - f0
            print(f"{path.name}: {nf} frames ({nf / args.fps:.1f} s) captured in {time.time() - t0:.0f} s "
                  f"= {nf / max(0.01, time.time() - t0):.1f} capture fps")
            clips.append(path)
        b.close()
    if errors:
        print("console errors:\n  " + "\n  ".join(errors[:20]))
    if len(clips) > 1:
        lst = out / "concat.txt"
        lst.write_text("".join(f"file '{c.resolve()}'\n" for c in clips))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                        "-movflags", "+faststart", str(out / "full_demo.mp4")], check=True)
        lst.unlink()
        print("full_demo.mp4 written")


# ---------------------------------------------------------------- follow mode
def probe_fps(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate,duration",
                        "-of", "json", str(path)], capture_output=True, text=True, check=True)
    st = json.loads(r.stdout)["streams"][0]
    n, d = st["r_frame_rate"].split("/")
    return float(n) / float(d), float(st.get("duration") or 0)


def follow(pg, args, base, out, w, h):
    fps = args.fps
    meta = json.loads(urllib.request.urlopen(base + "/data/scan/meta.json", timeout=20).read())
    src = out / ".follow_source.mp4"
    if not src.exists():
        urllib.request.urlretrieve(base + "/data/site.mp4", src)
    src_fps, dur = probe_fps(src)
    # LingBot-Map's loader keeps every round(src_fps / fps)-th source frame, so frame i is at i * interval / src_fps
    interval = max(1, round(src_fps / meta["fps"]))
    frame_dt = interval / src_fps
    last = (meta["frames"] - 1) * frame_dt
    if args.follow_range == "all":
        start, end = 0.0, last
    else:
        start, end = (float(x) for x in args.follow_range.split(","))
    start, end = max(0.0, start), min(end, last, dur)
    speed = args.follow_speed
    n = int(round((end - start) / speed * fps))
    print(f"follow: {start:.2f}-{end:.2f} s x{speed} -> {n} frames; frame_dt {frame_dt:.6f} s "
          f"(src {src_fps:.3f} fps, every {interval}th frame), window +-{args.follow_win:g} frames")

    t0 = time.time()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.4f}", "-i", str(src), "-an",
                    "-vf", f"setpts=PTS/{speed},fps={fps},scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}",
                    "-frames:v", str(n), "-c:v", "libx264", "-preset", "medium", "-crf", str(args.crf), "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(out / "follow_video.mp4")], check=True)

    layers = [l.strip() for l in args.follow_layers.split(",") if l.strip()]
    q = f"?capture=1&frame_dt={frame_dt:.7f}&win={args.follow_win:g}&smooth={args.follow_smooth:g}"
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(base + "/" + q)
    pg.wait_for_function("window.__demo && window.__demo.ready", timeout=180000)
    cdp = pg.context.new_cdp_session(pg)
    encs = {l: encoder(out / f"follow_{l.replace('+', '_')}.mp4", fps, args.crf) for l in layers}
    for k in range(n):
        t = start + k * speed / fps
        for l in layers:
            pg.evaluate("([l, t]) => window.__demo.follow(l, t)", [l, t])
            img = cdp.send("Page.captureScreenshot", {"format": "jpeg", "quality": 94, "optimizeForSpeed": True})["data"]
            encs[l].stdin.write(base64.b64decode(img))
        if k % 150 == 0:
            print(f"  frame {k}/{n}", flush=True)
    for e in encs.values():
        e.stdin.close(); e.wait()
    print(f"follow layers: {n} frames x {len(layers) + 1} in {time.time() - t0:.0f} s")
    if errors:
        print("page errors:", errors[:5])
    names = {"video": "VIDEO", "cloud": "POINT CLOUD", "bim": "AS-BUILT BIM", "design": "DESIGN"}
    have = [l for l in ["video", "cloud", "bim", "design"] if (out / f"follow_{l}.mp4").exists()]
    if len(have) >= 2:
        matchcut(out, have, names, n, fps, w, h, args.crf)
        split(out, have, names, fps, w, h, args.crf)
    src.unlink(missing_ok=True)


def _font(size):
    from PIL import ImageFont
    for f in ("/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def _label(text, scale=1.0):
    """Small rounded pill as an RGBA numpy array."""
    import numpy as np
    from PIL import Image, ImageDraw
    f = _font(int(17 * scale))
    x0, y0, x1, y1 = f.getbbox(text)
    pw, ph = int(14 * scale), int(8 * scale)
    W, H = x1 - x0 + 2 * pw + 2 * int(15 * scale), y1 - y0 + 2 * ph
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, W - 1, H - 1], radius=H // 2, fill=(14, 17, 21, 200), outline=(255, 255, 255, 40))
    r = int(4 * scale); cy = H // 2; cx = pw + r
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 138, 61, 255))
    d.text((pw + int(15 * scale) - x0, ph - y0), text, font=f, fill=(238, 241, 244, 255))
    return np.asarray(im).astype(np.float32) / 255.0


def _paste(frame, lab, x, y, alpha=1.0):
    h, w = lab.shape[:2]
    H, W = frame.shape[:2]
    x, y = int(max(0, min(W - w, x))), int(max(0, min(H - h, y)))
    a = lab[..., 3:4] * alpha
    reg = frame[y:y + h, x:x + w].astype(np.float32)
    frame[y:y + h, x:x + w] = (reg * (1 - a) + lab[..., :3] * 255 * a).astype(np.uint8)


def _reader(path, w, h):
    return subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                            stdout=subprocess.PIPE, bufsize=w * h * 3 * 2)


def _writer(path, fps, w, h, crf):
    return subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
                             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)


def matchcut(out, layers, names, n, fps, w, h, crf):
    """One continuous move: each layer wipes in from the left over the previous one, mid-motion."""
    t0 = time.time()
    D = n / fps
    m = len(layers)
    # wipe i (into layers[i]) starts at fraction cuts[i-1] of the clip; the last layer gets the final ~28%
    cuts = [0.22 + (0.72 - 0.22) * i / max(1, m - 2) for i in range(m - 1)] if m > 2 else [0.45]
    wipe_s = 1.1
    labs = {l: _label(names[l]) for l in layers}
    rd = {l: _reader(out / f"follow_{l}.mp4", w, h) for l in layers}
    wr = _writer(out / "follow_matchcut.mp4", fps, w, h, crf)
    size = w * h * 3
    for k in range(n):
        fr = {l: np.frombuffer(rd[l].stdout.read(size), np.uint8).reshape(h, w, 3) for l in layers}
        t = k / fps
        cur, nxt, p = 0, None, 0.0
        for i, c in enumerate(cuts):
            a = c * D
            if t >= a + wipe_s:
                cur = i + 1
            elif t >= a:
                cur, nxt = i, i + 1
                x = (t - a) / wipe_s
                p = x * x * (3 - 2 * x)
                break
        old = fr[layers[cur]]
        if nxt is None:
            img = old.copy()
            _paste(img, labs[layers[cur]], 28, h - 64)
        else:
            X = int(round(p * w))
            img = old.copy()
            img[:, :X] = fr[layers[nxt]][:, :X]
            if 0 < X < w:              # thin divider with a soft shadow
                for dx, a in ((-3, 0.25), (-2, 0.45), (2, 0.45), (3, 0.25)):
                    xx = X + dx
                    if 0 <= xx < w:
                        img[:, xx] = (img[:, xx] * (1 - a)).astype(np.uint8)
                img[:, max(0, X - 1):min(w, X + 1)] = 255
            ln, lo = labs[layers[nxt]], labs[layers[cur]]
            _paste(img, ln, min(28, X - ln.shape[1] - 16), h - 64, alpha=min(1, X / (ln.shape[1] + 44)))
            _paste(img, lo, max(X + 16, w - lo.shape[1] - 28), h - 64, alpha=min(1, (w - X) / (lo.shape[1] + 44)))
        wr.stdin.write(img.tobytes())
    wr.stdin.close(); wr.wait()
    for r in rd.values():
        r.kill()
    print(f"follow_matchcut.mp4: {n} frames, wipes at {', '.join(f'{c * D:.1f}s' for c in cuts)} ({time.time() - t0:.0f} s)")


def split(out, layers, names, fps, w, h, crf):
    """2x2 (or 1x3 / 1x2) synced panels with small labels."""
    t0 = time.time()
    font = "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"
    ff = f":fontfile={font}" if os.path.exists(font) else ""
    k = len(layers)
    pw, ph = (w // 2, h // 2) if k == 4 else (w // k, int(h // k)) if k < 4 else (w // 2, h // 2)
    chains, ins = [], []
    for i, l in enumerate(layers):
        ins += ["-i", str(out / f"follow_{l}.mp4")]
        chains.append(f"[{i}:v]scale={pw}:{ph},drawbox=x=0:y=0:w=iw:h=ih:color=0x0b0d10@1:t=2,"
                      f"drawtext=text='{names[l]}'{ff}:fontsize={max(14, ph // 26)}:fontcolor=0xeef1f4:x=18:y=h-th-18:"
                      f"box=1:boxcolor=0x0e1115@0.75:boxborderw=8[p{i}]")
    if k == 4:
        layout = "0_0|w0_0|0_h0|w0_h0"
    else:
        layout = "|".join("+".join(["w0"] * i) if i else "0" for i in range(k)).replace("|", "_0|") + "_0"
    tail = f"{''.join(f'[p{i}]' for i in range(k))}xstack=inputs={k}:layout={layout}"
    if k < 4:
        tail += f",pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=0x0b0d10"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *ins, "-filter_complex", ";".join(chains) + ";" + tail + "[v]",
                    "-map", "[v]", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(out / "follow_split.mp4")], check=True)
    print(f"follow_split.mp4 written ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
