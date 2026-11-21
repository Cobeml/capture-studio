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


if __name__ == "__main__":
    main()
