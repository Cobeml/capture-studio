"""Record the live-demo console (web/live) frame by frame at 1920x1080 / 30 fps, one clip per beat plus a stitched take.

    python scripts/record_live.py OUT_DIR [BASE_URL] [options]

Beats (in this order, pick with --beats): capture, reconstruct, explore, hero, replan, modeledit, deliver.
Writes OUT_DIR/<nn>_<beat>.mp4 and OUT_DIR/full_take.mp4.

The page is opened with ?capture=1&clean=1. With capture=1 its clock is virtual and only advances when this script
calls window.__demo.frame(ms): that seeks the <video> to the matching time (capture mode only), renders, and pins CSS
animations to the same clock. Each frame is screenshotted over CDP and piped to ffmpeg, so the output is smooth however
slowly the GPU (or SwiftShader) renders. Hovers and clicks are real Playwright mouse events, shown with a drawn cursor.
clean=1 hides presenter-only chrome (agent-activity hint, keyboard hint).

Mutating steps never run against the production server (port 8642) unless --allow-mutate is given:
  --send-prompts        really POST the typed requests (otherwise they are typed for the camera only)
  --start-scenario N    load_scenario N before recording
  --replan-scenario N   load_scenario N after the layout request (a pre-generated agent result: its points glide)
  --edit-scenario N     load_scenario N after the model-edit request (parts appear on the open machine)
  --export              click "Export IFC package" in the deliver beat
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import math
import os
import subprocess
import sys
import functools
print = functools.partial(print, flush=True)  # noqa: A001  (progress lines appear immediately)
import urllib.parse
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
BEATS = ["capture", "reconstruct", "explore", "hero", "replan", "modeledit", "deliver"]
GL_ARGS = {
    "gpu": ["--use-gl=angle", "--use-angle=gl-egl", "--enable-gpu", "--ignore-gpu-blocklist", "--enable-gpu-rasterization"],
    "swiftshader": ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"],
}


def parse():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out")
    ap.add_argument("base", nargs="?", default="http://127.0.0.1:8642")
    ap.add_argument("--page", default="/web/live/index.html")
    ap.add_argument("--beats", default=",".join(BEATS))
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--gl", choices=["auto", "gpu", "swiftshader"], default="auto")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--speed", type=float, default=4, help="video playback speed in the reconstruct beat")
    ap.add_argument("--allow-mutate", action="store_true")
    ap.add_argument("--send-prompts", action="store_true")
    ap.add_argument("--start-scenario")
    ap.add_argument("--replan-scenario")
    ap.add_argument("--edit-scenario")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--replan-prompt", default=None, help="typed layout request (default: the first suggestion chip)")
    ap.add_argument("--edit-prompt", default="Add a bar feeder on the left and an interlocked guard at the front.")
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


def mcp(base, tool, args):
    env = {**os.environ, "CAPTURE_STUDIO_MCP": base.rstrip("/") + "/mcp"}
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "mcp_call.py"), tool, json.dumps(args)], env=env, capture_output=True, text=True, timeout=120)
    if r.returncode:
        raise RuntimeError(f"{tool} failed: {r.stderr[-400:]}")
    return r.stdout


class Rec:
    def __init__(self, pg, fps, crf):
        self.pg, self.fps, self.crf, self.t = pg, fps, crf, 0.0
        self.cdp = pg.context.new_cdp_session(pg)
        self.enc, self.cur = None, [960.0, 560.0]

    def open(self, path):
        self.enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(self.fps), "-c:v", "mjpeg", "-i", "-",
                                     "-c:v", "libx264", "-preset", "medium", "-crf", str(self.crf), "-pix_fmt", "yuv420p", "-r", str(self.fps),
                                     "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)

    def close(self):
        self.enc.stdin.close(); self.enc.wait(); self.enc = None

    def frame(self):
        self.t += 1000.0 / self.fps
        st = self.pg.evaluate("t => window.__demo.frame(t)", self.t)
        img = self.cdp.send("Page.captureScreenshot", {"format": "jpeg", "quality": 94, "optimizeForSpeed": True})["data"]
        self.enc.stdin.write(base64.b64decode(img))
        return st

    def hold(self, s, until=None, max_s=None):
        n, st = 0, None
        limit = round((max_s if until else s) * self.fps)
        while n < limit:
            st = self.frame(); n += 1
            if until and n >= round(s * self.fps) and until(st):
                break
        return st

    def move(self, x, y, s=0.8):    # drawn cursor + real pointer moves, eased
        x0, y0 = self.cur; n = max(1, round(s * self.fps))
        self.pg.evaluate("() => window.__demo.cursor(%f, %f, 1, true)" % (x0, y0))
        for i in range(1, n + 1):
            k = i / n; k = 4 * k ** 3 if k < 0.5 else 1 - (-2 * k + 2) ** 3 / 2
            cx, cy = x0 + (x - x0) * k, y0 + (y - y0) * k
            self.pg.mouse.move(cx, cy)
            self.pg.evaluate("([x, y]) => window.__demo.cursor(x, y, 1, true)", [cx, cy])
            self.frame()
        self.cur = [x, y]

    def click(self, x=None, y=None, shift=False):
        if x is not None:
            self.move(x, y)
        if shift:
            self.pg.keyboard.down("Shift")
        self.pg.mouse.click(*self.cur)
        if shift:
            self.pg.keyboard.up("Shift")
        self.pg.evaluate("() => window.__demo.click()")
        self.hold(0.25)

    def click_el(self, sel, s=0.7):
        xy = self.pg.evaluate("s => window.__demo.rect(s)", sel)
        if not xy:
            return False
        self.move(*xy, s=s); self.click(); return True

    def type_into(self, sel, text, cps=28):
        self.pg.evaluate("s => { const e = document.querySelector(s); e.value = ''; }", sel)
        for i in range(1, len(text) + 1):
            self.pg.evaluate("([s, v]) => { document.querySelector(s).value = v; }", [sel, text[:i]])
            if i % max(1, round(cps / self.fps)) == 0 or i == len(text):
                self.frame()


def main():
    a = parse()
    w, h = map(int, a.size.split("x"))
    beats = [b for b in a.beats.split(",") if b]
    port = urllib.parse.urlparse(a.base).port
    mutating = a.send_prompts or a.start_scenario or a.replan_scenario or a.edit_scenario or a.export
    if mutating and port == 8642 and not a.allow_mutate:
        sys.exit("refusing to send prompts / load scenarios / export on port 8642 (production); use a test server or --allow-mutate")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.start_scenario:
        mcp(a.base, "load_scenario", {"name": a.start_scenario})
    clips = []
    with sync_playwright() as p:
        mode = "gpu" if a.gl in ("auto", "gpu") else "swiftshader"
        b, pg, gl = launch(p, mode, w, h)
        if a.gl == "auto" and (not gl or "swiftshader" in gl.lower()):
            b.close(); mode = "swiftshader"; b, pg, gl = launch(p, mode, w, h)
        print(f"renderer: {gl} ({mode})")
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto(a.base.rstrip("/") + a.page + "?capture=1&clean=1#capture")
        pg.wait_for_function("window.__demo && window.__demo.ready", timeout=180000)
        R = Rec(pg, a.fps, a.crf)
        demo = lambda js, arg=None: pg.evaluate(js, arg)  # noqa: E731
        for i, beat in enumerate(beats, 1):
            path = out / f"{i:02d}_{beat}.mp4"
            st = demo("() => window.__demo.status()")
            if beat in ("hero", "modeledit") and not demo("() => document.querySelectorAll('#ex-heroes .hero-btn').length || window.__live.OBJ.list.some(o => o.mesh)"):
                print(f"skip {beat}: no object with a generated 3D model yet"); continue
            print(f"recording {path.name}"); R.open(path)
            if beat == "capture":
                demo("() => window.__demo.go('capture')"); R.hold(6)
            elif beat == "reconstruct":
                demo("() => window.__demo.go('cloud')"); demo(f"() => window.__demo.speed({a.speed})")
                R.hold(4, until=lambda s: s and s.get("videoT") is not None and s["videoT"] >= demo("() => document.querySelector('video').duration") - 0.3, max_s=150)
                R.hold(1.5)
            elif beat == "explore":
                demo("() => window.__demo.go('explore')"); R.hold(2.5)
                objs = sorted([o for o in demo("() => window.__demo.objects()") if o["label"]], key=lambda o: -o["area"])[:3]
                for o in objs:
                    R.move(o["x"], o["y"], 1.0); R.hold(1.3)
                if objs:
                    R.click(objs[min(1, len(objs) - 1)]["x"], objs[min(1, len(objs) - 1)]["y"]); R.hold(4)
                pg.keyboard.press("Escape"); R.hold(2.5)
            elif beat == "hero":
                demo("() => window.__demo.go('explore')"); R.hold(1.5)
                R.click_el("#ex-heroes .hero-btn"); R.hold(7)
                R.click_el("#hero-close"); R.hold(1)
            elif beat == "replan":
                demo("() => window.__demo.go('design')"); R.hold(2)
                demo("() => window.__demo.view('cloud')"); R.hold(1)
                objs = sorted([o for o in demo("() => window.__demo.objects()") if o["label"] and not o["id"].startswith("obj:")], key=lambda o: -o["area"])[:2]
                for k, o in enumerate(objs):
                    R.click(o["x"], o["y"], shift=k > 0); R.hold(0.8)
                R.hold(1)
                if a.send_prompts:
                    if a.replan_prompt:
                        R.type_into("#brief-text", a.replan_prompt); R.click_el("#btn-send")
                    else:
                        R.click_el("#suggest .sugg")
                else:
                    text = a.replan_prompt or demo("() => document.querySelector('#suggest .sugg')?.textContent || ''")
                    R.type_into("#brief-text", text); xy = demo("() => window.__demo.rect('#btn-send')"); R.move(*xy, 0.7)
                R.hold(2)
                if a.replan_scenario:
                    mcp(a.base, "load_scenario", {"name": a.replan_scenario}); demo("() => window.__demo.poll()")
                R.hold(3, until=lambda s: s and s["queue"] == 0 and s["settled"], max_s=90); R.hold(2.5)
            elif beat == "modeledit":
                demo("() => window.__demo.go('explore')"); R.hold(1.5)
                R.click_el("#ex-heroes .hero-btn"); R.hold(3)
                R.type_into("#hero-prompt", a.edit_prompt)
                if a.send_prompts:
                    R.click_el("#hero-send")
                else:
                    xy = demo("() => window.__demo.rect('#hero-send')"); R.move(*xy, 0.7)
                R.hold(2)
                if a.edit_scenario:
                    mcp(a.base, "load_scenario", {"name": a.edit_scenario}); demo("() => window.__demo.poll()")
                R.hold(6)
            elif beat == "deliver":
                demo("() => window.__demo.go('deliver')"); R.hold(3)
                if a.export:
                    R.click_el("#btn-export")
                    R.hold(2, until=lambda s: demo("() => document.querySelectorAll('#files li').length > 0"), max_s=120)
                R.hold(4)
            R.close(); clips.append(path)
        b.close()
        if errors:
            print("page errors:", errors[:5])
    if clips:
        lst = out / "clips.txt"; lst.write_text("".join(f"file '{c.name}'\n" for c in clips))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out / "full_take.mp4")], check=True)
        print("wrote", out / "full_take.mp4")


if __name__ == "__main__":
    main()
