"""Record the recorded-run demo page (web/live/next/index.html) frame by frame at 1920x1080 / 30 fps, one clip per beat.

    python scripts/record_live.py OUT_DIR [BASE_URL] [--beats capture,reconstruct,understand,click,hero_lathe,design,replan,model_edit]

Writes OUT_DIR/<beat>.mp4 (capture, reconstruct, understand, click, hero_lathe, design, replan, model_edit, deliver),
OUT_DIR/clips.json (durations, and for design / replan the second at which each recorded step starts).

The page is opened with ?capture=1&clean=1: its clock is virtual and only advances when this script calls
window.__demo.frame(ms), which seeks the <video> to the matching time (capture mode only), renders, and pins CSS
animations to the same clock. Each frame is screenshotted over CDP and piped to ffmpeg, so the output is smooth however
slowly the GPU renders. Hovers and clicks are real Playwright mouse events, shown with a drawn cursor that moves on eased
paths. The page only plays local recordings (the Design step replays the bundled run), so nothing on the server changes.
"""
from __future__ import annotations

import argparse
import base64
import functools
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

print = functools.partial(print, flush=True)  # noqa: A001
BEATS = ["capture", "reconstruct", "understand", "click", "hero_lathe", "design", "replan", "model_edit"]
GL_ARGS = {
    "gpu": ["--use-gl=angle", "--use-angle=gl-egl", "--enable-gpu", "--ignore-gpu-blocklist", "--enable-gpu-rasterization"],
    "swiftshader": ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"],
}


def parse():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out")
    ap.add_argument("base", nargs="?", default="http://127.0.0.1:8642")
    ap.add_argument("--page", default="/web/live/next/index.html")
    ap.add_argument("--beats", default=",".join(BEATS))
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--gl", choices=["auto", "gpu", "swiftshader"], default="auto")
    ap.add_argument("--crf", type=int, default=17)
    ap.add_argument("--speed", type=float, default=4, help="video playback speed in the reconstruct beat")
    ap.add_argument("--pace", type=float, default=1.6, help="design replay speed-up (events per second x pace)")
    ap.add_argument("--bright", type=float, default=1.12, help="CSS brightness of the 3D views")
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


class Rec:
    def __init__(self, pg, fps, crf):
        self.pg, self.fps, self.crf, self.t = pg, fps, crf, 0.0
        self.cdp = pg.context.new_cdp_session(pg)
        self.enc, self.cur, self.n = None, [1500.0, 900.0], 0

    def open(self, path):
        self.n = 0
        self.enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(self.fps), "-c:v", "mjpeg", "-i", "-",
                                     "-c:v", "libx264", "-preset", "medium", "-crf", str(self.crf), "-pix_fmt", "yuv420p", "-r", str(self.fps),
                                     "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)

    def close(self):
        self.enc.stdin.close(); self.enc.wait(); self.enc = None
        return self.n / self.fps

    def frame(self):
        self.t += 1000.0 / self.fps
        st = self.pg.evaluate("t => window.__demo.frame(t)", self.t)
        img = self.cdp.send("Page.captureScreenshot", {"format": "jpeg", "quality": 94, "optimizeForSpeed": True})["data"]
        self.enc.stdin.write(base64.b64decode(img)); self.n += 1
        return st

    def hold(self, s, until=None, max_s=None):
        n, st, limit = 0, None, round((max_s if until else s) * self.fps)
        while n < limit:
            st = self.frame(); n += 1
            done = until(st) if until else False          # called every frame (it may also log progress)
            if until and n >= round(s * self.fps) and done:
                break
        return st

    def cursor(self, show=True):
        self.pg.evaluate("([x, y, s]) => window.__demo.cursor(x, y, 1, s)", [self.cur[0], self.cur[1], show])

    def move(self, x, y, s=0.9, button_down=False):   # eased path, one pointer event per frame (no jitter)
        x0, y0 = self.cur; n = max(1, round(s * self.fps))
        for i in range(1, n + 1):
            k = i / n; k = 4 * k ** 3 if k < 0.5 else 1 - (-2 * k + 2) ** 3 / 2
            self.cur = [x0 + (x - x0) * k, y0 + (y - y0) * k]
            self.pg.mouse.move(*self.cur); self.cursor(); self.frame()

    def click(self, x=None, y=None, s=0.9):
        if x is not None:
            self.move(x, y, s)
        self.pg.mouse.click(*self.cur)
        self.pg.evaluate("() => window.__demo.click()")
        self.hold(0.3)

    def click_el(self, sel, s=0.9):
        xy = self.pg.evaluate("s => window.__demo.rect(s)", sel)
        if not xy:
            print("  (missing", sel, ")"); return False
        self.click(*xy, s=s); return True

    def drag(self, x0, y0, dx, dy, s=2.5):   # orbit a canvas: press, eased drag, release
        self.move(x0, y0, 0.7)
        self.pg.mouse.down()
        n = round(s * self.fps)
        for i in range(1, n + 1):
            k = i / n; k = k * k * (3 - 2 * k)
            self.cur = [x0 + dx * k, y0 + dy * k]
            self.pg.mouse.move(*self.cur); self.cursor(); self.frame()
        self.pg.mouse.up()


def main():
    a = parse()
    w, h = map(int, a.size.split("x"))
    beats = [b for b in a.beats.split(",") if b]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    clips = []
    with sync_playwright() as p:
        mode = "gpu" if a.gl in ("auto", "gpu") else "swiftshader"
        b, pg, gl = launch(p, mode, w, h)
        if a.gl == "auto" and (not gl or "swiftshader" in gl.lower()):
            b.close(); mode = "swiftshader"; b, pg, gl = launch(p, mode, w, h)
        print(f"renderer: {gl} ({mode})")
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto(a.base.rstrip("/") + a.page + f"?capture=1&clean=1&pace={a.pace}&bright={a.bright}#capture")
        pg.wait_for_function("window.__demo && window.__demo.ready", timeout=240000)
        pg.wait_for_timeout(2500)
        R = Rec(pg, a.fps, a.crf)
        demo = lambda js, arg=None: pg.evaluate(js, arg)  # noqa: E731
        for i, beat in enumerate(beats, 1):
            path = out / f"{beat}.mp4"
            marks = []
            print(f"recording {path.name}"); R.open(path)
            def follow(st):          # remember when each recorded step first shows up (for INDEX.md)
                if st and st.get("tag") and (not marks or marks[-1][1] != st["tag"] + " · " + (st.get("caption") or "")):
                    marks.append([round(R.n / a.fps, 2), st["tag"] + " · " + (st.get("caption") or "")])
                return st and st["queue"] == 0 and st["settled"]
            if beat == "capture":
                demo("() => window.__demo.go('capture')"); R.cursor(False); R.hold(9)
            elif beat == "reconstruct":
                demo("() => window.__demo.go('cloud')"); demo(f"() => window.__demo.speed({a.speed})"); R.cursor(False); R.hold(18)
            elif beat == "understand":       # hover several objects: all points of each light up, label + function
                demo("() => window.__demo.go('explore')"); R.hold(1.5)
                objs = sorted([o for o in demo("() => window.__demo.objects()") if o["label"] and not o["hero"]], key=lambda o: -o["area"])[:4]
                for o in objs:
                    xy = demo("id => window.__demo.project(id)", o["id"]) or [o["x"], o["y"]]
                    R.move(*xy, 0.9); R.hold(1.2)
                R.move(R.cur[0] + 260, R.cur[1] + 120, 0.8); R.cursor(False); R.hold(0.6)
            elif beat == "click":            # click one object: fly to it, its stills from the video
                demo("() => window.__demo.go('explore')"); R.hold(1.2)
                objs = sorted([o for o in demo("() => window.__demo.objects()") if o["label"] and not o["hero"]], key=lambda o: -o["area"])
                if objs:
                    xy = demo("id => window.__demo.project(id)", objs[0]["id"]) or [objs[0]["x"], objs[0]["y"]]
                    R.click(*xy, 1.0); R.cursor(False); R.hold(6)
                pg.keyboard.press("Escape"); R.hold(0.5)
            elif beat == "hero_lathe":       # the lathe: textured generated model in place of its points, rotated
                demo("() => window.__demo.go('explore')"); R.hold(1.2)
                lathe = demo("() => window.__demo.objects().find(o => o.hero) || null")
                if not lathe:
                    print("  no hero object on screen")
                else:
                    R.click(lathe["x"], lathe["y"], 1.0)                 # its label anchor: the centre can be occluded
                    if not demo("() => window.__demo.status().hero"):        # the pick landed on a point in front of it
                        demo("id => window.__demo.openHero(id)", lathe["id"])
                    R.hold(3)
                    xy = demo("() => window.__demo.rect('#hero-gl')")
                    if xy:
                        R.drag(xy[0] - 140, xy[1], 280, -20, 3.0); R.cursor(False); R.hold(3)
                    demo("() => document.querySelector('#hero-close').click()"); R.hold(0.3)
            elif beat == "design":           # one continuous take of the recorded run
                demo("() => window.__demo.go('design')"); R.hold(1.2)
                marks.append([round(R.n / a.fps + 0.9, 2), "click · brief sent"])
                R.click_el("#btn-send"); R.cursor(False)
                R.hold(4, until=follow, max_s=30); R.hold(2)
            elif beat in ("replan", "model_edit"):
                if demo("() => window.__demo.status().stage") != "design":
                    demo("() => window.__demo.go('design')"); R.hold(1)
                if not demo("() => !document.querySelector('#followups')?.hidden"):
                    print("  follow-on requests not shown (run the design beat first)"); R.hold(0.5); R.close(); path.unlink(missing_ok=True); continue
                R.hold(0.6); R.click_el(f".fu[data-k={beat}]"); R.cursor(False)
                R.hold(3, until=follow, max_s=24); R.hold(3.5)       # ends on the settled wide view
            elif beat == "deliver":
                demo("() => window.__demo.go('deliver')"); R.cursor(False); R.hold(8)
            dur = R.close(); clips.append({"file": path.name, "beat": beat, "seconds": round(dur, 2), **({"steps": marks} if marks else {})})
            print(f"  {dur:.1f} s")
        b.close()
        if errors:
            print("page errors:", errors[:5])
    if clips:
        (out / "clips.json").write_text(json.dumps(clips, indent=1))
        lst = out / "clips.txt"; lst.write_text("".join(f"file '{c['file']}'\n" for c in clips))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out / "full_take.mp4")], check=True)
        print("wrote", out / "full_take.mp4")


if __name__ == "__main__":
    main()
