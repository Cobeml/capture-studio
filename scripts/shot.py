"""Headless screenshots of the viewer:  python scripts/shot.py OUT_DIR [BASE_URL]  (dev aid)."""
import glob
import os
import sys

from playwright.sync_api import sync_playwright

out = sys.argv[1]
base = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8642"
plan = [("scan", 1500), ("cloud", 9000), ("bim", 3500), ("design", 2500), ("design", 9000), ("design", 14000)]
exe = sorted(glob.glob(os.path.expanduser("~/.cache/ms-playwright/chromium-*/chrome-linux*/chrome")))
with sync_playwright() as p:
    b = p.chromium.launch(executable_path=exe[-1] if exe else None,
                          args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist",
                                "--autoplay-policy=no-user-gesture-required"])
    pg = b.new_page(viewport={"width": 1440, "height": 860})
    pg.on("console", lambda m: m.type == "error" and print("console error:", m.text))
    pg.on("pageerror", lambda e: print("page error:", e))
    pg.goto(base)
    pg.wait_for_timeout(2500)
    last = None
    for i, (stage, wait) in enumerate(plan):
        if stage != last:
            pg.click(f'#steps button[data-s="{stage}"]')
            last = stage
        pg.wait_for_timeout(wait)
        pg.screenshot(path=f"{out}/{i}_{stage}.png")
    b.close()
