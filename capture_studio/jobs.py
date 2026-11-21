"""Live pipeline for an uploaded video: normalise -> LingBot-Map -> dense cloud -> objects -> as-built model.

Runs as its own process (`python -m capture_studio.jobs`) so the GPU is released between stages and the
server stays responsive. Progress is written to <DATA>/job.json after every step; the viewer polls it and
loads each product (video, cloud, objects, model) the moment its file appears. Nothing here is replayed:
every stage is computed from the uploaded file.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import traceback

from . import store

JOB = store.DATA / "job.json"
STAGES = [("video", "Preparing video"), ("reconstruct", "LingBot-Map reconstruction"), ("cloud", "Dense point cloud"),
          ("objects", "Gemini segments and names objects"), ("model", "As-built model"), ("done", "Ready")]


def read() -> dict:
    return json.loads(JOB.read_text()) if JOB.exists() else {"state": "idle", "stages": []}


def _write(job: dict):
    JOB.parent.mkdir(parents=True, exist_ok=True)
    tmp = JOB.with_suffix(".tmp")
    tmp.write_text(json.dumps(job))
    tmp.replace(JOB)


def start(upload_path: str, name: str = "Uploaded site", site_type: str = "factory") -> dict:
    """Called by the server: launch the pipeline process for an uploaded file."""
    cur = read()
    if cur.get("state") == "running":
        return {"error": "a pipeline run is already in progress", "job": cur}
    job = {"state": "running", "name": name, "site_type": site_type, "upload": upload_path, "started": time.time(),
           "stage": "video", "stages": [{"key": k, "title": t, "state": "pending"} for k, t in STAGES[:-1]], "log": []}
    _write(job)
    log = open(store.DATA / "job.log", "w")
    subprocess.Popen([sys.executable, "-m", "capture_studio.jobs"], cwd=store.ROOT, stdout=log, stderr=subprocess.STDOUT,
                     start_new_session=True)
    return job


class Runner:
    def __init__(self):
        self.job = read()

    def stage(self, key, **info):
        for s in self.job["stages"]:
            if s["key"] == key:
                s.update(info)
        self.job["stage"] = key
        _write(self.job)

    def step(self, key, fn):
        t = time.time()
        self.stage(key, state="running", started=t)
        try:
            detail = fn() or {}
            self.stage(key, state="done", seconds=round(time.time() - t, 1), **detail)
            return True
        except Exception as ex:                                  # a failed optional stage must not hide the rest
            self.stage(key, state="failed", seconds=round(time.time() - t, 1), error=f"{type(ex).__name__}: {str(ex)[:400]}")
            self.job["log"].append(traceback.format_exc()[-1500:])
            _write(self.job)
            return False

    # ---- stages
    def video(self):
        scan = store.DATA / "scan"
        for p in (scan, store.DATA / "project", store.DATA / "export", store.DATA / "site_frames"):
            shutil.rmtree(p, ignore_errors=True)
        out = store.DATA / "site.mp4"
        out.unlink(missing_ok=True)
        # phones store rotation as metadata: bake it in, cap the long side at 1280, constant 30 fps, no audio
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", self.job["upload"], "-an", "-r", "30",
                        "-vf", "scale='if(gt(iw,ih),min(1280,iw),-2)':'if(gt(iw,ih),-2,min(1280,ih))'",
                        "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], check=True)
        pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height:format=duration",
                             "-of", "json", str(out)], capture_output=True, text=True, check=True)
        j = json.loads(pr.stdout)
        self.duration = float(j["format"]["duration"])
        return {"width": j["streams"][0]["width"], "height": j["streams"][0]["height"], "duration": round(self.duration, 1)}

    def reconstruct(self):
        fps = max(1, min(4, int(220 / max(self.duration, 1))))           # keep within 12 GB: <= ~220 frames
        p = subprocess.run([sys.executable, "-m", "capture_studio.reconstruct", "--video", str(store.DATA / "site.mp4"),
                            "--out", str(store.DATA / "scan" / "raw_predictions.npz"), "--fps", str(fps),
                            "--keyframe_interval", "2", "--window", "24"], cwd=store.ROOT, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(p.stderr[-600:])
        line = next((ln for ln in p.stdout.splitlines() if ln.startswith("LingBot-Map:")), "")
        return {"fps": fps, "detail": line}

    def cloud(self):
        from . import fit
        m = fit.prepare_cloud()
        return {"points": m["n"], "frames": m["frames"], "inference_seconds": m["inference_seconds"]}

    def objects(self):
        from . import objects
        r = objects.run(write_previews=True)      # montages of the object in the real video frames, for the viewer
        return {"objects": r.get("objects") if isinstance(r, dict) else None}

    def model(self):
        from . import bim, codes, fit
        model, report = fit.fit_shell()
        model["project"]["name"] = self.job.get("name", "Uploaded site")
        model["project"]["code_profile"] = "OSHA_1910" if self.job.get("site_type") == "factory" else "NBC_INDIA_2016"
        model["project"]["site_type"] = self.job.get("site_type", "factory")
        model["revision"] = 1
        s = store.fresh()
        s["model"], s["stage"] = model, "bim"
        code = codes.check(model, {})
        s["validation"] = {"code_ok": code["ok"], "safety_ok": True, "physics_ok": None, "physics_stale": True, "desires_ok": False,
                           "counts": code["counts"], "profile": code["profile"], "issues": code["issues"], "egress": code["egress"],
                           "physics": None, "review": None, "review_ok": None, "ok": False}
        store.record(s, "agent", "fit_bim", "As-built model fitted to the scan", [])
        store.save(s)
        return {"counts": bim.summary(model)["counts"], "fit_p50_mm": report["scan"].get("fit_p50_mm")}

    def run(self):
        ok = self.step("video", self.video) and self.step("reconstruct", self.reconstruct) and self.step("cloud", self.cloud)
        if ok:
            self.step("objects", self.objects)          # optional: needs a working Gemini key; the model still fits without it
            ok = self.step("model", self.model)
        self.job["state"] = "done" if ok else "failed"
        self.job["stage"] = "done" if ok else self.job["stage"]
        self.job["finished"] = time.time()
        self.job["seconds"] = round(self.job["finished"] - self.job["started"], 1)
        _write(self.job)


if __name__ == "__main__":
    Runner().run()
