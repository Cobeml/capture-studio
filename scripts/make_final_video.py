#!/usr/bin/env python3
"""Build the 60-second Capture Studio YouTube video, reproducibly.

    python scripts/make_final_video.py all  [--voice qwen:Ryan] [--design-src PATH] [--design-cues JSON]

Stages (each can be run alone; results are cached under demo/final/cache):

    fetch    pull only the needed seconds of the public-domain / Pexels footage
             (ffmpeg seeks the remote file; nothing is downloaded whole)
    tts      generate the voice-over with one engine/voice  (--engine kokoro|qwen --voice NAME)
             -> demo/final/tts_candidates/<engine>_<voice>/chunk_XX.wav + <engine>_<voice>.wav preview
             runs inside its own venv (demo/final/.venv_tts for kokoro + whisper,
             demo/final/.venv_qwen for Qwen3-TTS) so the project venv is never touched
    align    faster-whisper word timestamps + accuracy / artefact report for a candidate
    build    VO polish + layout, picture edit, music synth, mix, loudness, mux, captions
    check    ffprobe, loudness, frames contact sheet, whisper on the final mix

`all` runs fetch -> (tts+align if the chosen voice is missing) -> build -> check.

Matched-camera "follow" clips (follow_video/cloud/bim/design.mp4, same frame counts) are
used for the video -> point cloud -> BIM section when found in any FOLLOW_DIRS entry
(or --follow-dir); otherwise the stage clips demo/02_point_cloud.mp4 / 03_bim.mp4 are used.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"
FINAL = DEMO / "final"
CACHE = FINAL / "cache"
SRCDIR = FINAL / "sources"
TTSDIR = FINAL / "tts_candidates"
ASSETS = FINAL / "assets"
OUT = FINAL / "capture_studio_1min.mp4"
OUT_NOMUSIC = FINAL / "capture_studio_1min_nomusic.mp4"

SCRATCH_FOLLOW = Path("/tmp/claude-1000/-home-cobe-liu-Developing-scanBuild/"
                      "aba40cac-4c60-4f76-aacf-40b1c83928e6/scratchpad/demo_follow")
FOLLOW_DIRS = [FINAL / "rec" / "follow_4_28", DEMO / "follow", DEMO, SCRATCH_FOLLOW]
# rec/follow_4_28 = `record_demo.py OUT http://127.0.0.1:8642 --mode follow --follow-range 4,28
#                    --follow-layers cloud,bim,cloud+bim`  (steadiest forward move, Fadal row)
FOLLOW_LOCK = 9.5      # clip second where the video->cloud wipe lands (machines 3-10 m away lock best)
STAGE_DIRS = [FINAL / "rec" / "stages", DEMO]   # re-recorded stage clips (soft shadows) first


def stage_clip(name):
    for d in STAGE_DIRS:
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(name)

VENV_TTS = FINAL / ".venv_tts"      # kokoro + faster-whisper + pyloudnorm
VENV_QWEN = FINAL / ".venv_qwen"    # qwen-tts (pins transformers 4.57.3)
QWEN_MODEL = Path.home() / ".cache/capture_studio_tts/Qwen3-TTS-12Hz-1.7B-CustomVoice"

FPS = 30
W, H = 1920, 1080
DUR = 60.0
NF = int(DUR * FPS)
SR = 48000

DEFAULT_VOICE = "qwen:Aiden"

# --------------------------------------------------------------------------------------
# Voice-over.  "text" is what is shown (captions / voiceover.txt); the per-engine "say"
# strings only respell words the TTS would otherwise mispronounce.
# --------------------------------------------------------------------------------------
VO = [
    dict(id="intro", text="This is Capture Studio."),
    dict(id="walk", text="Walk a factory floor with your phone."),
    dict(id="cloud", text="LingBot-Map turns that video into a 3D point cloud in under a minute.",
         kokoro="[LingBot](/lˈɪŋbɑt/) Map turns that video into a three-D point cloud in under a minute.",
         qwen="LingBot Map turns that video into a three-D point cloud in under a minute."),
    dict(id="bim", text="Capture Studio rebuilds it as an editable BIM model: the walls, the floor, and every machine.",
         kokoro="Capture Studio rebuilds it as an editable [BIM](/bˈɪm/) model: the walls, the floor, and every machine.",
         qwen="Capture Studio rebuilds it as an editable Bim model: the walls, the floor, and every machine."),
    dict(id="agent", text="Then a Gemini agent designs the change with you."),
    dict(id="cnc", text="Ask for a new CNC cell, and it places the machines, checks the safety rules, keeps the exit "
                        "routes clear, and runs a physics test in MuJoCo on every change.",
         kokoro="Ask for a new C N C cell, and it places the machines, checks the safety rules, keeps the exit routes "
                "clear, and runs a physics test in [MuJoCo](/muʤˈOkO/) on every change.",
         qwen="Ask for a new C.N.C. cell, and it places the machines, checks the safety rules, keeps the exit routes "
              "clear, and runs a physics test in Moo-Joe-Co on every change."),
    dict(id="risky", text="Anything risky, it asks you first."),
    dict(id="intel", text="Intelligence is matched to the task: Gemini Carbon for the hard design decisions, with Neon, "
                          "Argon and Hydrogen for the lighter steps.",
         qwen="Intelligence is matched to the task: Gemini Carbon, for the hard design decisions, with Neon, Argon "
              "and Hydrogen for the lighter steps."),
    dict(id="ifc", text="You get an IFC model and next steps.",
         kokoro="You get an I F C model and next steps.",
         qwen="You get an I-F-C model and next steps."),
    dict(id="end", text="Capture Studio. From phone video to a buildable plan."),
]

# v2 recut (segmented point cloud instead of box models; no export / IFC claim)
VO_V2 = [
    dict(id="intro", text="This is Capture Studio."),
    dict(id="walk", text="Walk a factory floor with your phone."),
    dict(id="cloud", text="LingBot-Map turns the video into a dense 3D point cloud.",
         qwen="LingBot Map turns the video into a dense three-D point cloud."),
    dict(id="scene", text="Then Gemini reads the scene: every machine segmented, named, and understood."),
    dict(id="cnc", text="Ask for a change, and a Gemini agent designs it with you: placing machines, guarding them, "
                        "keeping exit routes clear, and running a physics test in MuJoCo on every change.",
         qwen="Ask for a change, and a Gemini agent designs it with you: placing machines, guarding them, "
              "keeping exit routes clear, and running a physics test in Moo-Joe-Co on every change."),
    dict(id="risky", text="Anything risky, it asks you first."),
    dict(id="intel", text="Intelligence is matched to the task: Gemini Carbon for the hard design decisions, with Neon, "
                          "Argon and Hydrogen for the lighter steps.",
         qwen="Intelligence is matched to the task: Gemini Carbon, for the hard design decisions, with Neon, Argon "
              "and Hydrogen for the lighter steps."),
    dict(id="end", text="Capture Studio. From phone video to a buildable plan."),
]
CUT = "v1"
TTS_SUFFIX = ""

QWEN_TAKES = 4
QWEN_INSTRUCT = ("Confident, warm documentary narrator for a product launch film. Clear, natural and steady pace, "
                 "slight energy, no exaggeration.")

# VO layout on the 60 s timeline
VO_START = 10.30          # "This is Capture Studio." starts over the title card
END_CARD_MAX = 56.70      # end card fades in (moved earlier if the last line is long); last line starts after it
END_CARD = END_CARD_MAX
GAP_AFTER = dict(intro=0.30, walk=0.30, cloud=0.35, bim=0.35, agent=0.30, cnc=0.35, risky=0.35, intel=0.35, ifc=0.0,
                 scene=0.35)
SPARE_AFTER = ("risky", 1.2)    # (chunk id, max seconds): unused VO time is given to the shot after this line
EXTRA_GAP_W = dict(intro=0.3, walk=0.3, cloud=0.5, cnc=0.8, risky=0.6, intel=0.5)   # v2 only


def use_cut(cut):
    """Switch every output path / VO table to the v2 recut (v1 files stay untouched)."""
    global CUT, VO, TTS_SUFFIX, OUT, OUT_NOMUSIC, CACHE, SPARE_AFTER, FOLLOW_DIRS, STAGE_DIRS
    CUT = cut
    if cut != "v2":
        return
    VO = VO_V2
    TTS_SUFFIX = "_v2"
    OUT = FINAL / "capture_studio_1min_v2.mp4"
    OUT_NOMUSIC = FINAL / "capture_studio_1min_v2_nomusic.mp4"
    CACHE = FINAL / "cache_v2"
    SPARE_AFTER = ("scene", 2.5)  # spare time goes to the segmentation centrepiece
    FOLLOW_DIRS = [FINAL / "rec" / "v2_follow"]
    STAGE_DIRS = [FINAL / "rec" / "v2_stages"]

# --------------------------------------------------------------------------------------
# Part 1 sources: public domain (US Gov / Prelinger) and Pexels-licensed modern footage.
# key: (url, t_in seconds, seconds to pull, licence tag)
# --------------------------------------------------------------------------------------
IA = "https://archive.org/download"
NASA = "http://images-assets.nasa.gov/video"
ARCHIVE = {
    "mh2_sparks": (f"{IA}/MasterHa1936_2/MasterHa1936_2.mp4", 40.0, 6.0),
    "mh2_pour": (f"{IA}/MasterHa1936_2/MasterHa1936_2.mp4", 296.0, 8.0),
    "mh4_line": (f"{IA}/MasterHa1936_4/MasterHa1936_4.mp4", 60.0, 6.0),
    "mh4_hall": (f"{IA}/MasterHa1936_4/MasterHa1936_4.mp4", 144.0, 6.0),
    "mh4_drop": (f"{IA}/MasterHa1936_4/MasterHa1936_4.mp4", 455.0, 7.0),
    "mh4_overhead": (f"{IA}/MasterHa1936_4/MasterHa1936_4.mp4", 444.0, 6.0),
    "bomber_noses": (f"{IA}/gov.archives.arc.38638/gov.archives.arc.38638.mpeg", 347.0, 7.0),
    "lib_rows": (f"{IA}/InTheServiceOfAmerica/200-fc-1101%20In%20The%20Service%20of%20America%20-%20Liberator"
                 f"%20Bomber%20Production%20ca1944%20no%20sound.mp4", 299.0, 8.0),
    "lib_floor": (f"{IA}/InTheServiceOfAmerica/200-fc-1101%20In%20The%20Service%20of%20America%20-%20Liberator"
                  f"%20Bomber%20Production%20ca1944%20no%20sound.mp4", 155.0, 8.0),
}
PEXELS = {  # Pexels video id -> local file name (pulled by `fetch` via the pexels download redirect)
    "6450803": "pexels_6450803.mp4",     # Automated Car Factory - Engin Altundag
    "32386518": "pexels_32386518.mp4",   # Industrial Robotics in Action in Modern Facility - Usman AbdulrasheedGambo
    "32386532": "pexels_32386532.mp4",   # Industrial Robot Arm in High-Tech Factory - Usman AbdulrasheedGambo
    "32386590": "pexels_32386590.mp4",   # Advanced Robotics in Automated Factory - Usman AbdulrasheedGambo
    "32386627": "pexels_32386627.mp4",   # Futuristic Factory Floor with Advanced Machinery - Usman AbdulrasheedGambo
    "32386521": "pexels_32386521.mp4",   # Automated Manufacturing Process in Factory - Usman AbdulrasheedGambo
}

# Hype edit, 0:00-0:09 at 120 bpm (0.5 s beats).  (key, duration s, src in-offset s, zoom0, zoom1, cx, cy, grade)
# grade: "bw" archival black & white, "col" modern colour.  cx/cy = crop centre (0..1 of source).
HYPE = [
    ("mh2_sparks", 1.00, 0.5, 1.00, 1.08, 0.50, 0.45, "bw"),
    ("mh2_pour", 0.75, 3.5, 1.05, 1.12, 0.55, 0.40, "bw"),
    ("mh4_line", 0.75, 3.0, 1.00, 1.06, 0.50, 0.40, "bw"),
    ("mh4_drop", 0.75, 3.0, 1.00, 1.05, 0.50, 0.40, "bw"),
    ("bomber_noses", 0.75, 2.0, 1.00, 1.06, 0.50, 0.40, "bw"),
    ("lib_rows", 0.50, 2.0, 1.28, 1.32, 0.50, 0.34, "bw"),
    ("lib_floor", 0.50, 3.0, 1.28, 1.32, 0.50, 0.34, "bw"),
    ("apollo", 0.50, 2.8, 1.37, 1.43, 0.50, 0.45, "bw"),
    ("maf", 0.50, 1.0, 1.00, 1.05, 0.50, 0.50, "col"),
    ("p6450803", 0.50, 2.2, 1.00, 1.05, 0.50, 0.55, "col"),
    ("p32386518", 0.50, 9.0, 1.00, 1.04, 0.50, 0.50, "col"),
    ("p32386532", 0.50, 6.2, 1.00, 1.04, 0.45, 0.50, "col"),
    ("p32386590", 0.50, 0.5, 1.00, 1.04, 0.50, 0.50, "col"),
    ("p32386521", 0.50, 8.0, 1.00, 1.04, 0.50, 0.50, "col"),
    ("p32386627", 0.50, 0.5, 1.00, 1.05, 0.50, 0.50, "col"),
]
HYPE_END = 9.0
HYPE_TEXT = [("AMERICA BUILT IT.", 1.75, 4.0), ("NOW BUILD IT FASTER.", 6.0, 8.95)]


def log(*a):
    print("[make_final_video]", *a, flush=True)


def run(cmd, **kw):
    kw.setdefault("check", True)
    return subprocess.run([str(c) for c in cmd], **kw)


def ffprobe_json(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
                        str(path)], capture_output=True, text=True)
    return json.loads(r.stdout or "{}")


def nframes(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-count_packets", "-show_entries",
                        "stream=nb_read_packets", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    try:
        return int(r.stdout.strip().split(",")[0])
    except ValueError:
        return 0


# ======================================================================================
# fetch
# ======================================================================================
def _archive_extra():
    """NASA clips are addressed by asset path (spaces/& need quoting)."""
    import urllib.parse as up
    apollo = "KSC-1960s-MH-NAS01-0001-Apollo_Saturn_V_Crawler,_VAB_Pad_Film_To_Tape"
    maf = "MAF_20191014_CS2_LH2_Move from VAC-complete"
    return {
        "apollo": (f"{NASA}/{apollo}/{apollo}~large.mp4", APOLLO_IN, 5.0),
        "maf": (f"{NASA}/{up.quote(maf)}/{up.quote(maf)}~large.mp4", MAF_IN, 5.0),
    }


APOLLO_IN = 37.0
MAF_IN = 84.0


def all_archive():
    d = dict(ARCHIVE)
    d.update(_archive_extra())
    return d


def fetch(force=False):
    (SRCDIR / "archive").mkdir(parents=True, exist_ok=True)
    (SRCDIR / "pexels").mkdir(parents=True, exist_ok=True)
    jobs = []
    for key, (url, t_in, dur) in all_archive().items():
        out = SRCDIR / "archive" / f"{key}.mp4"
        meta = out.with_suffix(".json")
        want = dict(url=url, t_in=t_in, dur=dur)
        if out.exists() and meta.exists() and json.loads(meta.read_text()) == want and not force:
            continue
        cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-ss", f"{t_in:.3f}", "-i", url, "-t", f"{dur:.3f}",
               "-an", "-vf", "yadif=deint=interlaced,fps=30", "-c:v", "libx264", "-crf", "12", "-preset", "fast",
               "-pix_fmt", "yuv420p", str(out)]
        jobs.append((cmd, meta, want, key))
    for pid, name in PEXELS.items():
        out = SRCDIR / "pexels" / name
        if out.exists() and not force:
            continue
        url = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{redirect_url}", "-A", "Mozilla/5.0",
                              f"https://www.pexels.com/download/video/{pid}/"], capture_output=True,
                             text=True).stdout.strip()
        cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-user_agent", "Mozilla/5.0", "-i", url, "-t", "20",
               "-an", "-vf", "scale=1920:-2,fps=30", "-c:v", "libx264", "-crf", "16", "-preset", "fast", str(out)]
        jobs.append((cmd, None, None, pid))
    procs = []
    for cmd, meta, want, key in jobs:
        log("fetch", key)
        procs.append((subprocess.Popen(cmd), meta, want, key))
    for p, meta, want, key in procs:
        if p.wait() != 0:
            log("FETCH FAILED", key)
        elif meta is not None:
            meta.write_text(json.dumps(want))


def hype_source(key):
    if key.startswith("p") and key[1:] in PEXELS:
        return SRCDIR / "pexels" / PEXELS[key[1:]]
    return SRCDIR / "archive" / f"{key}.mp4"


# ======================================================================================
# TTS (runs inside the engine venv; re-execs itself there)
# ======================================================================================
def _in_venv(venv: Path):
    return Path(sys.prefix).resolve() == venv.resolve()


def _reexec(venv: Path, argv):
    py = venv / "bin" / "python"
    if not py.exists():
        sys.exit(f"missing venv {venv}; see header of this script")
    env = dict(os.environ, HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    return subprocess.run([str(py), str(Path(__file__).resolve())] + argv + ["--cut", CUT], env=env).returncode


def say_text(chunk, engine):
    return chunk.get(engine, chunk["text"])


def tts(engine, voice, speed=1.0):
    venv = VENV_QWEN if engine == "qwen" else VENV_TTS
    if not _in_venv(venv):
        rc = _reexec(venv, ["tts", "--engine", engine, "--voice", voice, "--speed", str(speed)])
        if rc:
            sys.exit(rc)
        return
    import numpy as np
    import soundfile as sf
    import torch
    outdir = TTSDIR / f"{engine}_{voice}{TTS_SUFFIX}"
    outdir.mkdir(parents=True, exist_ok=True)
    if engine == "kokoro":
        from kokoro import KPipeline
        pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
        sr = 24000
        for i, c in enumerate(VO):
            parts = [a.cpu().numpy() if hasattr(a, "cpu") else np.asarray(a)
                     for _, _, a in pipe(say_text(c, "kokoro"), voice=voice, speed=speed, split_pattern=None)]
            sf.write(outdir / f"chunk_{i:02d}.wav", np.concatenate(parts), sr)
            log("kokoro", voice, i, c["id"])
    elif engine == "qwen":
        from qwen_tts import Qwen3TTSModel
        model = Qwen3TTSModel.from_pretrained(str(QWEN_MODEL), device_map="cuda:0", dtype=torch.bfloat16)
        # sampling-based model: render QWEN_TAKES takes per sentence; `align` keeps the best take
        for i, c in enumerate(VO):
            for k in range(QWEN_TAKES):
                torch.manual_seed(1000 * k + i)
                wavs, sr = model.generate_custom_voice(text=say_text(c, "qwen"), language="English", speaker=voice,
                                                       instruct=QWEN_INSTRUCT)
                a = np.asarray(wavs[0], dtype=np.float32)
                sf.write(outdir / f"chunk_{i:02d}_t{k}.wav", a, sr)
                if k == 0:
                    sf.write(outdir / f"chunk_{i:02d}.wav", a, sr)
                log("qwen", voice, i, c["id"], "take", k, f"{len(a) / sr:.2f}s")
    else:
        sys.exit(f"unknown engine {engine}")
    # preview: chunks joined with short pauses, so the user can audition the whole read
    chunks = [sf.read(outdir / f"chunk_{i:02d}.wav") for i in range(len(VO))]
    sr = chunks[0][1]
    gap = np.zeros(int(0.35 * sr), dtype=np.float32)
    full = np.concatenate([np.concatenate([c[0].astype(np.float32), gap]) for c in chunks])
    sf.write(TTSDIR / f"{engine}_{voice}{TTS_SUFFIX}.wav", full, sr)


def _norm_words(s):
    s = s.lower().replace("3d", "three d").replace("-", " ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return s.split()


def _wer(ref, hyp):
    r, h = _norm_words(ref), _norm_words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(1, len(r))


def align(name):
    """Word timestamps + quality report for tts_candidates/<name>/chunk_XX.wav."""
    if not _in_venv(VENV_TTS):
        rc = _reexec(VENV_TTS, ["align", "--name", name])
        if rc:
            sys.exit(rc)
        return
    import numpy as np
    import soundfile as sf
    from faster_whisper import WhisperModel
    model = WhisperModel("small", device="cuda", compute_type="float16")
    d = TTSDIR / name
    rep = dict(name=name, chunks=[])
    for i, c in enumerate(VO):
        p = d / f"chunk_{i:02d}.wav"
        takes = sorted(d.glob(f"chunk_{i:02d}_t*.wav"))
        if takes:   # choose the take with the lowest WER, then the most natural rate (closest to 2.75 words/s)
            scored = []
            for tp in takes:
                sg, _ = model.transcribe(str(tp), language="en", beam_size=5)
                hyp_t = " ".join(x.text for x in sg)
                ta, tsr = sf.read(tp)
                rate = len(_norm_words(c["text"])) / (len(ta) / tsr)
                scored.append((round(_wer(c["text"], hyp_t), 3), abs(rate - 2.75), tp.name, hyp_t))
            scored.sort()
            log(f"  takes {c['id']}: " + " | ".join(f"{n} wer={w} dr={r:.2f}" for w, r, n, _ in scored))
            shutil.copy(d / scored[0][2], p)
            rep.setdefault("takes", {})[c["id"]] = [dict(take=n, wer=w, hyp=h) for w, r, n, h in scored]
        a, sr = sf.read(p)
        if a.ndim > 1:
            a = a.mean(1)
        segs, _ = model.transcribe(str(p), language="en", word_timestamps=True, beam_size=5,
                                   initial_prompt="Capture Studio, LingBot-Map, BIM, Gemini, MuJoCo, CNC, IFC.")
        words = [dict(w=w.word.strip(), s=round(w.start, 3), e=round(w.end, 3)) for sg in segs for w in sg.words]
        hyp = " ".join(w["w"] for w in words)
        # artefacts: clipping, longest internal silence, speech start/end
        env = np.sqrt(np.convolve(a ** 2, np.ones(int(0.02 * sr)) / int(0.02 * sr), mode="same"))
        voiced = env > (10 ** (-40 / 20)) * max(1e-9, np.abs(a).max())
        idx = np.flatnonzero(voiced)
        s0, s1 = (idx[0] / sr, idx[-1] / sr) if len(idx) else (0, len(a) / sr)
        runs, cur = [], 0
        for v in voiced[idx[0]:idx[-1]] if len(idx) else []:
            cur = 0 if v else cur + 1
            runs.append(cur)
        max_sil = (max(runs) / sr) if runs else 0
        rep["chunks"].append(dict(id=c["id"], ref=c["text"], hyp=hyp, wer=round(_wer(c["text"], hyp), 3),
                                  dur=round(len(a) / sr, 3), speech=[round(s0, 3), round(s1, 3)],
                                  peak=round(float(np.abs(a).max()), 3),
                                  clipped=int((np.abs(a) > 0.999).sum()), max_internal_silence=round(max_sil, 3),
                                  words=words))
    tot = sum(len(_norm_words(c["ref"])) for c in rep["chunks"])
    rep["wer"] = round(sum(c["wer"] * len(_norm_words(c["ref"])) for c in rep["chunks"]) / tot, 4)
    rep["speech_seconds"] = round(sum(c["speech"][1] - c["speech"][0] for c in rep["chunks"]), 2)
    (d / "align.json").write_text(json.dumps(rep, indent=1))
    chunks = [sf.read(d / f"chunk_{i:02d}.wav") for i in range(len(VO))]     # preview of the selected takes
    gap = np.zeros(int(0.35 * chunks[0][1]), dtype=np.float32)
    sf.write(TTSDIR / f"{name}.wav", np.concatenate([np.concatenate([c[0].astype(np.float32), gap])
                                                    for c in chunks]), chunks[0][1])
    log(f"{name}: WER {rep['wer']:.3f}, speech {rep['speech_seconds']} s")
    for c in rep["chunks"]:
        log(f"  {c['id']:6s} wer={c['wer']:.2f} dur={c['dur']:.2f} sil={c['max_internal_silence']:.2f} "
            f"peak={c['peak']:.2f} | {c['hyp']}")


# ======================================================================================
# Audio helpers (project venv: numpy/scipy/soundfile)
# ======================================================================================
def load_mono(path, sr_out=SR):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(1)
    if sr != sr_out:
        g = math.gcd(sr, sr_out)
        a = resample_poly(a, sr_out // g, sr // g).astype(np.float32)
    return a


def trim_silence(a, sr=SR, thr_db=-45, pad=0.04):
    import numpy as np
    env = np.sqrt(np.convolve(a ** 2, np.ones(int(0.01 * sr)) / int(0.01 * sr), mode="same"))
    idx = np.flatnonzero(env > 10 ** (thr_db / 20) * np.abs(a).max())
    if not len(idx):
        return a, 0.0
    s0 = max(0, idx[0] - int(pad * sr))
    s1 = min(len(a), idx[-1] + int(pad * sr))
    out = a[s0:s1].copy()
    f = int(0.01 * sr)
    out[:f] *= np.linspace(0, 1, f)
    out[-f:] *= np.linspace(1, 0, f)
    return out, s0 / sr


def declick(a, sr=SR):
    """Replace isolated sample spikes (>> local derivative median) by interpolation."""
    import numpy as np
    d = np.abs(np.diff(a, prepend=a[0]))
    med = np.convolve(d, np.ones(64) / 64, mode="same") + 1e-6
    bad = np.flatnonzero(d > 12 * med + 0.05)
    a = a.copy()
    for i in bad:
        lo, hi = max(0, i - 3), min(len(a) - 1, i + 3)
        a[lo:hi + 1] = np.linspace(a[lo], a[hi], hi - lo + 1)
    return a, len(bad)


def compress(a, sr=SR, thr_db=-20, ratio=3.0, att=0.005, rel=0.12, makeup_db=0.0):
    import numpy as np
    from scipy.signal import lfilter
    env = np.abs(a)
    # one-pole peak follower with separate attack/release (vectorised approx: attack filter then release filter)
    aa, ar = math.exp(-1 / (att * sr)), math.exp(-1 / (rel * sr))
    e1 = lfilter([1 - aa], [1, -aa], env)
    e = np.maximum(e1, lfilter([1 - ar], [1, -ar], e1))
    lvl = 20 * np.log10(e + 1e-9)
    over = np.maximum(0, lvl - thr_db)
    gain_db = -over * (1 - 1 / ratio) + makeup_db
    return (a * 10 ** (gain_db / 20)).astype(np.float32)


def highpass(a, f=80, sr=SR, order=2):
    from scipy.signal import butter, sosfilt
    return sosfilt(butter(order, f, "hp", fs=sr, output="sos"), a).astype("float32")


def time_stretch(a, factor, sr=SR):
    """factor > 1 = faster.  ffmpeg atempo (WSOLA) keeps pitch; only used for small factors."""
    import numpy as np
    import soundfile as sf
    if abs(factor - 1) < 1e-3:
        return a
    CACHE.mkdir(parents=True, exist_ok=True)
    i, o = CACHE / "_ts_in.wav", CACHE / "_ts_out.wav"
    sf.write(i, a, sr)
    run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", i, "-af", f"atempo={factor:.4f}", o])
    b, _ = sf.read(o, dtype="float32")
    return b.astype(np.float32)


# ======================================================================================
# VO layout -> vo.wav + word timeline
# ======================================================================================
def build_vo(voice):
    import numpy as np
    import soundfile as sf
    engine, vname = voice.split(":", 1)
    name = f"{engine}_{vname}{TTS_SUFFIX}"
    d = TTSDIR / name
    rep = json.loads((d / "align.json").read_text())
    chunks, offs, nclicks = [], [], 0
    for i, c in enumerate(VO):
        a = load_mono(d / f"chunk_{i:02d}.wav")
        a = highpass(a, 80)
        a, nc = declick(a)
        nclicks += nc
        a, t0 = trim_silence(a)
        chunks.append(a)
        offs.append(t0)
    # the last line ends 0.45 s before the end of the film; the end card fades in 0.30 s before it starts
    global END_CARD
    END_CARD = round(min(END_CARD_MAX, DUR - 0.45 - len(chunks[-1]) / SR - 0.30), 3)
    # fit: chunks 0..len-2 must end by END_CARD-0.15; last chunk starts at END_CARD+0.30
    body = sum(len(c) for c in chunks[:-1]) / SR + sum(GAP_AFTER[c["id"]] for c in VO[:-1])
    avail = (END_CARD - 0.15) - VO_START
    factor = max(1.0, body / avail)
    if factor > 1.0:
        log(f"VO body {body:.2f}s > {avail:.2f}s, tempo x{factor:.3f}")
        chunks[:-1] = [time_stretch(c, factor) for c in chunks[:-1]]
    gaps = {k: v / factor for k, v in GAP_AFTER.items()}
    spare = avail - (sum(len(c) for c in chunks[:-1]) / SR + sum(gaps[c["id"]] for c in VO[:-1]))
    if CUT == "v2" and spare > SPARE_AFTER[1]:
        # v2 script is shorter: after the centrepiece hold, spread what is left over the other pauses
        rest = spare - SPARE_AFTER[1]
        tot_w = sum(EXTRA_GAP_W.values())
        for k, wgt in EXTRA_GAP_W.items():
            gaps[k] = gaps.get(k, 0) + min(1.0, rest * wgt / tot_w)
        log(f"v2: {spare:.2f}s spare -> {SPARE_AFTER[1]}s after {SPARE_AFTER[0]}, rest over pauses")
    total = np.zeros(int(DUR * SR), dtype=np.float32)
    t = VO_START
    timeline = []
    for i, (c, a) in enumerate(zip(VO, chunks)):
        if c["id"] == "end":
            t = END_CARD + 0.30
            fac = 1.0
        else:
            fac = factor
        s = int(round(t * SR))
        n = min(len(a), len(total) - s)
        total[s:s + n] += a[:n]
        words = []
        for w in rep["chunks"][i]["words"]:
            ws = t + (w["s"] - offs[i]) / fac
            we = t + (w["e"] - offs[i]) / fac
            words.append(dict(w=w["w"], s=round(ws, 3), e=round(we, 3)))
        timeline.append(dict(id=c["id"], text=c["text"], start=round(t, 3), end=round(t + len(a) / SR, 3),
                             words=words))
        t += len(a) / SR + gaps.get(c["id"], 0)
        if c["id"] == SPARE_AFTER[0] and spare > 0:   # give spare time to the shot after this line
            t += min(spare, SPARE_AFTER[1])
    # polish whole track: gentle compression, then peak-normalise to -3 dBFS (loudness set in the mix)
    total = compress(total, thr_db=-22, ratio=2.5, att=0.004, rel=0.15)
    total *= 10 ** (-3 / 20) / max(1e-6, np.abs(total).max())
    sf.write(FINAL / f"vo{TTS_SUFFIX}.wav", np.stack([total, total], 1), SR, subtype="PCM_24")
    (CACHE / "vo_timeline.json").write_text(json.dumps(dict(voice=voice, tempo=factor, declicked=nclicks,
                                                            chunks=timeline), indent=1))
    log(f"vo.wav written (tempo x{factor:.3f}, {nclicks} clicks repaired)")
    return timeline, total


def word_time(timeline, cid, word, which="s", nth=0, default=None):
    for c in timeline:
        if c["id"] != cid:
            continue
        hits = [w for w in c["words"] if re.sub(r"[^a-z0-9]", "", w["w"].lower()).startswith(word.lower())]
        if len(hits) > nth:
            return hits[nth][which]
        if default == "start":
            return c["start"]
    if default is not None and not isinstance(default, str):
        return default
    raise KeyError(f"word {word!r} not found in chunk {cid}")


def chunk(timeline, cid):
    return next(c for c in timeline if c["id"] == cid)


# ======================================================================================
# Music: original synthesized bed (numpy).  120 bpm, A minor.
# ======================================================================================
def synth_music(cuts, vo, timeline):
    import numpy as np
    from scipy.signal import butter, sosfilt
    rng = np.random.default_rng(7)
    n = int(DUR * SR)
    L = np.zeros(n, np.float32)
    R = np.zeros(n, np.float32)
    t_all = np.arange(n) / SR

    def add(sig, t0, gain=1.0, pan=0.0):
        s = int(t0 * SR)
        if s >= n:
            return
        m = min(len(sig), n - s)
        L[s:s + m] += sig[:m] * gain * math.sqrt(0.5 * (1 - pan))
        R[s:s + m] += sig[:m] * gain * math.sqrt(0.5 * (1 + pan))

    def bp(x, lo, hi, order=2):
        return sosfilt(butter(order, [lo, hi], "bp", fs=SR, output="sos"), x)

    def lp(x, f, order=2):
        return sosfilt(butter(order, f, "lp", fs=SR, output="sos"), x)

    def hp(x, f, order=2):
        return sosfilt(butter(order, f, "hp", fs=SR, output="sos"), x)

    def kick(dur=0.45, f0=110, f1=42, level=1.0):
        t = np.arange(int(dur * SR)) / SR
        f = f1 + (f0 - f1) * np.exp(-t / 0.035)
        ph = 2 * np.pi * np.cumsum(f) / SR
        body = np.sin(ph) * np.exp(-t / 0.16)
        click = hp(rng.standard_normal(len(t)), 2000) * np.exp(-t / 0.004) * 0.25
        return (np.tanh(1.6 * (body + click)) * level).astype(np.float32)

    def snare(dur=0.25, level=0.5):
        t = np.arange(int(dur * SR)) / SR
        nz = bp(rng.standard_normal(len(t)), 900, 7000) * np.exp(-t / 0.06)
        tone = np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.05) * 0.5
        return ((nz + tone) * level).astype(np.float32)

    def hat(dur=0.06, level=0.18):
        t = np.arange(int(dur * SR)) / SR
        return (hp(rng.standard_normal(len(t)), 7000) * np.exp(-t / 0.015) * level).astype(np.float32)

    def tom(dur=0.5, f=70, level=0.6):
        t = np.arange(int(dur * SR)) / SR
        fr = f * (1 + 0.6 * np.exp(-t / 0.05))
        return (np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-t / 0.18) * level).astype(np.float32)

    def saw(freq, t, detune=0.0):
        ph = (freq * (1 + detune) * t) % 1.0
        return 2 * ph - 1

    beat = 0.5
    # --- hype section 0..9 s -------------------------------------------------------------
    for k in range(int(HYPE_END / beat)):
        tb = k * beat
        add(hp(kick(level=0.8), 38).astype(np.float32), tb)
        if k % 2 == 1 and tb >= 2.0:
            add(snare(level=0.65), tb, pan=0.1)
        sub = 2 if tb < 4 else (4 if tb < 7 else 8)          # hats accelerate
        for j in range(sub):
            add(hat(level=0.16 + 0.08 * (tb / HYPE_END)), tb + j * beat / sub, pan=-0.3 + 0.6 * (j % 2))
        if tb >= 1.0:                                          # off-beat chord stabs for mid-range drive
            ts = np.arange(int(0.2 * SR)) / SR
            ch = [(220.0, 261.63, 329.63), (220.0, 261.63, 329.63), (174.61, 220.0, 261.63),
                  (196.0, 246.94, 293.66)][int(tb // 1.0) % 4]
            st = sum(saw(f, ts, dt) for f in ch for dt in (-0.006, 0.006)) / 6 * np.exp(-ts / 0.07)
            add(lp(hp(st, 200), 3800).astype(np.float32), tb + beat / 2, gain=0.34 + 0.16 * tb / HYPE_END,
                pan=0.2 if k % 2 else -0.2)
    for tc in cuts[1:]:                                        # hit on every picture cut
        add(tom(level=0.35), tc)
    # driving bass: 8ths, A-A-F-G progression, two beats each
    roots = [55.0, 55.0, 43.65, 49.0]
    bass = np.zeros(n, np.float32)
    for k in range(int(HYPE_END / (beat / 2))):
        t0 = k * beat / 2
        f = roots[int(t0 // 1.0) % 4]
        seg = np.arange(int(beat / 2 * SR)) / SR
        env = np.exp(-seg / 0.09)
        s = int(t0 * SR)
        bass[s:s + len(seg)] += (saw(f, seg) + 0.5 * saw(f * 2, seg, 0.003)) * env
    bass = lp(bass, 600) * 0.26
    L += bass
    R += bass
    # riser 5.5 -> 9.0: filtered noise + rising tone
    r0, r1 = 5.5, HYPE_END
    tr = np.arange(int((r1 - r0) * SR)) / SR
    prog = tr / (r1 - r0)
    nz = rng.standard_normal(len(tr))
    riser = np.zeros_like(nz)
    for a0 in range(0, len(tr), 2400):                      # sweep bandpass in blocks
        fc = 400 + 7000 * prog[a0] ** 2
        blk = slice(a0, min(len(tr), a0 + 2400))
        riser[blk] = bp(nz, fc * 0.7, min(fc * 1.4, 20000))[blk]
    tone = np.sin(2 * np.pi * np.cumsum(220 + 660 * prog ** 2) / SR) * 0.25
    add(((riser * 0.8 + tone) * prog ** 2.2 * 0.5).astype(np.float32), r0)
    # impact at 9.0: sub boom + kick + noise burst, then silence gap into the title
    ti = np.arange(int(2.5 * SR)) / SR
    boom = np.sin(2 * np.pi * np.cumsum(38 + 30 * np.exp(-ti / 0.08)) / SR) * np.exp(-ti / 0.9)
    burst = lp(rng.standard_normal(len(ti)), 3000) * np.exp(-ti / 0.25) * 0.5
    add(np.tanh(1.3 * (boom * 0.6 + burst)).astype(np.float32) * 0.8, HYPE_END)
    cym = hp(rng.standard_normal(len(ti)), 4000) * np.exp(-ti / 0.6) * 0.25
    add(cym.astype(np.float32), HYPE_END, pan=0.0)
    add(kick(level=1.0), HYPE_END)

    # --- bed under the voice-over: Am - F - C - G pad + soft pulse ---------------------------
    chords = [(220.0, 261.63, 329.63), (174.61, 220.0, 261.63), (196.0, 261.63, 329.63), (196.0, 246.94, 293.66)]
    pad = np.zeros(n, np.float32)
    bed0, bed1 = 9.6, END_CARD
    bar = 4.0
    t = t_all
    for ci in range(int((bed1 - bed0) / bar) + 2):
        t0 = bed0 + ci * bar
        notes = chords[ci % 4]
        s0, s1 = int(t0 * SR), min(n, int((t0 + bar + 0.6) * SR))
        tt = t[s0:s1] - t0
        env = np.minimum(1, tt / 0.8) * np.minimum(1, np.maximum(0, (bar + 0.6 - tt)) / 0.8)
        sig = sum(saw(f, tt, dt) for f in notes for dt in (-0.004, 0.004)) / 6
        pad[s0:s1] += (sig * env).astype(np.float32)
    pad = lp(pad, 900) * 0.30
    fade_in = np.clip((t_all - bed0) / 1.5, 0, 1)
    mask = (t_all < bed1 + 0.6) * fade_in
    pad *= mask.astype(np.float32)
    # soft 8th pulse (filtered root) + quiet kick on beats 1 & 3
    pulse = np.zeros(n, np.float32)
    for k in range(int((bed1 - bed0) / (beat / 2))):
        t0 = bed0 + k * beat / 2
        f = [110.0, 87.31, 130.81, 98.0][int((t0 - bed0) // bar) % 4] / 2
        seg = np.arange(int(beat / 2 * SR)) / SR
        s = int(t0 * SR)
        pulse[s:s + len(seg)] += saw(f, seg) * np.exp(-seg / 0.07)
    pulse = lp(pulse, 300) * 0.22 * fade_in.astype(np.float32)
    for k in range(int((bed1 - bed0 - 1.5) / (2 * beat))):
        add(hp(kick(level=0.30, f0=90), 50).astype(np.float32), bed0 + 1.5 + k * 2 * beat)
    L += pad + pulse
    R += pad * 0.95 + pulse
    # shimmer hit on cuts in part 2 (very soft)
    # --- end card resolve: A minor -> C major (add9) swell, ring out to 60 s -----------------
    te = np.arange(int((DUR - END_CARD) * SR)) / SR
    res = sum(saw(f, te, dt) for f in (130.81, 196.0, 261.63, 329.63, 392.0, 587.33) for dt in (-0.003, 0.003)) / 12
    env = np.minimum(1, te / 0.25) * np.exp(-te / 2.2)
    res = lp(res, 3500) * env * 0.6
    bell = sum(np.sin(2 * np.pi * f * te) * np.exp(-te / 1.2) for f in (523.25, 784.0, 1046.5)) * 0.06
    add((res + bell).astype(np.float32), END_CARD)
    add(boom[: len(te)].astype(np.float32) * 0.25, END_CARD)

    # --- duck under the voice: -12 dB with 60 ms attack / 350 ms release ---------------------
    voenv = np.abs(vo)
    win = int(0.05 * SR)
    voenv = np.convolve(voenv, np.ones(win) / win, mode="same")
    active = (voenv > 10 ** (-38 / 20)).astype(np.float32)
    from scipy.ndimage import maximum_filter1d, uniform_filter1d
    active = maximum_filter1d(active, int(0.25 * SR))           # bridge short gaps between words
    duck_db = -12 * uniform_filter1d(active, int(0.12 * SR))
    g = (10 ** (duck_db / 20)).astype(np.float32)
    L *= g
    R *= g
    # fade-out at the very end
    fo = np.clip((DUR - t_all) / 0.6, 0, 1).astype(np.float32)
    L *= fo
    R *= fo
    mus = np.stack([L, R], 1)
    mus = np.tanh(mus * 0.9) / 0.9                                # soft safety limiter
    mus *= 10 ** (-1.0 / 20) / max(1e-6, np.abs(mus).max())
    return mus.astype(np.float32)


# ======================================================================================
# Graphics (PIL)
# ======================================================================================
def font(size, weight="Bold"):
    from PIL import ImageFont
    f = ImageFont.truetype(str(ASSETS / "Inter.ttf"), size)
    try:
        f.set_variation_by_name(weight)
    except Exception:
        pass
    return f


ORANGE = (255, 138, 61)
INK = (11, 13, 18)


def text_rgba(text, size, weight="Bold", fill=(255, 255, 255), tracking=0, shadow=True):
    from PIL import Image, ImageDraw, ImageFilter
    import numpy as np
    f = font(size, weight)
    widths = [f.getlength(ch) for ch in text]
    w = int(sum(widths) + tracking * (len(text) - 1)) + 40
    asc, desc = f.getmetrics()
    h = asc + desc + 40
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    x = 20
    for ch, cw in zip(text, widths):
        d.text((x, 20), ch, font=f, fill=fill + (255,))
        x += cw + tracking
    if shadow:
        sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
        a = img.split()[3].filter(ImageFilter.GaussianBlur(size / 10))
        sh.putalpha(a.point(lambda v: int(v * 0.65)))
        out = Image.alpha_composite(sh, img)
    else:
        out = img
    return np.array(out)


def pill_rgba(text, size=26, fg=(235, 238, 245), bg=(16, 19, 26, 210), border=(255, 255, 255, 40), dot=None,
              pad=(18, 10), weight="SemiBold", tracking=1.5):
    from PIL import Image, ImageDraw
    import numpy as np
    f = font(size, weight)
    tw = sum(f.getlength(c) for c in text) + tracking * (len(text) - 1)
    asc, desc = f.getmetrics()
    dot_w = (size * 0.45 + 12) if dot else 0
    w, h = int(tw + 2 * pad[0] + dot_w), int(asc + desc * 0.4 + 2 * pad[1])
    img = Image.new("RGBA", (w + 4, h + 4), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((2, 2, w + 1, h + 1), radius=h // 2, fill=bg, outline=border, width=2)
    x = pad[0] + 2
    if dot:
        r = size * 0.22
        cy = 2 + h / 2
        d.ellipse((x, cy - r, x + 2 * r, cy + r), fill=dot)
        x += dot_w
    for ch in text:
        d.text((x, pad[1] + 2), ch, font=f, fill=fg)
        x += f.getlength(ch) + tracking
    return np.array(img)


def title_card_rgba(main="CAPTURE STUDIO", sub=None, size=150):
    """Big centred wordmark with an orange accent bar (no third-party logos)."""
    from PIL import Image, ImageDraw
    import numpy as np
    t = text_rgba(main, size, "Black", tracking=size * 0.06, shadow=False)
    hh, ww = t.shape[:2]
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    x0, y0 = (W - ww) // 2, (H - hh) // 2 - (30 if sub else 0)
    canvas.alpha_composite(Image.fromarray(t), (x0, y0))
    d = ImageDraw.Draw(canvas)
    bw = int(ww * 0.12)
    d.rectangle(((W - bw) // 2, y0 + hh - 6, (W + bw) // 2, y0 + hh + 4), fill=ORANGE + (255,))
    if sub:
        s = text_rgba(sub, int(size * 0.30), "Medium", fill=(210, 215, 225), tracking=1, shadow=False)
        canvas.alpha_composite(Image.fromarray(s), ((W - s.shape[1]) // 2, y0 + hh + 34))
    return np.array(canvas)


def model_ladder_rgba(active_upto=4, highlight=True):
    """Four labelled chips: Carbon (highlighted, hard tasks) -> Neon, Argon, Hydrogen (lighter steps)."""
    from PIL import Image, ImageDraw
    import numpy as np
    names = ["Carbon", "Neon", "Argon", "Hydrogen"]
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(canvas)
    head = text_rgba("INTELLIGENCE MATCHED TO THE TASK", 30, "SemiBold", fill=(200, 205, 215), tracking=4,
                     shadow=False)
    canvas.alpha_composite(Image.fromarray(head), ((W - head.shape[1]) // 2, 330))
    f = font(54, "Bold")
    fs = font(24, "Medium")
    chip_w, chip_h, gap = 330, 120, 36
    x0 = (W - (4 * chip_w + 3 * gap)) // 2
    y0 = 430
    for i, nm in enumerate(names):
        if i >= active_upto:
            continue
        x = x0 + i * (chip_w + gap)
        hot = highlight and i == 0
        fill = ORANGE + (255,) if hot else (18, 22, 30, 225)
        outline = ORANGE + (255,) if hot else (255, 255, 255, 90)
        d.rounded_rectangle((x, y0, x + chip_w, y0 + chip_h), radius=22, fill=fill, outline=outline, width=3)
        label = "GEMINI 4"
        lw = sum(fs.getlength(c) for c in label) + 3 * (len(label) - 1)
        lx = x + (chip_w - lw) / 2
        for ch in label:
            d.text((lx, y0 + 14), ch, font=fs, fill=(30, 20, 10) if hot else (170, 178, 190))
            lx += fs.getlength(ch) + 3
        tw = f.getlength(nm)
        d.text((x + (chip_w - tw) / 2, y0 + 40), nm, font=f, fill=(20, 12, 6) if hot else (240, 242, 246))
    # difficulty scale under the chips
    yb = y0 + chip_h + 44
    xa, xb = x0, x0 + 4 * chip_w + 3 * gap
    for k in range(xb - xa):
        a = 1 - k / (xb - xa)
        col = tuple(int(ORANGE[j] * a + 120 * (1 - a)) for j in range(3))
        d.line((xa + k, yb, xa + k, yb + 6), fill=col + (230,))
    sl = font(26, "Medium")
    d.text((xa, yb + 18), "hard design decisions", font=sl, fill=(235, 238, 245))
    txt = "lighter steps"
    d.text((xb - sl.getlength(txt), yb + 18), txt, font=sl, fill=(190, 196, 206))
    return np.array(canvas)


# ======================================================================================
# Picture: frame compositor (cv2 + numpy), piped into ffmpeg
# ======================================================================================
class Reader:
    """Sequential frame reader for one shot (ffmpeg decodes from t_in at the requested speed)."""

    def __init__(self, path, t_in, n, speed=1.0, maxw=1920):
        self.path, self.n = str(path), n
        info = ffprobe_json(path)
        vs = next(s for s in info["streams"] if s["codec_type"] == "video")
        sw, sh = int(vs["width"]), int(vs["height"])
        sar = vs.get("sample_aspect_ratio", "1:1")
        if sar not in ("1:1", "0:1", None):
            a, b = map(int, sar.split(":"))
            sw = int(round(sw * a / b))
        if sw > maxw:
            sh, sw = int(round(sh * maxw / sw / 2) * 2), maxw
        self.w, self.h = sw, sh
        vf = f"setpts=(PTS-STARTPTS)/{speed},fps={FPS},scale={sw}:{sh}:flags=lanczos,setsar=1"
        cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", f"{max(0, t_in):.3f}", "-i", self.path,
               "-vf", vf, "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10 ** 8)
        self.last = None
        self.k = 0

    def read(self):
        import numpy as np
        buf = self.p.stdout.read(self.w * self.h * 3)
        if len(buf) == self.w * self.h * 3:
            self.last = np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3)
        elif self.last is None:
            self.last = np.zeros((self.h, self.w, 3), np.uint8)
        self.k += 1
        return self.last   # freeze last frame if the source runs short

    def close(self):
        try:
            self.p.stdout.close()
            self.p.kill()
        except Exception:
            pass


def ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def warp(frame, zoom, cx, cy):
    """Scale-to-cover * zoom, centred on (cx, cy) in source-normalised coords, clamped to stay inside."""
    import cv2
    import numpy as np
    sh, sw = frame.shape[:2]
    s = max(W / sw, H / sh) * zoom
    vw, vh = W / s, H / s                    # visible source window
    x = min(max(cx * sw, vw / 2), sw - vw / 2)
    y = min(max(cy * sh, vh / 2), sh - vh / 2)
    M = np.float32([[s, 0, W / 2 - s * x], [0, s, H / 2 - s * y]])
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    return cv2.warpAffine(frame, M, (W, H), flags=interp, borderMode=cv2.BORDER_REFLECT)


class Grader:
    def __init__(self):
        import numpy as np
        rng = np.random.default_rng(3)
        self.grain = [(rng.standard_normal((H // 2, W // 2)) * 1.0).astype(np.float32) for _ in range(6)]
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        self.vig = np.clip(1 - 0.28 * np.maximum(0, r - 0.55) ** 1.6, 0, 1).astype(np.float32)[..., None]
        lut = np.arange(256, dtype=np.float32) / 255
        s = 1 / (1 + np.exp(-(lut - 0.5) * 6.0))
        s = (s - s.min()) / (s.max() - s.min())
        self.bw_lut = np.clip(s * 255, 0, 255).astype(np.uint8)
        c = 1 / (1 + np.exp(-(lut - 0.5) * 4.0))
        c = (c - c.min()) / (c.max() - c.min())
        self.col_lut = np.clip((0.55 * c + 0.45 * lut) * 255, 0, 255).astype(np.uint8)

    def bw(self, f, k):
        import cv2
        import numpy as np
        g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        # auto-level each frame a little so mixed archival sources match
        lo, hi = np.percentile(g[::4, ::4], (1, 99.5))
        g = np.clip((g.astype(np.float32) - lo) * (235.0 / max(30.0, hi - lo)) + 8, 0, 255).astype(np.uint8)
        g = cv2.LUT(g, self.bw_lut).astype(np.float32)
        gr = cv2.resize(self.grain[k % len(self.grain)], (W, H), interpolation=cv2.INTER_LINEAR)
        g = g + gr * 9.0
        flick = 1 + 0.02 * math.sin(k * 2.3)
        out = np.stack([g * 1.02, g * 1.0, g * 0.95], 2) * flick   # faint warm silver tone
        return np.clip(out * self.vig, 0, 255).astype(np.uint8)

    def col(self, f, k):
        import cv2
        import numpy as np
        f = cv2.LUT(f, self.col_lut).astype(np.float32)
        lum = f.mean(2, keepdims=True)
        f = lum + (f - lum) * 0.9                                    # slight desat for cohesion
        f[..., 2] *= 1.03
        f[..., 0] *= 0.98                                            # cool, clean
        gr = cv2.resize(self.grain[k % len(self.grain)], (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
        return np.clip((f + gr * 3.0) * self.vig, 0, 255).astype(np.uint8)


def overlay(dst, rgba, x, y, alpha=1.0):
    import numpy as np
    h, w = rgba.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0 or alpha <= 0:
        return dst
    src = rgba[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
    a = src[..., 3:4] / 255.0 * alpha
    region = dst[y0:y1, x0:x1].astype(np.float32)
    dst[y0:y1, x0:x1] = (region * (1 - a) + src[..., :3] * a).astype(np.uint8)
    return dst


def find_follow(extra=None):
    names = ["follow_video.mp4", "follow_cloud.mp4", "follow_bim.mp4"]
    dirs = ([Path(extra)] if extra else []) + FOLLOW_DIRS
    for d in dirs:
        ps = [d / n for n in names]
        if all(p.exists() for p in ps):
            if (d / "follow_cloud_bim.mp4").exists():
                ps[2] = d / "follow_cloud_bim.mp4"
            counts = [nframes(p) for p in ps]
            if min(counts) > 60 and len(set(counts)) == 1:
                return dict(zip(["video", "cloud", "bim"], ps)), counts[0]
            log(f"follow clips in {d} incomplete/unequal: {counts}")
    return None, 0


# Design-section presets: source clip + cue points (seconds in that clip) + punch-in regions
# (cx, cy, zoom; cx/cy are 0..1 of the 1920x1080 viewer recording, clamped so the window stays inside).
DESIGN_PRESETS = {
    # Gemini 4 Carbon session (http://127.0.0.1:8646, recorded read-only with
    # `record_demo.py demo/final/rec/design_8646 http://127.0.0.1:8646 --no-hitl --stages design`)
    "gemini": dict(
        src=FINAL / "rec" / "design_8646" / "04_agentic_design.mp4",
        cues=dict(overview=0.9, cnc=4.2, safety=10.8, physics=14.2, risky=16.9, bg=9.0, nextsteps=19.4,
                  end=21.8),
        ui=dict(scene=(0.50, 0.55, 1.40), cnc=(0.37, 0.45, 2.10), safety=(1.00, 0.50, 1.04),
                physics=(1.00, 0.00, 1.25), risky=(0.40, 0.45, 1.85), nextsteps=(1.00, 0.00, 1.12),
                end=(0.50, 0.55, 1.45)),
    ),
    # earlier non-Gemini session (demo/04_agentic_design.mp4): has the rack topple + question card
    "legacy": dict(
        src=DEMO / "04_agentic_design.mp4",
        cues=dict(overview=0.3, cnc=3.0, safety=19.5, physics=8.6, risky=30.0, bg=16.0, nextsteps=32.6,
                  end=32.6),
        ui=dict(scene=(0.45, 0.50, 1.50), cnc=(0.56, 0.38, 2.0), safety=(1.00, 0.50, 1.04),
                physics=(0.52, 0.27, 2.05), risky=(1.00, 1.00, 2.20), nextsteps=(1.00, 0.275, 2.30),
                end=(0.45, 0.50, 1.45)),
    ),
}
# v2: Gemini 4 Carbon session on the segmented model, recorded read-only from the live project server:
# `record_demo.py demo/final/rec/v2_stages http://127.0.0.1:8642 --no-hitl --stages cloud,understand,design`
DESIGN_PRESETS["v2"] = dict(
    src=FINAL / "rec" / "v2_stages" / "04_agentic_design.mp4",
    # 0.3 wishes + guard fence going up | 6.6 VMC + robot loader placed | 11.6 interlocked gate, aisle in view |
    # 17.6 Physics + Wishes pills turn green at 18.7 ("All checks pass") | 20.0 settled cell
    cues=dict(overview=0.3, cnc=6.6, safety=11.6, physics=17.6, risky=20.0, bg=9.0, end=21.8),
    ui=dict(scene=(0.45, 0.50, 1.30), cnc=(0.54, 0.40, 2.00), safety=(0.58, 0.45, 1.70),
            physics=(1.00, 0.00, 1.25), risky=(0.50, 0.48, 1.50), end=(0.50, 0.55, 1.45)),
)
DESIGN_PRESET = "gemini"


def build_shots(timeline, follow, follow_n, design_src, cues, preset=None):
    preset = preset or DESIGN_PRESETS[DESIGN_PRESET]
    """Return list of shots: dict(t0, t1, layers=[...], wipe=None|(tw0, tw1), label=...)."""
    S = []
    # ---- Part 1: hype -------------------------------------------------------------------
    t = 0.0
    cuts = []
    for key, dur, tin, z0, z1, cx, cy, grade in HYPE:
        cuts.append(t)
        S.append(dict(t0=t, t1=t + dur, layers=[dict(src=hype_source(key), tin=tin, z0=z0, z1=z1, c0=(cx, cy),
                                                     c1=(cx, cy), grade=grade)], tag=key))
        t += dur
    assert abs(t - HYPE_END) < 0.02, f"hype durations sum to {t}"
    cuts.append(HYPE_END)
    # ---- Title card (dark, faint point cloud) until "Walk" ------------------------------------
    t_walk = chunk(timeline, "walk")["start"] - 0.08
    S.append(dict(t0=HYPE_END, t1=t_walk, layers=[dict(src=stage_clip("02_point_cloud.mp4"), tin=10.0, z0=1.55, z1=1.7,
                                                       c0=(0.62, 0.52), c1=(0.62, 0.52), grade="dim")],
                  tag="title"))
    # ---- video -> cloud -> BIM ----------------------------------------------------------
    t_point = word_time(timeline, "cloud", "point", default="start")
    t_bimc = chunk(timeline, "bim")["start"]
    t_agent = chunk(timeline, "agent")["start"] - 0.10
    try:
        t_bimw = word_time(timeline, "bim", "bim")
    except KeyError:
        t_bimw = t_bimc + 1.4
    if follow:
        t_hand = t_bimw + 0.75                      # hand-off from eye level to the orbit view
        speed = 1.0
        tin = max(0.0, FOLLOW_LOCK - (t_point - t_walk) * speed)
        tin = min(tin, follow_n / FPS - (t_hand - t_walk) * speed - 0.1)
        common = dict(tin=tin, speed=speed, z0=1.0, z1=1.0, c0=(0.5, 0.5), c1=(0.5, 0.5), grade="none")
        S.append(dict(t0=t_walk, t1=t_hand,
                      layers=[dict(src=follow["video"], **common), dict(src=follow["cloud"], **common),
                              dict(src=follow["bim"], **common)],
                      wipes=[(t_point - 0.30, t_point + 0.60, "wipe"), (t_bimw - 0.45, t_bimw + 0.10, "fade")],
                      tag="follow",
                      label=[("PHONE VIDEO", t_walk), ("POINT CLOUD · LINGBOT-MAP", t_point),
                             ("BIM MODEL", t_bimw - 0.2)]))
        S.append(dict(t0=t_hand, t1=t_agent, layers=[dict(src=stage_clip("03_bim.mp4"), tin=0.9, speed=1.0,
                                                           z0=1.55, z1=1.62, c0=(0.47, 0.5), c1=(0.47, 0.5),
                                                           grade="none")],
                      tag="bim", label=[("BIM MODEL", t_hand)]))
    else:
        site = ROOT / "data" / "site.mp4"
        vid = site if site.exists() else DEMO / "01_video.mp4"
        S.append(dict(t0=t_walk, t1=t_point - 0.05, layers=[dict(src=vid, tin=20.0 if vid == site else 0.0, z0=1.0,
                                                                  z1=1.06, c0=(0.5, 0.5), c1=(0.5, 0.5),
                                                                  grade="none")],
                      tag="video", label=[("PHONE VIDEO", t_walk)]))
        S.append(dict(t0=t_point - 0.05, t1=t_bimc - 0.05,
                      layers=[dict(src=stage_clip("02_point_cloud.mp4"), tin=4.0, speed=1.6, z0=1.55, z1=1.75,
                                   c0=(0.6, 0.47), c1=(0.6, 0.5), grade="none")],
                      tag="cloud", label=[("POINT CLOUD · LINGBOT-MAP", t_point)]))
        S.append(dict(t0=t_bimc - 0.05, t1=t_agent, layers=[dict(src=stage_clip("03_bim.mp4"), tin=0.0, speed=1.0,
                                                                  z0=1.5, z1=1.6, c0=(0.58, 0.47),
                                                                  c1=(0.58, 0.47), grade="none")],
                      tag="bim", label=[("BIM MODEL", t_bimw)]))
    # ---- design section -------------------------------------------------------------------
    t_cnc = word_time(timeline, "cnc", "c", default="start") - 0.35
    try:
        t_cnc = word_time(timeline, "cnc", "cnc") - 0.35
    except KeyError:
        try:
            t_cnc = word_time(timeline, "cnc", "new") - 0.1
        except KeyError:
            pass
    t_safety = word_time(timeline, "cnc", "checks", default=None) - 0.15
    t_phys = word_time(timeline, "cnc", "physics") - 0.45
    t_risky = chunk(timeline, "risky")["start"] - 0.1
    t_intel = chunk(timeline, "intel")["start"] - 0.1
    t_ifc = chunk(timeline, "ifc")["start"] - 0.1

    UI = preset["ui"]

    def dlay(cue, ui, dz=0.08, speed=1.0, grade="none"):
        cx, cy, z = UI[ui]
        return dict(src=design_src, tin=cues[cue], speed=speed, z0=z, z1=z + dz, c0=(cx, cy), c1=(cx, cy),
                    grade=grade)

    S.append(dict(t0=t_agent, t1=t_cnc, layers=[dlay("overview", "scene", 0.06)], tag="agent", agent_tag=True))
    S.append(dict(t0=t_cnc, t1=t_safety, layers=[dlay("cnc", "cnc", 0.10)], tag="cnc", agent_tag=True))
    S.append(dict(t0=t_safety, t1=t_phys, layers=[dlay("safety", "safety", 0.08)], tag="safety", agent_tag=True))
    S.append(dict(t0=t_phys, t1=t_risky, layers=[dlay("physics", "physics", 0.06)], tag="physics", agent_tag=True))
    S.append(dict(t0=t_risky, t1=t_intel, layers=[dlay("risky", "risky", 0.10, speed=0.7)], tag="risky",
                  agent_tag=True))
    S.append(dict(t0=t_intel, t1=t_ifc, layers=[dlay("bg", "scene", 0.10, speed=0.6, grade="dimblur")],
                  tag="ladder"))
    S.append(dict(t0=t_ifc, t1=END_CARD, layers=[dlay("nextsteps", "nextsteps", 0.06)], tag="ifc",
                  agent_tag=True))
    S.append(dict(t0=END_CARD, t1=DUR, layers=[dlay("end", "end", 0.10, speed=0.3, grade="dimblur")], tag="end"))
    return S, cuts


# v2: hover beats in rec/v2_stages/05_understand.mp4  (clip second, duration, cx, cy, zoom) - punched in on the
# highlighted machine + its Gemini label, exposure lifted ("lift" grade)
UND_BEATS = [(4.1, 2.2, 0.69, 0.66, 2.2), (6.95, 1.5, 0.72, 0.38, 2.2), (1.55, 1.0, 0.667, 0.78, 2.1)]
# (tin, weight, cx, cy, zoom)  4.1 "Fadal VMC 5020A vertical machining centre" | 7.1 "Fadal VMC 4020A ..." | 1.6 "Fadal CNC vertical machining center"
UND_TINT = (13.0, 1.25, 0.53, 0.53)       # "segmented into 49 objects" beat (tin, zoom, cx, cy)
PHONE_TIN = 3.0                           # follow_video second at "Walk" (band saw -> Fadal VMC 4020A)
PC_AT_WIPE = 9.5                          # 02_point_cloud second when the wipe starts (cloud assembling)
PC_UI = (0.56, 0.55, 1.30)                # punch-in on the cloud (cx, cy, zoom)


def find_follow_v2(extra=None):
    names = dict(video="follow_video.mp4", cloud="follow_cloud.mp4", tint="follow_cloud_tint.mp4",
                 labels="follow_cloud_labeled.mp4")
    for d in ([Path(extra)] if extra else []) + FOLLOW_DIRS:
        ps = {k: d / n for k, n in names.items()}
        if all(p.exists() for p in ps.values()):
            counts = [nframes(p) for p in ps.values()]
            if min(counts) > 60 and len(set(counts)) == 1:
                return ps, counts[0]
            log(f"follow clips in {d} incomplete/unequal: {counts}")
    return None, 0


def build_shots_v2(timeline, follow, follow_n, design_src, cues):
    """v2 recut: hype + title identical to v1; product section shows the segmented point cloud (no box models)."""
    preset = DESIGN_PRESETS["v2"]
    S = []
    t = 0.0
    cuts = []
    for key, dur, tin, z0, z1, cx, cy, grade in HYPE:
        cuts.append(t)
        S.append(dict(t0=t, t1=t + dur, layers=[dict(src=hype_source(key), tin=tin, z0=z0, z1=z1, c0=(cx, cy),
                                                     c1=(cx, cy), grade=grade)], tag=key))
        t += dur
    cuts.append(HYPE_END)
    t_walk = chunk(timeline, "walk")["start"] - 0.08
    # title background = v1's clip, so 0:00-0:10 is the same picture as v1
    S.append(dict(t0=HYPE_END, t1=t_walk, layers=[dict(src=FINAL / "rec" / "stages" / "02_point_cloud.mp4", tin=TITLE_TIN,
                                                       z0=1.55, z1=1.7, c0=(0.62, 0.52), c1=(0.62, 0.52),
                                                       grade="dim")], tag="title"))
    # ---- phone video -> dense cloud (cloud stage) -> segments (tint) -> named machines (hover beats) ----
    # NOTE: the follow-mode cloud layers do not register to the phone video on the current scan (dense cloud vs
    # camera track), so the matched-camera wipe is not used; the phone video wipes into the cloud-stage assembly.
    t_point = word_time(timeline, "cloud", "point", default="start")
    t_dense = word_time(timeline, "cloud", "dense", default=None) or (t_point - 0.4)
    sc = chunk(timeline, "scene")
    t_then = sc["start"] - 0.15
    t_named = word_time(timeline, "scene", "named", default=sc["start"] + 2.6) - 0.25
    t_ask = chunk(timeline, "cnc")["start"] - 0.10
    t_hover = t_named
    pc = stage_clip("02_point_cloud.mp4")
    tw0 = t_dense - 0.25
    S.append(dict(t0=t_walk, t1=t_then,
                  layers=[dict(src=follow["video"], tin=PHONE_TIN, speed=1.0, z0=1.0, z1=1.04, c0=(0.5, 0.5),
                               c1=(0.5, 0.5), grade="none"),
                          dict(src=pc, tin=max(0.0, PC_AT_WIPE - (tw0 - t_walk)), speed=1.0, z0=PC_UI[2],
                               z1=PC_UI[2] + 0.12, c0=PC_UI[:2], c1=PC_UI[:2], grade="lift")],
                  wipes=[(tw0, t_dense + 0.65, "wipe")], tag="video_cloud",
                  label=[("PHONE VIDEO", t_walk), ("DENSE POINT CLOUD \u00b7 2.7 M POINTS \u00b7 LINGBOT-MAP", t_dense)]))
    und = stage_clip("05_understand.mp4")
    tin_, z_, cx_, cy_ = UND_TINT
    S.append(dict(t0=t_then, t1=t_hover, layers=[dict(src=und, tin=tin_, speed=1.0, z0=z_, z1=z_ + 0.10,
                                                     c0=(cx_, cy_), c1=(cx_, cy_), grade="lift")],
                  tag="segments", label=[("SEGMENTED BY GEMINI \u00b7 49 OBJECTS", t_then + 0.3)]))
    # ---- hover beats from the understand stage, punched in ----
    beats = list(UND_BEATS)
    n_b = len(beats)
    wsum = sum(b[1] for b in beats)
    tt = t_hover
    for i, (b_tin, wgt, cx, cy, z) in enumerate(beats):
        t1 = t_ask if i == n_b - 1 else tt + (t_ask - t_hover) * wgt / wsum
        S.append(dict(t0=tt, t1=t1, layers=[dict(src=und, tin=b_tin, speed=1.0, z0=z, z1=z + 0.08, c0=(cx, cy),
                                                 c1=(cx, cy), grade="lift")], tag=f"hover{i}",
                      label=[("NAMED + EXPLAINED BY GEMINI 4 CARBON", tt)] if i == 0 else []))
        tt = t1
    # ---- design section (Gemini 4 Carbon session on the segmented model) ----
    t_place = word_time(timeline, "cnc", "placing", default=None) - 0.25
    t_guard = word_time(timeline, "cnc", "guarding", default=None) - 0.2
    t_phys = word_time(timeline, "cnc", "physics") - 0.45
    t_risky = chunk(timeline, "risky")["start"] - 0.1
    t_intel = chunk(timeline, "intel")["start"] - 0.1
    UI = preset["ui"]

    def dlay(cue, ui, dz=0.08, speed=1.0, grade="none"):
        cx, cy, z = UI[ui]
        return dict(src=design_src, tin=cues[cue], speed=speed, z0=z, z1=z + dz, c0=(cx, cy), c1=(cx, cy),
                    grade=grade)

    S.append(dict(t0=t_ask, t1=t_place, layers=[dlay("overview", "scene", 0.06)], tag="agent", agent_tag=True))
    S.append(dict(t0=t_place, t1=t_guard, layers=[dlay("cnc", "cnc", 0.10, speed=1.6)], tag="cnc", agent_tag=True))
    S.append(dict(t0=t_guard, t1=t_phys, layers=[dlay("safety", "safety", 0.08)], tag="safety", agent_tag=True))
    S.append(dict(t0=t_phys, t1=t_risky, layers=[dlay("physics", "physics", 0.06)], tag="physics", agent_tag=True))
    S.append(dict(t0=t_risky, t1=t_intel, layers=[dlay("risky", "risky", 0.10, speed=0.7)], tag="risky",
                  agent_tag=True))
    S.append(dict(t0=t_intel, t1=END_CARD, layers=[dlay("bg", "scene", 0.10, speed=0.6, grade="dimblur")],
                  tag="ladder"))
    S.append(dict(t0=END_CARD, t1=DUR, layers=[dlay("end", "end", 0.10, speed=0.3, grade="dimblur")], tag="end"))
    return S, cuts


TITLE_TIN = 10.0
LIFT_LUT = __import__("numpy").clip(255 * (__import__("numpy").arange(256) / 255) ** 0.80 * 1.10, 0, 255).astype("uint8")


def render(timeline, follow, follow_n, design_src, cues, audio_path, out_path):
    import cv2
    import numpy as np
    if CUT == "v2":
        shots, cuts = build_shots_v2(timeline, follow, follow_n, design_src, cues)
    else:
        shots, cuts = build_shots(timeline, follow, follow_n, design_src, cues)
    for a, b in zip(shots, shots[1:]):
        assert abs(a["t1"] - b["t0"]) < 1e-6, (a["tag"], b["tag"])
    gr = Grader()
    # pre-render overlays
    hype_txt = [(text_rgba(s, 124, "Black", tracking=10), t0, t1) for s, t0, t1 in HYPE_TEXT]
    title = title_card_rgba("CAPTURE STUDIO", None, 150)
    endc = title_card_rgba("CAPTURE STUDIO", "From phone video to a buildable plan.", 132)
    agent_pill = pill_rgba("AGENT · GEMINI 4 CARBON", 28, dot=ORANGE + (255,))
    t_agent0 = next(x["t0"] for x in shots if x["tag"] == "agent")
    ladder = [model_ladder_rgba(k) for k in range(1, 5)]
    ifc_pill = pill_rgba("IFC MODEL  +  NEXT STEPS", 30, dot=(52, 211, 153, 255), bg=(16, 19, 26, 230))
    label_cache = {}
    t_words = {}
    ic = chunk(timeline, "intel")
    for nm in ("carbon", "neon", "argon", "hydrogen"):
        try:
            t_words[nm] = word_time(timeline, "intel", nm)
        except KeyError:
            t_words[nm] = ic["start"] + 1.5
    enc = subprocess.Popen(
        ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-i", str(audio_path), "-map", "0:v", "-map", "1:a",
         "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-profile:v", "high", "-pix_fmt", "yuv420p",
         "-x264-params", "keyint=60:min-keyint=30", "-maxrate", "20M", "-bufsize", "30M",
         "-c:a", "aac", "-b:a", "320k", "-ar", str(SR), "-ac", "2", "-movflags", "+faststart",
         "-t", f"{DUR:.3f}", str(out_path)], stdin=subprocess.PIPE)
    fi = 0
    for si, sh in enumerate(shots):
        f0, f1 = int(round(sh["t0"] * FPS)), int(round(sh["t1"] * FPS))
        n = f1 - f0
        if n <= 0:
            continue
        readers = [Reader(L["src"], L["tin"], n, L.get("speed", 1.0)) for L in sh["layers"]]
        for k in range(n):
            gt = (f0 + k) / FPS
            u = ease(k / max(1, n - 1))
            imgs = []
            for L, rd in zip(sh["layers"], readers):
                fr = rd.read()
                z = L["z0"] + (L["z1"] - L["z0"]) * u
                cx = L["c0"][0] + (L["c1"][0] - L["c0"][0]) * u
                cy = L["c0"][1] + (L["c1"][1] - L["c0"][1]) * u
                im = warp(fr, z, cx, cy)
                g = L.get("grade", "none")
                if g == "bw":
                    im = gr.bw(im, f0 + k)
                elif g == "col":
                    im = gr.col(im, f0 + k)
                elif g == "dim":
                    im = (im.astype(np.float32) * 0.28).astype(np.uint8)
                elif g == "lift":      # v2 understand beats: lift the dark top view a little
                    im = cv2.LUT(im, LIFT_LUT)
                elif g == "dimblur":
                    im = (cv2.GaussianBlur(im, (0, 0), 9).astype(np.float32) * 0.35).astype(np.uint8)
                imgs.append(im)
            frame = imgs[0].copy()
            for wi, (tw0, tw1, kind) in enumerate(sh.get("wipes", [])):
                p = (gt - tw0) / (tw1 - tw0)
                if p <= 0:
                    continue
                nxt = imgs[wi + 1]
                if p >= 1:
                    frame = nxt.copy()
                    continue
                pe = ease(p)
                if kind == "fade":
                    frame = (frame.astype(np.float32) * (1 - pe) + nxt.astype(np.float32) * pe).astype(np.uint8)
                    continue
                edge = -200 + pe * (W + 400)
                xs = np.arange(W, dtype=np.float32)[None, :] + (np.arange(H, dtype=np.float32)[:, None] - H / 2) * 0.18
                m = np.clip((edge - xs) / 120 + 0.5, 0, 1)[..., None]
                frame = (frame.astype(np.float32) * (1 - m) + nxt.astype(np.float32) * m)
                line = np.exp(-((xs - edge) / 6) ** 2)[..., None] * 160
                frame = np.clip(frame + line, 0, 255).astype(np.uint8)
            # --- overlays ---
            for img, t0, t1 in hype_txt:
                if t0 <= gt < t1:
                    a = min(1, (gt - t0) / 0.12, (t1 - gt) / 0.10)
                    overlay(frame, img, (W - img.shape[1]) // 2, int(H * 0.70) - img.shape[0] // 2, a)
            if sh["tag"] == "title":
                a = min(1, (gt - HYPE_END) / 0.05, (sh["t1"] - gt) / 0.25)
                overlay(frame, title, 0, 0, a)
            if sh["tag"] == "end":
                overlay(frame, endc, 0, 0, min(1, (gt - END_CARD) / 0.5))
            if sh.get("agent_tag"):
                a = min(1, (gt - t_agent0) / 0.25) if gt - t_agent0 < 0.25 else 1.0
                overlay(frame, agent_pill, 64, H - 64 - agent_pill.shape[0], a)
            if sh["tag"] == "ladder":
                stage = 1 + sum(gt >= t_words[nm] - 0.1 for nm in ("neon", "argon", "hydrogen"))
                a = min(1, (gt - sh["t0"]) / 0.3, (sh["t1"] - gt) / 0.25)
                overlay(frame, ladder[stage - 1], 0, 0, a)
            if sh["tag"] == "ifc":
                a = min(1, (gt - sh["t0"] - 0.4) / 0.3, (sh["t1"] - gt) / 0.2)
                overlay(frame, ifc_pill, 64, H - 64 - agent_pill.shape[0] - 18 - ifc_pill.shape[0], max(0, a))
            for txt, tl in sh.get("label", []):
                nxt_t = min([t for _, t in sh.get("label", []) if t > tl] + [sh["t1"]])
                if tl <= gt < nxt_t:
                    if txt not in label_cache:
                        label_cache[txt] = pill_rgba(txt, 28)
                    lab = label_cache[txt]
                    a = min(1, (gt - tl) / 0.2, (nxt_t - gt) / 0.15)
                    overlay(frame, lab, 64, H - 64 - lab.shape[0], a)
            # flashes: white flash into the title, small exposure pops on hype cuts
            fl = 0.0
            if HYPE_END <= gt < HYPE_END + 0.30:
                fl = max(fl, 1 - (gt - HYPE_END) / 0.30)
            for tc in cuts[1:-1]:
                if tc <= gt < tc + 2 / FPS:
                    fl = max(fl, 0.18)
            if END_CARD - 0.05 <= gt < END_CARD + 0.2:
                fl = max(fl, 0.25 * (1 - abs(gt - END_CARD) / 0.2))
            if fl > 0:
                frame = np.clip(frame.astype(np.float32) + 255 * fl, 0, 255).astype(np.uint8)
            if gt < 0.25:                                            # quick fade from black at the top
                frame = (frame.astype(np.float32) * (gt / 0.25)).astype(np.uint8)
            enc.stdin.write(np.ascontiguousarray(frame).tobytes())
            fi += 1
        for rd in readers:
            rd.close()
        log(f"shot {si:2d} {sh['tag']:<14s} {sh['t0']:6.2f}-{sh['t1']:6.2f} ({n} f)")
    enc.stdin.close()
    enc.wait()
    assert fi == NF, f"rendered {fi} frames, expected {NF}"
    return shots, cuts


# ======================================================================================
# Mix / loudness / captions
# ======================================================================================
def loudnorm(inp, out, target=-14.0, tp=-1.5):
    r = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-i", str(inp), "-af",
                        f"loudnorm=I={target}:TP={tp}:LRA=11:print_format=json", "-f", "null", "-"],
                       capture_output=True, text=True)
    m = json.loads(r.stderr[r.stderr.rindex("{"):])
    af = (f"loudnorm=I={target}:TP={tp}:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
          f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:"
          f"linear=true:print_format=json")
    run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", inp, "-af", af, "-ar", str(SR), "-c:a",
         "pcm_s24le", out])


def ebur128(path):
    r = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-i", str(path), "-af", "ebur128=peak=true",
                        "-f", "null", "-"], capture_output=True, text=True)
    s = r.stderr[r.stderr.rfind("Summary:"):]
    i = float(re.search(r"I:\s+(-?[\d.]+) LUFS", s).group(1))
    tp = float(re.search(r"Peak:\s+(-?[\d.]+) dBFS", s).group(1))
    lra = float(re.search(r"LRA:\s+(-?[\d.]+) LU", s).group(1))
    return dict(I=i, TP=tp, LRA=lra)


def srt_time(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def write_srt(timeline):
    items = []
    for c in timeline:
        words = c["words"]
        text = c["text"]
        if len(text) <= 84 or not words:
            items.append((c["start"], c["end"], text))
            continue
        # split long sentences at commas/colons into caption lines of <= ~84 chars
        parts = re.split(r"(?<=[,:])\s+", text)
        lines, cur = [], ""
        for p in parts:
            if cur and len(cur) + 1 + len(p) > 84:
                lines.append(cur)
                cur = p
            else:
                cur = (cur + " " + p).strip()
        lines.append(cur)
        wi = 0
        for ln in lines:
            nw = len(ln.split())
            seg = words[wi:wi + nw] or words[-1:]
            items.append((seg[0]["s"], seg[-1]["e"] + 0.15, ln))
            wi += nw
    with open(FINAL / f"captions{TTS_SUFFIX}.srt", "w") as f:
        for i, (a, b, t) in enumerate(items, 1):
            f.write(f"{i}\n{srt_time(a)} --> {srt_time(min(b, DUR))}\n{t}\n\n")


def build(voice, design_src, cues, follow_dir=None):
    import numpy as np
    import soundfile as sf
    CACHE.mkdir(parents=True, exist_ok=True)
    timeline, vo = build_vo(voice)
    if CUT == "v2":
        follow, follow_n = find_follow_v2(follow_dir)
        assert follow, "v2 needs rec/v2_follow/follow_{video,cloud,cloud_tint,cloud_labeled}.mp4"
        shots, cuts = build_shots_v2(timeline, follow, follow_n, design_src, cues)
    else:
        follow, follow_n = find_follow(follow_dir)
        shots, cuts = build_shots(timeline, follow, follow_n, design_src, cues)
    log("follow clips:", "YES " + str(follow["video"].parent) if follow else "no (stage clips)")
    mus = synth_music(cuts, vo, timeline)
    sf.write(FINAL / f"music{TTS_SUFFIX}.wav", mus, SR, subtype="PCM_24")
    vo_st = np.stack([vo, vo], 1)
    mix = vo_st * 1.0 + mus * 10 ** (-9 / 20)
    sf.write(CACHE / "mix_raw.wav", mix.astype(np.float32), SR, subtype="FLOAT")
    loudnorm(CACHE / "mix_raw.wav", CACHE / "mix.wav")
    sf.write(CACHE / "nomusic_raw.wav", vo_st.astype(np.float32), SR, subtype="FLOAT")
    loudnorm(CACHE / "nomusic_raw.wav", CACHE / "nomusic.wav")
    render(timeline, follow, follow_n, design_src, cues, CACHE / "mix.wav", OUT)
    run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", OUT, "-i", CACHE / "nomusic.wav", "-map", "0:v",
         "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "320k", "-ar", str(SR), "-movflags", "+faststart",
         "-t", f"{DUR:.3f}", OUT_NOMUSIC])
    write_srt(timeline)
    (CACHE / "shots.json").write_text(json.dumps(
        [dict(tag=s["tag"], t0=round(s["t0"], 3), t1=round(s["t1"], 3),
              srcs=[f"{Path(L['src']).name}@{L['tin']:.2f}s x{L.get('speed', 1.0):.2f}" for L in s["layers"]])
         for s in shots], indent=1))
    log("built", OUT)


def whisper_final():
    """faster-whisper word timestamps on the final mix; prints when each cue word is heard vs. the shot cut."""
    if not _in_venv(VENV_TTS):
        rc = _reexec(VENV_TTS, ["whisperfinal"])
        if rc:
            sys.exit(rc)
        return
    from faster_whisper import WhisperModel
    model = WhisperModel("small", device="cuda", compute_type="float16")
    segs, _ = model.transcribe(str(OUT), language="en", word_timestamps=True, beam_size=5)
    words = [dict(w=w.word.strip(), s=round(w.start, 2), e=round(w.end, 2)) for sg in segs for w in sg.words]
    hyp = " ".join(w["w"] for w in words)
    ref = " ".join(c["text"] for c in VO)
    shots = json.loads((CACHE / "shots.json").read_text())
    rep = dict(wer=round(_wer(ref, hyp), 4), transcript=hyp, words=words)
    (CACHE / "whisper_final.json").write_text(json.dumps(rep, indent=1))
    print(f"final-mix WER vs script: {rep['wer']:.3f}")
    print(hyp)
    cues = ["walk", "point", "bim", "gemini", "cnc", "checks", "exit", "physics", "risky", "intelligence",
            "carbon", "neon", "argon", "hydrogen", "ifc", "buildable"]
    if CUT == "v2":
        cues = ["walk", "dense", "point", "gemini", "segmented", "named", "understood", "ask", "placing",
                "guarding", "exit", "physics", "risky", "intelligence", "carbon", "neon", "argon", "hydrogen",
                "buildable"]
    for cw in cues:
        hit = next((w for w in words if re.sub(r"[^a-z0-9]", "", w["w"].lower()).startswith(cw)), None)
        if hit:
            sh = next((x for x in shots if x["t0"] <= hit["s"] < x["t1"]), None)
            print(f"  {cw:13s} heard {hit['s']:6.2f}s  -> on shot '{sh['tag'] if sh else '?'}' "
                  f"({sh['t0'] if sh else 0:.2f}-{sh['t1'] if sh else 0:.2f})")


def check():
    info = ffprobe_json(OUT)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    a = next(s for s in info["streams"] if s["codec_type"] == "audio")
    rep = dict(duration=float(info["format"]["duration"]), bitrate_mbps=round(int(info["format"]["bit_rate"]) / 1e6, 2),
               video=f"{v['codec_name']} {v.get('profile')} {v['width']}x{v['height']} {v['r_frame_rate']} "
                     f"{v['pix_fmt']} frames={v.get('nb_frames')}",
               audio=f"{a['codec_name']} {a['sample_rate']} Hz {a['channels']} ch",
               loudness=ebur128(OUT), loudness_nomusic=ebur128(OUT_NOMUSIC))
    sheet_dir = CACHE / "check_frames"
    shutil.rmtree(sheet_dir, ignore_errors=True)
    sheet_dir.mkdir(parents=True)
    run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", OUT, "-vf", "fps=0.5,scale=640:-2", sheet_dir / "f_%02d.jpg"])
    (CACHE / "check.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def main():
    global DESIGN_PRESET
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fetch", "tts", "align", "build", "check", "whisperfinal", "all"])
    ap.add_argument("--engine", default="kokoro")
    ap.add_argument("--voice", default=None, help="tts: engine voice name; build/all: ENGINE:VOICE")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--design-preset", default=DESIGN_PRESET, choices=sorted(DESIGN_PRESETS))
    ap.add_argument("--design-src", default=None, help="override the preset's design clip")
    ap.add_argument("--design-cues", default=None, help="JSON dict overriding the preset cue points")
    ap.add_argument("--follow-dir", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cut", default="v1", choices=["v1", "v2"])
    a = ap.parse_args()
    use_cut(a.cut)
    if a.cut == "v2" and a.design_preset == "gemini":
        a.design_preset = "v2"
    DESIGN_PRESET = a.design_preset
    preset = DESIGN_PRESETS[a.design_preset]
    cues = dict(preset["cues"])
    a.design_src = a.design_src or str(preset["src"])
    if a.design_cues:
        cues.update(json.loads(Path(a.design_cues).read_text() if Path(a.design_cues).exists() else a.design_cues))
    if a.stage == "fetch":
        fetch(a.force)
    elif a.stage == "tts":
        tts(a.engine, a.voice or "am_michael", a.speed)
    elif a.stage == "align":
        align(a.name)
    elif a.stage == "build":
        build(a.voice or DEFAULT_VOICE, Path(a.design_src), cues, a.follow_dir)
    elif a.stage == "check":
        check()
        whisper_final()
    elif a.stage == "whisperfinal":
        whisper_final()
    elif a.stage == "all":
        voice = a.voice or DEFAULT_VOICE
        fetch(a.force)
        eng, vn = voice.split(":", 1)
        if not (TTSDIR / f"{eng}_{vn}" / "align.json").exists():
            tts(eng, vn)
            align(f"{eng}_{vn}")
        build(voice, Path(a.design_src), cues, a.follow_dir)
        check()
        whisper_final()


if __name__ == "__main__":
    main()
