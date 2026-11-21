# Capture Studio

Capture Studio turns a phone RGB video of a construction site into an editable BIM model that
an AI agent designs on through an MCP server. Every design edit is checked against a building-code
profile, site-safety rules and a MuJoCo rigid-body stability test; decisions that need a human
are put to the user in a minimal three.js viewer, and the result is exported as IFC4 + GLB + MJCF.

Pipeline:

1. **Scan** — phone video (`data/site.mp4`).
2. **Point cloud** — LingBot-Map streaming reconstruction: coloured points + camera path, gravity-aligned and scaled to metres.
3. **As-built BIM** — slab, walls, doors and windows fitted to the cloud (`provenance: scanned`, with P50/P95 fit residuals in mm).
4. **Design** — an agent adds spaces, walls, openings, fixtures (`provenance: proposed`), validates, gets sign-offs, commits revisions and exports.

## Quick start

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt                        # includes mcp[cli]<2, mujoco, open3d, ifcopenshell
uv pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128

git clone https://github.com/robbyant/lingbot-map third_party/lingbot-map
uv pip install -e third_party/lingbot-map                 # optional: flashinfer-python, else use --use_sdpa
hf download robbyant/lingbot-map lingbot-map.pt --local-dir models/lingbot-map

python -m capture_studio.server                           # viewer http://127.0.0.1:8642/  MCP http://127.0.0.1:8642/mcp
python -m capture_studio.server --stdio                   # same tools over stdio (state shared via data/project/state.json)
```

Reconstruction needs a CUDA GPU (about 1–2 min, cached in `data/scan/raw_predictions.npz`); the rest
runs on CPU. `python scripts/mcp_call.py --list` lists the tools. Tests: `pytest tests`.

## Driving it with an agent

The server's MCP `instructions` describe the workflow; any MCP client picks them up.

**Claude Code** — `.mcp.json` in the repo root (or `claude mcp add --transport http capture-studio http://127.0.0.1:8642/mcp`):

```json
{ "mcpServers": { "capture-studio": { "type": "http", "url": "http://127.0.0.1:8642/mcp" } } }
```

**Gemini (google-genai SDK)** — `scripts/agent_gemini.py` lists the MCP tools, converts them to Gemini
function declarations and runs an explicit tool-call loop, printing each call:

```bash
uv pip install google-genai
export GEMINI_API_KEY=...                  # or GOOGLE_API_KEY; GEMINI_MODEL overrides the default model
python scripts/agent_gemini.py --dry-run   # connect + convert all tools, no model call
python scripts/agent_gemini.py             # follow the server workflow end to end
python scripts/agent_gemini.py "add a 10 m2 office with its own door" --watch   # then act on viewer messages
```

`CAPTURE_STUDIO_MCP` overrides the server URL; `--max-steps` caps model calls per turn.

**Gemini CLI** — `~/.gemini/settings.json` (or `.gemini/settings.json` in the project):

```json
{ "mcpServers": { "capture-studio": { "httpUrl": "http://127.0.0.1:8642/mcp" } } }
```

## Tools (29)

| Purpose | Tools |
|---|---|
| Pipeline | `pipeline_status`, `reconstruct_scan`, `fit_bim`, `set_scale` |
| Read the model | `get_model`, `get_element`, `get_quantities` |
| Design | `add_space`, `add_wall`, `add_opening`, `add_fixture`, `add_column`, `update_element`, `remove_element`, `undo` |
| Checks | `validate`, `set_code_profile`, `get_code_rules` |
| Human in the loop | `get_user_messages`, `add_requirement`, `resolve_requirement`, `list_requirements`, `ask_user`, `get_user_answer`, `record_approval`, `say` |
| Deliver | `set_next_steps`, `commit_revision`, `export_package` |

Design tools return the new revision and the issues they introduced or cleared (physics re-runs only
on `validate`). Units: metres, Z up, origin at the shell's min corner.

## What the checks do

`validate` returns issues with a severity: **fail** (change the design), **hold** (needs the user's
sign-off) or **warn** (site action, carried into next steps).

- **Code** (`capture_studio/codes.py`) — profile `NBC_INDIA_2016` (default) or `IRC_2021`, each a demo
  subset of the cited code: minimum room area / width / height per use, light-and-ventilation ratio,
  door clear width and height, egress clear width and travel distance (grid path search), guard height,
  lintel requirement over openings wider than 1.2 m.
- **Safety** — cutting a new opening into a scanned wall or demolishing one raises a structural **hold**;
  low-sill openings raise a fall-protection **warn**; missing lintels **fail**. A hold clears only through
  `ask_user` → an affirmative human click in the viewer → `record_approval`; the answer text is stored.
- **Physics** (`capture_studio/physics.py`, MuJoCo) — the scanned shell is static; each proposed element is
  a free rigid body. Walls abutting other walls are welded, `head_restraint` ties to the soffit, `anchored`
  fixes to the slab. Tests: initial clashes; a service tilt test (gravity gains 0.15 g horizontally in
  4 directions; > 20 mm slide or > 2° tilt fails); an erection test with all ties removed at 0.05 g
  (what needs temporary propping); and, on export, a drop test (boxes dropped on the slab must come to rest).
- **Requirements** — client wishes recorded with `add_requirement`, auto-checked (space use + area,
  or a named proposed element) or closed manually with evidence.

`commit_revision` refuses while any fail or hold is open.

## Export package

`export_package` writes `data/export/`:

- `capture_studio_R<n>.ifc` — IFC4, property set `CaptureStudio_Element` (id, provenance, fit P50/P95, material)
- `capture_studio_R<n>.glb` — same geometry for viewers, proposed work in orange
- `capture_studio_R<n>.mjcf.xml` — MuJoCo scene, ties as weld constraints
- `model.json` — parametric source model
- `validation.json` — checks, open issues, scan-fit residuals, physics bodies/ties, drop test, approvals, requirements, next steps, quantities
- `README.md`, `SHA256SUMS`

IFC, GLB and MJCF are read back after writing and their element counts compared with the model.

## Repo layout

```
capture_studio/  server.py (MCP tools + viewer routes)  reconstruct.py (LingBot-Map)  fit.py (cloud prep, shell fitting)
                 bim.py (element model, quantities)  codes.py  physics.py  export.py  store.py (project state)
web/             index.html, app.js, vendor/three.js viewer
scripts/         mcp_call.py (CLI MCP client)  agent_gemini.py (Gemini agent loop)
tests/           pytest
data/            site.mp4, scan/, project/state.json, export/
third_party/     lingbot-map (git clone)     models/  lingbot-map checkpoint
```

## Limitations

- **Scale.** Video-only reconstruction has no absolute scale. The default assumes the phone was held
  1.5 m above the floor; treat all dimensions as approximate until one is tape-measured and set with
  `set_scale` (true floor-to-ceiling height), which refits the as-built model.
- **Code rules** are a demo subset of NBC India 2016 / IRC 2021, not the full codes, and not a compliance
  certificate. Verify values against the adopted edition before any real submission.
- **Physics** is a rigid-body stability screen (sliding, toppling, missing ties), not structural design:
  no member stresses, foundations, seismic or wind design.
- **As-built fitting** produces one storey of slab, walls, doors and windows; nothing else in the cloud is modelled.
- **Demo video.** `data/site.mp4` is a third-party YouTube site walkthrough used for demonstration only;
  it is not ours and is not licensed for redistribution.
