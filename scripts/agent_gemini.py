"""Let a Google Gemini model drive Capture Studio through its MCP server.

    export GEMINI_API_KEY=...                              # or GOOGLE_API_KEY
    python scripts/agent_gemini.py                         # follow the server's own workflow
    python scripts/agent_gemini.py "add a 10 m2 office in the north-east corner"
    python scripts/agent_gemini.py --watch                 # then keep acting on viewer messages
    python scripts/agent_gemini.py --dry-run               # no model call: connect, list + convert tools
    python scripts/agent_gemini.py --route auto            # tiered: strongest model only where it matters
    python scripts/agent_gemini.py --route neon            # one tier for everything (default: carbon)
    python scripts/agent_gemini.py --model gemini-3.8-flash

Env (also read from the repo's .env if not already set):
     GEMINI_API_KEY / GOOGLE_API_KEY,
     GEMINI_MODEL (single model for everything, same as --model),
     GEMINI_MODEL_CARBON / _NEON / _ARGON / _HYDROGEN (tier -> model id, see TIERS),
     GEMINI_THINKING (minimal|low|medium|high; model default if unset),
     GEMINI_KEEP_THOUGHTS (recent model turns whose thought signature is replayed; default 2, or 'all'),
     CAPTURE_STUDIO_MCP (default http://127.0.0.1:8642/mcp).

Explicit loop, no SDK magic: list the MCP tools, convert each inputSchema to a Gemini
FunctionDeclaration, call the model, execute every function call on the MCP server,
send the results back, repeat until the model answers in text or --max-steps is hit.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time

import httpx

from google import genai
from google.genai import errors, types
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# ---------------------------------------------------------------- model tiers
# The user's four Gemini tiers, strongest first. IDs verified against models.list() for this key
# (display names "Gemini 4 Carbon" etc.). Override any of them with GEMINI_MODEL_<TIER>.
TIER_ORDER = ("carbon", "neon", "argon", "hydrogen")
TIERS = {t: os.environ.get(f"GEMINI_MODEL_{t.upper()}", f"gemini-4-{t}") for t in TIER_ORDER}
DEFAULT_ROUTE = "carbon"  # quality first; --route auto for tiered orchestration

# --route auto: which tier handles which phase. Decided per model call from deterministic signals in
# the tool results so far (no extra LLM call). Phases, roughly in workflow order:
#   orient  - first call of a turn: read status / messages / requirements      -> lightest
#   plan    - context read (get_model seen) but nothing built yet this turn    -> strongest
#   fix     - the last validate reported 'fail' issues                         -> strongest
#   edit    - design under way, no open fail                                   -> middle
#   signoff - waiting on / handling 'hold' answers                             -> light-middle
#   wrapup  - last validate had no fail/hold and nothing changed since         -> lightest
PHASE_TIER = {"orient": "hydrogen", "plan": "carbon", "fix": "carbon", "edit": "neon",
              "signoff": "argon", "wrapup": "hydrogen"}
MCP_URL = os.environ.get("CAPTURE_STUDIO_MCP", "http://127.0.0.1:8642/mcp")

DEFAULT_GOAL = (
    "Follow the Capture Studio workflow. Start with pipeline_status, get_user_messages and "
    "list_requirements. If there is no as-built model yet, run reconstruct_scan / fit_bim. "
    "Record each client wish with add_requirement, design it, then validate. Fix every 'fail' "
    "and re-validate. For every 'hold' ask the user with ask_user, wait with get_user_answer "
    "and only call record_approval after an affirmative human answer. When validation is ok: "
    "set_next_steps, commit_revision, export_package, then summarise in two lines. If the design "
    "has not changed since the last committed revision, do not commit or export again."
)

RULES = """\
Operating rules for this session (you are an autonomous agent; nobody reads your prose until the end):
- Act through tool calls. Do not describe what you would do, and do not stop to ask the operator
  for permission to continue: keep calling tools until the workflow is finished or you are blocked
  on a human answer.
- Read the model once with get_model (it lists every element with coordinates) and get_code_rules
  once; do not fetch elements one by one unless you need a field get_model lacks. Issue independent
  calls together in one turn.
- Deliver what the client actually asked for. Relabelling existing space is fine only if it truly
  meets the wish; if they ask to split, partition, add or move something, build it (walls, doors,
  fixtures) and record a requirement for each wish. Avoiding a check by not building is not a fix.
- Derive geometry from get_model: new walls must end on existing walls (tied in) or be restrained,
  spaces must follow wall centrelines, openings must fit inside their host wall.
- After design edits run validate. Every 'fail' must be fixed by changing the design (follow the
  fix text in the issue), then validate again. Loop until there are no 'fail' items.
- ask_user is only for 'hold' sign-offs and genuine design choices - never test questions.
- 'hold' items need a human sign-off: ask_user (one clear question naming the issue, with options),
  then get_user_answer with wait_seconds=60. Only an affirmative human answer counts; then
  record_approval with that issue key and question id. If the answer is null, negative or
  ambiguous, never record approval and never pretend it was given: either change the design to
  remove the hold, or say() that you are waiting for the human and end your turn.
- Safety attributes (guarded, anchored, head_restraint, ...) are design commitments, not switches
  to silence a check: provide what justifies them. Model the physical measure when a tool can
  (e.g. a guard fence as tied-in walls with a gate opening around a hazardous cell), and name the
  rest (guard type, anchor spec) in set_next_steps.
- Keep new items clear of existing (scanned) equipment and of each other, with the profile's
  clearance; check coordinates against get_model before placing.
- 'warn' items go into set_next_steps as site actions.
- When validate is ok: set_next_steps, commit_revision, export_package, then reply with a
  two-line summary. Keep say() lines under 12 words.
"""

RESULT_CHAR_LIMIT = 60_000  # per tool result sent back to the model
# Gemini 3+ thought signatures replay the model's whole reasoning as prompt tokens on every later
# call (measured: a 5.7k-token think adds ~5.7k prompt tokens to each following step). Only the
# newest KEEP_THOUGHTS model turns keep their real signature; older function-call parts get
# Google's documented placeholder, which the API accepts. GEMINI_KEEP_THOUGHTS=all disables this.
SKIP_SIGNATURE = b"skip_thought_signature_validator"
MAX_NUDGES = 3       # times per turn the loop pushes a model that stops with work left
TRANSCRIPT_RESULT_CHARS = 4000  # per tool result in --transcript (args are written in full)
REPEAT_LIMIT = 4     # identical consecutive calls before the loop guard speaks up
MUTATING = {"add_wall", "add_opening", "add_fixture", "add_column", "add_space", "update_element",
            "remove_element", "undo", "set_code_profile", "record_approval", "set_scale", "fit_bim"}


# ---------------------------------------------------------------- schema conversion
#
# Gemini function parameters take an OpenAPI-3 subset. It rejects JSON-schema keywords
# such as additionalProperties, anyOf (in practice), $defs/$ref, title, default, and
# OBJECT types with no properties. MCP (pydantic) schemas use all of those:
#   height: float | None   ->  {"anyOf": [{"type":"number"}, {"type":"null"}], "default": null}
#   changes: dict          ->  {"type":"object", "additionalProperties": true}
# Free-form objects are therefore declared as JSON *strings* and decoded before the MCP call.

KEEP = ("description", "enum", "format", "minimum", "maximum", "minItems", "maxItems")
SNAKE = {"minItems": "min_items", "maxItems": "max_items"}


def _resolve(s: dict, defs: dict) -> dict:
    """Inline $ref and collapse anyOf/oneOf: drop null branches, mark nullable, keep the first."""
    s = dict(s)
    while "$ref" in s:
        ref = defs.get(s.pop("$ref").split("/")[-1], {})
        s = {**ref, **s}
    for key in ("anyOf", "oneOf"):
        if key in s:
            options = s.pop(key)
            branches = [b for b in options if b.get("type") != "null"]
            merged = _resolve(branches[0], defs) if branches else {"type": "string"}
            s = {**merged, **{k: v for k, v in s.items() if k not in merged}}
            if len(branches) < len(options):
                s["nullable"] = True
    if s.get("allOf"):
        merged = {}
        for b in s.pop("allOf"):
            merged.update(_resolve(b, defs))
        s = {**merged, **s}
    if isinstance(s.get("type"), list):  # {"type": ["number","null"]}
        types_ = [t for t in s["type"] if t != "null"]
        s["nullable"] = len(types_) < len(s["type"])
        s["type"] = types_[0] if types_ else "string"
    return s


def _is_free_object(s: dict) -> bool:
    return s.get("type") == "object" and not s.get("properties")


def to_gemini_schema(s: dict, defs: dict) -> dict:
    s = _resolve(s, defs)
    desc = s.get("description", "")
    if "default" in s and s["default"] is not None:
        desc = f"{desc} (default: {json.dumps(s['default'])})".strip()
    if _is_free_object(s):
        return {"type": "STRING", "description": f"{desc} JSON-encoded object, e.g. '{{\"key\": 1}}'.".strip()}
    out = {"type": (s.get("type") or "string").upper()}
    for k in KEEP:
        if k in s and k != "description":
            out[SNAKE.get(k, k)] = s[k]
    if desc:
        out["description"] = desc
    if s.get("nullable"):
        out["nullable"] = True
    if out["type"] == "OBJECT":
        out["properties"] = {n: to_gemini_schema(p, defs) for n, p in s["properties"].items()}
        if s.get("required"):
            out["required"] = [r for r in s["required"] if r in s["properties"]]
    if out["type"] == "ARRAY":
        out["items"] = to_gemini_schema(s.get("items") or {"type": "string"}, defs)
    return out


def to_declaration(tool) -> types.FunctionDeclaration:
    schema = tool.inputSchema or {}
    decl = {"name": tool.name, "description": (tool.description or "").strip()}
    if schema.get("properties"):
        decl["parameters"] = types.Schema(**to_gemini_schema(schema, schema.get("$defs", {})))
    return types.FunctionDeclaration(**decl)


def decode_args(value, s: dict, defs: dict):
    """Undo the free-object -> JSON string trick, guided by the original MCP schema."""
    s = _resolve(s, defs)
    if value is None:
        return None
    if _is_free_object(s):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except ValueError:
                return value  # let the server report the bad argument
        return value
    if s.get("type") == "object" and isinstance(value, dict):
        props = s.get("properties", {})
        return {k: decode_args(v, props[k], defs) if k in props else v for k, v in value.items()}
    if s.get("type") == "array" and isinstance(value, list):
        return [decode_args(v, s.get("items") or {}, defs) for v in value]
    return value


# ---------------------------------------------------------------- MCP side

def _result_payload(res) -> dict:
    text = "".join(getattr(c, "text", "") for c in res.content)
    data = res.structuredContent
    if data is None:
        try:
            data = json.loads(text)
        except ValueError:
            data = text
    if isinstance(data, dict) and set(data) == {"result"}:  # FastMCP wraps non-dict returns
        data = data["result"]
    blob = json.dumps(data, default=str)
    if len(blob) > RESULT_CHAR_LIMIT:
        data = blob[:RESULT_CHAR_LIMIT] + " …(truncated)"
    return {"error": data} if res.isError else {"result": data}


def _short(obj, n=220) -> str:
    s = json.dumps(obj, default=str, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + "…"


class Agent:
    def __init__(self, session: ClientSession, tools, instructions: str, route: str, max_steps: int):
        """route: 'auto' (tiered by phase), a tier name, or a literal model id."""
        self.session, self.route, self.max_steps = session, route, max_steps
        self.model = TIERS.get(route, route)
        self.tier = route if route in TIERS else "fixed"
        self.per_tier: dict[str, dict] = {}
        self.last_validate: dict | None = None  # {"fail": n, "hold": n} of the latest validate
        self.edited_since_validate = False
        self.read_model = False
        self.transcript = None  # open file: full (untruncated) tool calls for the record
        self.schemas = {t.name: t.inputSchema or {} for t in tools}
        self.tool = types.Tool(function_declarations=[to_declaration(t) for t in tools])
        self.client = genai.Client(api_key=api_key(), http_options=types.HttpOptions(timeout=240_000))
        level = os.environ.get("GEMINI_THINKING")  # minimal | low | medium | high (model default if unset)
        keep = os.environ.get("GEMINI_KEEP_THOUGHTS", "2")
        self.keep_thoughts = None if keep == "all" else max(int(keep), 1)
        self.config = types.GenerateContentConfig(
            system_instruction=f"{instructions.strip()}\n\n{RULES}",
            tools=[self.tool],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=types.ThinkingConfig(thinking_level=level) if level else None,
        )
        self.contents: list[types.Content] = []
        self.totals = {"in": 0, "out": 0, "sec": 0.0, "calls": 0}
        self.dirty = self.exported = self.committed_this_run = self.built_this_run = False
        self.pending_q: set[str] = set()

    async def _generate(self):
        """generate_content with backoff on 429 / 5xx / network errors and on empty or
        malformed candidates (Gemini occasionally returns no parts, or MALFORMED_FUNCTION_CALL)."""
        delay = 3.0
        for attempt in range(1, 7):
            try:
                resp = await self.client.aio.models.generate_content(
                    model=self.model, contents=self.contents, config=self.config)
            except errors.APIError as ex:
                code = getattr(ex, "code", None)
                if code == 404 and self.route == "auto" and self.model != TIERS["carbon"]:
                    print(f"   ({self.model} not available; falling back to {TIERS['carbon']})")
                    self.model, self.tier = TIERS["carbon"], "carbon"
                    continue
                if code not in (408, 429, 500, 502, 503, 504) or attempt == 6:
                    raise
                wait = _retry_delay(ex) or delay
                print(f"   (gemini {code}; retry {attempt} in {wait:.0f}s)")
            except (httpx.TransportError, asyncio.TimeoutError) as ex:
                if attempt == 6:
                    raise
                wait = delay
                print(f"   ({type(ex).__name__}; retry {attempt} in {wait:.0f}s)")
            else:
                cand = resp.candidates[0] if resp.candidates else None
                if cand is not None and cand.content is not None and cand.content.parts:
                    return resp
                reason = cand.finish_reason if cand else getattr(resp.prompt_feedback, "block_reason", None)
                if attempt >= 3 or (reason and "SAFETY" in str(reason)):
                    return resp  # give up; caller handles the empty turn
                wait = 1.0
                print(f"   (empty model turn, finish_reason={reason}; regenerating)")
            await asyncio.sleep(wait + random.uniform(0, 1))
            delay = min(delay * 2, 60)

    def _log(self, line: str) -> None:
        if self.transcript:
            self.transcript.write(line + "\n")
            self.transcript.flush()

    def _phase(self, step: int) -> str:
        v = self.last_validate
        if v and v["fail"]:
            return "fix"  # including the edits that follow, until a validate comes back without fails
        if step == 1:
            return "orient"
        if v and v.get("ok") and not self.edited_since_validate:
            return "wrapup"  # everything passes (requirements included): publishing only
        if not self.built_this_run:
            return "plan" if self.read_model else "orient"
        if self.pending_q or (v and v["hold"] and not self.edited_since_validate):
            return "signoff"
        if v and not self.edited_since_validate:
            return "wrapup"
        return "edit"

    def _route(self, step: int) -> None:
        if self.route != "auto":
            return
        phase = self._phase(step)
        tier = PHASE_TIER[phase]
        if tier != self.tier:
            print(f"   (route: {phase} -> {tier})")
        self.tier, self.model = tier, TIERS[tier]

    def _compact(self) -> None:
        """Swap thought signatures of older model turns for the placeholder (see SKIP_SIGNATURE)."""
        if self.keep_thoughts is None:
            return
        model_turns = [c for c in self.contents if c.role == "model"]
        for c in model_turns[:-self.keep_thoughts]:
            for p in c.parts or []:
                if p.thought_signature and p.thought_signature != SKIP_SIGNATURE:
                    p.thought_signature = SKIP_SIGNATURE if p.function_call else None

    def _log_usage(self, step: int, resp, dt: float) -> None:
        u = resp.usage_metadata
        if u is None:
            return
        p, o, th = u.prompt_token_count or 0, u.candidates_token_count or 0, u.thoughts_token_count or 0
        self.totals["in"] += p
        self.totals["out"] += o + th
        self.totals["sec"] += dt
        self.totals["calls"] += 1
        t = self.per_tier.setdefault(f"{self.tier}:{self.model}", {"calls": 0, "in": 0, "out": 0, "sec": 0.0})
        t["calls"] += 1
        t["in"] += p
        t["out"] += o + th
        t["sec"] += dt
        line = f"   · step {step} [{self.tier} {self.model}]: {dt:.1f}s, in {p} / out {o} / think {th} tok"
        print(line)
        self._log(line)

    async def call_tool(self, name: str, args: dict) -> dict:
        if name not in self.schemas:
            return {"error": f"unknown tool {name!r}; available: {sorted(self.schemas)}"}
        schema = self.schemas[name]
        args = decode_args(args or {}, schema, schema.get("$defs", {}))
        print(f"→ {name} {_short(args)}")
        try:
            res = await self.session.call_tool(name, args)
        except Exception as ex:  # transport / validation errors go back to the model
            payload = {"error": f"{type(ex).__name__}: {ex}"}
        else:
            payload = _result_payload(res)
        print(f"  {'✗' if 'error' in payload else '←'} {_short(payload.get('error', payload.get('result')))}")
        self._log(f"→ {name} {json.dumps(args, ensure_ascii=False, default=str)}")
        self._log(f"  {'✗' if 'error' in payload else '←'} "
                  f"{_short(payload.get('error', payload.get('result')), TRANSCRIPT_RESULT_CHARS)}")
        self._track(name, args, payload)
        return payload

    def _track(self, name: str, args: dict, payload: dict) -> None:
        """Book-keeping the loop uses to decide whether a text-only turn really is the end."""
        res = payload.get("result")
        if "error" in payload or (isinstance(res, dict) and res.get("ok") is False and name != "validate"):
            return
        if name == "validate" and isinstance(res, dict):
            c = res.get("counts") or {}
            self.last_validate = {"fail": c.get("fail", 0), "hold": c.get("hold", 0), "ok": bool(res.get("ok"))}
            self.edited_since_validate = False
        elif name == "get_model":
            self.read_model = True
        if name in MUTATING:
            self.dirty = True
            self.edited_since_validate = True
            self.built_this_run = True
        elif name == "commit_revision":
            self.dirty, self.exported = False, False
        elif name == "export_package":
            self.exported = True
        elif name == "ask_user" and isinstance(res, dict):
            qid = res.get("id") or res.get("question_id")
            if qid:
                self.pending_q.add(qid)
        elif name == "get_user_answer" and isinstance(res, dict) and res.get("answer"):
            self.pending_q.discard(args.get("question_id"))

    def _unfinished(self) -> str | None:
        """Why the workflow is not done yet (None = fine to stop)."""
        if self.pending_q:
            return None  # legitimately waiting for a human
        if self.dirty:
            return ("the design was changed but not committed. Continue with tool calls: validate, fix "
                    "every 'fail', ask_user for every 'hold', then set_next_steps, commit_revision and "
                    "export_package.")
        if self.committed_this_run and not self.exported:
            return "a revision was committed but export_package has not been called."
        return None

    async def run(self, prompt: str) -> str:
        self._log(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} user turn ===\n{prompt}\n")
        self.contents.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))
        nudges, recent = 0, []
        self.committed_this_run = self.built_this_run = False
        for step in range(1, self.max_steps + 1):
            self._route(step)
            self._compact()
            t0 = time.monotonic()
            resp = await self._generate()
            self._log_usage(step, resp, time.monotonic() - t0)
            cand = resp.candidates[0] if resp.candidates else None
            parts_in = (cand.content.parts if cand and cand.content else None) or []
            if parts_in:
                # keep the model turn verbatim: it carries the thought signatures Gemini needs back
                self.contents.append(cand.content)
            calls = [p.function_call for p in parts_in if p.function_call]
            text = "".join(p.text for p in parts_in if p.text and not p.thought).strip()
            if text:
                print(f"[step {step}] {text}")
                self._log(f"[step {step}] {text}")
            if not calls:
                why = self._unfinished() if nudges < MAX_NUDGES else None
                if why is None:
                    if not parts_in:
                        reason = cand.finish_reason if cand else getattr(resp.prompt_feedback, "block_reason", "no candidates")
                        print(f"[step {step}] model returned nothing ({reason})")
                    return text
                nudges += 1
                print(f"   (nudge {nudges}: {why.split('.')[0]})")
                self.contents.append(types.Content(role="user", parts=[types.Part.from_text(
                    text=f"Not finished: {why} Do not reply in prose; call the next tool now.")]))
                continue
            parts = []
            for fc in calls:
                args = dict(fc.args or {})
                payload = await self.call_tool(fc.name, args)
                if fc.name == "commit_revision" and "error" not in payload:
                    self.committed_this_run = True
                sig = (fc.name, json.dumps(args, sort_keys=True, default=str))
                recent = (recent + [sig])[-REPEAT_LIMIT:]
                part = types.Part.from_function_response(name=fc.name, response=payload)
                if fc.id:
                    part.function_response.id = fc.id
                parts.append(part)
            if len(recent) == REPEAT_LIMIT and len(set(recent)) == 1 and recent[0][0] != "get_user_answer":
                print(f"   (loop guard: {recent[0][0]} repeated {REPEAT_LIMIT}x)")
                parts.append(types.Part.from_text(text=(
                    f"You have called {recent[0][0]} with identical arguments {REPEAT_LIMIT} times in a row. "
                    "Repeating it will not change the result; try a different approach or stop.")))
                recent = []
            self.contents.append(types.Content(role="user", parts=parts))
        print(f"[stopped after --max-steps {self.max_steps}]")
        return ""

    def summary(self) -> str:
        t = self.totals
        lines = [f"[usage] {t['calls']} model calls, {t['sec']:.0f}s model time, "
                 f"{t['in']} prompt tok (cumulative), {t['out']} output+thinking tok"]
        for k, v in sorted(self.per_tier.items(), key=lambda kv: -kv[1]["sec"]):
            lines.append(f"[usage]   {k:28s} {v['calls']:3d} calls {v['sec']:7.1f}s  "
                         f"in {v['in']:8d}  out {v['out']:7d}")
        return "\n".join(lines)


def _retry_delay(ex) -> float | None:
    """Honour google.rpc.RetryInfo ('retryDelay': '17s') when the API sends one."""
    try:
        for d in (ex.details or {}).get("error", {}).get("details", []):
            if str(d.get("@type", "")).endswith("RetryInfo"):
                return min(float(str(d.get("retryDelay", "0s")).rstrip("s")), 90.0)
    except (AttributeError, TypeError, ValueError):
        pass
    return None


# ---------------------------------------------------------------- entry points

def load_dotenv(path: str | None = None) -> None:
    """Minimal .env reader (KEY=VALUE, optional 'export ', quotes, # comments).
    Never overrides a variable that is already set in the environment."""
    path = path or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    try:
        with open(path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        elif " #" in val:
            val = val.split(" #", 1)[0].rstrip()
        if key and key not in os.environ:
            os.environ[key] = val


def api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        sys.exit("error: set GEMINI_API_KEY (or GOOGLE_API_KEY); get one at https://aistudio.google.com/apikey")
    return key


async def _peek_messages(session: ClientSession) -> list[dict]:
    res = await session.call_tool("get_user_messages", {"mark_read": False})
    data = _result_payload(res).get("result") or {}
    return data.get("messages", []) if isinstance(data, dict) else []


async def main(a):
    async with streamablehttp_client(MCP_URL) as (r, w, _):
        async with ClientSession(r, w) as session:
            init = await session.initialize()
            tools = (await session.list_tools()).tools
            print(f"MCP {init.serverInfo.name} {init.serverInfo.version} at {MCP_URL}: {len(tools)} tools")

            if a.dry_run:
                decls = [to_declaration(t) for t in tools]
                tool = types.Tool(function_declarations=decls)
                types.GenerateContentConfig(tools=[tool])  # validates the whole tool payload
                for d in decls:
                    params = d.parameters.properties if d.parameters else {}
                    print(f"  {d.name:22s} {', '.join(params) or '-'}")
                print(f"OK: {len(decls)} FunctionDeclarations in one types.Tool; no model called")
                if a.verbose:
                    print(json.dumps(tool.model_dump(mode="json", exclude_none=True), indent=1))
                return

            route = a.model or (a.route if a.route_given else None) or os.environ.get("GEMINI_MODEL") or a.route
            agent = Agent(session, tools, init.instructions or "", route, a.max_steps)
            if a.transcript:
                agent.transcript = open(a.transcript, "a", encoding="utf-8")
            desc = ("auto: " + ", ".join(f"{ph}->{TIERS[t]}" for ph, t in PHASE_TIER.items())
                    if route == "auto" else agent.model)
            print(f"route {desc}; max {a.max_steps} steps per turn")
            agent._log(f"MCP {MCP_URL}; route {desc}")
            await agent.run(a.goal or DEFAULT_GOAL)  # the loop prints the model's text itself
            print(agent.summary())
            agent._log(agent.summary())

            while a.watch:
                await asyncio.sleep(a.interval)
                prompt = await _watch_prompt(session, agent)
                if prompt:
                    await agent.run(prompt)
                    print(agent.summary())
                    agent._log(agent.summary())


async def _watch_prompt(session: ClientSession, agent: Agent) -> str | None:
    """New viewer messages, or answers to questions the agent asked, as one user turn."""
    answered = []
    for qid in sorted(agent.pending_q):
        res = await session.call_tool("get_user_answer", {"question_id": qid, "wait_seconds": 0})
        data = _result_payload(res).get("result")
        if isinstance(data, dict) and data.get("answer"):
            agent.pending_q.discard(qid)
            answered.append(data)
    msgs, open_q = [], []
    if await _peek_messages(session):
        # consume them here so the model sees them exactly once
        res = await session.call_tool("get_user_messages", {"mark_read": True})
        data = _result_payload(res).get("result") or {}
        msgs, open_q = data.get("messages", []), data.get("open_questions", [])
    if not msgs and not answered:
        return None
    stamp = time.strftime("%H:%M:%S")
    print(f"\n[{stamp}] {len(msgs)} new message(s), {len(answered)} answer(s) from the viewer")
    out = []
    if msgs:
        out.append("New messages from the human in the viewer:\n" + json.dumps(msgs, indent=1))
    if answered:
        out.append("The human answered your question(s):\n" + json.dumps(answered, indent=1))
    if open_q:
        out.append("Still-open questions: " + json.dumps(open_q))
    out.append("Act on this: record new requirements, design, validate, fix every fail, ask_user for holds "
               "(record_approval only after an affirmative answer), then set_next_steps, commit_revision "
               "and export_package once everything passes. If nothing needs changing, reply briefly.")
    return "\n".join(out)


def cli():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("goal", nargs="?", help="what to do (default: follow the server workflow)")
    ap.add_argument("--watch", action="store_true", help="after finishing, keep acting on new viewer messages")
    ap.add_argument("--interval", type=float, default=4.0, help="--watch poll interval, seconds")
    ap.add_argument("--max-steps", type=int, default=60, help="model calls per turn before giving up")
    ap.add_argument("--route", default=None,
                    help=f"'auto' (tiered by workflow phase) or one tier of {'/'.join(TIER_ORDER)} "
                         f"for every call (default {DEFAULT_ROUTE})")
    ap.add_argument("--model", help="one literal Gemini model id for every call (overrides --route)")
    ap.add_argument("--transcript", metavar="FILE", help="append a full transcript (untruncated calls) to FILE")
    ap.add_argument("--dry-run", action="store_true", help="connect, list and convert tools; no model call")
    ap.add_argument("-v", "--verbose", action="store_true", help="with --dry-run: print the converted tool JSON")
    a = ap.parse_args()
    a.route_given = a.route is not None
    a.route = a.route or DEFAULT_ROUTE
    if a.route != "auto" and a.route not in TIERS:
        ap.error(f"--route must be auto or one of {', '.join(TIER_ORDER)}")
    load_dotenv()
    TIERS.update({t: os.environ.get(f"GEMINI_MODEL_{t.upper()}", TIERS[t]) for t in TIER_ORDER})
    global MCP_URL
    MCP_URL = os.environ.get("CAPTURE_STUDIO_MCP", MCP_URL)
    if not a.dry_run:
        api_key()  # fail fast, before connecting
    try:
        asyncio.run(main(a))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    cli()
