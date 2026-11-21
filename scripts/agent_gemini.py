"""Let a Google Gemini model drive Capture Studio through its MCP server.

    export GEMINI_API_KEY=...                              # or GOOGLE_API_KEY
    python scripts/agent_gemini.py                         # follow the server's own workflow
    python scripts/agent_gemini.py "add a 10 m2 office in the north-east corner"
    python scripts/agent_gemini.py --watch                 # then keep acting on viewer messages
    python scripts/agent_gemini.py --dry-run               # no model call: connect, list + convert tools

Env: GEMINI_MODEL (default below), GEMINI_API_KEY / GOOGLE_API_KEY,
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
import sys
import time

from google import genai
from google.genai import errors, types
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# Any Gemini model that supports function calling works; override with GEMINI_MODEL.
DEFAULT_MODEL = "gemini-3.8-flash"
MCP_URL = os.environ.get("CAPTURE_STUDIO_MCP", "http://127.0.0.1:8642/mcp")

DEFAULT_GOAL = (
    "Follow the Capture Studio workflow. Start with pipeline_status, get_user_messages and "
    "list_requirements. If there is no as-built model yet, run reconstruct_scan / fit_bim. "
    "Record each client wish with add_requirement, design it, then validate. Fix every 'fail' "
    "and re-validate. For every 'hold' ask the user with ask_user, wait with get_user_answer "
    "and only call record_approval after an affirmative human answer. When validation is ok: "
    "set_next_steps, commit_revision, export_package, then summarise in two lines."
)

RULES = (
    "Rules: never invent or assume user approval - only a human answer from get_user_answer counts. "
    "Fix every 'fail' issue; do not stop while any remain. Keep say() lines under 12 words. "
    "Call tools rather than describing what you would do."
)

RESULT_CHAR_LIMIT = 60_000  # per tool result sent back to the model


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
    def __init__(self, session: ClientSession, tools, instructions: str, model: str, max_steps: int):
        self.session, self.model, self.max_steps = session, model, max_steps
        self.schemas = {t.name: t.inputSchema or {} for t in tools}
        self.tool = types.Tool(function_declarations=[to_declaration(t) for t in tools])
        self.client = genai.Client(api_key=api_key())
        self.config = types.GenerateContentConfig(
            system_instruction=f"{instructions.strip()}\n\n{RULES}",
            tools=[self.tool],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            temperature=0.2,
        )
        self.contents: list[types.Content] = []

    async def _generate(self):
        for attempt in range(5):
            try:
                return await self.client.aio.models.generate_content(
                    model=self.model, contents=self.contents, config=self.config)
            except errors.APIError as ex:
                if getattr(ex, "code", None) in (429, 500, 502, 503, 504) and attempt < 4:
                    wait = 2 ** attempt * 3
                    print(f"   (gemini {ex.code}; retrying in {wait}s)")
                    await asyncio.sleep(wait)
                    continue
                raise

    async def call_tool(self, name: str, args: dict) -> dict:
        if name not in self.schemas:
            return {"error": f"unknown tool {name!r}"}
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
        return payload

    async def run(self, prompt: str) -> str:
        self.contents.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))
        for step in range(1, self.max_steps + 1):
            resp = await self._generate()
            cand = resp.candidates[0] if resp.candidates else None
            if cand is None or cand.content is None or not cand.content.parts:
                reason = cand.finish_reason if cand else getattr(resp.prompt_feedback, "block_reason", "no candidates")
                print(f"[step {step}] model returned nothing ({reason})")
                return ""
            # keep the model turn verbatim: it carries thought signatures Gemini needs back
            self.contents.append(cand.content)
            calls = [p.function_call for p in cand.content.parts if p.function_call]
            for p in cand.content.parts:
                if p.text and not p.thought:
                    print(f"[step {step}] {p.text.strip()}")
            if not calls:
                return resp.text or ""
            parts = []
            for fc in calls:
                payload = await self.call_tool(fc.name, dict(fc.args or {}))
                part = types.Part.from_function_response(name=fc.name, response=payload)
                if fc.id:
                    part.function_response.id = fc.id
                parts.append(part)
            self.contents.append(types.Content(role="user", parts=parts))
        print(f"[stopped after --max-steps {self.max_steps}]")
        return ""


# ---------------------------------------------------------------- entry points

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

            model = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
            agent = Agent(session, tools, init.instructions or "", model, a.max_steps)
            print(f"model {model}, max {a.max_steps} steps per turn")
            await agent.run(a.goal or DEFAULT_GOAL)  # the loop prints the model's text itself

            while a.watch:
                await asyncio.sleep(a.interval)
                if not await _peek_messages(session):
                    continue
                # consume them here so the model sees them exactly once
                res = await session.call_tool("get_user_messages", {"mark_read": True})
                data = _result_payload(res).get("result") or {}
                msgs = data.get("messages", [])
                if not msgs:
                    continue
                stamp = time.strftime("%H:%M:%S")
                print(f"\n[{stamp}] {len(msgs)} new message(s) from the viewer")
                prompt = ("New messages from the human in the viewer:\n"
                          + json.dumps(msgs, indent=1)
                          + "\nOpen questions: " + json.dumps(data.get("open_questions", []))
                          + "\nAct on them: record requirements, design, validate, fix fails, ask for holds, "
                            "then set_next_steps and commit_revision when everything passes.")
                await agent.run(prompt)


def cli():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("goal", nargs="?", help="what to do (default: follow the server workflow)")
    ap.add_argument("--watch", action="store_true", help="after finishing, keep acting on new viewer messages")
    ap.add_argument("--interval", type=float, default=4.0, help="--watch poll interval, seconds")
    ap.add_argument("--max-steps", type=int, default=60, help="model calls per turn before giving up")
    ap.add_argument("--dry-run", action="store_true", help="connect, list and convert tools; no model call")
    ap.add_argument("-v", "--verbose", action="store_true", help="with --dry-run: print the converted tool JSON")
    a = ap.parse_args()
    if not a.dry_run:
        api_key()  # fail fast, before connecting
    try:
        asyncio.run(main(a))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    cli()
